import Foundation
import MLXLMCommon
import MLXNN
import MLXVLM

@preconcurrency import MLX

// Qwen3-VL and Qwen3 restructured for yes/no scoring. Parameter names match the
// checkpoints, so the weights load unchanged. What differs from
// `MLXVLM.Qwen3VL` and why:
//
// - The score is `(E[yes] − E[no]) · h` in float32 at the scored position —
//   the `score_linear` of `qwen3_vl_reranker.py`. The stock path projects every
//   position onto 151,936 tokens, ~20 % of the language model's work, and
//   subtracts two float16 logits in the tens, where float16's spacing is
//   0.03–0.06.
// - The last decoder layer computes keys and values for every position and
//   everything else only at the scored one.
// - A shared prefix's keys and values are passed in, so a query's system
//   prompt, instruction, text and pictures run once for all its candidates.
//   That is most of the gain for picture queries: the stock path ran the query
//   image through the vision tower once per candidate.
// - M-RoPE tables once per pass instead of once per layer; image features
//   placed by gathers whose indices come from the token ids on the CPU, so the
//   pass never waits on the GPU.
// - The vision MLP uses the activation the checkpoint names
//   (`gelu_pytorch_tanh`). Stock mlx-swift-lm used the sigmoid approximation,
//   the largest single error against a float32 PyTorch reference.
//
// With float32 weights and activations the port matched Hugging Face
// transformers to 1e-4 logits on all 50 pairs of the experiment's workload.

/// Arrays derived from the weights. `Module` reflection treats `MLXArray`
/// properties as parameters; a value held in a plain class is not one.
final class DerivedArray {
    var value: MLXArray?
    init(_ value: MLXArray? = nil) { self.value = value }
}

/// Keys and values of one decoder layer, `[B, kvHeads, L, headDim]`.
struct LayerKeysValues {
    let k: MLXArray
    let v: MLXArray

    static func concatenate(_ a: LayerKeysValues, _ b: LayerKeysValues) -> LayerKeysValues {
        LayerKeysValues(k: concatenated([a.k, b.k], axis: 2), v: concatenated([a.v, b.v], axis: 2))
    }

    static func join(_ past: LayerKeysValues?, _ new: LayerKeysValues) -> LayerKeysValues {
        guard let past else { return new }
        return concatenate(past, new)
    }
}

/// Element-wise chains compiled into single kernels: 4 % of the whole
/// workload, scores bit-identical. A shapeless compile cannot trace the slices
/// of `rotateHalf`, so the rotation stays outside.
enum CompiledKernels {
    static let ropeCombine: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) {
        [($0[0] * $0[2]) + ($0[1] * $0[3])]
    }

    /// Vision variant: float32 tables, cast back to the activation's dtype.
    static let visionRopeCombine: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: true
    ) {
        [(($0[0] * $0[2]) + ($0[1] * $0[3])).asType($0[0].dtype)]
    }

    static let swiglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
        silu($0) * $1
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0..<half]], axis: -1)
    }
}

// MARK: - Model

final class FastQwen3VL: Module {
    @ModuleInfo(key: "vision_tower") var vision: FastVision
    @ModuleInfo(key: "language_model") var language: FastLanguage

    init(_ config: Qwen3VLConfiguration) {
        _vision.wrappedValue = FastVision(config.visionConfiguration)
        _language.wrappedValue = FastLanguage(config.textConfiguration)
    }

    /// After the weights are loaded.
    func prepare(yesToken: Int, noToken: Int) {
        vision.patchEmbed.prepare()
        language.prepareScoreHead(yes: yesToken, no: noToken)
    }
}

// MARK: - Vision

/// Consecutive images with the same patch count. Attention runs within each
/// image, batched across the images of a run.
struct VisionRun {
    let start: Int
    let images: Int
    let length: Int
}

final class FastPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d
    let flatWeight = DerivedArray()

    init(_ c: Qwen3VLConfiguration.VisionConfiguration) {
        let k = IntOrTriple([c.temporalPatchSize, c.patchSize, c.patchSize])
        _proj.wrappedValue = Conv3d(
            inputChannels: c.inChannels, outputChannels: c.hiddenSize, kernelSize: k, stride: k,
            bias: true)
    }

    /// Kernel equals stride, so the convolution is a matmul over flattened
    /// patches. The `[O, T, P, P, C]` kernel is reordered once to the
    /// `(C, T, P, P)` order `QwenVL.patchify` produces.
    func prepare() {
        let w = proj.weight
        flatWeight.value = w.transposed(0, 4, 1, 2, 3).reshaped(w.dim(0), -1).transposed()
        eval(flatWeight.value!)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        addMM(proj.bias!, x, flatWeight.value!)
    }
}

final class FastVisionAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(dim: Int, heads: Int) {
        self.heads = heads
        self.headDim = dim / heads
        self.scale = pow(Float(dim / heads), -0.5)
        _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
        _proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    /// `x`: `[N, dim]`; `cos`/`sin`: `[N, 1, headDim]` float32.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, runs: [VisionRun])
        -> MLXArray
    {
        let n = x.dim(0)
        let states = qkv(x).reshaped(n, 3, heads, headDim)
        func rotate(_ t: MLXArray) -> MLXArray {
            CompiledKernels.visionRopeCombine([t, CompiledKernels.rotateHalf(t), cos, sin])[0]
        }
        let q = rotate(states[0..., 0])
        let k = rotate(states[0..., 1])
        let v = states[0..., 2]

        var parts: [MLXArray] = []
        for run in runs {
            let range = run.start..<(run.start + run.images * run.length)
            func split(_ t: MLXArray) -> MLXArray {
                t[range].reshaped(run.images, run.length, heads, headDim).transposed(0, 2, 1, 3)
            }
            parts.append(
                MLXFast.scaledDotProductAttention(
                    queries: split(q), keys: split(k), values: split(v), scale: scale, mask: .none
                ).transposed(0, 2, 1, 3).reshaped(run.images * run.length, -1))
        }
        return proj(parts.count == 1 ? parts[0] : concatenated(parts))
    }
}

final class FastVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    let gelu: GELU

    init(dim: Int, hidden: Int, gelu: GELU.Approximation) {
        _fc1.wrappedValue = Linear(dim, hidden, bias: true)
        _fc2.wrappedValue = Linear(hidden, dim, bias: true)
        self.gelu = GELU(approximation: gelu)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class FastVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: FastVisionAttention
    @ModuleInfo(key: "mlp") var mlp: FastVisionMLP

    init(_ c: Qwen3VLConfiguration.VisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _attn.wrappedValue = FastVisionAttention(dim: c.hiddenSize, heads: c.numHeads)
        _mlp.wrappedValue = FastVisionMLP(
            dim: c.hiddenSize, hidden: c.intermediateSize,
            gelu: FastVision.approximation(c.hiddenAct))
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, runs: [VisionRun])
        -> MLXArray
    {
        let h = x + attn(norm1(x), cos: cos, sin: sin, runs: runs)
        return h + mlp(norm2(h))
    }
}

final class FastPatchMerger: Module {
    let hidden: Int
    let postShuffleNorm: Bool
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    let gelu = GELU()

    init(_ c: Qwen3VLConfiguration.VisionConfiguration, postShuffleNorm: Bool) {
        hidden = c.hiddenSize * c.spatialMergeSize * c.spatialMergeSize
        self.postShuffleNorm = postShuffleNorm
        _norm.wrappedValue = LayerNorm(
            dimensions: postShuffleNorm ? hidden : c.hiddenSize, eps: 1e-6)
        _fc1.wrappedValue = Linear(hidden, hidden)
        _fc2.wrappedValue = Linear(hidden, c.outHiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm(postShuffleNorm ? x.reshaped(-1, hidden) : x).reshaped(-1, hidden)
        return fc2(gelu(fc1(h)))
    }
}

/// Vision tower: attention within each image rather than one `[N, N]`-masked
/// attention over every image of the prompt; rotary tables once per call
/// rather than in each of the 24 blocks; bilinear position-embedding indices
/// computed on the CPU; patch embedding as one matmul.
final class FastVision: Module {
    let config: Qwen3VLConfiguration.VisionConfiguration
    let gridPerSide: Int
    let merge: Int

    @ModuleInfo(key: "patch_embed") var patchEmbed: FastPatchEmbed
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    @ModuleInfo(key: "blocks") var blocks: [FastVisionBlock]
    @ModuleInfo(key: "merger") var merger: FastPatchMerger
    @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [FastPatchMerger]

    let inverseFrequencies: DerivedArray

    init(_ c: Qwen3VLConfiguration.VisionConfiguration) {
        config = c
        gridPerSide = Int(sqrt(Double(c.numPositionEmbeddings)))
        merge = c.spatialMergeSize
        _patchEmbed.wrappedValue = FastPatchEmbed(c)
        _posEmbed.wrappedValue = Embedding(
            embeddingCount: c.numPositionEmbeddings, dimensions: c.hiddenSize)
        _blocks.wrappedValue = (0..<c.depth).map { _ in FastVisionBlock(c) }
        _merger.wrappedValue = FastPatchMerger(c, postShuffleNorm: false)
        _deepstackMergers.wrappedValue = c.deepstackVisualIndexes.map { _ in
            FastPatchMerger(c, postShuffleNorm: true)
        }
        // The stock VisionRotaryEmbedding(dimension: headDim / 2).
        let dim = c.hiddenSize / c.numHeads / 2
        inverseFrequencies = DerivedArray(
            1.0
                / pow(
                    MLXArray(Float(10_000)),
                    MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)))
    }

    /// The checkpoint's `hidden_act`. Qwen3-VL names `gelu_pytorch_tanh`,
    /// which is MLX's `.precise`.
    static func approximation(_ activation: String) -> GELU.Approximation {
        switch activation {
        case "gelu_pytorch_tanh", "gelu_new", "gelu_fast": .precise
        case "quick_gelu": .fast
        default: .none
        }
    }

    /// (row, column) of every patch, in the merged order of `QwenVL.patchify`.
    private func patchCoordinates(_ g: THW) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        out.reserveCapacity(g.product)
        for _ in 0..<g.t {
            for bh in 0..<(g.h / merge) {
                for bw in 0..<(g.w / merge) {
                    for ih in 0..<merge {
                        for iw in 0..<merge { out.append((bh * merge + ih, bw * merge + iw)) }
                    }
                }
            }
        }
        return out
    }

    private func tables(_ grids: [THW]) -> (pos: MLXArray, cos: MLXArray, sin: MLXArray) {
        let maxIndex = gridPerSide - 1
        var indices: [[Int32]] = Array(repeating: [], count: 4)
        var weights: [[Float]] = Array(repeating: [], count: 4)
        var rows: [Int32] = []
        var columns: [Int32] = []
        for g in grids {
            // Bilinear interpolation of the learned 48×48 grid.
            let hs = (0..<g.h).map { Float($0) * Float(maxIndex) / Float(max(1, g.h - 1)) }
            let ws = (0..<g.w).map { Float($0) * Float(maxIndex) / Float(max(1, g.w - 1)) }
            for (r, c) in patchCoordinates(g) {
                let hf = Int(hs[r])
                let wf = Int(ws[c])
                let hc = min(hf + 1, maxIndex)
                let wc = min(wf + 1, maxIndex)
                let dh = hs[r] - Float(hf)
                let dw = ws[c] - Float(wf)
                indices[0].append(Int32(hf * gridPerSide + wf))
                weights[0].append((1 - dh) * (1 - dw))
                indices[1].append(Int32(hf * gridPerSide + wc))
                weights[1].append((1 - dh) * dw)
                indices[2].append(Int32(hc * gridPerSide + wf))
                weights[2].append(dh * (1 - dw))
                indices[3].append(Int32(hc * gridPerSide + wc))
                weights[3].append(dh * dw)
                rows.append(Int32(r))
                columns.append(Int32(c))
            }
        }
        let dtype = posEmbed.weight.dtype
        var pos = MLXArray.zeros([rows.count, config.hiddenSize], dtype: dtype)
        for k in 0..<4 {
            pos =
                pos
                + posEmbed(MLXArray(indices[k]))
                * MLXArray(weights[k]).asType(dtype).expandedDimensions(axis: -1)
        }
        let inverse = inverseFrequencies.value!
        let frequencies = concatenated(
            [
                MLXArray(rows).asType(.float32).expandedDimensions(axis: 1) * inverse,
                MLXArray(columns).asType(.float32).expandedDimensions(axis: 1) * inverse,
            ], axis: -1)
        let embedding = concatenated([frequencies, frequencies], axis: -1)
            .expandedDimensions(axis: 1)
        return (pos, MLX.cos(embedding), MLX.sin(embedding))
    }

    static func runs(_ grids: [THW]) -> [VisionRun] {
        var runs: [VisionRun] = []
        var start = 0
        for g in grids {
            if let last = runs.last, last.length == g.product,
                last.start + last.images * last.length == start
            {
                runs[runs.count - 1] = VisionRun(
                    start: last.start, images: last.images + 1, length: last.length)
            } else {
                runs.append(VisionRun(start: start, images: 1, length: g.product))
            }
            start += g.product
        }
        return runs
    }

    /// Merged features `[N/4, out]` and the deepstack features.
    func callAsFunction(_ pixels: MLXArray, grids: [THW]) -> (MLXArray, [MLXArray]) {
        let t = tables(grids)
        var h = patchEmbed(pixels.asType(posEmbed.weight.dtype)) + t.pos
        let runs = Self.runs(grids)
        var deepstack: [MLXArray] = []
        for (i, block) in blocks.enumerated() {
            h = block(h, cos: t.cos, sin: t.sin, runs: runs)
            if let k = config.deepstackVisualIndexes.firstIndex(of: i) {
                deepstack.append(deepstackMergers[k](h))
            }
        }
        return (merger(h), deepstack)
    }
}

// MARK: - Language

/// Rotary embedding of one pass: M-RoPE tables `[B, 1, L, headDim]`, or plain
/// RoPE through the fused kernel from a start position.
struct RotaryTables {
    var cos: MLXArray?
    var sin: MLXArray?
    /// Plain RoPE: `[B]` position of each row's first token.
    var offset: MLXArray?
    var base: Float = 0

    func apply(_ x: MLXArray) -> MLXArray {
        if let offset {
            return MLXFast.RoPE(
                x, dimensions: x.dim(-1), traditional: false, base: base, scale: 1,
                offset: offset)
        }
        return CompiledKernels.ropeCombine([x, CompiledKernels.rotateHalf(x), cos!, sin!])[0]
    }

    /// The embedding at one position per row; `index`: `[B, 1]`.
    func rows(_ index: MLXArray) -> RotaryTables {
        if let offset {
            return RotaryTables(
                offset: offset + index.reshaped(-1).asType(offset.dtype), base: base)
        }
        let i = index.reshaped(-1, 1, 1, 1)
        return RotaryTables(cos: takeAlong(cos!, i, axis: 2), sin: takeAlong(sin!, i, axis: 2))
    }
}

final class FastAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: Qwen3VLConfiguration.TextConfiguration) {
        heads = c.numAttentionHeads
        kvHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        _wq.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rmsNormEps))
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rmsNormEps))
    }

    /// Keys and values of `x` (`[B, L, D]`, normed), after k-norm and RoPE.
    func keysValues(_ x: MLXArray, rope: RotaryTables) -> LayerKeysValues {
        let (b, l) = (x.dim(0), x.dim(1))
        let k = kNorm(wk(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        return LayerKeysValues(
            k: rope.apply(k), v: wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3))
    }

    func queries(_ x: MLXArray, rope: RotaryTables) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        return rope.apply(qNorm(wq(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3))
    }

    func output(
        _ q: MLXArray, _ keys: MLXArray, _ values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (b, l) = (q.dim(0), q.dim(2))
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: mask)
        return wo(o.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class FastMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ d: Int, _ h: Int) {
        _gate.wrappedValue = Linear(d, h, bias: false)
        _up.wrappedValue = Linear(d, h, bias: false)
        _down.wrappedValue = Linear(h, d, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(CompiledKernels.swiglu(gate(x), up(x)))
    }
}

final class FastDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attn: FastAttention
    @ModuleInfo(key: "mlp") var mlp: FastMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ c: Qwen3VLConfiguration.TextConfiguration) {
        _attn.wrappedValue = FastAttention(c)
        _mlp.wrappedValue = FastMLP(c.hiddenSize, c.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: Float(c.rmsNormEps))
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: Float(c.rmsNormEps))
    }

    /// Every position. Returns the output and this chunk's keys and values.
    func callAsFunction(
        _ x: MLXArray, rope: RotaryTables, past: LayerKeysValues?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> (MLXArray, LayerKeysValues) {
        let xn = inputNorm(x)
        let kv = attn.keysValues(xn, rope: rope)
        let all = LayerKeysValues.join(past, kv)
        let h = x + attn.output(attn.queries(xn, rope: rope), all.k, all.v, mask: mask)
        return (h + mlp(postNorm(h)), kv)
    }

    /// Causal mask for `queries` rows that follow `keys − queries` past keys.
    ///
    /// Explicit, never `.causal`, whenever there are past keys: in the MLX that
    /// mlx-swift bundles up to 0.31.6, the fused kernel with `.causal` leaves
    /// future keys unmasked for many shapes where the query is shorter than
    /// the keys and the key count is a multiple of 16 (1,244 of the shapes in
    /// a sweep; fixed upstream in ml-explore/mlx b41b349b6). One pair of the
    /// experiment's 50 scored 0.12 logits off with it.
    static func causalMask(queries: Int, keys: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let q = MLXArray(Int32(keys - queries)..<Int32(keys)).reshaped(-1, 1)
        let k = MLXArray(Int32(0)..<Int32(keys)).reshaped(1, -1)
        return .array(k .<= q)
    }

    /// Output at one position per row (`last`: `[B]`). Keys and values are
    /// needed for every position, the rest only for `last`.
    func lastPosition(_ x: MLXArray, rope: RotaryTables, past: LayerKeysValues?, last: MLXArray)
        -> MLXArray
    {
        let xn = inputNorm(x)
        let all = LayerKeysValues.join(past, attn.keysValues(xn, rope: rope))
        let index = last.reshaped(-1, 1, 1)
        let xl = takeAlong(x, index, axis: 1)
        let q = attn.queries(takeAlong(xn, index, axis: 1), rope: rope.rows(last.reshaped(-1, 1)))
        let pastLength = past?.k.dim(2) ?? 0
        let keyIndex = MLXArray(Int32(0)..<Int32(all.k.dim(2))).reshaped(1, 1, 1, -1)
        let limit = (last.asType(.int32) + Int32(pastLength)).reshaped(-1, 1, 1, 1)
        let h = xl + attn.output(q, all.k, all.v, mask: .array(keyIndex .<= limit))
        return h + mlp(postNorm(h))
    }
}

final class FastTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [FastDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: Qwen3VLConfiguration.TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0..<c.numHiddenLayers).map { _ in FastDecoderLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: Float(c.rmsNormEps))
    }
}

/// Qwen3's decoder for scoring; Qwen3-VL's language model and Qwen3-Reranker
/// alike. Requires tied embeddings: the score head is two rows of the input
/// embedding.
final class FastLanguage: Module {
    @ModuleInfo(key: "model") var model: FastTextModel
    let inverseFrequencies: [Float]
    /// Which of t, h, w each rotary frequency reads (interleaved M-RoPE).
    let sections: [Int]
    let scoreDirection = DerivedArray()

    init(_ c: Qwen3VLConfiguration.TextConfiguration) {
        _model.wrappedValue = FastTextModel(c)
        let inverse =
            1.0
            / pow(
                MLXArray(Float(c.ropeTheta)),
                MLXArray(stride(from: 0, to: c.headDim, by: 2)).asType(.float32)
                    / Float(c.headDim))
        inverseFrequencies = inverse.asArray(Float.self)
        let m = c.ropeScaling?.mropeSection ?? [24, 20, 20]
        let n = c.headDim / 2
        sections = (0..<n).map { i in
            for (dim, offset) in [(1, 1), (2, 2)]
            where i >= offset && i < min(m[dim] * 3, n) && (i - offset) % 3 == 0 {
                return dim
            }
            return 0
        }
    }

    func prepareScoreHead(yes: Int, no: Int) {
        let rows = MLXArray([Int32(yes), Int32(no)])
        let embedding: MLXArray
        if let q = model.embedTokens as? QuantizedEmbedding {
            embedding = dequantized(
                q.weight[rows], scales: q.scales[rows], biases: q.biases?[rows],
                groupSize: q.groupSize, bits: q.bits, mode: q.mode)
        } else {
            embedding = model.embedTokens.weight[rows]
        }
        let rows32 = embedding.asType(.float32)
        scoreDirection.value = rows32[0] - rows32[1]
        eval(scoreDirection.value!)
    }

    /// M-RoPE tables from positions `[3][B·L]`.
    func rope(_ positions: [[Int32]], batch: Int, length: Int, dtype: DType) -> RotaryTables {
        let n = inverseFrequencies.count
        var frequencies = [Float](repeating: 0, count: batch * length * n)
        for p in 0..<(batch * length) {
            for i in 0..<n {
                frequencies[p * n + i] = Float(positions[sections[i]][p]) * inverseFrequencies[i]
            }
        }
        let f = MLXArray(frequencies, [batch, 1, length, n])
        let embedding = concatenated([f, f], axis: -1)
        return RotaryTables(
            cos: MLX.cos(embedding).asType(dtype), sin: MLX.sin(embedding).asType(dtype))
    }

    enum Output {
        /// Keys and values of every layer: a prefix pass.
        case keysValues([LayerKeysValues])
        /// `logit(yes) − logit(no)` per row at `last`.
        case scores(MLXArray)
    }

    /// - Parameters:
    ///   - h: input embeddings `[B, L, D]`
    ///   - deepstack: added after layers 0, 1, 2 (`[B, L, D]`, zero on text)
    ///   - past: per-layer keys and values of a shared prefix
    ///   - last: `[B]` scored position of each row; `nil` for a prefix pass
    func callAsFunction(
        _ h: MLXArray, rope: RotaryTables, deepstack: [MLXArray], past: [LayerKeysValues]?,
        last: MLXArray?
    ) -> Output {
        var h = h
        var keysValues: [LayerKeysValues] = []
        let layers = model.layers
        let pastLength = past?.first?.k.dim(2) ?? 0
        let mask: MLXFast.ScaledDotProductAttentionMaskMode =
            pastLength == 0
            ? .causal
            : FastDecoderLayer.causalMask(queries: h.dim(1), keys: pastLength + h.dim(1))
        for (i, layer) in layers.enumerated() {
            let p = past?[i]
            if i == layers.count - 1 {
                guard let last else {
                    // A prefix only has to provide the last layer's keys and values.
                    keysValues.append(layer.attn.keysValues(layer.inputNorm(h), rope: rope))
                    return .keysValues(keysValues)
                }
                let out = model.norm(layer.lastPosition(h, rope: rope, past: p, last: last))
                return .scores(score(out.squeezed(axis: 1)))
            }
            let (out, kv) = layer(h, rope: rope, past: p, mask: mask)
            h = out
            if last == nil { keysValues.append(kv) }
            if i < deepstack.count { h = h + deepstack[i] }
        }
        preconditionFailure("a decoder without layers")
    }

    /// `[B, D]` → `[B]`: `(E[yes] − E[no]) · h` in float32.
    func score(_ h: MLXArray) -> MLXArray {
        matmul(h.asType(.float32), scoreDirection.value!)
    }
}

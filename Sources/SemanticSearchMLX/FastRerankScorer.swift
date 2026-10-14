import CoreImage
import CryptoKit
import Foundation
import MLXLMCommon
import MLXNN
import MLXVLM
import SemanticSearch
import SemanticSearchApple

@preconcurrency import MLX

// Scoring with the models in FastQwen3Reranker.swift: a query's shared prefix
// runs once, each candidate runs only its own tokens against the prefix's keys
// and values, one candidate at a time. Batching candidates was measured and
// lost: a batch-1 pass of a hundred tokens already fills the GPU, and padding
// adds work (53.2 s against 48.5 s on the experiment's workload).

/// Loads a checkpoint's safetensors into `model`, quantizing the layers the
/// checkpoint stores quantized. `nil` when the checkpoint has an untied
/// `lm_head`: the fast path scores with two rows of the input embedding, which
/// is the output projection only when the two are tied.
func loadScoringWeights(
    directory: URL, quantization: BaseConfiguration.PerLayerQuantization?, into model: Module
) throws -> Bool {
    var weights: [String: MLXArray] = [:]
    let files = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil)
    for url in files where url.pathExtension == "safetensors" {
        for (key, value) in try loadArrays(url: url) { weights[key] = value }
    }
    if weights.keys.contains(where: { $0.contains("lm_head") }) { return false }
    if let quantization {
        quantize(model: model) { path, _ in
            guard weights["\(path).scales"] != nil,
                let layer = quantization.quantization(layer: path)
            else { return nil }
            return (layer.groupSize, layer.bits, layer.mode)
        }
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    eval(model)
    return true
}

/// Vision-tower outputs per image, least recently used first out.
///
/// They depend on the pixels and the weights alone — not on the query or the
/// prompt — so a hit returns exactly what the tower would compute. A text
/// query over keyframes spends about half its time in the tower; a repeated
/// or refined query over the same pool skips it. Keyed by the SHA-256 of the
/// image's bytes, so a frame is found again whatever its URL.
final class VisionFeatureCache {
    struct Features {
        let merged: MLXArray
        let deepstack: [MLXArray]
        let grid: THW
    }

    private struct Entry {
        let features: Features
        let bytes: Int
        var used: Int
    }

    let capacity: Int
    private var entries: [String: Entry] = [:]
    private(set) var bytes = 0
    private(set) var hits = 0
    private(set) var misses = 0
    private var clock = 0

    init(capacity: Int) { self.capacity = max(0, capacity) }

    func get(_ key: String) -> Features? {
        clock += 1
        guard var entry = entries[key] else {
            misses += 1
            return nil
        }
        hits += 1
        entry.used = clock
        entries[key] = entry
        return entry.features
    }

    func put(_ key: String, _ features: Features) {
        let size = ([features.merged] + features.deepstack).reduce(0) { $0 + $1.nbytes }
        guard size <= capacity else { return }
        while bytes + size > capacity,
            let oldest = entries.min(by: { $0.value.used < $1.value.used })
        {
            bytes -= oldest.value.bytes
            entries.removeValue(forKey: oldest.key)
        }
        clock += 1
        entries[key] = Entry(features: features, bytes: size, used: clock)
        bytes += size
    }

    func removeAll() {
        entries.removeAll()
        bytes = 0
    }
}

/// An image as the fast scorer takes it: decoded, and keyed by its content.
struct ScoringImage {
    let image: CIImage
    let key: String

    /// Upright, as the embedder sees it (`ImageDecoding.decodeUpright`).
    init?(data: Data) {
        guard
            let decoded = try? ImageDecoding.decodeUpright(
                .encoded(data, contentType: "public.image"))
        else { return nil }
        image = CIImage(cgImage: decoded)
        key = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// One candidate's prompt, split where the candidate begins.
struct VisionRerankPrompt {
    /// System turn, instruction, query and `\n<Document>:` — the same for
    /// every candidate that kept the same query images.
    let prefix: String
    let queryImages: [ScoringImage]
    /// The candidate's images and text, and the assistant turn's opening.
    let suffix: String
    let documentImages: [ScoringImage]
}

/// Qwen3-VL-Reranker.
final class FastVisionRerankScorer {
    let model: FastQwen3VL
    let tokenizer: any Tokenizer
    let processor: Qwen3VLProcessor
    let merge: Int
    let imagePad: Int
    let placeholder = "<|vision_start|><|image_pad|><|vision_end|>"

    init(
        model: FastQwen3VL, tokenizer: any Tokenizer, processor: Qwen3VLProcessor,
        configuration: Qwen3VLConfiguration, imagePad: Int
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.merge = configuration.visionConfiguration.spatialMergeSize
        self.imagePad = imagePad
    }

    /// `logit(yes) − logit(no)` per prompt, in order.
    func score(_ prompts: [VisionRerankPrompt], cache: VisionFeatureCache) throws -> [Float] {
        var features: [String: VisionFeatureCache.Features] = [:]
        for image in prompts.flatMap({ $0.queryImages + $0.documentImages })
        where features[image.key] == nil {
            features[image.key] = try vision(image, cache: cache)
        }

        // Candidates share a prefix when they kept the same query images; the
        // pair's image budget can drop one for a candidate with a picture.
        var groups: [String: [Int]] = [:]
        var order: [String] = []
        for (index, prompt) in prompts.enumerated() {
            let key = prompt.prefix + "\u{0}" + prompt.queryImages.map(\.key).joined(separator: ",")
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(index)
        }

        var pending: [(Int, MLXArray)] = []
        for key in order {
            let members = groups[key]!
            let first = prompts[members[0]]
            let queryGrids = first.queryImages.map { features[$0.key]!.grid }
            let prefixTokens = try expand(first.prefix, grids: queryGrids)

            var candidates: [(index: Int, tokens: [Int], grids: [THW], images: [ScoringImage])] =
                []
            var shared = prefixTokens.count
            for index in members {
                let prompt = prompts[index]
                let images = prompt.queryImages + prompt.documentImages
                let grids = images.map { features[$0.key]!.grid }
                // The whole prompt is tokenized as one string, as the stock
                // processor does; the prefix is what it has in common with the
                // prefix tokenized alone. A merge across the boundary costs a
                // token of sharing, never a different tokenization.
                let tokens = try expand(prompt.prefix + prompt.suffix, grids: grids)
                shared = min(shared, commonPrefix(prefixTokens, tokens), tokens.count - 1)
                candidates.append((index, tokens, grids, images))
            }
            // Never split an image: back off to the start of a run of pads.
            while shared > 0, shared < prefixTokens.count, prefixTokens[shared] == imagePad,
                prefixTokens[shared - 1] == imagePad
            {
                shared -= 1
            }
            let sharedImages = runs(of: imagePad, in: prefixTokens[..<shared])

            var past: [LayerKeysValues]?
            if shared > 0 {
                let tokens = Array(prefixTokens[..<shared])
                let positions = Self.ropePositions(tokens, grids: queryGrids, merge: merge, pad: imagePad)
                let images = Array(first.queryImages.prefix(sharedImages))
                guard
                    case .keysValues(let kv) = forward(
                        tokens: tokens, positions: positions, images: images,
                        features: features, past: nil, scoring: false)
                else { throw FastScoringError.unexpectedOutput }
                asyncEval(kv.flatMap { [$0.k, $0.v] })
                past = kv
            }

            for candidate in candidates.sorted(by: { $0.tokens.count < $1.tokens.count }) {
                let positions = Self.ropePositions(
                    candidate.tokens, grids: candidate.grids, merge: merge, pad: imagePad)
                guard
                    case .scores(let s) = forward(
                        tokens: Array(candidate.tokens[shared...]),
                        positions: positions.map { Array($0[shared...]) },
                        images: Array(candidate.images.dropFirst(sharedImages)),
                        features: features, past: past, scoring: true)
                else { throw FastScoringError.unexpectedOutput }
                // Submitted without waiting, so the CPU builds the next
                // candidate while the GPU runs this one.
                asyncEval(s)
                pending.append((candidate.index, s))
            }
        }

        var scores = [Float](repeating: .nan, count: prompts.count)
        for (index, s) in pending { scores[index] = s.item(Float.self) }
        return scores
    }

    /// Features of one image, from the cache or the vision tower. Lazy: the
    /// tower runs when the pass that needs it is evaluated.
    private func vision(_ image: ScoringImage, cache: VisionFeatureCache) throws
        -> VisionFeatureCache.Features
    {
        if let hit = cache.get(image.key) { return hit }
        let (pixels, grid) = try processor.preprocess(images: [image.image], processing: nil)
        let (merged, deepstack) = model.vision(pixels, grids: [grid])
        let features = VisionFeatureCache.Features(merged: merged, deepstack: deepstack, grid: grid)
        cache.put(image.key, features)
        return features
    }

    /// Tokens with each `<|image_pad|>` expanded to the image's merged patches.
    private func expand(_ text: String, grids: [THW]) throws -> [Int] {
        let raw = tokenizer.encode(text: text, addSpecialTokens: false)
        var out: [Int] = []
        out.reserveCapacity(raw.count + grids.reduce(0) { $0 + $1.product / (merge * merge) })
        var image = 0
        for token in raw {
            if token == imagePad, image < grids.count {
                out.append(
                    contentsOf: repeatElement(imagePad, count: grids[image].product / (merge * merge)))
                image += 1
            } else {
                out.append(token)
            }
        }
        // A document whose text spells out a vision token would shift every
        // feature after it.
        let expected = grids.reduce(0) { $0 + $1.product / (merge * merge) }
        guard image == grids.count, out.count(where: { $0 == imagePad }) == expected else {
            throw FastScoringError.imageTokenMismatch
        }
        return out
    }

    private func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var n = 0
        while n < a.count, n < b.count, a[n] == b[n] { n += 1 }
        return n
    }

    private func runs(of token: Int, in tokens: ArraySlice<Int>) -> Int {
        var count = 0
        var previous: Int?
        for t in tokens {
            if t == token, previous != token { count += 1 }
            previous = t
        }
        return count
    }

    private func forward(
        tokens: [Int], positions: [[Int32]], images: [ScoringImage],
        features: [String: VisionFeatureCache.Features], past: [LayerKeysValues]?,
        scoring: Bool
    ) -> FastLanguage.Output {
        let length = tokens.count
        var gather = (0..<Int32(length)).map { $0 }
        var deepstackGather = [Int32](repeating: 0, count: length)
        var feature: Int32 = 0
        for (i, t) in tokens.enumerated() where t == imagePad {
            gather[i] = Int32(length) + feature
            deepstackGather[i] = 1 + feature
            feature += 1
        }

        let text = model.language.model.embedTokens(MLXArray(tokens.map(Int32.init), [1, length]))
        let dim = text.dim(-1)
        var h = text
        var deepstack: [MLXArray] = []
        if !images.isEmpty {
            let parts = images.map { features[$0.key]! }
            let merged = parts.count == 1 ? parts[0].merged : concatenated(parts.map(\.merged))
            h = take(
                concatenated([text.reshaped(length, dim), merged.asType(text.dtype)]),
                MLXArray(gather), axis: 0
            ).reshaped(1, length, dim)
            let zero = MLXArray.zeros([1, dim], dtype: text.dtype)
            deepstack = (0..<parts[0].deepstack.count).map { k in
                let level =
                    parts.count == 1
                    ? parts[0].deepstack[k] : concatenated(parts.map { $0.deepstack[k] })
                return take(
                    concatenated([zero, level.asType(text.dtype)]), MLXArray(deepstackGather),
                    axis: 0
                ).reshaped(1, length, dim)
            }
        }
        let rope = model.language.rope(positions, batch: 1, length: length, dtype: h.dtype)
        let last = scoring ? MLXArray([Int32(length - 1)]) : nil
        return model.language(h, rope: rope, deepstack: deepstack, past: past, last: last)
    }

    /// M-RoPE positions `[3][L]` (t, h, w), as `Qwen3VLLanguage.getRopeIndex`
    /// computes them: text advances by one; an image of merged grid (t, h, w)
    /// starting at `s` takes `(s + ti, s + hi, s + wi)`, and the text after it
    /// continues at the image's largest position + 1.
    static func ropePositions(_ tokens: [Int], grids: [THW], merge: Int, pad: Int) -> [[Int32]] {
        var positions = [[Int32]](
            repeating: [Int32](repeating: 0, count: tokens.count), count: 3)
        var next: Int32 = 0
        var i = 0
        var image = 0
        while i < tokens.count {
            if tokens[i] == pad, image < grids.count {
                let g = grids[image]
                image += 1
                let (gt, gh, gw) = (g.t, g.h / merge, g.w / merge)
                for t in 0..<gt {
                    for y in 0..<gh {
                        for x in 0..<gw where i < tokens.count {
                            positions[0][i] = next + Int32(t)
                            positions[1][i] = next + Int32(y)
                            positions[2][i] = next + Int32(x)
                            i += 1
                        }
                    }
                }
                next += Int32(max(gt, gh, gw))
            } else {
                for d in 0..<3 { positions[d][i] = next }
                next += 1
                i += 1
            }
        }
        return positions
    }
}

/// Qwen3-Reranker (text).
final class FastTextRerankScorer {
    let language: FastLanguage
    let ropeBase: Float

    init(language: FastLanguage, ropeBase: Float) {
        self.language = language
        self.ropeBase = ropeBase
    }

    /// `logit(yes) − logit(no)` per row of prompt tokens, in order. The rows'
    /// longest common prefix — scaffolding, instruction and query — runs once.
    func score(_ rows: [[Int]]) throws -> [Float] {
        guard let shortest = rows.map(\.count).min(), shortest > 0 else {
            throw FastScoringError.emptyPrompt
        }
        var shared = 0
        while shared < shortest - 1, rows.allSatisfy({ $0[shared] == rows[0][shared] }) {
            shared += 1
        }
        var past: [LayerKeysValues]?
        if shared > 0 {
            guard
                case .keysValues(let kv) = forward(
                    Array(rows[0][..<shared]), start: 0, past: nil, scoring: false)
            else { throw FastScoringError.unexpectedOutput }
            asyncEval(kv.flatMap { [$0.k, $0.v] })
            past = kv
        }
        var pending: [(Int, MLXArray)] = []
        for (index, row) in rows.enumerated().sorted(by: { $0.element.count < $1.element.count }) {
            guard
                case .scores(let s) = forward(
                    Array(row[shared...]), start: shared, past: past, scoring: true)
            else { throw FastScoringError.unexpectedOutput }
            asyncEval(s)
            pending.append((index, s))
        }
        var scores = [Float](repeating: .nan, count: rows.count)
        for (index, s) in pending { scores[index] = s.item(Float.self) }
        return scores
    }

    private func forward(_ tokens: [Int], start: Int, past: [LayerKeysValues]?, scoring: Bool)
        -> FastLanguage.Output
    {
        let h = language.model.embedTokens(MLXArray(tokens.map(Int32.init), [1, tokens.count]))
        let rope = RotaryTables(offset: MLXArray([Int32(start)]), base: ropeBase)
        let last = scoring ? MLXArray([Int32(tokens.count - 1)]) : nil
        return language(h, rope: rope, deepstack: [], past: past, last: last)
    }
}

enum FastScoringError: Error, CustomStringConvertible {
    case unexpectedOutput
    case emptyPrompt
    case imageTokenMismatch

    var description: String {
        switch self {
        case .unexpectedOutput: "the scoring model returned the wrong kind of output"
        case .emptyPrompt: "an empty reranker prompt"
        case .imageTokenMismatch: "the prompt's image tokens do not match its images"
        }
    }
}

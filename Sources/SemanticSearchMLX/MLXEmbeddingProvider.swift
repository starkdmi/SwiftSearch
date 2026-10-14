import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import MLXEmbedders
import MLXLMCommon
import MLXVLM
import SemanticSearch
import SemanticSearchApple

@preconcurrency import MLX

private typealias TextEmbedderContainer = EmbedderModelContainer
private typealias VLMContainer = MLXLMCommon.ModelContainer

/// MLX-backed embedding provider for the Qwen3 families.
///
/// Two details make it correct, and both are silent when wrong:
///
/// - **Attention masks come from true sequence lengths, not token identity.**
///   Qwen3's tokenizer uses PAD == EOS. Building the mask with `token != padToken`
///   masks out the real EOS token, and last-token pooling then reads a padding
///   position. The vectors still look plausible and retrieval quietly degrades.
/// - **Last-token pooling over hidden states.** Qwen3-Embedding pools the final
///   hidden state at the last *valid* token. Mean pooling, or the last position
///   of a padded batch, produces a different space — one that still has a
///   cosine similarity and no relationship to the model's training objective.
///
/// Both text-only and vision-language checkpoints load through here. The
/// backend is chosen by what the weights turn out to be rather than by
/// configuration, so a caller swapping `Qwen3-Embedding-0.6B` for
/// `Qwen3-VL-Embedding-2B` changes one string.
public actor MLXEmbeddingProvider: EmbeddingProvider {

    private enum Backend {
        case text(TextEmbedderContainer)
        case vlm(VLMContainer)
    }

    public struct Configuration: Sendable {
        public var model: ModelReference
        /// Instruction prefixed to queries when none is supplied per call.
        public var defaultQueryInstruction: String
        /// Instruction prefixed to documents at index time. Part of the space's
        /// identity — changing it invalidates every vector in the space.
        public var documentInstruction: String?
        /// Dimensions the weights produce. Read from the first embedding when
        /// left `nil`, which costs one extra inference on `prepare`.
        public var dimensions: Int?
        /// Matryoshka prefixes this checkpoint supports. Empty disables
        /// dimension reduction, which is the safe default: truncating a
        /// non-Matryoshka vector yields something with no meaning.
        public var matryoshkaDimensions: [Int]
        public var maximumBatchSize: Int
        public var maximumInputTokens: Int
        /// Most frames one video input is reduced to. The reference
        /// implementation samples 1 frame per second up to 64; the Qwen3-VL
        /// processor in mlx-swift-lm samples 2 per second with no ceiling, so a
        /// two-minute clip would arrive as 270 frames and a feature-length file
        /// as thousands, each several hundred vision tokens. Longer videos are
        /// sampled evenly across their length and reported as truncated.
        public var maximumVideoFrames: Int
        /// Identity recorded in the descriptor. Defaults to the model
        /// reference, which is not portable for a local directory; set it
        /// explicitly when two machines hold the same weights in different
        /// places.
        public var modelIdentity: String?
        public var weightsVariant: String?
        /// Whether the checkpoint is a vision-language model. Decides the
        /// capabilities reported before the weights are loaded; `nil` guesses
        /// from the model id. Set it when the weights live in a directory whose
        /// name does not say "VL".
        public var multimodal: Bool?
        public var runtime: MLXRuntime.Configuration

        public init(
            model: ModelReference,
            defaultQueryInstruction: String = MLXEmbeddingProvider.qwen3DefaultQueryInstruction,
            documentInstruction: String? = nil,
            dimensions: Int? = nil,
            matryoshkaDimensions: [Int] = [],
            maximumBatchSize: Int = 8,
            maximumInputTokens: Int = 512,
            maximumVideoFrames: Int = 64,
            modelIdentity: String? = nil,
            weightsVariant: String? = nil,
            multimodal: Bool? = nil,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) {
            self.model = model
            self.defaultQueryInstruction = defaultQueryInstruction
            self.documentInstruction = documentInstruction
            self.dimensions = dimensions
            self.matryoshkaDimensions = matryoshkaDimensions.filter { $0 > 0 }.sorted()
            self.maximumBatchSize = max(1, maximumBatchSize)
            self.maximumInputTokens = max(1, maximumInputTokens)
            self.maximumVideoFrames = max(1, maximumVideoFrames)
            self.modelIdentity = modelIdentity
            self.weightsVariant = weightsVariant
            self.multimodal = multimodal
            self.runtime = runtime
        }
    }

    /// Qwen3-Embedding's own default query prompt.
    public static let qwen3DefaultQueryInstruction =
        "Given a web search query, retrieve relevant passages that answer the query"
    /// Qwen3-VL-Embedding's system prompt.
    public static let qwen3VLDefaultInstruction = "Represent the user's input."

    /// Bumped whenever prompt wire format, pooling or image preprocessing
    /// changes, because any of those changes every vector the provider produces.
    /// v2: no LayerNorm before L2.
    private static let preprocessingVersion = "qwen3.v2"

    /// The vision-language backend's version also names its video sampling:
    /// changing the frame ceiling changes every long video's vector.
    /// v3: images upright by their EXIF orientation, their stored values, alpha
    /// on white (`ImageDecoding.decodeUpright`).
    private var effectivePreprocessingVersion: String {
        guard isVisual else { return Self.preprocessingVersion }
        return "qwen3-vl.v3.genprompt.fps1.frames\(configuration.maximumVideoFrames)"
    }

    private var isVisual: Bool {
        loadedVisual ?? configuration.multimodal ?? Self.looksMultimodal(configuration.model)
    }

    private let configuration: Configuration
    private var backend: Backend?
    private var resolvedDimensions: Int
    /// What the weights turned out to be, remembered across `release()`.
    private var loadedVisual: Bool?

    public init(configuration: Configuration) {
        self.configuration = configuration
        self.resolvedDimensions = configuration.dimensions ?? 0
    }

    public init(model: ModelReference) {
        self.init(configuration: Configuration(model: model))
    }

    // MARK: - Contract

    public var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: configuration.modelIdentity ?? configuration.model.identity,
            weightsVariant: configuration.weightsVariant,
            // The tokenizer ships with the weights, so its version is the
            // weights' version. A checkpoint that changed its tokenizer without
            // changing its id would defeat this, and so would every other
            // identity scheme short of hashing the files.
            tokenizerVersion: configuration.modelIdentity ?? configuration.model.identity,
            preprocessingVersion: effectivePreprocessingVersion,
            documentInstruction: configuration.documentInstruction,
            pooling: "last-token",
            normalized: true,
            dimensions: resolvedDimensions)
    }

    public var capabilities: EmbeddingCapabilities {
        // The same answer loaded or not. The engine reads capabilities before
        // it makes a model resident, and a released model — sequential
        // residency releases one on nearly every query — must not look like a
        // text-only one.
        return EmbeddingCapabilities(
            acceptsText: true,
            acceptsImages: isVisual,
            acceptsVideo: isVisual,
            // Qwen3-VL-Embedding puts text and vision in one space, which is
            // what makes image-to-text retrieval work at all.
            sharedTextVisualSpace: isVisual,
            supportsDimensionReduction: !configuration.matryoshkaDimensions.isEmpty,
            supportedDimensions: configuration.matryoshkaDimensions,
            maximumBatchSize: configuration.maximumBatchSize,
            maximumInputTokens: configuration.maximumInputTokens)
    }

    public var isPrepared: Bool { backend != nil }

    public func prepare() async throws {
        guard backend == nil else { return }
        MLXRuntime.apply(configuration.runtime)

        // EmbeddingGemma 2 is registered with `EmbedderModelFactory`, so it
        // would load here — under Qwen's prompts and a last-token descriptor
        // for a vector the model has already pooled, projected and
        // normalised. Every vector would be wrong and look fine. A local
        // directory is checked before anything loads, a Hub download once its
        // files are on disk.
        if let directory = configuration.model.directoryURL,
            let refusal = Self.embeddingGemma2Refusal(at: directory)
        {
            throw SemanticSearchError.modelUnavailable(
                configuration.model.identity, underlying: refusal)
        }

        // Text embedders first. A VLM checkpoint fails that load with a
        // specific error, which is a cheaper and more reliable test than
        // guessing from the model id.
        do {
            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: RefusingEmbeddingGemma2(
                    MLXRuntime.downloader(configuration.runtime, for: .embedder)),
                using: MLXRuntime.tokenizerLoader,
                configuration: modelConfiguration())
            backend = .text(container)
            loadedVisual = false
        } catch let error as ModelFactoryError {
            guard case .unsupportedModelType = error else {
                throw SemanticSearchError.modelUnavailable(
                    configuration.model.identity, underlying: error)
            }
            do {
                let container = try await VLMModelFactory.shared.loadContainer(
                    from: MLXRuntime.downloader(configuration.runtime, for: .languageModel),
                    using: MLXRuntime.tokenizerLoader,
                    configuration: modelConfiguration())
                backend = .vlm(container)
                loadedVisual = true
            } catch {
                throw SemanticSearchError.modelUnavailable(
                    configuration.model.identity, underlying: error)
            }
        } catch {
            throw SemanticSearchError.modelUnavailable(
                configuration.model.identity, underlying: error)
        }

        if resolvedDimensions == 0 {
            // One probe so the descriptor is complete. The engine calls
            // `prepare` before registering a space whose provider reports zero
            // dimensions, so this runs before the space is recorded; setting
            // `Configuration.dimensions` avoids both the probe and the load.
            let probe = try await embed([EmbeddingInput(text: "dimension probe")])
            resolvedDimensions = probe.first?.vector.count ?? 0
            guard resolvedDimensions > 0 else {
                throw SemanticSearchError.providerReturnedNoOutput(
                    configuration.model.identity)
            }
        }
    }

    public func release() async {
        backend = nil
        MLXRuntime.releaseCache()
    }

    public func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard let backend else {
            throw SemanticSearchError.providerNotPrepared(configuration.model.identity)
        }
        guard !inputs.isEmpty else { return [] }

        switch backend {
        case .text(let container):
            guard !inputs.contains(where: { !$0.images.isEmpty || !$0.videos.isEmpty }) else {
                throw SemanticSearchError.unsupportedModality(
                    itemID: "query", kind: .image, provider: configuration.model.identity)
            }
            return try await embedText(inputs, container: container)

        case .vlm(let container):
            var outputs: [EmbeddingOutput] = []
            outputs.reserveCapacity(inputs.count)
            for input in inputs {
                try Task.checkCancellation()
                outputs.append(try await embedMultimodal(input, container: container))
            }
            return outputs
        }
    }

    // MARK: - Text backend

    private func embedText(_ inputs: [EmbeddingInput], container: TextEmbedderContainer)
        async throws -> [EmbeddingOutput]
    {
        let prompts = inputs.map { formatted($0) }
        let indexed = prompts.enumerated().filter { !$0.element.isEmpty }
        guard !indexed.isEmpty else {
            return Array(repeating: EmbeddingOutput(vector: []), count: inputs.count)
        }

        let limit = configuration.maximumInputTokens
        let rows = await container.perform {
            context -> [(vector: [Float], tokens: Int, truncated: Bool)] in
            let model = context.model
            let tokenizer = context.tokenizer
            let pooling = context.pooling

            var tokenized: [[Int]] = []
            var truncations: [Bool] = []
            tokenized.reserveCapacity(indexed.count)
            for (_, prompt) in indexed {
                var tokens = tokenizer.encode(text: prompt, addSpecialTokens: true)
                let truncated = tokens.count > limit
                if truncated {
                    // Keep the head and the final token. Qwen3 pools the last
                    // position, so dropping the tail outright would pool a word
                    // from the middle of the passage.
                    let last = tokens[tokens.count - 1]
                    tokens = Array(tokens.prefix(limit - 1)) + [last]
                }
                tokenized.append(tokens)
                truncations.append(truncated)
            }

            let padToken = tokenizer.eosTokenId ?? 0
            let maxLength = max(1, tokenized.map(\.count).max() ?? 1)

            let padded: [MLXArray] = tokenized.map { tokens in
                MLXArray(
                    (tokens + Array(repeating: padToken, count: maxLength - tokens.count))
                        .map(Int32.init))
            }
            // Masks are built from true sequence lengths. Qwen3's PAD is its
            // EOS, so `token != padToken` would mask the real EOS and pooling
            // would then read a padding position.
            let maskRows: [MLXArray] = tokenized.map { tokens in
                MLXArray(
                    Array(repeating: Int32(1), count: tokens.count)
                        + Array(repeating: Int32(0), count: maxLength - tokens.count))
            }

            let inputIDs = stacked(padded)
            let attentionMask = stacked(maskRows) .== Int32(1)
            let outputs = model(
                inputIDs,
                positionIds: nil,
                tokenTypeIds: MLXArray.zeros(like: inputIDs),
                attentionMask: attentionMask)
            // Last token, then L2, as the model's sentence-transformers
            // modules run it. The LayerNorm MLXEmbedders' README passes here
            // is not part of Qwen3-Embedding: before L2 it only takes each
            // vector's mean out, which moved every vector from the reference.
            var pooled = pooling(
                outputs, mask: attentionMask, normalize: true, applyLayerNorm: false)

            if pooled.ndim == 1 { pooled = expandedDimensions(pooled, axis: 0) }
            eval(pooled)

            let shape = pooled.shape
            if shape.count == 3 {
                // Some embedders have no pooling configuration and return
                // [B, T, D]. Qwen3 wants the last valid token of each row.
                let batch = shape[0]
                let tokens = shape[1]
                let dimension = shape[2]
                guard batch > 0, tokens > 0, dimension > 0 else { return [] }
                let flat = pooled.asArray(Float.self)
                guard flat.count == batch * tokens * dimension else { return [] }
                return (0..<batch).map { row in
                    let length = min(tokens, max(1, tokenized[row].count))
                    let start = ((row * tokens) + (length - 1)) * dimension
                    return (
                        vector: Self.l2Normalize(Array(flat[start..<(start + dimension)])),
                        tokens: tokenized[row].count,
                        truncated: truncations[row]
                    )
                }
            }

            guard shape.count == 2, shape[0] > 0, shape[1] > 0 else { return [] }
            let batch = shape[0]
            let dimension = shape[1]
            let flat = pooled.asArray(Float.self)
            guard flat.count == batch * dimension else { return [] }
            return (0..<batch).map { row in
                let start = row * dimension
                return (
                    vector: Array(flat[start..<(start + dimension)]),
                    tokens: tokenized[row].count,
                    truncated: truncations[row]
                )
            }
        }

        guard rows.count == indexed.count else {
            throw SemanticSearchError.providerReturnedNoOutput(configuration.model.identity)
        }

        var outputs = Array(repeating: EmbeddingOutput(vector: []), count: inputs.count)
        for (offset, entry) in indexed.enumerated() {
            let row = rows[offset]
            outputs[entry.offset] = EmbeddingOutput(
                vector: Self.reduced(
                    row.vector, to: inputs[entry.offset].requestedDimensions,
                    supported: configuration.matryoshkaDimensions),
                consumedTokens: row.tokens,
                wasTruncated: row.truncated)
        }
        return outputs
    }

    // MARK: - Vision-language backend

    private func embedMultimodal(_ input: EmbeddingInput, container: VLMContainer) async throws
        -> EmbeddingOutput
    {
        guard !input.isEmpty else { return EmbeddingOutput(vector: []) }

        let images = try input.images.map { try Self.userInputImage($0) }
        var videos: [UserInput.Video] = []
        var videoTruncated = false
        for url in input.videos {
            let sampled = try await Self.sampledVideo(
                url, maximumFrames: configuration.maximumVideoFrames)
            videos.append(sampled.video)
            videoTruncated = videoTruncated || sampled.truncated
        }
        let (text, truncated) = try await Self.budgeted(
            Self.normalized(input.text), limit: configuration.maximumInputTokens,
            container: container)

        var output = try await embedMultimodal(
            text: text, images: images, videos: videos, role: input.role, container: container)
        output.wasTruncated = truncated || videoTruncated
        return output
    }

    /// Frames for one video, at the reference 1 fps up to `maximumFrames`,
    /// and evenly spread across the whole length beyond it.
    ///
    /// Handed to the processor as decoded frames rather than a URL: given a
    /// URL it samples at 2 fps with no ceiling, and given frames it keeps at
    /// most as many as it was handed.
    static func sampledVideo(_ url: URL, maximumFrames: Int) async throws
        -> (video: UserInput.Video, truncated: Bool)
    {
        if url.isFileURL, !FileManager.default.fileExists(atPath: url.path) {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        let asset = AVURLAsset(url: url)
        let seconds = try await asset.load(.duration).seconds
        guard seconds.isFinite, seconds > 0 else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "video has no duration")
        }

        let times = videoSampleTimes(duration: seconds, maximumFrames: maximumFrames)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var frames: [UserInput.VideoFrame] = []
        frames.reserveCapacity(times.count)
        for await result in generator.images(
            for: times.map { CMTime(seconds: $0, preferredTimescale: 600) })
        {
            try Task.checkCancellation()
            if case .success(_, let image, let actual) = result {
                let frame = CIImage(
                    cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                frames.append(UserInput.VideoFrame(image: .ciImage(frame), timeStamp: actual))
            }
        }
        guard !frames.isEmpty else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "no video frames could be decoded")
        }
        let natural = Int(seconds.rounded())
        return (.frames(frames), natural > maximumFrames)
    }

    /// Sample times: 1 per second, or `maximumFrames` spread evenly from the
    /// first frame to the last when that would be more.
    static func videoSampleTimes(duration: TimeInterval, maximumFrames: Int) -> [TimeInterval] {
        let count = max(1, min(maximumFrames, Int(duration.rounded())))
        guard count > 1 else { return [0] }
        // Stop short of the very end: a request at the duration itself often
        // decodes nothing.
        let last = max(0, duration - 0.05)
        return (0..<count).map { last * Double($0) / Double(count - 1) }
    }

    /// Cut `text` to `limit` tokens.
    ///
    /// The text backend truncates token ids directly; this backend hands a
    /// string to the chat template, so the budget is applied by encoding,
    /// keeping the head and decoding it back. Vision tokens are not counted —
    /// their budget is the processor's resolution limit — so this bounds what a
    /// long prepared text can cost, not the full sequence.
    private static func budgeted(_ text: String, limit: Int, container: VLMContainer) async throws
        -> (text: String, truncated: Bool)
    {
        guard !text.isEmpty else { return (text, false) }
        return await container.perform { context in
            let tokens = context.tokenizer.encode(text: text, addSpecialTokens: false)
            guard tokens.count > limit else { return (text, false) }
            return (
                context.tokenizer.decode(tokenIds: Array(tokens.prefix(limit))), true
            )
        }
    }

    private func embedMultimodal(
        text: String, images: [UserInput.Image], videos: [UserInput.Video],
        role: EmbeddingRole, container: VLMContainer
    ) async throws -> EmbeddingOutput {
        let primary = Self.userInput(
            text: text, images: images, videos: videos,
            instruction: instruction(for: role))

        do {
            return try await pool(primary, container: container)
        } catch {
            guard Self.isPlaceholderMismatch(error), !images.isEmpty || !videos.isEmpty else {
                throw error
            }

            // Some converted Qwen3-VL checkpoints ship chat templates that do
            // not emit the expected number of `<|image_pad|>` / `<|video_pad|>`
            // placeholders, and the processor then rejects the frame count.
            // Writing the placeholders into a plain prompt bypasses the
            // template; failing that, a raw message list bypasses the chat
            // wrapper as well.
            let prompt = Self.explicitVisionPlaceholders(
                text, images: images.count, videos: videos.count)
            do {
                return try await pool(
                    UserInput(
                        prompt: .text(prompt), images: images, videos: videos,
                        additionalContext: ["add_generation_prompt": true]),
                    container: container)
            } catch {
                guard Self.isPlaceholderMismatch(error) else { throw error }
                return try await pool(
                    UserInput(
                        messages: [["role": "user", "content": prompt]],
                        images: images, videos: videos,
                        additionalContext: ["add_generation_prompt": true]),
                    container: container)
            }
        }
    }

    private func pool(_ input: consuming UserInput, container: VLMContainer) async throws
        -> EmbeddingOutput
    {
        let box = UncheckedBox(input)
        let prepared = try await container.prepare(input: box.take())
        let supported = configuration.matryoshkaDimensions

        return try await container.perform(nonSendable: prepared) { context, prepared in
            let prepared = Self.endingOnEndOfText(prepared, tokenizer: context.tokenizer)
            let cache = try context.model.newCache(parameters: nil)
            let hiddenStates: MLXArray
            let mask: MLXArray?

            // One forward over the whole prompt, as 2.x ran it. 3.x prefills
            // a prompt longer than one step in windows and returns only the
            // last window's states.
            let oneForward = PrefillParameters(stepSize: .max)
            switch try context.model.prepare(
                prepared, cache: cache, state: nil, prefill: oneForward)
            {
            case .logits(let output):
                guard let states = output.hiddenStates else {
                    throw SemanticSearchError.modelUnavailable(
                        "multimodal embedding model did not expose hidden states",
                        underlying: nil)
                }
                hiddenStates = states
                mask = prepared.text.mask
            case .tokens(let textInput):
                let output = context.model(
                    textInput, cache: cache.isEmpty ? nil : cache, state: nil)
                guard let states = output.hiddenStates else {
                    throw SemanticSearchError.modelUnavailable(
                        "multimodal embedding model did not expose hidden states",
                        underlying: nil)
                }
                hiddenStates = states
                mask = textInput.mask
            }

            let pooled = try Self.lastTokenEmbedding(hiddenStates: hiddenStates, mask: mask)
            return EmbeddingOutput(
                vector: Self.reduced(pooled.vector, to: nil, supported: supported),
                consumedTokens: pooled.sequenceLength,
                wasTruncated: false)
        }
    }

    /// The prompt as Qwen3-VL-Embedding pools it: ending on `<|endoftext|>`,
    /// which Hugging Face's processor appends with `add_special_tokens=True`,
    /// after a final `\n\n` token is reduced to `\n`. mlx-swift-lm 2.x did this
    /// in a patched processor; 3.x's renders the template and stops there.
    static func endingOnEndOfText(_ input: LMInput, tokenizer: any MLXLMCommon.Tokenizer)
        -> LMInput
    {
        var tokens = input.text.tokens.asArray(Int.self)
        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false)
        let doubleNewline = tokenizer.encode(text: "\n\n", addSpecialTokens: false)
        if newline.count == 1, doubleNewline.count == 1, tokens.last == doubleNewline[0] {
            tokens[tokens.count - 1] = newline[0]
        }
        let endOfText = tokenizer.encode(text: "<|endoftext|>", addSpecialTokens: false)
        if let end = endOfText.count == 1 ? endOfText[0] : tokenizer.eosTokenId,
            tokens.last != end
        {
            tokens.append(end)
        }
        let array = MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: array, mask: ones(like: array).asType(.int8)),
            image: input.image, video: input.video, audio: input.audio)
    }

    /// Pool the final hidden state at the last unmasked position.
    static func lastTokenEmbedding(hiddenStates: MLXArray, mask: MLXArray?) throws -> (
        vector: [Float], sequenceLength: Int
    ) {
        eval(hiddenStates)
        let shape = hiddenStates.shape

        let sequenceLength: Int
        let dimension: Int
        switch shape.count {
        case 2:
            sequenceLength = shape[0]
            dimension = shape[1]
        case 3:
            guard shape[0] == 1 else {
                throw SemanticSearchError.modelUnavailable(
                    "multimodal embedding expects batch size 1, got \(shape[0])", underlying: nil)
            }
            sequenceLength = shape[1]
            dimension = shape[2]
        default:
            throw SemanticSearchError.modelUnavailable(
                "unexpected hidden state shape \(shape)", underlying: nil)
        }

        guard sequenceLength > 0, dimension > 0 else {
            throw SemanticSearchError.modelUnavailable(
                "hidden states are empty", underlying: nil)
        }

        let index = lastValidIndex(mask: mask, sequenceLength: sequenceLength)
        let flat = hiddenStates.asArray(Float.self)
        let start = index * dimension
        guard start >= 0, start + dimension <= flat.count else {
            throw SemanticSearchError.modelUnavailable(
                "hidden state index \(index) is out of range for shape \(shape)", underlying: nil)
        }
        return (
            vector: l2Normalize(Array(flat[start..<(start + dimension)])),
            sequenceLength: sequenceLength
        )
    }

    /// The last position the attention mask marks as real.
    ///
    /// Not `sequenceLength - 1`: with right padding that position holds a pad
    /// token, and pooling it yields a vector that varies with batch
    /// composition — the same input embedded alone and in a batch would land in
    /// different places.
    static func lastValidIndex(mask: MLXArray?, sequenceLength: Int) -> Int {
        guard sequenceLength > 0 else { return 0 }
        guard let mask else { return sequenceLength - 1 }

        let boolMask = mask.asType(.bool)
        eval(boolMask)
        let shape = boolMask.shape
        let values = boolMask.asArray(Bool.self)

        let usable: ArraySlice<Bool>
        switch shape.count {
        case 1:
            usable = values.prefix(min(sequenceLength, shape[0]))
        case 2 where shape[0] >= 1:
            usable = values.prefix(min(sequenceLength, shape[1]))
        default:
            return sequenceLength - 1
        }

        if let index = usable.lastIndex(where: { $0 }) { return index }
        return max(0, usable.count - 1)
    }

    // MARK: - Prompting

    private func instruction(for role: EmbeddingRole) -> String? {
        switch role {
        case .document:
            return configuration.documentInstruction
        case .query(let instruction):
            let resolved = (instruction ?? configuration.defaultQueryInstruction)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return resolved.isEmpty ? nil : resolved
        }
    }

    /// Text-backend wire format.
    ///
    /// `Instruct: …\nQuery:…` verbatim — no space after the colon. That is what
    /// Qwen3-Embedding was trained with, and a space there is a different
    /// prompt with measurably different vectors.
    private func formatted(_ input: EmbeddingInput) -> String {
        let text = Self.normalized(input.text)
        guard !text.isEmpty else { return "" }
        switch input.role {
        case .document:
            guard let instruction = configuration.documentInstruction,
                !instruction.isEmpty
            else { return text }
            return "Instruct: \(instruction)\nDocument:\(text)"
        case .query:
            guard let instruction = instruction(for: input.role) else { return text }
            return "Instruct: \(instruction)\nQuery:\(text)"
        }
    }

    private static func userInput(
        text: String, images: [UserInput.Image], videos: [UserInput.Video], instruction: String?
    ) -> UserInput {
        UserInput(
            chat: [
                .system(instruction ?? qwen3VLDefaultInstruction),
                .user(text, images: images, videos: videos),
            ],
            additionalContext: [
                "add_generation_prompt": true,
            ])
    }

    /// Not `.url`: the runtime decodes one with `CIImage(contentsOf:)`, which
    /// leaves an iPhone portrait on its side and converts a Display P3 photo
    /// to sRGB values the reference never sees.
    private static func userInputImage(_ payload: ImagePayload) throws -> UserInput.Image {
        .ciImage(CIImage(cgImage: try ImageDecoding.decodeUpright(payload)))
    }

    private static func explicitVisionPlaceholders(
        _ text: String, images: Int, videos: Int
    ) -> String {
        var prefix = ""
        for _ in 0..<max(0, images) {
            prefix += "<|vision_start|><|image_pad|><|vision_end|>"
        }
        for _ in 0..<max(0, videos) {
            prefix += "<|vision_start|><|video_pad|><|vision_end|>"
        }
        return prefix + text
    }

    private static func isPlaceholderMismatch(_ error: any Error) -> Bool {
        let message = "\(error.localizedDescription) \(String(describing: error))".lowercased()
        return message.contains("placeholder tokens") && message.contains("number of frames")
    }

    // MARK: - Helpers

    /// Why the checkpoint in `directory` is not loaded here, or `nil` when it
    /// may be. Only an EmbeddingGemma 2 `model_type` is refused; a missing or
    /// unreadable `config.json` is left for the loader to report.
    ///
    /// Read as `EmbedderModelFactory` reads it — the same type, the same
    /// JSON5 decoder — so the two cannot disagree: a strict decoder fails on
    /// files the factory accepts (`NaN` from Python's `json.dumps`, a
    /// trailing comma), and the guard would wave those through to load.
    static func embeddingGemma2Refusal(at directory: URL) -> SemanticSearchError? {
        guard
            let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
            let type = try? JSONDecoder.json5()
                .decode(MLXLMCommon.BaseConfiguration.self, from: data).modelType,
            type.hasPrefix("embedding_gemma2")
        else { return nil }
        guard type == "embedding_gemma2" else {
            return .configuration(
                "config.json has model_type '\(type)', an EmbeddingGemma 2 checkpoint "
                    + "that neither this provider nor EmbeddingGemma2Provider loads; for "
                    + "text alone load the full 'embedding_gemma2' checkpoint with "
                    + "EmbeddingGemma2Provider and vision: false")
        }
        return .configuration(
            "config.json has model_type '\(type)', an EmbeddingGemma 2 checkpoint; load it "
                + "with EmbeddingGemma2Provider, which applies its prompts and pooling")
    }

    private func modelConfiguration() -> MLXLMCommon.ModelConfiguration {
        configuration.model.runtimeConfiguration
    }

    /// Heuristic used only before the weights load, to advertise capabilities
    /// to a caller that has not called `prepare()` yet. Once loaded, the actual
    /// backend answers.
    private static func looksMultimodal(_ model: ModelReference) -> Bool {
        let identity = model.identity.lowercased()
        return identity.contains("-vl") || identity.contains("_vl") || identity.contains("vlm")
    }

    private static func normalized(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }

    /// Apply a Matryoshka prefix, renormalising afterwards.
    private static func reduced(_ vector: [Float], to requested: Int?, supported: [Int])
        -> [Float]
    {
        guard let requested, requested > 0, requested < vector.count,
            supported.contains(requested)
        else { return vector }
        return l2Normalize(Array(vector.prefix(requested)))
    }

    static func l2Normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }
        var sum: Float = 0
        for value in vector { sum += value * value }
        let norm = sum.squareRoot()
        guard norm > 0, norm.isFinite else { return vector }
        return vector.map { $0 / norm }
    }
}

/// The text-embedder downloader, refusing an EmbeddingGemma 2 checkpoint once
/// its files are on disk and before any weight loads. Its files land where
/// `EmbeddingGemma2Provider` looks for them, so the download is not wasted.
private struct RefusingEmbeddingGemma2: MLXLMCommon.Downloader {
    let base: any MLXLMCommon.Downloader

    init(_ base: any MLXLMCommon.Downloader) {
        self.base = base
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let directory = try await base.download(
            id: id, revision: revision, matching: patterns, useLatest: useLatest,
            progressHandler: progressHandler)
        if let refusal = MLXEmbeddingProvider.embeddingGemma2Refusal(at: directory) {
            throw refusal
        }
        return directory
    }
}

/// Carries a non-`Sendable` `UserInput` across one isolation hop.
///
/// `UserInput` holds `CIImage`s and is not `Sendable`; it is constructed here
/// and consumed exactly once on the model's executor, which is safe and which
/// the compiler cannot see.
final class UncheckedBox<T>: @unchecked Sendable {
    private var value: T?

    init(_ value: T) { self.value = value }

    func take() -> T {
        precondition(value != nil, "UncheckedBox can only be taken once")
        defer { value = nil }
        return value!
    }
}

import CoreGraphics
@preconcurrency import CoreML
import FluidUseEmbeddingGemma2
import Foundation
import SemanticSearch
import SemanticSearchApple

/// EmbeddingGemma 2 on Core ML: FluidInference's conversion
/// (`FluidInference/embeddinggemma-2-coreml`) run by FluidUse's Swift runtime,
/// vendored with its patches in `Vendor/FluidUseEmbeddingGemma2`.
///
/// The second EmbeddingGemma 2 provider, beside the MLX one, so the two can be
/// compared on one index. Same model, not the same vectors:
///
/// - **512 tokens, prompt and specials included.** Every function of the
///   conversion has a fixed shape and the longest is 512; the reference reads
///   8,192. A longer text keeps `<bos>`, its head and `<eos>`, and says so.
/// - **fp16 weights and activations.** Google's model card warns against
///   fp16; the conversion keeps RMSNorm in range by dividing each row by its
///   largest value first. On the Neural Engine, which computes in fp16 alone,
///   that is not enough past 256 tokens, so everything runs on the CPU by
///   default (``Configuration/textComputeUnits``). A vector that still comes
///   back non-finite is returned empty, which the engine reports, rather than
///   normalised into NaNs that would rank.
/// - **Images at one budget**, 280 soft tokens by default as in the reference,
///   decoded as transformers' `load_image` does it (`ReferenceImage`) and
///   resized with Pillow's bicubic, as its PIL backend does. Its torchvision
///   backend, the one it picks whenever torchvision is installed, resizes with
///   PyTorch: a level or two off on under 1% of values. A 16-bit image is
///   scaled to 8 bits, where Pillow clips 16-bit gray at 255, nearly all
///   white, and keeps the high byte of 16-bit colour. **No video**, and audio
///   only through ``embedAudio(_:)``: `EmbeddingInput` has no audio.
///
/// Text, images and their prompts are laid out as the reference processor
/// lays them out: `<bos>`, the prompt and the text, then each image as
/// `<|image>` soft tokens `<image|>`, then `<eos>`. Media alone gets no prompt,
/// as the model card says.
@available(macOS 15, iOS 18, *)
public actor CoreMLEmbeddingGemma2Provider: EmbeddingProvider {

    public struct Configuration: Sendable {
        /// Local copy of FluidInference/embeddinggemma-2-coreml. nil downloads the
        /// pinned revision into
        /// `<downloadBase>/models/FluidInference/embeddinggemma-2-coreml`.
        ///
        /// A local copy is checked against the pinned checksums, again whenever
        /// one of its files changes, and never downloaded into: the descriptor
        /// names `revision`, and a directory holding other weights would be
        /// indexed under that name. The compiled models are written beside the
        /// packages either way.
        public var modelDirectory: URL?
        /// Base for downloads, as `SEMANTIC_SEARCH_MODEL_DIRECTORY` and
        /// `MLXRuntime.Configuration.modelDirectory`: files land in
        /// `<downloadBase>/models/<org>/<name>`. nil reads
        /// `SEMANTIC_SEARCH_MODEL_DIRECTORY`, else `~/Library/Caches`, so
        /// `~/Library/Caches/models/…` as the MLX loaders' default.
        public var downloadBase: URL?
        /// Load the vision encoder and accept images, in the text space.
        public var vision: Bool
        /// Load the audio encoder for ``embedAudio(_:)``.
        public var audio: Bool
        /// Soft tokens per image: 70, 140 or 280, the functions the conversion
        /// ships. 280 is the reference's.
        public var imageTokenBudget: Int
        /// The task of a query that names none: `task: <this> | query: `.
        public var defaultQueryTask: String
        public var maximumBatchSize: Int
        /// Where the text functions run. The CPU, although they were converted
        /// for the Neural Engine: the Neural Engine computes in fp16
        /// throughout, and past 256 tokens its vectors leave the reference's —
        /// cosine 0.962 at 257–512 tokens against transformers in fp32, and
        /// 0.990 between two runs of the same input (M1 Pro). On the CPU the
        /// same functions give 0.9997 or better at every length, at 41 ms an
        /// input against the Neural Engine's 33. The GPU gives 1.00000 but
        /// takes 250 ms and 1.9 GB. Units that let the Neural Engine in
        /// (`.cpuAndNeuralEngine`, `.all`) are named in the descriptor, so an
        /// index made with them is a space of its own.
        public var textComputeUnits: MLComputeUnits
        /// Where the vision and audio encoders run. The CPU too: images 0.9996
        /// against the reference there, 0.982 on the Neural Engine, and the
        /// GPU's 0.9999 costs 2.5 GB resident against 0.5 GB. The Neural
        /// Engine is named in the descriptor as for text.
        public var mediaComputeUnits: MLComputeUnits
        /// Pack up to eight short texts into one `pack_256` call. Off by
        /// default: that function computes its rotary angles in fp16, so a text's
        /// vector would depend on whether it was packed. Off, `pack_256` is not
        /// even loaded and every text runs alone through `embed_<S>`.
        public var usesPackedBatches: Bool

        public init(
            modelDirectory: URL? = nil,
            downloadBase: URL? = nil,
            vision: Bool = true,
            audio: Bool = false,
            imageTokenBudget: Int = 280,
            defaultQueryTask: String = "search result",
            maximumBatchSize: Int = 8,
            textComputeUnits: MLComputeUnits = .cpuOnly,
            mediaComputeUnits: MLComputeUnits = .cpuOnly,
            usesPackedBatches: Bool = false
        ) {
            self.modelDirectory = modelDirectory
            self.downloadBase = downloadBase
            self.vision = vision
            self.audio = audio
            self.imageTokenBudget = imageTokenBudget
            self.defaultQueryTask = defaultQueryTask
            self.maximumBatchSize = max(1, maximumBatchSize)
            self.textComputeUnits = textComputeUnits
            self.mediaComputeUnits = mediaComputeUnits
            self.usesPackedBatches = usesPackedBatches
        }
    }

    /// The Hugging Face revision of FluidInference/embeddinggemma-2-coreml the
    /// checksums pin and the descriptor names.
    public static let revision = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"
    /// `config_sentence_transformers.json`'s document prompt.
    public static let documentPrefix = "title: none | text: "

    static let modelIdentity = "google/embeddinggemma-2"
    static let repository = "FluidInference/embeddinggemma-2-coreml"
    static let dimensions = 768
    /// The widths EmbeddingGemma 2 was trained to be cut to (model card).
    static let matryoshkaDimensions = [128, 256, 512, 768]
    /// `embed_512`, the longest function: the whole sequence, `<bos>`, prompt,
    /// text, media and `<eos>`, has to fit it.
    static let maximumTokens = 512

    /// The vendored tokenizer: FluidUse's byte-keyed BPE over Google's
    /// tokenizer.json (the Core ML repository ships the same file), with the
    /// patch that splits added tokens out of the text as Hugging Face does, and
    /// media placeholders reached from text looked up as `<pad>` as the
    /// reference model does.
    static let tokenizerVersion =
        "tokenizer.json@4d777ef5;fluiduse-bpe@e974b38;added-tokens-split;placeholders-as-pad"

    private struct Loaded: Sendable {
        let text: EmbeddingGemma2Manager
        let vision: EmbeddingGemma2Vision?
        let audio: EmbeddingGemma2Audio?
    }

    private let configuration: Configuration
    private var loaded: Loaded?
    /// The load in flight, so two jobs that find the provider unloaded at
    /// once share one load instead of making two copies resident.
    private var loading: Task<Loaded, any Error>?

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    // MARK: - Contract

    public var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: Self.modelIdentity,
            // The conversion is part of the identity. It is the same model as
            // the MLX provider's and not the same numerics — fp16 activations,
            // a 512-token cap, its own image path — so its vectors must never
            // be compared with those, and the two fingerprints must differ.
            weightsVariant: "\(Self.repository)@\(Self.revision.prefix(8)):fp16"
                + neuralEngineVariant,
            tokenizerVersion: Self.tokenizerVersion,
            preprocessingVersion: preprocessingVersion,
            documentInstruction: Self.documentPrefix,
            pooling: "mean",
            normalized: true,
            dimensions: Self.dimensions)
    }

    /// The functions the Neural Engine may run, named, or nothing. It computes
    /// in fp16 throughout, and measured against transformers in fp32 its text
    /// vectors fall to cosine 0.962 past 256 tokens and are not the same twice
    /// (images 0.982): an index made there cannot be queried from the CPU, or
    /// the other way round. The CPU (0.9997) and the GPU (1.0000) both match
    /// the reference, so they share the default name and an index moves
    /// between them. Media counts only when an encoder is loaded: a text-only
    /// provider's vectors do not depend on where images would have run.
    private var neuralEngineVariant: String {
        var variant = ""
        if Self.mayUseNeuralEngine(configuration.textComputeUnits) { variant += ":ane-text" }
        if configuration.vision || configuration.audio,
            Self.mayUseNeuralEngine(configuration.mediaComputeUnits)
        {
            variant += ":ane-media"
        }
        return variant
    }

    private static func mayUseNeuralEngine(_ units: MLComputeUnits) -> Bool {
        switch units {
        case .cpuOnly, .cpuAndGPU: return false
        case .cpuAndNeuralEngine, .all: return true
        // Units this provider was not measured on stay out of the space the
        // CPU and the GPU share.
        @unknown default: return true
        }
    }

    /// Everything between an input and the sequence the model sees. Where the
    /// functions run is in `weightsVariant`. Packing is in neither: it is off
    /// by default, and on, a text's vector depends on the batch it came in,
    /// which no name can describe.
    private var preprocessingVersion: String {
        [
            "eg2-coreml.v1",
            // The conversion's cap, prompt and specials included; a cut keeps
            // <bos>, the head and <eos>.
            "max512.keep-bos-head-eos",
            // Text as given; media alone without a prompt; text before images.
            "text-verbatim.media-unprompted.text-then-images",
            // 16-bit is scaled to 8, not clipped or truncated as Pillow does
            // it: a gray scan the reference reads as white is not embedded so.
            "image\(configuration.imageTokenBudget).pillow-bicubic"
                + ".exif-upright.stored-rgb.alpha-dropped.16bit-scaled",
            "audio.16k-mono.first-10s",
        ].joined(separator: ";")
    }

    public var capabilities: EmbeddingCapabilities {
        // Fixed by the configuration, loaded or not: the engine routes on
        // capabilities before it makes a model resident.
        EmbeddingCapabilities(
            acceptsText: true,
            acceptsImages: configuration.vision,
            acceptsVideo: false,
            sharedTextVisualSpace: configuration.vision,
            supportsDimensionReduction: true,
            supportedDimensions: Self.matryoshkaDimensions,
            maximumBatchSize: batchSize,
            maximumInputTokens: Self.maximumTokens,
            acceptsAudio: false)
    }

    /// `maximumBatchSize`, at least one. The init clamps it, but the property
    /// can be set to zero afterwards, and a batch of zero texts never moves
    /// `embed` past the first.
    private var batchSize: Int { max(1, configuration.maximumBatchSize) }

    public var isPrepared: Bool { loaded != nil }

    public func prepare() async throws {
        guard loaded == nil else { return }
        guard EmbeddingGemma2Vision.Budget(rawValue: configuration.imageTokenBudget) != nil else {
            throw SemanticSearchError.configuration(
                "EmbeddingGemma 2 on Core ML has image budgets of 70, 140 and 280 tokens, "
                    + "not \(configuration.imageTokenBudget)")
        }
        // The descriptor names this provider's revision and the store verifies
        // its own: a vendored update that moves one and not the other would
        // index new weights under the old name.
        guard Self.revision == EmbeddingGemma2ModelStore.revision else {
            throw SemanticSearchError.configuration(
                "the descriptor names \(Self.revision), the model store pins "
                    + EmbeddingGemma2ModelStore.revision)
        }
        let task: Task<Loaded, any Error>
        if let loading {
            task = loading
        } else {
            let configuration = configuration
            task = Task { try await Self.load(configuration) }
            loading = task
        }
        do {
            let result = try await task.value
            if loading == task { loading = nil }
            if loaded == nil { loaded = result }
        } catch {
            if loading == task { loading = nil }
            throw SemanticSearchError.modelUnavailable(Self.modelIdentity, underlying: error)
        }
    }

    public func release() async {
        loaded = nil
    }

    public func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard let loaded else { throw SemanticSearchError.providerNotPrepared(Self.modelIdentity) }
        for input in inputs { try Self.checkDimensions(input.requestedDimensions) }

        var outputs = Array(repeating: EmbeddingOutput(vector: []), count: inputs.count)
        var texts: [(index: Int, ids: [Int32], truncated: Bool)] = []
        for (index, input) in inputs.enumerated() {
            try Task.checkCancellation()
            // Video has no path here; text beside it alone would be a narrower
            // input than the one given. Empty is what the engine reports.
            guard input.videos.isEmpty else { continue }
            if input.images.isEmpty {
                guard let sequence = textSequence(input, text: loaded.text) else { continue }
                texts.append((index, sequence.ids, sequence.truncated))
            } else {
                outputs[index] = try await embedWithImages(input, loaded: loaded)
            }
        }

        var start = 0
        while start < texts.count {
            try Task.checkCancellation()
            let chunk = texts[start..<min(start + batchSize, texts.count)]
            let vectors = try await loaded.text.embed(sequences: chunk.map { $0.ids })
            for (entry, vector) in zip(chunk, vectors) {
                outputs[entry.index] = Self.output(
                    vector, tokens: entry.ids.count, truncated: entry.truncated,
                    dimensions: inputs[entry.index].requestedDimensions)
            }
            start += batchSize
        }
        return outputs
    }

    // MARK: - Beyond the contract

    /// The ids an input is embedded as, `<bos>` to `<eos>`, after any cut —
    /// for comparing with another runtime's or the reference's `input_ids`.
    ///
    /// For a text-only input these are exactly the ids looked up. With images,
    /// each soft-token slot appears as `<|image|>` between `<|image>` and
    /// `<image|>`, as in the reference processor's `input_ids`; the model
    /// reads the image's tokens there instead. Empty for an input with neither.
    public func tokenIDs(_ input: EmbeddingInput) async throws -> [Int] {
        guard let loaded else { throw SemanticSearchError.providerNotPrepared(Self.modelIdentity) }
        guard input.videos.isEmpty else {
            throw SemanticSearchError.unsupportedModality(
                itemID: "input", kind: .video, provider: Self.modelIdentity)
        }
        if input.images.isEmpty {
            return textSequence(input, text: loaded.text)?.ids.map(Int.init) ?? []
        }
        guard loaded.vision != nil,
            let budget = EmbeddingGemma2Vision.Budget(rawValue: configuration.imageTokenBudget)
        else {
            throw SemanticSearchError.unsupportedModality(
                itemID: "input", kind: .image, provider: Self.modelIdentity)
        }
        let images = try input.images.map { try ReferenceImage.decode($0) }
        guard let plan = layout(input, images: images, budget: budget, text: loaded.text) else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "input",
                detail: "the images alone need more than \(Self.maximumTokens) tokens")
        }
        var ids = [loaded.text.tokenizer.bosId] + plan.textTokens
        for count in plan.imageTokens {
            ids.append(EmbeddingGemma2Manager.imageStartId)
            ids += Array(repeating: EmbeddingGemma2Manager.imageTokenId, count: count)
            ids.append(EmbeddingGemma2Manager.imageEndId)
        }
        ids.append(loaded.text.tokenizer.eosId)
        return ids.map(Int.init)
    }

    /// The first 10 s of a recording, 16 kHz mono, unprompted:
    /// `<bos> <|audio>` its tokens `<audio|> <eos>`. `wasTruncated` when the
    /// recording is longer. Throws unless the provider was configured with
    /// `audio` and prepared.
    ///
    /// Outside ``embed(_:)`` because `EmbeddingInput` carries no audio. One
    /// window and no more: the conversion's audio encoder takes 10 s, the
    /// reference 30, and averaging windows would be a vector neither makes.
    public func embedAudio(_ url: URL) async throws -> EmbeddingOutput {
        guard configuration.audio else {
            throw SemanticSearchError.configuration(
                "CoreMLEmbeddingGemma2Provider was not configured with audio")
        }
        guard let audio = loaded?.audio else {
            throw SemanticSearchError.providerNotPrepared(Self.modelIdentity)
        }
        let window = EmbeddingGemma2Audio.windowSamples
        let rate = Double(EmbeddingGemma2Audio.sampleRate)
        // AVFoundation resamples and mixes down, as for every other recording
        // this package reads, and reads no more of the file than the window.
        let extractor = AudioExtractor()
        let samples = try await extractor.samples(
            from: url, timeRange: SourceLocator.TimeRange(start: 0, end: Double(window) / rate))
        let duration = try await extractor.duration(of: url)
        let clip = Array(samples.prefix(window))
        let tokens = EmbeddingGemma2Audio.tokenCount(
            validFrames: EmbeddingGemma2Audio.validFrames(samples: clip.count))
        guard tokens > 0 else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "too short to make an audio token")
        }
        let vector = try await audio.embed(window: clip)
        return Self.output(
            vector, tokens: tokens + 4, truncated: (duration * rate).rounded() > Double(window),
            dimensions: nil)
    }

    // MARK: - Sequences

    /// The prompt in front of an input's text, as the MLX provider builds it.
    /// Queries take the model card's `task: … | query: ` form: no instruction
    /// is the default task, an instruction already written in that form is
    /// used as it is (with the space every prompt ends in), any other is the
    /// task's name.
    private func prompt(for role: EmbeddingRole) -> String {
        switch role {
        case .document:
            return Self.documentPrefix
        case .query(let instruction):
            let task = instruction?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if task.isEmpty { return "task: \(configuration.defaultQueryTask) | query: " }
            if task.hasPrefix("task:"), task.contains("| query:") { return task + " " }
            return "task: \(task) | query: "
        }
    }

    /// The prompt and the text tokenized as one string, as the reference
    /// renders them into one before tokenizing, without `<bos>`/`<eos>`.
    /// Placeholders typed in the text are written as `<pad>`, which is the row
    /// the model reads for them.
    ///
    /// The text is tokenized as given: the reference tokenizes the string it
    /// is handed, and collapsing whitespace here would change its ids.
    private func promptedTokens(_ input: EmbeddingInput, text: EmbeddingGemma2Manager) -> [Int32] {
        let padding = text.tokenizer.padId
        return text.tokenizer.tokenize(prompt(for: input.role) + input.text).map {
            EmbeddingGemma2Manager.placeholderIds.contains($0) ? padding : $0
        }
    }

    /// `<bos>`, prompt and text, `<eos>`, cut to 512 by dropping the tail of
    /// the text and keeping `<eos>`. nil for blank text.
    private func textSequence(_ input: EmbeddingInput, text: EmbeddingGemma2Manager)
        -> (ids: [Int32], truncated: Bool)?
    {
        guard Self.hasText(input.text) else { return nil }
        let tokens = promptedTokens(input, text: text)
        let room = Self.maximumTokens - 2
        let kept = tokens.count > room ? Array(tokens.prefix(room)) : tokens
        return (
            [text.tokenizer.bosId] + kept + [text.tokenizer.eosId], tokens.count > room
        )
    }

    private struct Layout {
        /// The prompted text's ids, without `<bos>`/`<eos>`, cut to fit.
        var textTokens: [Int32]
        /// Soft tokens of each image.
        var imageTokens: [Int]
        var truncated: Bool
        var length: Int
    }

    /// How a text with images fills 512 tokens: the images whole, each with its
    /// two markers, and the text in what is left, cut from the end, as media is
    /// never cut. nil when the images alone do not fit: one always does (282
    /// tokens at most with its markers), two only at 253 soft tokens each or
    /// fewer, which the 140 budget always is.
    private func layout(
        _ input: EmbeddingInput, images: [CGImage], budget: EmbeddingGemma2Vision.Budget,
        text: EmbeddingGemma2Manager
    ) -> Layout? {
        let imageTokens = images.map {
            EmbeddingGemma2Vision.tokenCount(width: $0.width, height: $0.height, budget: budget)
        }
        let media = imageTokens.reduce(0) { $0 + $1 + 2 }
        let room = Self.maximumTokens - 2 - media
        guard room >= 0 else { return nil }
        // Media alone goes unprompted (model card); a prompt is the text's.
        var tokens = Self.hasText(input.text) ? promptedTokens(input, text: text) : []
        let truncated = tokens.count > room
        if truncated { tokens = Array(tokens.prefix(room)) }
        return Layout(
            textTokens: tokens, imageTokens: imageTokens, truncated: truncated,
            length: 2 + tokens.count + media)
    }

    private func embedWithImages(_ input: EmbeddingInput, loaded: Loaded) async throws
        -> EmbeddingOutput
    {
        guard let vision = loaded.vision,
            let budget = EmbeddingGemma2Vision.Budget(rawValue: configuration.imageTokenBudget)
        else { return EmbeddingOutput(vector: []) }
        let images: [CGImage]
        do {
            images = try input.images.map { try ReferenceImage.decode($0) }
        } catch SemanticSearchError.unsupportedContent {
            // Bytes that are not an image: this input, not the batch. A file
            // that has gone missing still throws (`mediaUnavailable`).
            return EmbeddingOutput(vector: [])
        }
        guard let plan = layout(input, images: images, budget: budget, text: loaded.text) else {
            return EmbeddingOutput(vector: [])
        }
        let vector = try await vision.embed(
            textTokens: plan.textTokens, images: images, budget: budget)
        return Self.output(
            vector, tokens: plan.length, truncated: plan.truncated,
            dimensions: input.requestedDimensions)
    }

    // MARK: - Loading

    private static func load(_ configuration: Configuration) async throws -> Loaded {
        let (directory, downloads) = location(configuration)
        try await EmbeddingGemma2ModelStore.ensure(in: directory, downloads: downloads)
        if configuration.vision {
            try await EmbeddingGemma2ModelStore.ensureVision(in: directory, downloads: downloads)
        }
        if configuration.audio {
            try await EmbeddingGemma2ModelStore.ensureAudio(in: directory, downloads: downloads)
        }
        try Task.checkCancellation()
        let text = try await EmbeddingGemma2Manager.load(
            from: directory, computeUnits: configuration.textComputeUnits,
            packed: configuration.usesPackedBatches)
        var vision: EmbeddingGemma2Vision?
        if configuration.vision,
            let budget = EmbeddingGemma2Vision.Budget(rawValue: configuration.imageTokenBudget)
        {
            // One budget's function: an index embeds every image at one.
            vision = try await EmbeddingGemma2Vision.load(
                from: directory, text: text, budgets: [budget],
                computeUnits: configuration.mediaComputeUnits)
        }
        var audio: EmbeddingGemma2Audio?
        if configuration.audio {
            audio = try await EmbeddingGemma2Audio.load(
                from: directory, text: text, computeUnits: configuration.mediaComputeUnits)
        }
        return Loaded(text: text, vision: vision, audio: audio)
    }

    /// Where the model files are, and whether missing ones may be fetched.
    static func location(_ configuration: Configuration) -> (directory: URL, downloads: Bool) {
        if let directory = configuration.modelDirectory { return (directory, false) }
        let base = configuration.downloadBase ?? environmentBase ?? URL.cachesDirectory
        let directory = base.appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("FluidInference", isDirectory: true)
            .appendingPathComponent("embeddinggemma-2-coreml", isDirectory: true)
        return (directory, true)
    }

    private static var environmentBase: URL? {
        guard let raw = ProcessInfo.processInfo.environment["SEMANTIC_SEARCH_MODEL_DIRECTORY"],
            !raw.isEmpty
        else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
    }

    // MARK: - Vectors

    private static func hasText(_ text: String) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func checkDimensions(_ requested: Int?) throws {
        guard let requested, requested != dimensions else { return }
        guard matryoshkaDimensions.contains(requested) else {
            throw SemanticSearchError.dimensionReductionUnsupported(
                model: modelIdentity, requested: requested, native: dimensions)
        }
    }

    /// A model vector as the engine stores it: unit length, cut to a
    /// Matryoshka width when one was asked for and renormalised, or empty when
    /// fp16 overflowed somewhere and left a value that is not finite.
    private static func output(_ vector: [Float], tokens: Int, truncated: Bool, dimensions: Int?)
        -> EmbeddingOutput
    {
        let width = dimensions ?? Self.dimensions
        guard vector.count == Self.dimensions, vector.allSatisfy(\.isFinite) else {
            return EmbeddingOutput(vector: [], consumedTokens: tokens, wasTruncated: truncated)
        }
        let prefix = vector.prefix(width)
        let norm = prefix.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        guard norm.isFinite, norm > 0 else {
            return EmbeddingOutput(vector: [], consumedTokens: tokens, wasTruncated: truncated)
        }
        return EmbeddingOutput(
            vector: prefix.map { Float(Double($0) / norm) }, consumedTokens: tokens,
            wasTruncated: truncated)
    }
}

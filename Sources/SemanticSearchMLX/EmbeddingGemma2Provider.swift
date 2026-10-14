import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLXLMCommon
import MLXVLM
import SemanticSearch

@preconcurrency import MLX

/// EmbeddingGemma 2 on MLX: text, images and video in one 768-dimensional
/// space, with Matryoshka prefixes at 512, 256 and 128.
///
/// Built on the runtime's `EmbeddingGemma2` model and its media processors,
/// not on its `EmbeddingGemma2Embedding` actor, which would hold a second
/// copy of the weights, nor on `EmbedderModelFactory`, which makes every
/// tower resident. Between them the runtime does four things the reference
/// does not, each silent when it happens:
///
/// - **The actor passes text that starts with `task:` or `title:` through
///   unprompted.** A note whose first line reads "title: …" would be embedded
///   without the document prefix, away from every other document. Prompts are
///   built here, from the role alone.
/// - **`<|image|>`, `<|video|>` and `<|audio|>` typed in text are embedded as
///   themselves.** The reference maps them to padding before the embedding
///   lookup (`llm_input_ids` in transformers' `EmbeddingGemma2Model`), and so
///   does this provider.
/// - **The factory path cuts token ids at 8192 and can drop `<eos>`.** Here
///   the text is cut to ``Configuration/maximumInputTokens`` keeping the head
///   and `<eos>`, as the reference tokenizer truncates, and the cut is
///   reported.
/// - **The image processor ignores EXIF orientation and colour-manages
///   embedded profiles.** Images are decoded here the way PIL reads them
///   (`decodedImage(_:)`).
///
/// Pooling is the model's own: a masked mean over every real token — prompt,
/// `<bos>` and `<eos>` included, padding excluded — then the 512→768
/// projection and an L2 norm. There is no LayerNorm, unlike
/// `MLXEmbeddingProvider`'s text path; one over these vectors moves all of
/// them.
public actor EmbeddingGemma2Provider: EmbeddingProvider {

    public struct Configuration: Sendable {
        public var model: ModelReference
        /// Load the vision tower: images and video. Without it the provider
        /// is text-only and ~0.54 GB resident at bf16 instead of ~0.88 GB
        /// (~1.5 GB with the audio tower too). Does not change the space; see
        /// ``EmbeddingGemma2Provider/descriptor``.
        public var vision: Bool
        /// Load the audio tower, for ``EmbeddingGemma2Provider/embedAudio(_:)``.
        public var audio: Bool
        /// The task a query is prompted with when it brings no instruction:
        /// `task: <this> | query: `.
        public var defaultQueryTask: String
        /// Text tokens per input, counting the prompt, `<bos>` and `<eos>`.
        /// Longer text keeps its head and `<eos>` and is reported as
        /// truncated. Media tokens are not counted — the processors' budgets
        /// bound them — so for an input with media this is the text's share,
        /// and the whole sequence is further held to the model's 8K context.
        public var maximumInputTokens: Int
        /// Text-only inputs per forward pass. Inputs with media run one at a
        /// time: their soft tokens fill one row.
        public var maximumBatchSize: Int
        /// Identity recorded in the descriptor. Defaults to the model
        /// reference, which is not portable for a local directory.
        public var modelIdentity: String?
        /// Recorded in the descriptor as given, never read from the loaded
        /// files: the descriptor is fingerprinted before anything loads, and a
        /// value that appeared on `prepare()` would change the space under the
        /// vectors already in it.
        public var weightsVariant: String?
        public var runtime: MLXRuntime.Configuration

        public init(
            model: ModelReference,
            vision: Bool = true,
            audio: Bool = false,
            defaultQueryTask: String = "search result",
            maximumInputTokens: Int = 2_048,
            maximumBatchSize: Int = 8,
            modelIdentity: String? = nil,
            weightsVariant: String? = nil,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) {
            self.model = model
            self.vision = vision
            self.audio = audio
            self.defaultQueryTask = defaultQueryTask
            // `<bos>`, one token and `<eos>` at the least; the model's context
            // at the most.
            self.maximumInputTokens = min(
                max(3, maximumInputTokens),
                MLXLMCommon.EmbeddingGemma2Configuration.contextLength)
            self.maximumBatchSize = max(1, maximumBatchSize)
            self.modelIdentity = modelIdentity
            self.weightsVariant = weightsVariant
            self.runtime = runtime
        }
    }

    /// The document prompt of `config_sentence_transformers.json`. Per-item
    /// titles are not used: a parent's title is not part of a chunk's content
    /// fingerprint, so a renamed document would keep vectors made under its
    /// old title.
    public static let documentPrefix = "title: none | text: "

    static let dimensions = 768
    /// The widths the model card says it was trained at.
    static let matryoshkaDimensions = [128, 256, 512, 768]

    /// Names everything this provider does that changes a vector: the
    /// sentence-transformers prompts; placeholders in text as padding; 280
    /// image tokens; video at 1 fps, up to 32 frames, 140 tokens each; audio
    /// at 16 kHz, its first 30 s; media unprompted; text before media; images
    /// oriented, read as sRGB, alpha dropped. `load(directory:vision:audio:)`
    /// refuses a checkpoint whose processor config says otherwise.
    private static let preprocessingVersion =
        "embeddinggemma2.v1.prompts-st.placeholders-pad.img280.vid-fps1-max32-tok140"
        + ".aud16k-30s.media-unprompted.text-first.exif-srgb-noalpha"

    /// Everything `prepare()` loads, dropped together by `release()`.
    ///
    /// `@unchecked`: one load is shared through a `Task`, whose result has to
    /// be `Sendable`, and the model is a `Module`, which is not. The load
    /// builds it and hands it to this actor, the only place it is used.
    private struct Loaded: @unchecked Sendable {
        let model: MLXLMCommon.EmbeddingGemma2
        let tokenizer: any MLXLMCommon.Tokenizer
        /// `nil` when the tower that reads them did not load.
        let images: EmbeddingGemma2ImageProcessor?
        let videos: EmbeddingGemma2VideoProcessor?
        let audio: EmbeddingGemma2AudioProcessor?
        let beginOfSequence: Int
        let endOfSequence: Int
        let padding: Int
        /// `<|image|>`, `<|video|>` and `<|audio|>`, whichever towers loaded:
        /// text holding them is the same text either way.
        let placeholders: Set<Int>
    }

    private let configuration: Configuration
    private var loaded: Loaded?
    /// The load in flight, which every `prepare()` meanwhile waits for. The
    /// actor is reentrant across the download and the weights: two jobs that
    /// find the provider unloaded — an ingest and a search, with
    /// `maximumConcurrentInferenceJobs` above one — would otherwise each
    /// download into one directory and hold a copy of the weights until one
    /// was dropped.
    private var loading: Task<Loaded, any Error>?

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Contract

    /// The same with and without the vision and audio towers, on purpose. They
    /// add encoders; the weights, the tokenizer and the text path are one, so
    /// a text-only provider's vectors are the full one's and either opens a
    /// space the other wrote. Opening it is not searching all of it: the
    /// engine routes on ``capabilities`` and queries only the kinds a provider
    /// accepts, so the image and video vectors the full provider indexed are
    /// never searched through a text-only one — a frame or photo is found by
    /// its caption or its words alone — and no diagnostic says so.
    public var descriptor: EmbeddingSpaceDescriptor {
        let identity = configuration.modelIdentity ?? configuration.model.identity
        return EmbeddingSpaceDescriptor(
            modelIdentity: identity,
            weightsVariant: configuration.weightsVariant,
            // The tokenizer ships with the weights; see `MLXEmbeddingProvider`.
            tokenizerVersion: identity,
            preprocessingVersion: Self.preprocessingVersion,
            documentInstruction: Self.documentPrefix,
            pooling: "mean",
            normalized: true,
            dimensions: Self.dimensions)
    }

    /// From the configuration alone, loaded or not: the engine routes on
    /// capabilities before it makes a model resident.
    public var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(
            acceptsText: true,
            acceptsImages: configuration.vision,
            acceptsVideo: configuration.vision,
            sharedTextVisualSpace: configuration.vision,
            supportsDimensionReduction: true,
            supportedDimensions: Self.matryoshkaDimensions,
            maximumBatchSize: configuration.maximumBatchSize,
            maximumInputTokens: configuration.maximumInputTokens,
            // `EmbeddingInput` has no audio yet; recordings go through
            // `embedAudio(_:)`, which the engine does not call.
            acceptsAudio: false)
    }

    public var isPrepared: Bool { loaded != nil }

    public func prepare() async throws {
        guard loaded == nil else { return }
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
        } catch let error as CancellationError {
            if loading == task { loading = nil }
            throw error
        } catch {
            if loading == task { loading = nil }
            throw SemanticSearchError.modelUnavailable(
                configuration.model.identity, underlying: error)
        }
    }

    public func release() async {
        loaded = nil
        MLXRuntime.releaseCache()
    }

    /// Text-only inputs are batched, padded on the right and masked by their
    /// true lengths; inputs with images or videos run one at a time. An input
    /// this provider cannot embed — media without the vision tower, a file
    /// that does not decode, more media than the context holds — gets an
    /// empty vector, which the engine reports. A width the model was not
    /// trained at throws: that is a configuration error, not a bad input.
    public func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard let loaded else {
            throw SemanticSearchError.providerNotPrepared(configuration.model.identity)
        }
        guard !inputs.isEmpty else { return [] }
        for input in inputs {
            try checkDimensions(input.requestedDimensions)
        }

        var outputs = Array(repeating: EmbeddingOutput(vector: []), count: inputs.count)
        let texts = inputs.indices.filter {
            inputs[$0].images.isEmpty && inputs[$0].videos.isEmpty
                && !Self.isBlank(inputs[$0].text)
        }
        let media = inputs.indices.filter {
            !inputs[$0].images.isEmpty || !inputs[$0].videos.isEmpty
        }

        let batchSize = configuration.maximumBatchSize
        for start in stride(from: 0, to: texts.count, by: batchSize) {
            try Task.checkCancellation()
            let batch = Array(texts[start..<min(start + batchSize, texts.count)])
            let rows = batch.map { textTokens(inputs[$0].text, role: inputs[$0].role, loaded) }
            let vectors = try embedRows(rows.map { $0.tokens }, loaded)
            for (offset, index) in batch.enumerated() {
                outputs[index] = EmbeddingOutput(
                    vector: Self.finished(
                        vectors[offset], requested: inputs[index].requestedDimensions),
                    consumedTokens: rows[offset].tokens.count,
                    wasTruncated: rows[offset].truncated)
            }
        }

        for index in media {
            try Task.checkCancellation()
            outputs[index] = try await embedMedia(inputs[index], loaded)
        }
        return outputs
    }

    // MARK: - Side APIs

    /// The token ids the model is fed for a text-only input: `<bos>`, the
    /// prompt and the text tokenised as one string, `<eos>`, after
    /// truncation and with placeholders as padding. For parity tests against
    /// the reference tokenizer.
    public func tokenIDs(_ input: EmbeddingInput) async throws -> [Int] {
        guard let loaded else {
            throw SemanticSearchError.providerNotPrepared(configuration.model.identity)
        }
        guard input.images.isEmpty, input.videos.isEmpty else {
            throw SemanticSearchError.configuration(
                "tokenIDs(_:) covers text-only inputs; media tokens depend on the pixels")
        }
        guard !Self.isBlank(input.text) else { return [] }
        return textTokens(input.text, role: input.role, loaded).tokens
    }

    /// One recording's vector: its first 30 s, mono at 16 kHz, unprompted,
    /// as the reference embeds audio. `wasTruncated` when it is longer.
    ///
    /// A side API until `EmbeddingInput` carries audio. Throws when the
    /// provider was not configured with ``Configuration/audio``.
    public func embedAudio(_ url: URL) async throws -> EmbeddingOutput {
        guard let loaded else {
            throw SemanticSearchError.providerNotPrepared(configuration.model.identity)
        }
        guard let processor = loaded.audio else {
            throw SemanticSearchError.unsupportedModality(
                itemID: ItemID(url.lastPathComponent), kind: .audio,
                provider: configuration.model.identity)
        }
        if url.isFileURL, !FileManager.default.fileExists(atPath: url.path) {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }

        // AVFoundation's errors mapped, as every other failure here is: a
        // caller catching `SemanticSearchError` would otherwise let a
        // protected or unreadable file's `NSError` through.
        let decoded: (samples: [Float], truncated: Bool)
        do {
            decoded = try await Self.audioSamples(
                url, sampleRate: processor.sampleRate,
                limit: EmbeddingGemma2AudioProcessor.maximumSampleCount)
        } catch let error as CancellationError {
            throw error
        } catch let error as SemanticSearchError {
            throw error
        } catch {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "audio could not be decoded: \(error.localizedDescription)")
        }
        guard !decoded.samples.isEmpty else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "the audio track has no samples")
        }
        try Task.checkCancellation()

        let frames = try await processor.features(for: .array(MLXArray(decoded.samples)))
        let segment: EmbeddingGemma2Sequence.Segment
        let softTokens: MLXArray?
        if frames.dim(0) > 0 {
            guard let features = loaded.model.audioFeatures(frames) else {
                throw SemanticSearchError.unsupportedModality(
                    itemID: ItemID(url.lastPathComponent), kind: .audio,
                    provider: configuration.model.identity)
            }
            try MLX.checkedEval(features)
            segment = .audio(tokens: features.dim(1))
            softTokens = features.reshaped(-1, features.dim(-1))
        } else {
            // Shorter than one feature frame: the markers alone, as the
            // reference lays it out.
            segment = .audio(tokens: 0)
            softTokens = nil
        }

        let tokens = try EmbeddingGemma2Sequence.tokens(
            [segment], configuration: loaded.model.config,
            beginOfSequence: loaded.beginOfSequence, endOfSequence: loaded.endOfSequence,
            limit: MLXLMCommon.EmbeddingGemma2Configuration.contextLength)
        let vector = loaded.model.embed(
            inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
            attentionMask: nil, softTokens: softTokens)
        try MLX.checkedEval(vector)
        let values = Self.finished(vector.asArray(Float.self), requested: nil)
        guard !values.isEmpty else {
            throw SemanticSearchError.providerReturnedNoOutput(configuration.model.identity)
        }
        return EmbeddingOutput(
            vector: values, consumedTokens: tokens.count, wasTruncated: decoded.truncated)
    }

    // MARK: - Text

    /// The text with its prompt, tokenised as one string — the reference
    /// renders prompt and text into one string before tokenising, and
    /// "query: " followed by a word does not tokenise as "query: " and the
    /// word — with placeholders as padding, cut to the budget less `<bos>`
    /// and `<eos>`.
    private func promptedText(_ text: String, role: EmbeddingRole, _ loaded: Loaded) -> (
        tokens: [Int], truncated: Bool
    ) {
        let tokens = loaded.tokenizer
            .encode(text: prompt(for: role) + text, addSpecialTokens: false)
            .map { loaded.placeholders.contains($0) ? loaded.padding : $0 }
        let budget = configuration.maximumInputTokens - 2
        guard tokens.count > budget else { return (tokens, false) }
        return (Array(tokens.prefix(budget)), true)
    }

    /// `<bos>` + the prompted text + `<eos>`: what the reference tokenizer's
    /// post-processor builds, and with truncation the head it keeps.
    private func textTokens(_ text: String, role: EmbeddingRole, _ loaded: Loaded) -> (
        tokens: [Int], truncated: Bool
    ) {
        let content = promptedText(text, role: role, loaded)
        let tokens = [loaded.beginOfSequence] + content.tokens + [loaded.endOfSequence]
        return (tokens, content.truncated)
    }

    /// One forward over rows padded on the right, masked by their true
    /// lengths. Unpadded rows get no mask, as the reference runs one input.
    private func embedRows(_ rows: [[Int]], _ loaded: Loaded) throws -> [[Float]] {
        let length = rows.map(\.count).max() ?? 0
        guard length > 0 else { return rows.map { _ in [] } }
        let padded = rows.flatMap {
            $0 + Array(repeating: loaded.padding, count: length - $0.count)
        }
        let ids = MLXArray(padded.map(Int32.init), [rows.count, length])
        let mask: MLXArray? =
            rows.allSatisfy({ $0.count == length })
            ? nil
            : MLXArray(
                rows.flatMap {
                    Array(repeating: Int32(1), count: $0.count)
                        + Array(repeating: Int32(0), count: length - $0.count)
                }, [rows.count, length])
        let vectors = loaded.model.embed(inputIds: ids, attentionMask: mask)
        try MLX.checkedEval(vectors)
        let width = vectors.dim(1)
        let flat = vectors.asArray(Float.self)
        guard flat.count == rows.count * width else {
            throw SemanticSearchError.providerReturnedNoOutput(configuration.model.identity)
        }
        return rows.indices.map { Array(flat[($0 * width)..<(($0 + 1) * width)]) }
    }

    // MARK: - Media

    /// Text, then each image, then each video — the order the reference's
    /// chat template renders a message's content in — as one sequence whose
    /// placeholders the media's soft tokens fill. Without text, no prompt:
    /// the model card passes media unprompted.
    private func embedMedia(_ input: EmbeddingInput, _ loaded: Loaded) async throws
        -> EmbeddingOutput
    {
        guard let imageProcessor = loaded.images, let videoProcessor = loaded.videos else {
            // Not substituted with the text alone: that would be a narrower
            // input than the one asked for, under the same vector.
            return EmbeddingOutput(vector: [])
        }
        do {
            var segments: [EmbeddingGemma2Sequence.Segment] = []
            var softTokens: [MLXArray] = []
            var truncated = false
            if !Self.isBlank(input.text) {
                let text = promptedText(input.text, role: input.role, loaded)
                segments.append(.text(text.tokens))
                truncated = text.truncated
            }
            for payload in input.images {
                try Task.checkCancellation()
                // Decoded and rendered to float at full size before the
                // resize: ~32 bytes a source pixel at the peak, ~1.5 GB for a
                // 48 MP photo, beside the model. Not decoded smaller: ImageIO's
                // downsampling filters the pixels the reference resizes from,
                // which wants measuring against the reference and a new
                // `preprocessingVersion` before it is done.
                let image = try Self.decodedImage(payload)
                let pixels = try imageProcessor.pixels(for: .ciImage(image)).pixels
                guard let features = loaded.model.imageFeatures(pixels) else {
                    return EmbeddingOutput(vector: [])
                }
                // One encoder's activations alive at a time.
                try MLX.checkedEval(features)
                softTokens.append(features.reshaped(-1, features.dim(-1)))
                segments.append(.image(tokens: features.dim(1)))
            }
            for url in input.videos {
                try Task.checkCancellation()
                let frames = try await Self.videoPixels(url, processor: videoProcessor)
                guard let features = loaded.model.imageFeatures(frames) else {
                    return EmbeddingOutput(vector: [])
                }
                try MLX.checkedEval(features)
                softTokens.append(features.reshaped(-1, features.dim(-1)))
                segments.append(
                    .video(frames: features.dim(0), tokensPerFrame: features.dim(1)))
            }

            // Laid out once without a limit to see whether it fits the
            // context; the runtime cuts text to fit without saying so.
            let layout = loaded.model.config
            let context = MLXLMCommon.EmbeddingGemma2Configuration.contextLength
            let whole = try EmbeddingGemma2Sequence.tokens(
                segments, configuration: layout, beginOfSequence: loaded.beginOfSequence,
                endOfSequence: loaded.endOfSequence, limit: .max)
            let tokens: [Int]
            if whole.count <= context {
                tokens = whole
            } else {
                tokens = try EmbeddingGemma2Sequence.tokens(
                    segments, configuration: layout, beginOfSequence: loaded.beginOfSequence,
                    endOfSequence: loaded.endOfSequence, limit: context)
            }

            let vector = loaded.model.embed(
                inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
                attentionMask: nil, softTokens: concatenated(softTokens, axis: 0))
            try MLX.checkedEval(vector)
            return EmbeddingOutput(
                vector: Self.finished(
                    vector.asArray(Float.self), requested: input.requestedDimensions),
                consumedTokens: tokens.count,
                // Uniform sampling of a long video is not truncation: the 32
                // frames span all of it, as the reference samples them.
                wasTruncated: truncated || tokens.count < whole.count)
        } catch let error as CancellationError {
            throw error
        } catch {
            return EmbeddingOutput(vector: [])
        }
    }

    /// An image as the reference reads it, for the runtime's processor.
    ///
    /// The processor takes `CIImage(contentsOf:)` as it comes: EXIF
    /// orientation unapplied, the embedded profile colour-managed — a Display
    /// P3 photo arrives converted to sRGB. The reference opens files with
    /// PIL: transformers' `load_image` applies the EXIF orientation, the
    /// stored values are used as they are, and `convert("RGB")` drops alpha
    /// without compositing, so a transparent pixel keeps the colour it was
    /// stored with. Core Image cannot be asked for that: it premultiplies
    /// any alpha it is handed, and a pixel with none comes back black — the
    /// white background of a transparent logo turns black. So the decoded
    /// bitmap is relabelled before Core Image sees it (`storedRGB(_:)`) and
    /// then turned upright.
    ///
    /// What cannot be relabelled is read as Core Image reads it: grey and
    /// other colour models colour-managed, and an alpha ImageIO has already
    /// premultiplied divided back out, its fully transparent pixels black.
    private static func decodedImage(_ payload: ImagePayload) throws -> CIImage {
        let source: CGImageSource?
        switch payload {
        case .fileURL(let url): source = CGImageSourceCreateWithURL(url as CFURL, nil)
        case .encoded(let data, _): source = CGImageSourceCreateWithData(data as CFData, nil)
        }
        guard let source,
            let decoded = CGImageSourceCreateImageAtIndex(
                source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "image could not be decoded")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation =
            (properties?[kCGImagePropertyOrientation] as? NSNumber)
            .flatMap { CGImagePropertyOrientation(rawValue: $0.uint32Value) } ?? .up

        let image = CIImage(cgImage: storedRGB(decoded) ?? decoded).oriented(orientation)
        guard !image.extent.isEmpty, !image.extent.isInfinite else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "image could not be decoded")
        }
        // An identity for a relabelled image, which is opaque by now; for a
        // premultiplied one, a half-transparent pixel keeps its colour
        // instead of fading towards black.
        return image.unpremultiplyingAlpha().settingAlphaOne(in: image.extent)
            .cropped(to: image.extent)
    }

    /// `image`'s own bytes declared sRGB, whatever its profile says, with a
    /// straight alpha channel skipped rather than read; `nil` when it is not
    /// RGB or CoreGraphics refuses the layout. A premultiplied alpha is left
    /// alone: the colour is already scaled by it, and skipping it would
    /// darken every translucent pixel.
    private static func storedRGB(_ image: CGImage) -> CGImage? {
        guard image.colorSpace?.model == .rgb,
            let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
            let provider = image.dataProvider
        else { return nil }
        let alpha: CGImageAlphaInfo
        switch image.alphaInfo {
        case .last: alpha = .noneSkipLast
        case .first: alpha = .noneSkipFirst
        default: alpha = image.alphaInfo
        }
        let layout = CGBitmapInfo(
            rawValue: (image.bitmapInfo.rawValue & ~CGBitmapInfo.alphaInfoMask.rawValue)
                | alpha.rawValue)
        return CGImage(
            width: image.width, height: image.height,
            bitsPerComponent: image.bitsPerComponent, bitsPerPixel: image.bitsPerPixel,
            bytesPerRow: image.bytesPerRow, space: sRGB, bitmapInfo: layout,
            provider: provider, decode: image.decode,
            shouldInterpolate: image.shouldInterpolate, intent: image.renderingIntent)
    }

    /// `[frames, 3, H, W]` pixels of one video, as `EmbeddingGemma2VideoProcessor`
    /// builds them from a URL — but resized frame by frame.
    ///
    /// Given a URL, the processor decodes every sampled frame at full size and
    /// renders all of them to float pixels before resizing any: 32 frames of
    /// 4K are ~5 GB at once. Here each decoded frame goes through the
    /// processor alone and is evaluated before the next is read. The frames,
    /// their times and their pixels are still the processor's: the indices
    /// and times below are its own for a track with a frame rate (its
    /// `sampledIndices` is internal), and one decoded frame passes through
    /// its sampling unchanged and is sized by the same budget. The processor
    /// sizes every frame to the first; a frame of another size goes through
    /// it behind the first, so it is sized as it would be there.
    private static func videoPixels(_ url: URL, processor: EmbeddingGemma2VideoProcessor)
        async throws -> MLXArray
    {
        if url.isFileURL, !FileManager.default.fileExists(atPath: url.path) {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "no video track")
        }
        let (range, nominalRate) = try await track.load(.timeRange, .nominalFrameRate)
        let seconds = range.duration.seconds
        let frameRate = Double(nominalRate)
        // A damaged file can report an invalid or indefinite duration — NaN
        // or infinite seconds — and converting that to a frame count traps:
        // the whole process down, where the contract asks for an empty
        // vector. The runtime's own sampling has the same hole.
        guard seconds.isFinite, seconds > 0, frameRate.isFinite, frameRate > 0,
            let frameCount = Int(exactly: (seconds * frameRate).rounded())
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "video has no usable duration or frame rate")
        }
        let indices = sampledIndices(
            frameCount: frameCount, frameRate: frameRate,
            framesPerSecond: processor.framesPerSecond, maximumFrames: processor.maximumFrames)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        // The middle of each frame's interval, so a time selects its frame
        // without rounding to a neighbour.
        let times = indices.map {
            CMTime(
                seconds: range.start.seconds + (Double($0) + 0.5) / frameRate,
                preferredTimescale: 90_000)
        }
        // The decoded RGB values read as sRGB, as the processor reads them.
        let options: [CIImageOption: Any] =
            CGColorSpace(name: CGColorSpace.sRGB).map { [.colorSpace: $0] } ?? [:]

        var first: (frame: UserInput.VideoFrame, width: Int, height: Int)?
        var frames: [MLXArray] = []
        frames.reserveCapacity(times.count)
        for await result in generator.images(for: times) {
            try Task.checkCancellation()
            guard case .success(_, let image, let actual) = result else { continue }
            let frame = UserInput.VideoFrame(
                image: .ciImage(CIImage(cgImage: image, options: options)), timeStamp: actual)
            let pixels: MLXArray
            if let first, first.width != image.width || first.height != image.height {
                // A track whose frames change size — segments of two
                // resolutions joined without re-encoding. Sized by its own
                // extent, this frame can come out another shape than the
                // first, and MLX answers a concatenation of mismatched shapes
                // with a `fatalError`, the whole process down.
                pixels = try await processor.frames(for: .frames([first.frame, frame]))[1...]
            } else {
                pixels = try await processor.frames(for: .frames([frame]))
            }
            try MLX.checkedEval(pixels)
            if first == nil { first = (frame, image.width, image.height) }
            frames.append(pixels)
        }
        guard !frames.isEmpty else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "no video frames could be decoded")
        }
        return concatenated(frames, axis: 0)
    }

    /// `EmbeddingGemma2VideoProcessor.sampledIndices` for a known frame rate,
    /// after the reference's `sample_frames`: one frame every
    /// `frameRate / framesPerSecond`, then `maximumFrames` of those spread
    /// evenly, as `np.linspace` truncates.
    ///
    /// The candidates are computed where they are picked rather than listed
    /// first: a damaged header claiming a duration of centuries would
    /// allocate one per second of it. Same indices.
    static func sampledIndices(
        frameCount: Int, frameRate: Double, framesPerSecond: Double, maximumFrames: Int
    ) -> [Int] {
        guard frameCount > 0, frameRate.isFinite, frameRate > 0, framesPerSecond.isFinite,
            framesPerSecond > 0, maximumFrames > 0
        else { return [] }
        let sampled = (Double(frameCount) / frameRate * framesPerSecond).rounded(.towardZero)
        guard let candidates = Int(exactly: sampled) else { return [] }
        let count = max(1, candidates)
        let step = frameRate / framesPerSecond
        let last = Double(frameCount - 1)
        // The reference's `min(frameCount - 1, int(i * step))`, compared
        // before converting, so an index past the end cannot overflow.
        func candidate(_ position: Int) -> Int {
            let index = Double(position) * step
            return index < last ? Int(index) : frameCount - 1
        }
        guard count > maximumFrames else { return (0..<count).map(candidate) }
        let spread = Double(count - 1) / Double(max(maximumFrames - 1, 1))
        return (0..<maximumFrames).map { position in
            candidate(
                position > 0 && position == maximumFrames - 1
                    ? count - 1 : Int(Double(position) * spread))
        }
    }

    /// The first `limit` samples of the first audio track, mono at
    /// `sampleRate`, and whether there were more.
    ///
    /// Decoded with the settings of `UserInput.Audio.asMLXArray`, which the
    /// runtime's audio processor uses for a URL, so the samples are its
    /// samples — but reading stops one buffer past `limit` instead of
    /// holding an hour of audio to keep 30 seconds of it.
    private static func audioSamples(_ url: URL, sampleRate: Int, limit: Int) async throws -> (
        samples: [Float], truncated: Bool
    ) {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "no audio track")
        }
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: Double(sampleRate),
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
            ])
        let reader = try AVAssetReader(asset: asset)
        reader.add(output)
        reader.startReading()

        var samples: [Float] = []
        while samples.count <= limit, let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let byteCount = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)
            CMBlockBufferCopyDataBytes(
                block, atOffset: 0, dataLength: byteCount, destination: &chunk)
            samples.append(contentsOf: chunk)
        }
        if samples.count > limit {
            reader.cancelReading()
            return (Array(samples.prefix(limit)), true)
        }
        guard reader.status == .completed else {
            throw reader.error
                ?? SemanticSearchError.unsupportedContent(
                    itemID: ItemID(url.lastPathComponent), detail: "audio could not be decoded")
        }
        return (samples, false)
    }

    // MARK: - Prompts

    /// The model card's prefix for `role`, applied to text only.
    private func prompt(for role: EmbeddingRole) -> String {
        switch role {
        case .document:
            return Self.documentPrefix
        case .query(let instruction):
            return Self.queryPrefix(instruction, defaultTask: configuration.defaultQueryTask)
        }
    }

    /// `task: <instruction> | query: `.
    ///
    /// The instruction is a task description — "search result", "question
    /// answering", "code retrieval" — as the model card's prompts name them.
    /// One already written as a whole prompt, `task: … | query:`, is taken
    /// verbatim and given the space the prompts end with. `nil` or blank uses
    /// the configured default task.
    private static func queryPrefix(_ instruction: String?, defaultTask: String) -> String {
        let task = (instruction ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !task.isEmpty else { return "task: \(defaultTask) | query: " }
        if task.hasPrefix("task:"), task.contains("| query:") { return task + " " }
        return "task: \(task) | query: "
    }

    // MARK: - Helpers

    private func checkDimensions(_ requested: Int?) throws {
        guard let requested, !Self.matryoshkaDimensions.contains(requested) else { return }
        throw SemanticSearchError.configuration(
            "'\(configuration.model.identity)' was trained at "
                + "\(Self.matryoshkaDimensions.map { String($0) }.joined(separator: ", ")) "
                + "dimensions; \(requested) is not one of them")
    }

    /// The model's unit vector, or its renormalised Matryoshka prefix. Empty
    /// when it is not finite — float16 activations or a broken conversion —
    /// rather than stored, where it would sit nearest to nothing or to
    /// everything.
    private static func finished(_ vector: [Float], requested: Int?) -> [Float] {
        guard vector.count == dimensions, vector.allSatisfy(\.isFinite),
            vector.contains(where: { $0 != 0 })
        else { return [] }
        guard let requested, requested < vector.count else { return vector }
        return VectorMath.normalizedPrefix(vector, dimensions: requested)
    }

    private static func isBlank(_ text: String) -> Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: - Loading

    /// What `config.json` says beyond the model's own configuration: the
    /// placeholder ids, read from the root where the reference reads them —
    /// so a text-only load, whose model configuration drops the vision
    /// tower and its ids, still maps them to padding — and the padding id.
    private struct CheckpointHeader: Decodable {
        let modelType: String?
        let placeholders: [Int]
        let padding: Int

        private enum Keys: String, CodingKey {
            case modelType = "model_type"
            case image = "image_token_id"
            case video = "video_token_id"
            case audio = "audio_token_id"
            case text = "text_config"
        }

        private enum TextKeys: String, CodingKey {
            case padding = "pad_token_id"
        }

        init(from decoder: any Decoder) throws {
            let root = try decoder.container(keyedBy: Keys.self)
            modelType = try root.decodeIfPresent(String.self, forKey: .modelType)
            placeholders = try [Keys.image, .video, .audio].compactMap {
                try root.decodeIfPresent(Int.self, forKey: $0)
            }
            let text = try? root.nestedContainer(keyedBy: TextKeys.self, forKey: .text)
            padding = try text?.decodeIfPresent(Int.self, forKey: .padding) ?? 0
        }
    }

    /// The checkpoint `configuration` names, downloaded unless it is a local
    /// directory, loaded with its towers.
    private static func load(_ configuration: Configuration) async throws -> Loaded {
        MLXRuntime.apply(configuration.runtime)
        let directory: URL
        if let local = configuration.model.directoryURL {
            directory = local
        } else {
            let downloader = MLXRuntime.downloader(configuration.runtime, for: .embedder)
            directory = try await downloader.download(
                id: configuration.model.hubID ?? configuration.model.identity,
                revision: configuration.model.revision,
                matching: ["*.safetensors", "*.json", "*.jinja"], useLatest: false,
                progressHandler: { _ in })
        }
        return try await load(
            directory: directory, vision: configuration.vision, audio: configuration.audio)
    }

    /// One copy of the weights, with only the towers asked for.
    ///
    /// Not `EmbedderModelFactory`: its creator builds every tower in
    /// `config.json`, ~1.5 GB resident where text alone is ~0.54 GB. Every
    /// processor is read and checked whichever towers load, because the
    /// descriptor names their settings for every provider of this space.
    private static func load(directory: URL, vision: Bool, audio: Bool) async throws -> Loaded {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let header = try JSONDecoder().decode(CheckpointHeader.self, from: data)
        switch header.modelType {
        case "embedding_gemma2"?:
            break
        case "embedding_gemma2_text"?:
            // The runtime loads these, so say what to do instead rather than
            // fail later on a processor config the checkpoint need not have.
            throw SemanticSearchError.configuration(
                "config.json has model_type 'embedding_gemma2_text', a standalone text "
                    + "checkpoint, which EmbeddingGemma2Provider does not load; for text "
                    + "alone load the full 'embedding_gemma2' checkpoint with vision: false, "
                    + "which keeps only its text tower resident")
        default:
            throw SemanticSearchError.configuration(
                "config.json has model_type '\(header.modelType ?? "none")'; "
                    + "EmbeddingGemma2Provider loads 'embedding_gemma2' checkpoints")
        }
        let checkpoint = try JSONDecoder().decode(
            MLXLMCommon.EmbeddingGemma2Configuration.self, from: data)
        guard checkpoint.embeddingDim == dimensions else {
            throw SemanticSearchError.configuration(
                "the checkpoint embeds at \(checkpoint.embeddingDim) dimensions, "
                    + "not \(dimensions)")
        }
        if vision, checkpoint.vision?.videoTokenID == nil {
            throw SemanticSearchError.configuration(
                "the checkpoint has no vision tower; configure the provider with vision: false")
        }
        if audio, checkpoint.audio == nil {
            throw SemanticSearchError.configuration(
                "the checkpoint has no audio tower; configure the provider with audio: false")
        }

        let images = try EmbeddingGemma2ImageProcessor(directory: directory)
        let videos = try EmbeddingGemma2VideoProcessor(directory: directory)
        let sounds = try EmbeddingGemma2AudioProcessor(directory: directory)
        guard images.configuration.maxSoftTokens == 280,
            videos.framesPerSecond == 1, videos.maximumFrames == 32, videos.budget == 140,
            sounds.sampleRate == 16_000,
            EmbeddingGemma2AudioProcessor.maximumSampleCount == 480_000
        else {
            throw SemanticSearchError.configuration(
                "processor_config.json asks for \(images.configuration.maxSoftTokens) image "
                    + "tokens, video at \(videos.framesPerSecond) fps up to "
                    + "\(videos.maximumFrames) frames of \(videos.budget) tokens and audio at "
                    + "\(sounds.sampleRate) Hz; this provider's space is defined at 280, "
                    + "1 fps, 32 frames of 140 and 16 kHz")
        }

        let model = MLXLMCommon.EmbeddingGemma2(
            checkpoint.selectingEncoders(vision: vision, audio: audio))
        let quantization = try JSONDecoder()
            .decode(MLXLMCommon.BaseConfiguration.self, from: data)
            .perLayerQuantization
        try await loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: quantization)

        let tokenizer = try await MLXRuntime.tokenizerLoader.load(from: directory)
        guard
            let begin = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) })
                ?? tokenizer.convertTokenToId("<bos>"),
            let end = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>")
        else {
            throw SemanticSearchError.configuration("the tokenizer has no <bos> or <eos> token")
        }

        return Loaded(
            model: model,
            tokenizer: tokenizer,
            images: vision ? images : nil,
            videos: vision ? videos : nil,
            audio: audio ? sounds : nil,
            beginOfSequence: begin,
            endOfSequence: end,
            padding: header.padding,
            placeholders: Set(header.placeholders + checkpoint.softTokenIDs))
    }
}

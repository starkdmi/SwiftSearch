import CoreImage
import Foundation
import MLXLMCommon
import MLXVLM
import SemanticSearch

@preconcurrency import MLX

/// Captions images with an MLX vision-language model.
///
/// The default is SmolVLM2-500M with a short caption prompt, greedy decoding
/// and a tile-aware resize. Any VLM the MLX runtime
/// loads can be swapped in through ``Configuration/model`` and
/// ``Configuration/prompt``; both are part of the fingerprint, so changing
/// either re-captions on the next ingest.
public actor MLXImageCaptioner: ImageCaptioner {

    public struct Configuration: Sendable, Hashable {
        public var model: ModelReference
        public var prompt: String
        public var maximumTokens: Int
        /// Longest side the image is reduced to before the model sees it.
        /// SmolVLM's processor then scales it onto its full tiling canvas
        /// (2048 pixels, 512-pixel tiles) however small it is, so this sets
        /// how much detail the tiles hold, not how many there are.
        public var maximumDimension: Int
        /// Language the model writes in. SmolVLM2 answers in English.
        public var language: String?
        public var runtime: MLXRuntime.Configuration

        public init(
            model: ModelReference = MLXImageCaptioner.defaultModel,
            prompt: String = "Provide a caption for an image.",
            maximumTokens: Int = 128,
            maximumDimension: Int = 1_024,
            language: String? = "en",
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) {
            self.model = model
            self.prompt = prompt
            self.maximumTokens = max(1, maximumTokens)
            self.maximumDimension = max(64, maximumDimension)
            self.language = language
            self.runtime = runtime
        }
    }

    /// SmolVLM2-500M, 8-bit: ~0.75 s to load, ~1.4 s and ~14 tokens per
    /// caption on an M1 Pro.
    public static let defaultModel: ModelReference =
        "starkdmi/SmolVLM2-500M-Video-Instruct-mlx-8bit@2e8ce1273bb560451061cf3e33e0361b6132521c"

    public let configuration: Configuration
    private var container: ModelContainer?
    private var tilingCanvas: TilingCanvas?
    private let context = CIContext()

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    public var identifier: String { configuration.model.identity }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("mlx.caption.v1")
            hasher.append(configuration.model.identity)
            hasher.append(configuration.prompt)
            hasher.append(configuration.maximumTokens)
            hasher.append(configuration.maximumDimension)
            hasher.append("greedy")
        }
    }

    public var language: String? { configuration.language }

    public var isPrepared: Bool { container != nil }

    public func prepare() async throws {
        guard container == nil else { return }
        MLXRuntime.apply(configuration.runtime)
        do {
            let downloader = MLXRuntime.downloader(configuration.runtime, for: .languageModel)
            let directory = try await resolve(
                configuration: configuration.model.runtimeConfiguration,
                from: downloader, useLatest: false, progressHandler: { _ in }
            ).modelDirectory
            var modelConfiguration = ModelConfiguration(directory: directory)
            if Self.modelType(in: directory) == "smolvlm" {
                // SmolVLM2's processor renders the prompt with Qwen2-VL's
                // message layout, text before the image; the model was trained
                // with the image first.
                modelConfiguration.messageGenerator = MediaFirstMessageGenerator()
            }
            let container = try await VLMModelFactory.shared.loadContainer(
                from: downloader, using: MLXRuntime.tokenizerLoader,
                configuration: modelConfiguration)
            tilingCanvas = Self.tilingCanvas(in: directory)
            self.container = container
        } catch {
            throw SemanticSearchError.modelUnavailable(
                configuration.model.identity, underlying: error)
        }
    }

    public func release() async {
        container = nil
        MLXRuntime.releaseCache()
    }

    public func caption(_ image: ImagePayload) async throws -> String {
        guard let container else {
            throw SemanticSearchError.providerNotPrepared(configuration.model.identity)
        }
        var prepared = try preparedImage(image)
        if let tilingCanvas {
            prepared = Self.onCanvas(prepared, tilingCanvas)
        }
        let input = UserInput(chat: [.user(configuration.prompt, images: [.ciImage(prepared)])])
        let box = UncheckedBox(input)
        let lmInput = try await container.prepare(input: box.take())
        let stream = try await container.generate(
            input: lmInput,
            parameters: GenerateParameters(
                maxTokens: configuration.maximumTokens, temperature: 0))
        var text = ""
        for await generation in stream {
            if let chunk = generation.chunk { text += chunk }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Image

    private func preparedImage(_ payload: ImagePayload) throws -> CIImage {
        let image: CIImage?
        switch payload {
        case .fileURL(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
            }
            image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true])
        case .encoded(let data, _):
            image = CIImage(data: data, options: [.applyOrientationProperty: true])
        }
        guard let image else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "image bytes could not be decoded")
        }
        let extent = image.extent.integral
        let origin = image.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
        guard
            let size = Self.preferredSize(
                width: Int(extent.width), height: Int(extent.height),
                maximumLongSide: configuration.maximumDimension)
        else { return origin }
        let scaled = origin.transformed(
            by: CGAffineTransform(
                scaleX: CGFloat(size.width) / extent.width,
                y: CGFloat(size.height) / extent.height))
        // Rendered once so the model's processor resamples a small bitmap
        // rather than re-running the filter graph on the full frame.
        guard
            let rendered = context.createCGImage(
                scaled, from: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        else { return scaled }
        return CIImage(cgImage: rendered)
    }

    /// SmolVLM works in 512-pixel tiles, so aim the long side at
    /// `maximumLongSide`, but if that leaves the short side just over one tile
    /// (512…640) shrink until it is one tile — the extra row of tiles costs as
    /// much as the first and sees a sliver. Never upscales; the long side lands
    /// on a 32-pixel grid. `nil` when the image is already small enough.
    static func preferredSize(
        width: Int, height: Int, maximumLongSide: Int = 1_024, tile: Int = 512,
        alignment: Int = 32
    ) -> (width: Int, height: Int)? {
        guard width > 0, height > 0 else { return nil }
        let longSide = Double(max(width, height))
        let shortSide = Double(min(width, height))
        let target = Double(maximumLongSide)
        let shortAtTarget = target * shortSide / longSide

        let desiredLong: Double
        if shortAtTarget > Double(tile), shortAtTarget < Double(tile) * 1.25 {
            desiredLong = Double(tile) * longSide / shortSide
        } else {
            desiredLong = target
        }
        let clamped = min(longSide, desiredLong)
        guard clamped < longSide * 0.999 else { return nil }

        let rounded = max(1, Int((clamped + 1e-6).rounded(.down)))
        let aligned = rounded - rounded % alignment
        let targetLong = min(Int(longSide), aligned > 0 ? aligned : rounded)
        guard targetLong < Int(longSide) else { return nil }
        let scaledShort = max(1, Int((shortSide * Double(targetLong) / longSide + 1e-6).rounded(.down)))
        return width >= height ? (targetLong, scaledShort) : (scaledShort, targetLong)
    }

    // MARK: - Tiling canvas

    /// The canvas an Idefics3-style processor (SmolVLM) cuts into tiles: the
    /// long side resized to `size.longest_edge`, both sides rounded up to a
    /// whole tile of `max_image_size.longest_edge`.
    struct TilingCanvas: Sendable {
        var longestEdge: CGFloat
        var tile: CGFloat
    }

    static func modelType(in directory: URL) -> String? {
        struct Config: Decodable { let model_type: String? }
        return (try? Data(contentsOf: directory.appendingPathComponent("config.json")))
            .flatMap { try? JSONDecoder().decode(Config.self, from: $0).model_type }
    }

    /// Read from the model's `preprocessor_config.json`; `nil` for a model
    /// that does not split images.
    static func tilingCanvas(in directory: URL) -> TilingCanvas? {
        struct Edge: Decodable { let longest_edge: Int }
        struct Preprocessor: Decodable {
            let do_image_splitting: Bool?
            let size: Edge?
            let max_image_size: Edge?
        }
        guard
            let data = try? Data(
                contentsOf: directory.appendingPathComponent("preprocessor_config.json")),
            let preprocessor = try? JSONDecoder().decode(Preprocessor.self, from: data),
            preprocessor.do_image_splitting == true,
            let longest = preprocessor.size?.longest_edge,
            let tile = preprocessor.max_image_size?.longest_edge,
            longest > 0, tile > 0
        else { return nil }
        return TilingCanvas(longestEdge: CGFloat(longest), tile: CGFloat(tile))
    }

    /// The image resized onto the full canvas, smaller images included.
    ///
    /// Hugging Face's processor scales every image to the canvas, and so did
    /// mlx-swift-lm until 3.x stopped enlarging small ones (its #208): a 16:9
    /// keyframe then goes in as 4 tiles instead of 12 and is captioned in half
    /// the time — and worse: searching paraphrases of what a frame shows found
    /// it measurably less often. The size and the Lanczos resample are the
    /// processor's own, so 3.x resamples this at scale 1 and cuts the tiles 2.x
    /// cut.
    static func onCanvas(_ image: CIImage, _ canvas: TilingCanvas) -> CIImage {
        let size = image.extent.size
        guard size.width > 0, size.height > 0 else { return image }
        let fitted = MediaProcessing.bestFit(
            size, in: CGSize(width: canvas.longestEdge, height: canvas.longestEdge))
        let aspectRatio = fitted.width / fitted.height
        let target: CGSize
        if size.width >= size.height {
            let width = ceil(fitted.width / canvas.tile) * canvas.tile
            target = CGSize(
                width: width, height: ceil(width / aspectRatio / canvas.tile) * canvas.tile)
        } else {
            let height = ceil(fitted.height / canvas.tile) * canvas.tile
            target = CGSize(
                width: ceil(height * aspectRatio / canvas.tile) * canvas.tile, height: height)
        }
        guard target != size else { return image }
        // Resampled between the sRGB tone curve and its inverse, because the
        // processor applies the curve before it resamples: the pixels are
        // 2.x's. The captions still differ from 2.x's on some frames — MLX
        // 0.32's arithmetic and a bf16 vision tower move greedy decoding — and
        // score within the run-to-run spread.
        return MediaProcessing.resampleLanczos(image.toSRGB(), to: target).toLinear()
    }
}

/// The user's images and videos before their text, as SmolVLM2 was trained.
struct MediaFirstMessageGenerator: MessageGenerator {
    func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var content: [MLXLMCommon.Message] = []
        if message.role == .user {
            content += message.images.map { _ in ["type": "image"] }
            content += message.videos.map { _ in ["type": "video"] }
            content += [["type": "text", "text": message.content]]
        } else {
            content += [["type": "text", "text": message.content]]
            content += message.images.map { _ in ["type": "image"] }
            content += message.videos.map { _ in ["type": "video"] }
        }
        return ["role": message.role.rawValue, "content": content]
    }
}

import Foundation

/// Describes an image in words. Implement this to use a different captioning
/// model; the bundled one is SmolVLM2 (`MLXImageCaptioner`).
///
/// A caption is what lets a frame be found by a text-only profile: the 0.6B
/// text embedder and reranker cannot read pixels, but they read "a dog
/// running along a beach" as well as any transcript line. It is also the
/// cheap way to rerank frames under the multimodal profile — see
/// ``SearchOptions/reranksImages``.
public protocol ImageCaptioner: Actor {
    /// Model and prompt, e.g. `"starkdmi/SmolVLM2-500M-Video-Instruct-mlx-8bit"`.
    /// Recorded as the caption's ``Provenance/producer``.
    var identifier: String { get }
    /// Changes whenever the captions would: model, weights, prompt, image
    /// preprocessing, decoding. Part of the processor fingerprint, so a new
    /// captioner re-captions on the next ingest and an unchanged one does not.
    var configurationFingerprint: String { get }
    /// BCP-47 language the captions are written in, when the model has one.
    /// `nil` leaves it to detection.
    var language: String? { get }

    func prepare() async throws
    func release() async
    var isPrepared: Bool { get }

    /// One caption for one image. Empty when the model has nothing to say;
    /// an empty caption is not indexed.
    func caption(_ image: ImagePayload) async throws -> String
}

/// Wraps a processor and captions the images it produces.
///
/// Each visual representation gets a sibling `.caption` representation on the
/// same item, with the same locator: the frame stays where it was, and now has
/// words. Wrap the frame source — ``ContentProcessor``s for shot frames,
/// sampled frames or plain images — and combine the result with a transcriber
/// through ``CombinedContentProcessor`` as before.
public struct CaptioningProcessor: ContentProcessor {

    public struct Configuration: Sendable, Hashable {
        /// Representation kinds to caption. Video is left out by default: a
        /// clip's frames are captioned individually, and one sentence for a
        /// whole clip says less than each of them.
        public var kinds: Set<RepresentationKind>
        /// Release the captioner after each input. On a device where the
        /// captioner and the embedder should not be resident together — the
        /// engine embeds right after processing — turn this on; reloading a
        /// 500M captioner costs under a second.
        public var releasesAfterEachInput: Bool

        public init(kinds: Set<RepresentationKind> = [.image], releasesAfterEachInput: Bool = false) {
            self.kinds = kinds
            self.releasesAfterEachInput = releasesAfterEachInput
        }

        public static let `default` = Configuration()
    }

    public let base: any ContentProcessor
    public let captioner: any ImageCaptioner
    public let configuration: Configuration
    private let captionerIdentifier: String
    private let captionerFingerprint: String
    private let captionLanguage: String?

    /// The captioner's identity is read once, here, so the processor's
    /// fingerprint is a plain property like every other processor's.
    public init(
        wrapping base: any ContentProcessor,
        captioner: any ImageCaptioner,
        configuration: Configuration = .default
    ) async {
        self.base = base
        self.captioner = captioner
        self.configuration = configuration
        self.captionerIdentifier = await captioner.identifier
        self.captionerFingerprint = await captioner.configurationFingerprint
        self.captionLanguage = await captioner.language
    }

    public var identifier: String { base.identifier }

    public var version: String { base.version + "+caption.1" }

    public var capabilities: ProcessorCapabilities {
        var capabilities = base.capabilities
        capabilities.producedKinds.insert(.caption)
        return capabilities
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("captioning.v1")
            hasher.append(base.configurationFingerprint)
            hasher.append(captionerIdentifier)
            hasher.append(captionerFingerprint)
            for kind in configuration.kinds.map(\.rawValue).sorted() { hasher.append(kind) }
        }
    }

    public func canProcess(_ content: RawContent) -> Bool { base.canProcess(content) }

    /// A caption says what is in the picture, not where the file is: the
    /// wrapped processor's relocation is all there is to do.
    public func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem] {
        try await base.relocate(items, from: old, to: content)
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        let items = try await base.process(content, onProgress: onProgress)
        do {
            let captioned = try await caption(items, of: content, onProgress: onProgress)
            if configuration.releasesAfterEachInput { await captioner.release() }
            return captioned
        } catch {
            if configuration.releasesAfterEachInput { await captioner.release() }
            throw error
        }
    }

    private func caption(
        _ items: [SearchItem], of content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        var items = items
        let targets = items.indices.flatMap { itemIndex in
            items[itemIndex].representations.indices.compactMap {
                representationIndex -> (Int, Int, ImagePayload)? in
                let representation = items[itemIndex].representations[representationIndex]
                guard configuration.kinds.contains(representation.kind) else { return nil }
                switch representation.content {
                case .imageData(let data, let contentType):
                    return (itemIndex, representationIndex, .encoded(data, contentType: contentType))
                case .mediaReference(let url) where url.isFileURL:
                    return (itemIndex, representationIndex, .fileURL(url))
                default:
                    return nil
                }
            }
        }
        guard !targets.isEmpty else { return items }

        let provenance = Provenance(
            producer: captionerIdentifier, version: "1",
            configurationFingerprint: captionerFingerprint)

        for (done, (itemIndex, representationIndex, image)) in targets.enumerated() {
            try Task.checkCancellation()
            onProgress?(
                ProcessingProgress(
                    itemID: content.id, stage: "captioning",
                    fraction: Double(done) / Double(targets.count)))
            let source = items[itemIndex].representations[representationIndex]
            let captionID = RepresentationID("\(source.id.rawValue)-caption")
            // The processor underneath may already caption — a caller-supplied
            // caption is better evidence than a generated one.
            guard
                !items[itemIndex].representations.contains(where: {
                    $0.id == captionID || ($0.kind == .caption && $0.derivedFrom == source.id)
                })
            else { continue }

            // Per image: the captioner may be the engine's query captioner,
            // and `releaseModels()` between two images would otherwise fail
            // the rest of the file.
            if !(await captioner.isPrepared) { try await captioner.prepare() }
            let caption = try await captioner.caption(image)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !caption.isEmpty else { continue }
            items[itemIndex].representations.append(
                ContentRepresentation(
                    id: captionID, kind: .caption, content: .text(caption),
                    language: captionLanguage, provenance: provenance,
                    locator: source.locator, derivedFrom: source.id))
        }
        onProgress?(ProcessingProgress(itemID: content.id, stage: "captioning", fraction: 1))
        return items
    }
}

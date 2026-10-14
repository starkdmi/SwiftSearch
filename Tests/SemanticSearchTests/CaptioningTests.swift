import Foundation
import Testing

@testable import SemanticSearch

/// Captions from a table, so a test can say what the model "saw".
private actor TableCaptioner: ImageCaptioner {
    let captions: [Data: String]
    let fingerprint: String
    private(set) var prepared = false
    private(set) var releases = 0
    private(set) var captioned = 0

    init(_ captions: [Data: String], fingerprint: String = "table.v1") {
        self.captions = captions
        self.fingerprint = fingerprint
    }

    var identifier: String { "table-captioner" }
    var configurationFingerprint: String { fingerprint }
    var language: String? { "en" }
    var isPrepared: Bool { prepared }
    func prepare() async throws { prepared = true }
    func release() async {
        prepared = false
        releases += 1
    }

    func caption(_ image: ImagePayload) async throws -> String {
        captioned += 1
        guard case .encoded(let data, _) = image else { return "" }
        return captions[data] ?? ""
    }
}

/// Refuses to caption unloaded, as `MLXImageCaptioner` does, and takes its
/// time, so an idle timer can fire while a file is being captioned.
private actor SlowCaptioner: ImageCaptioner {
    private(set) var prepared = false
    private(set) var releases = 0

    var identifier: String { "slow-captioner" }
    var configurationFingerprint: String { "slow.v1" }
    var language: String? { "en" }
    var isPrepared: Bool { prepared }
    func prepare() async throws { prepared = true }
    func release() async {
        prepared = false
        releases += 1
    }

    func caption(_ image: ImagePayload) async throws -> String {
        guard prepared else { throw SemanticSearchError.providerNotPrepared(identifier) }
        try await Task.sleep(for: .milliseconds(300))
        return "a frame"
    }
}

/// A video with two frames and nothing else, standing in for shot frames.
private struct FramesProcessor: ContentProcessor {
    var frames: [Data]
    var existingCaption: String?

    var identifier: String { "frames" }
    var version: String { "1" }
    var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(acceptsVideo: true, producedKinds: [.image])
    }
    var configurationFingerprint: String { "frames.v1" }
    func canProcess(_ content: RawContent) -> Bool { true }

    func process(
        _ content: RawContent, onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        let provenance = Provenance(producer: "frames")
        var items = [
            SearchItem(
                id: content.id,
                representations: [
                    ContentRepresentation(
                        id: "title", kind: .originalText, content: .text("a video"),
                        provenance: provenance)
                ])
        ]
        for (index, frame) in frames.enumerated() {
            let locator = SourceLocator(
                source: "video.mp4", frameIndex: index,
                timeRange: .init(start: Double(index) * 10, end: Double(index) * 10 + 10))
            var representations = [
                ContentRepresentation(
                    id: "frame", kind: .image,
                    content: .imageData(frame, contentType: "public.png"),
                    provenance: provenance, locator: locator)
            ]
            if index == 0, let existingCaption {
                representations.append(
                    ContentRepresentation(
                        id: "mine", kind: .caption, content: .text(existingCaption),
                        locator: locator, derivedFrom: "frame"))
            }
            items.append(
                SearchItem(
                    id: ItemID("\(content.id.rawValue)#shot-\(index)"), parentID: content.id,
                    locator: locator, representations: representations, ordinal: index))
        }
        return items
    }
}

private let bar = Data([0x89, 0x50, 0x4E, 0x47, 0x01])
private let street = Data([0x89, 0x50, 0x4E, 0x47, 0x02])
private let blank = Data([0x89, 0x50, 0x4E, 0x47, 0x03])

@Suite("Captioning")
struct CaptioningProcessorTests {

    @Test("Each frame gets a caption beside it, at the frame's moment")
    func captionsFrames() async throws {
        let captioner = TableCaptioner([
            bar: "a bartender pouring a cocktail", street: "a busy street at night",
        ])
        let processor = await CaptioningProcessor(
            wrapping: FramesProcessor(frames: [bar, street]), captioner: captioner)

        let items = try await processor.process(RawContent(id: "v", payload: .text("")))
        #expect(items.count == 3)
        #expect(items[0].representations.map(\.kind) == [.originalText])

        let frame = items[1]
        let caption = try #require(frame.representations.first { $0.kind == .caption })
        #expect(caption.content.text == "a bartender pouring a cocktail")
        #expect(caption.id == "frame-caption")
        #expect(caption.derivedFrom == "frame")
        #expect(caption.language == "en")
        #expect(caption.locator == frame.representations[0].locator)
        #expect(caption.provenance.producer == "table-captioner")
        #expect(frame.representations.contains { $0.kind == .image })
        #expect(items[2].representations.first { $0.kind == .caption }?.content.text
            == "a busy street at night")
    }

    @Test("An empty caption is not indexed, and an existing caption is kept")
    func keepsExistingAndSkipsEmpty() async throws {
        let captioner = TableCaptioner([bar: "generated"])
        let processor = await CaptioningProcessor(
            wrapping: FramesProcessor(frames: [bar, blank], existingCaption: "written by hand"),
            captioner: captioner)

        let items = try await processor.process(RawContent(id: "v", payload: .text("")))
        #expect(items[1].representations.filter { $0.kind == .caption }.map(\.content.text)
            == ["written by hand"])
        #expect(!items[2].representations.contains { $0.kind == .caption })
        // The captioned frame was never sent; the blank one was and said nothing.
        #expect(await captioner.captioned == 1)
    }

    @Test("A different captioner changes the fingerprint; the processor id does not")
    func fingerprintFollowsCaptioner() async throws {
        let base = FramesProcessor(frames: [bar])
        let first = await CaptioningProcessor(wrapping: base, captioner: TableCaptioner([:]))
        let same = await CaptioningProcessor(wrapping: base, captioner: TableCaptioner([:]))
        let other = await CaptioningProcessor(
            wrapping: base, captioner: TableCaptioner([:], fingerprint: "table.v2"))

        #expect(first.configurationFingerprint == same.configurationFingerprint)
        #expect(first.configurationFingerprint != other.configurationFingerprint)
        #expect(first.configurationFingerprint != base.configurationFingerprint)
        #expect(first.identifier == base.identifier)
        #expect(first.capabilities.producedKinds.contains(.caption))
    }

    @Test("The captioner is released after each input when asked")
    func releasesAfterEachInput() async throws {
        let captioner = TableCaptioner([bar: "x"])
        let processor = await CaptioningProcessor(
            wrapping: FramesProcessor(frames: [bar]), captioner: captioner,
            configuration: .init(releasesAfterEachInput: true))
        _ = try await processor.process(RawContent(id: "v", payload: .text("")))
        #expect(await captioner.releases == 1)
        #expect(await !captioner.isPrepared)
    }

    @Test("Neither the idle timer nor a release fails a file mid-captioning")
    func releaseDuringCaptioning() async throws {
        // The engine's query captioner and the processor's are one model, as
        // in the text profile; the timer used to release it between frames.
        let captioner = SlowCaptioner()
        let processor = await CaptioningProcessor(
            wrapping: FramesProcessor(frames: [bar, street, blank]), captioner: captioner)
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: false),
                        kinds: [.originalText, .caption], indexKind: .exact)
                ],
                reranker: nil, processor: processor,
                runtime: RuntimePolicy(modelIdleTimeout: .milliseconds(50)),
                queryCaptioner: captioner))
        try await engine.prepare()

        let ingest = Task { try await engine.ingest([RawContent(id: "v", payload: .text(""))]) }
        try await Task.sleep(for: .milliseconds(450))
        await engine.releaseModels()
        let report = try await ingest.value
        #expect(report.skipped.isEmpty)

        let items = try await engine.listItems(
            matching: .withinParent("v"), topLevelOnly: false, limit: 10, includesRepresentations: true
        ).items
        #expect(items.filter { $0.representations.contains { $0.kind == .caption } }.count == 3)
        try await engine.close()
    }

    @Test("Captions make frames findable in a text-only space")
    func textSpaceFindsCaptionedFrame() async throws {
        let captioner = TableCaptioner([
            bar: "a bartender pouring a cocktail", street: "a busy street at night",
        ])
        let processor = await CaptioningProcessor(
            wrapping: FramesProcessor(frames: [bar, street]), captioner: captioner)
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: false),
                        kinds: [.originalText, .caption], indexKind: .exact)
                ],
                reranker: ScriptedRerankingProvider.termOverlap(),
                processor: processor))
        try await engine.prepare()
        let report = try await engine.ingest([RawContent(id: "v", payload: .text(""))])
        // The pixels are not searchable in a text space, but the frame is,
        // through its caption; it is not reported as lost.
        #expect(report.skipped.isEmpty)

        let results = try await engine.search("cocktail bartender")
        let top = try #require(results.hits.first)
        #expect(top.itemID == "v#shot-0")
        #expect(top.kind == .caption)
        #expect(top.evidence.rerankScore != nil)
        #expect(top.locator.timeRange?.start == 0)
        try await engine.close()
    }
}

@Suite("Reranking frames by their captions")
struct RerankByCaptionTests {

    private static func visualReranker() -> ScriptedRerankingProvider {
        ScriptedRerankingProvider(
            capabilities: RerankingCapabilities(acceptsImages: true, maximumImagesPerPair: 2)
        ) { query, candidate in
            let terms = Set(query.text.lowercased().split(separator: " ").map(String.init))
            let text = candidate.text.lowercased()
            return RerankScore(score: Float(terms.filter { text.contains($0) }.count))
        }
    }

    /// A frame whose only indexed representation is its pixels — the caption
    /// is stored as context (weight 0) — so the pool's evidence for it is the
    /// image and nothing else.
    private func engine(
        reranker: ScriptedRerankingProvider, reranksImages: Bool = true
    ) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "mm", provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: true),
                        kinds: [.originalText, .image], indexKind: .exact)
                ],
                reranker: reranker,
                reranksImages: reranksImages))
        try await engine.prepare()
        try await engine.upsert([
            textItem("note", "a cocktail recipe"),
            SearchItem(
                id: "frame",
                representations: [
                    ContentRepresentation(
                        id: "frame", kind: .image,
                        content: .imageData(bar, contentType: "public.png")),
                    ContentRepresentation(
                        id: "frame-caption", kind: .caption,
                        content: .text("a bartender pouring a cocktail"), weight: 0),
                ]),
        ])
        return engine
    }

    private func sentFrame(_ reranker: ScriptedRerankingProvider) async -> RerankCandidate? {
        await reranker.calls.flatMap(\.candidates).first { $0.kind == .image }
    }

    @Test("By default an image-capable reranker is shown the frame")
    func showsPixelsByDefault() async throws {
        let reranker = Self.visualReranker()
        let engine = try await engine(reranker: reranker)
        _ = try await engine.search(
            SearchQuery(text: "cocktail", images: [.encoded(street, contentType: "public.png")]))
        let frame = try #require(await sentFrame(reranker))
        #expect(frame.images.count == 1)
        // Shown the picture, it still reads what was written about it.
        #expect(frame.text == "a bartender pouring a cocktail")
        #expect(await reranker.calls.first?.query.images.count == 1)
        try await engine.close()
    }

    @Test("A caption that won retrieval brings its frame to a reranker that sees pixels")
    func captionBringsItsFrame() async throws {
        let reranker = Self.visualReranker()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: true),
                        kinds: [.originalText, .caption], indexKind: .exact)
                ],
                reranker: reranker))
        try await engine.prepare()
        try await engine.upsert([
            textItem("note", "a cocktail recipe"),
            SearchItem(
                id: "frame",
                representations: [
                    ContentRepresentation(
                        id: "frame", kind: .image,
                        content: .imageData(bar, contentType: "public.png")),
                    ContentRepresentation(
                        id: "frame-caption", kind: .caption,
                        content: .text("a bartender pouring a cocktail"),
                        derivedFrom: "frame"),
                ]),
        ])

        _ = try await engine.search("cocktail")
        var sent = try #require(
            await reranker.calls.flatMap(\.candidates).first { $0.kind == .caption })
        #expect(sent.images.count == 1)
        #expect(sent.text == "a bartender pouring a cocktail")

        await reranker.reset()
        _ = try await engine.search("cocktail", options: SearchOptions(reranksImages: false))
        sent = try #require(
            await reranker.calls.flatMap(\.candidates).first { $0.kind == .caption })
        #expect(sent.images.isEmpty)
        try await engine.close()
    }

    @Test("reranksImages: false reads the frame through its caption")
    func readsCaptionInstead() async throws {
        let reranker = Self.visualReranker()
        let engine = try await engine(reranker: reranker)
        let results = try await engine.search(
            SearchQuery(text: "cocktail", images: [.encoded(street, contentType: "public.png")]),
            options: SearchOptions(reranksImages: false))

        let frame = try #require(await sentFrame(reranker))
        #expect(frame.images.isEmpty)
        #expect(frame.text == "a bartender pouring a cocktail")
        #expect(await reranker.calls.first?.query.images.isEmpty == true)
        #expect(results.diagnostics.rerankIgnoredQueryImages)
        #expect(results.diagnostics.rerankUnscoredCandidates == 0)
        try await engine.close()
    }

    @Test("The engine default applies when the query does not say")
    func engineDefault() async throws {
        let reranker = Self.visualReranker()
        let engine = try await engine(reranker: reranker, reranksImages: false)
        _ = try await engine.search("cocktail")
        let frame = try #require(await sentFrame(reranker))
        #expect(frame.images.isEmpty)

        await reranker.reset()
        _ = try await engine.search("cocktail", options: SearchOptions(reranksImages: true))
        #expect(try #require(await sentFrame(reranker)).images.count == 1)
        try await engine.close()
    }

    @Test("An image-only query is not reranked by text")
    func imageOnlyQuerySkips() async throws {
        let reranker = Self.visualReranker()
        let engine = try await engine(reranker: reranker)
        let results = try await engine.search(
            SearchQuery(images: [.encoded(bar, contentType: "public.png")]),
            options: SearchOptions(reranksImages: false))
        #expect(results.diagnostics.rerankSkippedReason != nil)
        #expect(await reranker.calls.isEmpty)
        try await engine.close()
    }

    @Test("A frame with no words is left unscored rather than scored as empty")
    func uncaptionedFrameIsUnscored() async throws {
        let reranker = Self.visualReranker()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "mm", provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: true),
                        indexKind: .exact)
                ],
                reranker: reranker))
        try await engine.prepare()
        try await engine.upsert([
            textItem("note", "a cocktail recipe"),
            SearchItem(
                id: "frame",
                representations: [
                    ContentRepresentation(
                        id: "frame", kind: .image,
                        content: .imageData(bar, contentType: "public.png"))
                ]),
        ])
        let results = try await engine.search(
            "cocktail", options: SearchOptions(reranksImages: false))
        #expect(await sentFrame(reranker) == nil)
        #expect(results.diagnostics.rerankUnscoredCandidates == 1)
        try await engine.close()
    }
}

@Suite("Searching with a captioned query image")
struct QueryCaptionTests {

    /// A text-only engine holding one captioned frame and one transcript line,
    /// captioning query images with `captioner`.
    private func engine(_ captioner: TableCaptioner) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text", provider: HashingEmbeddingProvider(dimensions: 64),
                        kinds: [.caption, .transcript], indexKind: .exact)
                ],
                reranker: ScriptedRerankingProvider.termOverlap(),
                queryCaptioner: captioner))
        try await engine.prepare()
        try await engine.upsert([
            SearchItem(
                id: "shot",
                representations: [
                    ContentRepresentation(
                        id: "frame", kind: .image,
                        content: .imageData(bar, contentType: "public.png")),
                    ContentRepresentation(
                        id: "frame-caption", kind: .caption,
                        content: .text("a bartender pouring a cocktail"), derivedFrom: "frame"),
                ]),
            textItem("line", "the weather will be cold tomorrow", kind: .transcript),
        ])
        return engine
    }

    @Test("The query captioner follows residency, unless a processor loaded it")
    func captionerResidency() async throws {
        func engine(_ captioner: TableCaptioner) async throws -> SemanticSearchEngine {
            let engine = try await SemanticSearchEngine(
                configuration: EngineConfiguration(
                    spaces: [
                        EmbeddingSpaceConfiguration(
                            id: "text", provider: HashingEmbeddingProvider(dimensions: 64),
                            kinds: [.caption], indexKind: .exact)
                    ],
                    reranker: nil, runtime: RuntimePolicy(sequentialModelResidency: true),
                    queryCaptioner: captioner))
            try await engine.upsert(textItem("shot", "a bartender", kind: .caption))
            return engine
        }
        let query = SearchQuery(images: [.encoded(bar, contentType: "public.png")])

        // Loaded for the query: unloaded when the embedder takes over.
        let owned = TableCaptioner([bar: "a bartender"])
        let first = try await engine(owned)
        _ = try await first.search(query)
        #expect(await owned.prepared == false)
        #expect(await owned.releases == 1)

        // Already loaded — by a processor captioning what it indexes: left be,
        // until every model is released.
        let shared = TableCaptioner([bar: "a bartender"])
        try await shared.prepare()
        let second = try await engine(shared)
        _ = try await second.search(query)
        #expect(await shared.prepared)
        await second.releaseModels()
        #expect(await shared.prepared == false)
    }

    @Test("An image-only query is searched as its caption")
    func imageOnly() async throws {
        let engine = try await engine(TableCaptioner([bar: "a bartender pouring a cocktail"]))
        let results = try await engine.search(
            SearchQuery(images: [.encoded(bar, contentType: "public.png")]))
        #expect(results.query.text == "a bartender pouring a cocktail")
        #expect(results.query.images.isEmpty)
        #expect(results.diagnostics.queryCaptions == ["a bartender pouring a cocktail"])
        #expect(results.hits.first?.itemID == "shot")
        #expect(results.diagnostics.rerankSkippedReason == nil)
        try await engine.close()
    }

    @Test("A question keeps its words, with the caption after it as a frame")
    func questionAndImage() async throws {
        let engine = try await engine(TableCaptioner([bar: "a bartender pouring a cocktail"]))
        let results = try await engine.search(
            SearchQuery(
                text: "when is this made?", images: [.encoded(bar, contentType: "public.png")]))
        #expect(results.query.text == "when is this made?\nFrame: a bartender pouring a cocktail")
        #expect(results.hits.first?.itemID == "shot")
        try await engine.close()
    }

    @Test("An image the captioner has nothing to say about stays an image")
    func emptyCaption() async throws {
        let engine = try await engine(TableCaptioner([:]))
        let results = try await engine.search(
            SearchQuery(text: "cocktail", images: [.encoded(blank, contentType: "public.png")]))
        #expect(results.query.text == "cocktail")
        #expect(results.query.images.count == 1)
        #expect(results.diagnostics.queryCaptions.isEmpty)
        try await engine.close()
    }

    @Test("The question's language is detected, not the caption's")
    func languageFromQuestion() async throws {
        let engine = try await engine(TableCaptioner([bar: "a bartender pouring a cocktail"]))
        let results = try await engine.search(
            SearchQuery(
                text: "когда готовят этот коктейль?",
                images: [.encoded(bar, contentType: "public.png")]))
        #expect(results.diagnostics.detectedQueryLanguage == "ru")
        try await engine.close()
    }
}

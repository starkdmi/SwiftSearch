import Foundation
import SemanticSearch
import Testing

/// Returns fixed windows whatever the file.
actor ScriptedSoundClassifier: SoundClassifier {
    let windows: [SoundWindow]
    init(_ windows: [SoundWindow]) { self.windows = windows }
    var identifier: String { "scripted.sound" }
    var configurationFingerprint: String { "scripted" }
    var isPrepared: Bool { true }
    func prepare() async throws {}
    func release() async {}
    func classify(_ url: URL) async throws -> [SoundWindow] { windows }
}

@Suite("Sound events")
struct SoundEventTests {

    /// 3 s windows every 1.5 s, as Apple's classifier returns them.
    static func windows(_ scores: [[String: Float]]) -> [SoundWindow] {
        scores.enumerated().map { index, scores in
            SoundWindow(start: Double(index) * 1.5, end: Double(index) * 1.5 + 3, scores: scores)
        }
    }

    static func recording() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sound-\(UUID().uuidString).wav")
        try Data().write(to: url)
        return url
    }

    static let script = windows([
        ["music": 0.9, "speech": 0.8],
        ["music": 0.8, "glass_clink": 0.7],
        ["music": 0.4, "glass_clink": 0.6],  // music drops below 0.5 here
        ["music": 0.7, "dog_bark": 0.3],
        ["music": 0.6],
    ])

    @Test("Consecutive windows over the threshold are one event per class")
    func eventsFromWindows() async throws {
        let processor = await SoundEventProcessor(classifier: ScriptedSoundClassifier(Self.script))
        let items = try await processor.process(
            RawContent(id: "show", payload: .fileURL(try Self.recording())))
        let events = items.dropFirst().map {
            ($0.representations[0].content.text ?? "", $0.locator.timeRange!.start, $0.locator.timeRange!.end)
        }
        #expect(items.first?.id == "show")
        #expect(events.map(\.0) == ["music", "speech", "glass clink", "music"])
        #expect(events.map(\.1) == [0, 0, 1.5, 4.5])
        #expect(events.map(\.2) == [4.5, 3, 6, 9])
        // Peak score as confidence; English class names.
        let clink = items.first { $0.representations.first?.content.text == "glass clink" }!
        #expect(clink.representations[0].provenance.confidence == 0.7)
        #expect(clink.representations[0].kind == .soundEvent)
        #expect(clink.representations[0].language == "en")
        #expect(clink.id == "show#sound-2")
    }

    @Test("Per-class thresholds and excluded classes")
    func thresholdsAndExclusions() async throws {
        let processor = await SoundEventProcessor(
            classifier: ScriptedSoundClassifier(Self.script),
            configuration: .init(thresholds: ["dog_bark": 0.25], excludedClasses: ["speech"]))
        let items = try await processor.process(
            RawContent(id: "show", payload: .fileURL(try Self.recording())))
        let labels = items.dropFirst().compactMap { $0.representations.first?.content.text }
        #expect(!labels.contains("speech"))
        #expect(labels.contains("dog bark"))
    }

    @Test("Settings that change the events change the fingerprint")
    func fingerprint() async throws {
        let classifier = ScriptedSoundClassifier(Self.script)
        let a = await SoundEventProcessor(classifier: classifier)
        let b = await SoundEventProcessor(classifier: classifier, configuration: .init(threshold: 0.6))
        let c = await SoundEventProcessor(
            classifier: classifier, configuration: .init(excludedClasses: ["speech"]))
        #expect(a.configurationFingerprint != b.configurationFingerprint)
        #expect(a.configurationFingerprint != c.configurationFingerprint)
    }

    @Test("A sound is found by its name, at its time")
    func searchable() async throws {
        let processor = await SoundEventProcessor(classifier: ScriptedSoundClassifier(Self.script))
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: false),
                        kinds: [.originalText, .soundEvent], indexKind: .exact)
                ],
                reranker: ScriptedRerankingProvider.termOverlap(),
                processor: processor))
        try await engine.prepare()
        let report = try await engine.ingest([
            RawContent(id: "show", payload: .fileURL(try Self.recording()))
        ])
        #expect(report.skipped.isEmpty)

        let results = try await engine.search("glass clink")
        let top = try #require(results.hits.first)
        #expect(top.itemID == "show#sound-2")
        #expect(top.kind == .soundEvent)
        #expect(top.locator.timeRange?.start == 1.5)
        try await engine.close()
    }
}

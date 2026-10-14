import Foundation
import Testing

@testable import SemanticSearch

@Suite("Moments")
struct MomentTests {

    private func timed(
        _ id: String, _ text: String, _ start: Double, _ end: Double,
        kind: RepresentationKind = .transcript, source: String = "file:///talk.mp4"
    ) -> SearchItem {
        let locator = SourceLocator(source: source, timeRange: .init(start: start, end: end))
        return SearchItem(
            id: ItemID(id), parentID: "talk", locator: locator,
            representations: [
                ContentRepresentation(
                    id: "r", kind: kind, content: .text(text), locator: locator)
            ])
    }

    @Test("Hits close in time in one source form one moment; distant ones do not")
    func groupsByTimeAndSource() async throws {
        let engine = try await SemanticSearchEngine.testEngine(reranker: nil)
        try await engine.upsert([
            SearchItem(id: "talk", representations: []),
            timed("a", "harbour lights at dusk", 10, 12),
            timed("b", "the harbour master waves", 13, 15),
            // Stands in for a frame: another kind, overlapping the lines.
            timed("c", "harbour boats", 11, 14, kind: .caption),
            timed("d", "harbour tides tomorrow", 40, 42),
            timed("e", "harbour in another film", 10, 12, source: "file:///other.mp4"),
        ])

        let results = try await engine.search(
            "harbour", options: SearchOptions(mode: .lexicalOnly, momentGap: 3))
        let talk = results.moments.filter { $0.source == "file:///talk.mp4" }
        #expect(talk.count == 2)
        let early = try #require(talk.first { $0.timeRange.start == 10 })
        #expect(Set(early.hits.map(\.itemID)) == ["a", "b", "c"])
        #expect(early.timeRange.end == 15)
        #expect(early.bestHitByKind[.transcript] != nil)
        #expect(early.bestHitByKind[.caption]?.itemID == "c")
        #expect(results.moments.contains { $0.source == "file:///other.mp4" })

        // Ordered by the strongest member.
        let scores = results.moments.map(\.bestHit.evidence.finalScore)
        #expect(zip(scores, scores.dropFirst()).allSatisfy { $0 >= $1 })

        let plain = try await engine.search("harbour", options: SearchOptions(mode: .lexicalOnly))
        #expect(plain.moments.isEmpty)
    }

    @Test("A weaker hit beyond the limit still joins the moment it belongs to")
    func momentsDrawFromThePool() async throws {
        let engine = try await SemanticSearchEngine.testEngine(reranker: nil)
        var items = [SearchItem(id: "talk", representations: [])]
        for index in 0..<8 {
            let start = Double(index * 30)
            items.append(timed("line-\(index)", "harbour harbour harbour \(index)", start, start + 2))
        }
        // Mentions the query once: ranks below all eight lines.
        items.append(timed("frame", "harbour", 1, 3, kind: .caption))
        try await engine.upsert(items)

        let results = try await engine.search(
            "harbour",
            options: SearchOptions(mode: .lexicalOnly, rerankPoolSize: 20, limit: 3, momentGap: 3))
        #expect(!results.hits.contains { $0.itemID == "frame" })
        let first = try #require(results.moments.first { $0.timeRange.start == 0 })
        #expect(first.hits.contains { $0.itemID == "frame" })
        #expect(results.moments.count <= 3)
    }
}

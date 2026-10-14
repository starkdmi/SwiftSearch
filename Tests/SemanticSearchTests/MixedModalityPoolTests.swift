import Foundation
import Testing

@testable import SemanticSearch

/// Frames reach retrieval through one channel, visual similarity; transcript
/// lines also through BM25. These pin down that the extra channel cannot
/// crowd frames out: the two kinds fill the pool by turns, and the reranker
/// alone orders it.
///
/// The hashing embedder gives an encoded image the vector of
/// `img<byte count>`, so a query image with a frame's byte count matches that
/// frame exactly.
@Suite("Mixed-modality pool")
struct MixedModalityPoolTests {

    private static let frame = Data(repeating: 0x42, count: 7)

    private func engine(
        reranker: (any RerankingProvider)? = nil,
        kinds: Set<RepresentationKind> = [.transcript, .image]
    ) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine.testEngine(reranker: reranker, kinds: kinds)
        var items = [7, 11, 13].map { count in
            SearchItem(
                id: ItemID("shot-\(count)"),
                representations: [
                    ContentRepresentation(
                        id: "frame", kind: .image,
                        content: .imageData(
                            Data(repeating: 0x42, count: count), contentType: "public.png"))
                ])
        }
        for (index, words) in [
            "pour the gin", "gin and tonic", "more gin please", "the gin is local",
            "gin over ice", "a gin bottle", "gin tasting notes", "my brother makes gin",
        ].enumerated() {
            items.append(textItem("line-\(index)", words, kind: .transcript))
        }
        try await engine.upsert(items)
        return engine
    }

    private static func imageFirstReranker() -> ScriptedRerankingProvider {
        ScriptedRerankingProvider(
            capabilities: RerankingCapabilities(acceptsImages: true, maximumImagesPerPair: 2)
        ) { _, candidate in
            RerankScore(score: candidate.images.isEmpty ? 0 : 1)
        }
    }

    @Test("Without a reranker, an image query's identical frame leads though every line matches its words")
    func identicalFrameLeads() async throws {
        let engine = try await engine()
        let results = try await engine.search(
            SearchQuery(text: "gin", images: [.encoded(Self.frame, contentType: "public.png")]),
            options: SearchOptions(mode: .retrievalOnly, limit: 10))
        #expect(results.diagnostics.stratifiedPool)
        #expect(results.hits.first?.itemID == "shot-7")
        try await engine.close()
    }

    @Test("A text query's pool keeps room for frames")
    func poolHoldsBothKinds() async throws {
        let reranker = Self.imageFirstReranker()
        let engine = try await engine(reranker: reranker)
        _ = try await engine.search("gin", options: SearchOptions(rerankPoolSize: 4))
        let sent = try #require(await reranker.calls.first).candidates
        #expect(sent.count(where: { $0.kind == .image }) == 2)
        #expect(sent.count(where: { $0.kind == .transcript }) == 2)
        try await engine.close()
    }

    @Test("The reranker alone orders a mixed pool")
    func rerankerOrdersThePool() async throws {
        let engine = try await engine(reranker: Self.imageFirstReranker())
        let results = try await engine.search(
            "gin", options: SearchOptions(rerankPoolSize: 6, limit: 6))
        let kinds = results.hits.map(\.kind)
        #expect(kinds == [.image, .image, .image, .transcript, .transcript, .transcript])
        #expect(results.hits.map(\.evidence.rerankRank) == [1, 2, 3, 4, 5, 6])
        try await engine.close()
    }

    @Test("Scores follow the order, so moments rank by it too")
    func scoresFollowOrder() async throws {
        let engine = try await engine()
        let results = try await engine.search(
            "gin", options: SearchOptions(mode: .retrievalOnly, rerankPoolSize: 4, limit: 10))
        let scores = results.hits.map(\.evidence.finalScore)
        #expect(scores == scores.sorted(by: >))
        #expect(Set(results.hits.map(\.itemID)).count == results.hits.count)
        try await engine.close()
    }

    @Test("A frame reached by both lists through its caption is pooled once")
    func captionedFrameOnce() async throws {
        let engine = try await engine(kinds: [.transcript, .image, .caption])
        try await engine.upsert([
            SearchItem(
                id: "shot-captioned",
                representations: [
                    ContentRepresentation(
                        id: "frame", kind: .image,
                        content: .imageData(Self.frame, contentType: "public.png")),
                    ContentRepresentation(
                        id: "caption", kind: .caption, content: .text("a glass of gin"),
                        derivedFrom: "frame"),
                ])
        ])
        let results = try await engine.search(
            SearchQuery(text: "gin", images: [.encoded(Self.frame, contentType: "public.png")]),
            options: SearchOptions(mode: .retrievalOnly, limit: 20))
        let ids = results.hits.map(\.itemID)
        #expect(ids.count(where: { $0 == "shot-captioned" }) == 1)
        #expect(Set(ids).count == ids.count)
        try await engine.close()
    }

    @Test("An index without pictures ranks by plain fusion")
    func textOnlyIsUnchanged() async throws {
        let engine = try await engine(kinds: [.transcript])
        let results = try await engine.search(
            "gin", options: SearchOptions(mode: .retrievalOnly, limit: 10))
        #expect(!results.diagnostics.stratifiedPool)
        let fused = results.hits.map(\.evidence.fusedScore)
        #expect(fused == fused.sorted(by: >))
        #expect(results.hits.allSatisfy { $0.kind == .transcript })
        try await engine.close()
    }
}

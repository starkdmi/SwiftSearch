import Foundation
import Testing

@testable import SemanticSearch

@Suite("Rank fusion")
struct FusionTests {

    private func candidate(_ item: String, _ representation: String, score: Float = 1)
        -> RetrievedCandidate
    {
        RetrievedCandidate(
            slot: abs(item.hashValue &+ representation.hashValue) % 100_000,
            key: RepresentationKey(
                item: ItemID(item), representation: RepresentationID(representation)),
            score: score)
    }

    private func channel(
        _ id: RetrievalChannelID, _ items: [(String, String)]
    ) -> RetrievalChannelResult {
        RetrievalChannelResult(
            channel: id,
            candidates: items.enumerated().map {
                candidate($0.element.0, $0.element.1, score: Float(100 - $0.offset))
            })
    }

    // MARK: - The formula

    @Test("A single channel scores 1 / (k + rank)")
    func singleChannelReciprocalRank() {
        let fused = RankFusion.fuse(
            [channel(.lexical, [("a", "text"), ("b", "text")])],
            configuration: FusionConfiguration(k: 60))

        #expect(abs(fused[0].fusedScore - 1 / 61) < 1e-6)
        #expect(abs(fused[1].fusedScore - 1 / 62) < 1e-6)
    }

    @Test("k defaults to 60")
    func defaultK() {
        // Cormack et al.'s value, and the extracted implementation's.
        #expect(FusionConfiguration.default.k == 60)
    }

    // MARK: - Collapsing

    @Test("Repeated representations of one unit vote once per channel")
    func collapseWithinChannel() {
        // Three representations of `a` at ranks 1–3 would otherwise collect
        // three votes from one channel and beat a unit that genuinely ranked
        // first everywhere.
        let fused = RankFusion.fuse(
            [
                channel(.lexical, [("a", "ocr"), ("a", "caption"), ("a", "summary"), ("b", "text")])
            ],
            configuration: FusionConfiguration(k: 60))

        #expect(fused.count == 2)
        let a = fused.first { $0.itemID == "a" }!
        #expect(a.evidence.count == 1)
        #expect(abs(a.fusedScore - 1 / 61) < 1e-6)
    }

    @Test("Collapsing re-ranks the surviving candidates densely")
    func collapseRepacksRanks() {
        // After `a`'s duplicates are dropped, `b` must be rank 2 — not rank 4,
        // the position it held in the uncollapsed list.
        let fused = RankFusion.fuse(
            [
                channel(.lexical, [("a", "one"), ("a", "two"), ("a", "three"), ("b", "text")])
            ],
            configuration: FusionConfiguration(k: 60))

        let b = fused.first { $0.itemID == "b" }!
        #expect(b.evidence[0].rank == 2)
        #expect(abs(b.fusedScore - 1 / 62) < 1e-6)
    }

    @Test("The strongest contributing representation becomes the hit's evidence")
    func bestRepresentationIsTheStrongest() {
        let text = RetrievalChannelID(family: .text, space: "s")
        let transcript = RetrievalChannelID(family: .transcript, space: "s")
        let fused = RankFusion.fuse(
            [
                RetrievalChannelResult(
                    channel: text,
                    candidates: [candidate("z", "filler"), candidate("a", "weak")]),
                RetrievalChannelResult(
                    channel: transcript, candidates: [candidate("a", "strong")]),
            ],
            configuration: FusionConfiguration(k: 60))

        let a = fused.first { $0.itemID == "a" }!
        #expect(a.bestKey.representation == "strong")
    }

    // MARK: - Family normalisation

    @Test("Two channels in one family split that family's weight")
    func familyWeightIsSplit() {
        // Adding a second embedding space must not double the dense side's say
        // in the ranking.
        let one = RetrievalChannelID(family: .text, space: "a")
        let two = RetrievalChannelID(family: .text, space: "b")

        let single = RankFusion.fuse(
            [channel(one, [("x", "text")])], configuration: FusionConfiguration(k: 60))
        let paired = RankFusion.fuse(
            [channel(one, [("x", "text")]), channel(two, [("x", "text")])],
            configuration: FusionConfiguration(k: 60))

        #expect(abs(single[0].fusedScore - paired[0].fusedScore) < 1e-6)
    }

    @Test("Query variants in one family split its weight too")
    func variantsSplitWeight() {
        let original = RetrievalChannelID(family: .text, space: "a", variant: nil)
        let expanded = RetrievalChannelID(family: .text, space: "a", variant: "expanded")

        let weights = RankFusion.channelWeights(
            for: [channel(original, [("x", "t")]), channel(expanded, [("x", "t")])],
            configuration: .default)

        #expect(abs((weights[original] ?? 0) - 0.5) < 1e-6)
        #expect(abs((weights[expanded] ?? 0) - 0.5) < 1e-6)
    }

    @Test("Different families keep their full weight")
    func familiesAreIndependent() {
        let text = RetrievalChannelID(family: .text, space: "a")
        let visual = RetrievalChannelID(family: .visual, space: "a")
        let weights = RankFusion.channelWeights(
            for: [channel(text, [("x", "t")]), channel(visual, [("x", "v")])],
            configuration: .default)

        #expect(abs((weights[text] ?? 0) - 1) < 1e-6)
        #expect(abs((weights[visual] ?? 0) - 1) < 1e-6)
    }

    @Test("An empty channel does not absorb its family's weight")
    func emptyChannelsAreExcludedFromTheSplit() {
        // A dense channel a filter left with nothing must not halve the
        // lexical channel's weight on the way out.
        let full = RetrievalChannelID(family: .text, space: "a")
        let empty = RetrievalChannelID(family: .text, space: "b")
        let weights = RankFusion.channelWeights(
            for: [
                channel(full, [("x", "t")]),
                RetrievalChannelResult(channel: empty, candidates: []),
            ],
            configuration: .default)

        #expect(abs((weights[full] ?? 0) - 1) < 1e-6)
    }

    @Test("Family weights are configurable")
    func explicitFamilyWeights() {
        let text = RetrievalChannelID(family: .text, space: "a")
        let configuration = FusionConfiguration(k: 60, familyWeights: [.text: 2, .lexical: 1])
        let fused = RankFusion.fuse(
            [channel(text, [("x", "t")]), channel(.lexical, [("y", "t")])],
            configuration: configuration)

        let x = fused.first { $0.itemID == "x" }!
        let y = fused.first { $0.itemID == "y" }!
        #expect(abs(x.fusedScore - 2 * y.fusedScore) < 1e-6)
    }

    @Test("Per-representation weight scales its contribution")
    func representationWeightApplies() {
        let demoted = RetrievalChannelResult(
            channel: .lexical,
            candidates: [
                RetrievedCandidate(
                    slot: 1,
                    key: RepresentationKey(item: "a", representation: "noisy"),
                    score: 1, weight: 0.5)
            ])
        let fused = RankFusion.fuse(demoted.candidates.isEmpty ? [] : [demoted],
            configuration: FusionConfiguration(k: 60))
        #expect(abs(fused[0].fusedScore - 0.5 / 61) < 1e-6)
    }

    @Test("The evidence shown is what earned the most, representation weight included")
    func bestKeyCountsRepresentationWeight() {
        // One item, two representations: a near-zero-weight one first in the
        // lexical channel, a full-weight one second in a dense channel.
        let lexical = RetrievalChannelResult(
            channel: .lexical,
            candidates: [
                RetrievedCandidate(
                    slot: 1, key: RepresentationKey(item: "a", representation: "faint"),
                    score: 1, weight: 0.01)
            ])
        let dense = RetrievalChannelResult(
            channel: RetrievalChannelID(family: .text, space: "s"),
            candidates: [
                RetrievedCandidate(
                    slot: 9, key: RepresentationKey(item: "b", representation: "other"),
                    score: 1),
                RetrievedCandidate(
                    slot: 2, key: RepresentationKey(item: "a", representation: "full"),
                    score: 1),
            ])
        let fused = RankFusion.fuse([lexical, dense], configuration: FusionConfiguration(k: 60))
        let a = fused.first { $0.itemID == "a" }!
        #expect(a.bestKey.representation == "full")
        #expect(a.bestSlot == 2)
    }

    // MARK: - Determinism

    @Test("Identical inputs produce an identical ordering")
    func orderingIsDeterministic() {
        let channels = [
            channel(.lexical, (1...40).map { ("item-\($0)", "text") }),
            channel(
                RetrievalChannelID(family: .text, space: "a"),
                (1...40).reversed().map { ("item-\($0)", "text") }),
        ]
        let first = RankFusion.fuse(channels).map(\.itemID)
        for _ in 0..<20 {
            #expect(RankFusion.fuse(channels).map(\.itemID) == first)
        }
    }

    @Test("Exact score ties break on the item id")
    func tiesBreakOnItemID() {
        // Two items at the same rank in different channels tie exactly. The
        // last tier of the comparator never ties, so the order is total.
        let one = RetrievalChannelID(family: .text, space: "a")
        let two = RetrievalChannelID(family: .visual, space: "a")
        let fused = RankFusion.fuse(
            [channel(one, [("zebra", "t")]), channel(two, [("apple", "v")])],
            configuration: FusionConfiguration(k: 60))

        #expect(fused.map(\.itemID) == ["apple", "zebra"])
    }

    @Test("Raw channel scores never cross channels")
    func rawScoresAreNotCompared() {
        // A BM25 score of 40 and an inner product of 0.4 are on different
        // scales; fusing on ranks is what makes them combinable at all. A
        // channel whose scores are enormous must not outrank one whose scores
        // are small when the ranks say otherwise.
        let big = RetrievalChannelResult(
            channel: .lexical, candidates: [candidate("a", "t", score: 10_000)])
        let small = RetrievalChannelResult(
            channel: RetrievalChannelID(family: .text, space: "s"),
            candidates: [candidate("b", "t", score: 0.001)])

        let fused = RankFusion.fuse([big, small], configuration: FusionConfiguration(k: 60))
        #expect(abs(fused[0].fusedScore - fused[1].fusedScore) < 1e-9)
    }

    @Test("Fusing nothing yields nothing")
    func emptyInput() {
        #expect(RankFusion.fuse([]).isEmpty)
        #expect(
            RankFusion.fuse([RetrievalChannelResult(channel: .lexical, candidates: [])]).isEmpty)
    }
}

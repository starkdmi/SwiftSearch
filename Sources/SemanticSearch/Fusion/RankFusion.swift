import Foundation

/// Where a candidate came from.
public struct RetrievalChannelID: Sendable, Hashable, Codable, CustomStringConvertible {
    public var family: ChannelFamily
    /// Embedding space for a dense channel; `nil` for the lexical channel.
    public var space: EmbeddingSpaceID?
    /// Distinguishes several channels within one family and space — one per
    /// query variant, say. Part of identity so the fusion weight can be split
    /// between them.
    public var variant: String?

    public init(family: ChannelFamily, space: EmbeddingSpaceID? = nil, variant: String? = nil) {
        self.family = family
        self.space = space
        self.variant = variant
    }

    public static let lexical = RetrievalChannelID(family: .lexical)

    public var description: String {
        var parts = [family.rawValue]
        if let space { parts.append(space.rawValue) }
        if let variant { parts.append(variant) }
        return parts.joined(separator: ":")
    }
}

/// One retrieved representation, before fusion.
public struct RetrievedCandidate: Sendable, Hashable {
    public var slot: Int
    public var key: RepresentationKey
    /// Raw channel score — an inner product, or a BM25 score. Reported in
    /// diagnostics and used only to break ties *within* its own channel. It is
    /// never compared across channels; that is the whole reason fusion happens
    /// on ranks.
    public var score: Float
    /// Per-representation multiplier from ``ContentRepresentation/weight``.
    public var weight: Float

    public init(slot: Int, key: RepresentationKey, score: Float, weight: Float = 1) {
        self.slot = slot
        self.key = key
        self.score = score
        self.weight = weight
    }
}

/// A channel's ranked output.
public struct RetrievalChannelResult: Sendable {
    public var channel: RetrievalChannelID
    /// Best first. Ranks are assigned from this order.
    public var candidates: [RetrievedCandidate]

    public init(channel: RetrievalChannelID, candidates: [RetrievedCandidate]) {
        self.channel = channel
        self.candidates = candidates
    }
}

/// One channel's contribution to a fused candidate, kept for diagnostics.
public struct ChannelEvidence: Sendable, Hashable {
    public var channel: RetrievalChannelID
    /// Rank within the channel after collapsing, 1-based.
    public var rank: Int
    public var score: Float
    /// Which representation of the item was the one that ranked.
    public var key: RepresentationKey
    /// The channel's effective weight after family normalisation.
    public var appliedWeight: Float
    /// The representation's own ``ContentRepresentation/weight``, which
    /// scales its vote as the channel's weight does.
    public var representationWeight: Float

    public init(
        channel: RetrievalChannelID, rank: Int, score: Float, key: RepresentationKey,
        appliedWeight: Float, representationWeight: Float = 1
    ) {
        self.channel = channel
        self.rank = rank
        self.score = score
        self.key = key
        self.appliedWeight = appliedWeight
        self.representationWeight = representationWeight
    }

    /// What this evidence added to the fused score.
    func contribution(k: Float) -> Float {
        appliedWeight * representationWeight * (1 / (k + Float(rank)))
    }
}

/// A unit after fusion.
public struct FusedCandidate: Sendable {
    public var itemID: ItemID
    /// The representation with the strongest evidence, used as the hit's
    /// primary evidence and as the reranker's input.
    public var bestKey: RepresentationKey
    public var bestSlot: Int
    public var fusedScore: Float
    public var evidence: [ChannelEvidence]

    public init(
        itemID: ItemID, bestKey: RepresentationKey, bestSlot: Int, fusedScore: Float,
        evidence: [ChannelEvidence]
    ) {
        self.itemID = itemID
        self.bestKey = bestKey
        self.bestSlot = bestSlot
        self.fusedScore = fusedScore
        self.evidence = evidence
    }
}

public struct FusionConfiguration: Sendable, Hashable, Codable {
    /// RRF's rank offset. 60 is the value from Cormack et al. and the value the
    /// extracted implementation used.
    public var k: Float
    /// Weight per channel family, before it is split among the channels in that
    /// family. A family missing from the map gets 1.
    public var familyWeights: [ChannelFamily: Float]

    public init(k: Float = 60, familyWeights: [ChannelFamily: Float] = [:]) {
        self.k = k
        self.familyWeights = familyWeights
    }

    public static let `default` = FusionConfiguration()

    func weight(for family: ChannelFamily) -> Float {
        max(0, familyWeights[family] ?? 1)
    }
}

/// Weighted reciprocal rank fusion.
///
/// Two properties the plain formula does not have:
///
/// **Collapsing.** A unit with an OCR pass, a caption and a translation would
/// otherwise appear three times in one channel and collect three votes from it.
/// Within a channel only the best-ranked representation of a unit survives, and
/// the surviving candidates are re-ranked densely so the vote positions are the
/// ones the collapsed list actually has.
///
/// **Family normalisation.** Adding a second embedding space, or a second query
/// variant, would otherwise increase that family's total say in the ranking.
/// Each family has a fixed budget, split evenly among the channels that belong
/// to it, so the ranking is stable as configuration grows.
public enum RankFusion {

    public static func fuse(
        _ results: [RetrievalChannelResult],
        configuration: FusionConfiguration = .default
    ) -> [FusedCandidate] {
        let collapsed = results.map(collapse(_:))
        let weights = channelWeights(for: collapsed, configuration: configuration)

        var byItem: [ItemID: FusedCandidate] = [:]
        byItem.reserveCapacity(collapsed.reduce(0) { $0 + $1.candidates.count })

        for result in collapsed {
            let weight = weights[result.channel] ?? 0
            for (offset, candidate) in result.candidates.enumerated() {
                let evidence = ChannelEvidence(
                    channel: result.channel, rank: offset + 1, score: candidate.score,
                    key: candidate.key, appliedWeight: weight,
                    representationWeight: candidate.weight)
                let contribution = evidence.contribution(k: configuration.k)

                if var existing = byItem[candidate.key.item] {
                    existing.fusedScore += contribution
                    existing.evidence.append(evidence)
                    byItem[candidate.key.item] = existing
                } else {
                    byItem[candidate.key.item] = FusedCandidate(
                        itemID: candidate.key.item,
                        bestKey: candidate.key,
                        bestSlot: candidate.slot,
                        fusedScore: contribution,
                        evidence: [evidence])
                }
            }
        }

        // The representation that earned the most of the fused score is the one
        // shown as evidence — not the first channel that happened to find the
        // item, which would make the hit's quoted text depend on dictionary
        // ordering.
        var fused = byItem.values.map { candidate -> FusedCandidate in
            var copy = candidate
            if let best = strongestEvidence(candidate.evidence, configuration: configuration) {
                copy.bestKey = best.key
                copy.bestSlot =
                    slotForKey(best.key, in: collapsed) ?? candidate.bestSlot
            }
            return copy
        }

        fused.sort(by: isOrderedBefore)
        return fused
    }

    /// Deterministic ordering: fused score, then the strongest single channel
    /// contribution, then the item id. The last tier never ties, so two runs
    /// over one corpus always produce the same list.
    static func isOrderedBefore(_ lhs: FusedCandidate, _ rhs: FusedCandidate) -> Bool {
        if lhs.fusedScore != rhs.fusedScore { return lhs.fusedScore > rhs.fusedScore }
        let lhsBest = lhs.evidence.map(\.rank).min() ?? Int.max
        let rhsBest = rhs.evidence.map(\.rank).min() ?? Int.max
        if lhsBest != rhsBest { return lhsBest < rhsBest }
        if lhs.evidence.count != rhs.evidence.count {
            return lhs.evidence.count > rhs.evidence.count
        }
        return lhs.itemID.rawValue < rhs.itemID.rawValue
    }

    // MARK: - Collapsing

    /// Keep the best-ranked representation per item, then re-rank densely.
    private static func collapse(_ result: RetrievalChannelResult) -> RetrievalChannelResult {
        var seen: Set<ItemID> = []
        var kept: [RetrievedCandidate] = []
        kept.reserveCapacity(result.candidates.count)
        for candidate in result.candidates where seen.insert(candidate.key.item).inserted {
            kept.append(candidate)
        }
        return RetrievalChannelResult(channel: result.channel, candidates: kept)
    }

    // MARK: - Weighting

    /// Split each family's budget evenly among its channels.
    static func channelWeights(
        for results: [RetrievalChannelResult],
        configuration: FusionConfiguration
    ) -> [RetrievalChannelID: Float] {
        var countsByFamily: [ChannelFamily: Int] = [:]
        for result in results where !result.candidates.isEmpty {
            countsByFamily[result.channel.family, default: 0] += 1
        }

        var weights: [RetrievalChannelID: Float] = [:]
        for result in results {
            let family = result.channel.family
            // An empty channel is excluded from the split rather than allowed
            // to absorb part of the budget. A dense channel that a filter left
            // with nothing must not quietly halve the lexical channel's weight.
            guard let count = countsByFamily[family], count > 0 else {
                weights[result.channel] = 0
                continue
            }
            weights[result.channel] = configuration.weight(for: family) / Float(count)
        }
        return weights
    }

    private static func strongestEvidence(
        _ evidence: [ChannelEvidence], configuration: FusionConfiguration
    ) -> ChannelEvidence? {
        // By what each earned, representation weight included: a weight of
        // 0.01 at rank one earns less than a full-weight rank two, and this
        // choice is also what the reranker reads.
        evidence.max { lhs, rhs in
            let lhsValue = lhs.contribution(k: configuration.k)
            let rhsValue = rhs.contribution(k: configuration.k)
            if lhsValue != rhsValue { return lhsValue < rhsValue }
            // Same contribution: prefer the lower rank, then the lower key, so
            // the choice does not depend on evidence insertion order.
            if lhs.rank != rhs.rank { return lhs.rank > rhs.rank }
            return lhs.key.description > rhs.key.description
        }
    }

    private static func slotForKey(_ key: RepresentationKey, in results: [RetrievalChannelResult])
        -> Int?
    {
        for result in results {
            if let match = result.candidates.first(where: { $0.key == key }) { return match.slot }
        }
        return nil
    }
}

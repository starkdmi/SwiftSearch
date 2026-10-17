import Foundation

/// Everything known about why a hit ranked where it did.
public struct RankingEvidence: Sendable {
    /// Weighted RRF score across retrieval channels.
    public var fusedScore: Float
    /// Per-channel rank, score and weight.
    public var channels: [ChannelEvidence]
    /// Reranker score, when the reranker ran on this candidate.
    ///
    /// Not a probability and not a confidence. See
    /// ``RerankingCapabilities/producesCalibratedScores``.
    public var rerankScore: Float?
    /// Position in the reranker's ordering, 1-based.
    public var rerankRank: Int?
    /// Final ordering value: the reranker's reciprocal rank, or — with
    /// `rerankerOrderingOnly: false` — that plus the fused retrieval score.
    public var finalScore: Float
    /// True when the reranker truncated this candidate's pair.
    public var rerankTruncated: Bool

    public init(
        fusedScore: Float,
        channels: [ChannelEvidence],
        rerankScore: Float? = nil,
        rerankRank: Int? = nil,
        finalScore: Float,
        rerankTruncated: Bool = false
    ) {
        self.fusedScore = fusedScore
        self.channels = channels
        self.rerankScore = rerankScore
        self.rerankRank = rerankRank
        self.finalScore = finalScore
        self.rerankTruncated = rerankTruncated
    }
}

/// Bounded context from the enclosing unit, for display and for reranking.
public struct ParentContext: Sendable {
    public var itemID: ItemID
    public var metadata: Metadata
    public var locator: SourceLocator
    /// Text drawn from the parent's own representations, truncated to the
    /// caller's budget.
    public var text: String?

    public init(
        itemID: ItemID, metadata: Metadata, locator: SourceLocator, text: String? = nil
    ) {
        self.itemID = itemID
        self.metadata = metadata
        self.locator = locator
        self.text = text
    }
}

/// One result.
public struct SearchHit: Sendable {
    public var itemID: ItemID
    /// The representation that matched, not merely the item that contains it.
    public var representationID: RepresentationID
    public var kind: RepresentationKind
    /// Matched text, where the representation has any.
    public var text: String?
    public var language: String?
    /// Where to find it, as precisely as the representation knows.
    public var locator: SourceLocator
    public var metadata: Metadata
    public var parent: ParentContext?
    public var provenance: Provenance
    public var evidence: RankingEvidence
    /// Other representations of the same item that also matched, best first.
    /// Empty unless the item matched through more than one channel.
    public var supportingRepresentations: [RepresentationID]

    public init(
        itemID: ItemID,
        representationID: RepresentationID,
        kind: RepresentationKind,
        text: String?,
        language: String?,
        locator: SourceLocator,
        metadata: Metadata,
        parent: ParentContext?,
        provenance: Provenance,
        evidence: RankingEvidence,
        supportingRepresentations: [RepresentationID] = []
    ) {
        self.itemID = itemID
        self.representationID = representationID
        self.kind = kind
        self.text = text
        self.language = language
        self.locator = locator
        self.metadata = metadata
        self.parent = parent
        self.provenance = provenance
        self.evidence = evidence
        self.supportingRepresentations = supportingRepresentations
    }
}

/// Hits belonging to one parent, when `groupByParent` is set.
public struct SearchGroup: Sendable {
    public var parentID: ItemID
    public var parent: ParentContext?
    /// The strongest hit in the group. Groups are ordered by this hit alone —
    /// never by how many hits they contain, which would rank a long document
    /// with many weak matches above a short one with the right answer.
    public var bestHit: SearchHit
    public var hits: [SearchHit]

    public init(parentID: ItemID, parent: ParentContext?, bestHit: SearchHit, hits: [SearchHit]) {
        self.parentID = parentID
        self.parent = parent
        self.bestHit = bestHit
        self.hits = hits
    }
}

/// Hits from one stretch of one source, when `momentGap` is set.
///
/// The transcript line, the frame and anything else that happened within a
/// few seconds of each other in the same recording are one moment, however
/// each ranked on its own. A moment is ordered by its strongest hit, never by
/// how many it holds.
public struct SearchMoment: Sendable {
    /// ``SourceLocator/source`` shared by the hits, or the parent's id when
    /// they have none.
    public var source: String
    public var timeRange: SourceLocator.TimeRange
    public var bestHit: SearchHit
    /// The strongest hit of each kind — the transcript line and the frame for
    /// the same moment, side by side.
    public var bestHitByKind: [RepresentationKind: SearchHit]
    /// Best first.
    public var hits: [SearchHit]

    public init(
        source: String, timeRange: SourceLocator.TimeRange, bestHit: SearchHit,
        bestHitByKind: [RepresentationKind: SearchHit], hits: [SearchHit]
    ) {
        self.source = source
        self.timeRange = timeRange
        self.bestHit = bestHit
        self.bestHitByKind = bestHitByKind
        self.hits = hits
    }
}

/// What the engine did, and what it could not do.
public struct SearchDiagnostics: Sendable {
    public struct ChannelDiagnostics: Sendable {
        public var channel: RetrievalChannelID
        public var retrieved: Int
        /// After collapsing repeated representations of one unit.
        public var uniqueUnits: Int
        public var duration: Duration
        /// Weight this channel contributed at, after family normalisation.
        public var appliedWeight: Float

        public init(
            channel: RetrievalChannelID, retrieved: Int, uniqueUnits: Int, duration: Duration,
            appliedWeight: Float
        ) {
            self.channel = channel
            self.retrieved = retrieved
            self.uniqueUnits = uniqueUnits
            self.duration = duration
            self.appliedWeight = appliedWeight
        }
    }

    public var mode: SearchMode
    public var queryTerms: [String]
    public var detectedQueryLanguage: String?
    public var channels: [ChannelDiagnostics]
    public var fusedCandidates: Int
    public var rerankedCandidates: Int
    /// Candidates whose pair the reranker had to truncate.
    public var rerankTruncatedCandidates: Int
    /// Candidate images the reranker could not fit in its visual budget.
    public var rerankDroppedImages: Int
    /// Why a hybrid search ran without its reranking stage: no reranker is
    /// configured, or the reranker cannot read the query at all (an image-only
    /// query and a text-only reranker). `nil` when reranking ran or was not
    /// asked for.
    public var rerankSkippedReason: String?
    /// True when the query had images the reranker cannot take, so it judged
    /// the candidates against the text of the query alone.
    public var rerankIgnoredQueryImages: Bool
    /// Candidates that offered the reranker nothing to read — an image hit
    /// with no text, in front of a text-only reranker. They are not scored;
    /// they keep their retrieval position in the reranker's ordering rather
    /// than being judged on an empty string.
    public var rerankUnscoredCandidates: Int
    /// True when pictures and text filled the pool by taking turns, each
    /// from its own fused list, and the reranker alone ordered it. Happens
    /// when visual and text channels both returned candidates.
    public var stratifiedPool: Bool
    /// True when ``SearchOptions/quickLexicalAnswer`` applied: the hits are
    /// BM25's alone, and nothing was embedded or reranked.
    public var answeredLexically = false
    /// Captions the engine's ``EngineConfiguration/queryCaptioner`` wrote for
    /// the query images, in order. The search ran on the text with these
    /// appended, in place of the images.
    public var queryCaptions: [String]
    public var retrievalDuration: Duration
    public var rerankDuration: Duration
    public var totalDuration: Duration
    /// Waiting for the inference permit — behind another search, an ingest
    /// batch or a release — summed over the query's model calls. Like
    /// ``modelLoadDuration``, counted in ``totalDuration`` too, and in
    /// ``retrievalDuration`` or ``rerankDuration`` for the embedder and the
    /// reranker.
    public var inferenceWaitDuration: Duration = .zero
    /// Making the models resident: next to nothing when they are, a full load
    /// after a release or under sequential residency.
    public var modelLoadDuration: Duration = .zero
    /// Spaces that were skipped, and why. A space missing from a search shows
    /// up here rather than silently narrowing the result set.
    public var skippedSpaces: [SkippedSpace]
    /// Spaces that answered only part of the query — the text of a
    /// text-and-image query sent to a space that cannot embed images. Their
    /// channels are in the results; this says what those channels did not see.
    public var partiallyAnsweredSpaces: [SkippedSpace]
    /// Representations the engine could not evaluate — missing media, a
    /// modality the provider does not accept.
    public var unavailableRepresentations: [UnavailableRepresentation]

    public struct SkippedSpace: Sendable {
        public var space: EmbeddingSpaceID
        public var reason: String

        public init(space: EmbeddingSpaceID, reason: String) {
            self.space = space
            self.reason = reason
        }
    }

    public struct UnavailableRepresentation: Sendable {
        public var key: RepresentationKey
        public var reason: String

        public init(key: RepresentationKey, reason: String) {
            self.key = key
            self.reason = reason
        }
    }

    public init(
        mode: SearchMode,
        queryTerms: [String] = [],
        detectedQueryLanguage: String? = nil,
        channels: [ChannelDiagnostics] = [],
        fusedCandidates: Int = 0,
        rerankedCandidates: Int = 0,
        rerankTruncatedCandidates: Int = 0,
        rerankDroppedImages: Int = 0,
        rerankSkippedReason: String? = nil,
        rerankIgnoredQueryImages: Bool = false,
        rerankUnscoredCandidates: Int = 0,
        stratifiedPool: Bool = false,
        queryCaptions: [String] = [],
        retrievalDuration: Duration = .zero,
        rerankDuration: Duration = .zero,
        totalDuration: Duration = .zero,
        skippedSpaces: [SkippedSpace] = [],
        partiallyAnsweredSpaces: [SkippedSpace] = [],
        unavailableRepresentations: [UnavailableRepresentation] = []
    ) {
        self.mode = mode
        self.queryTerms = queryTerms
        self.detectedQueryLanguage = detectedQueryLanguage
        self.channels = channels
        self.fusedCandidates = fusedCandidates
        self.rerankedCandidates = rerankedCandidates
        self.rerankTruncatedCandidates = rerankTruncatedCandidates
        self.rerankDroppedImages = rerankDroppedImages
        self.rerankSkippedReason = rerankSkippedReason
        self.rerankIgnoredQueryImages = rerankIgnoredQueryImages
        self.rerankUnscoredCandidates = rerankUnscoredCandidates
        self.stratifiedPool = stratifiedPool
        self.queryCaptions = queryCaptions
        self.retrievalDuration = retrievalDuration
        self.rerankDuration = rerankDuration
        self.totalDuration = totalDuration
        self.skippedSpaces = skippedSpaces
        self.partiallyAnsweredSpaces = partiallyAnsweredSpaces
        self.unavailableRepresentations = unavailableRepresentations
    }
}

public struct SearchResults: Sendable {
    public var query: SearchQuery
    public var hits: [SearchHit]
    /// Populated when `groupByParent` is set.
    public var groups: [SearchGroup]
    /// Populated when `momentGap` is set.
    public var moments: [SearchMoment]
    public var diagnostics: SearchDiagnostics

    public init(
        query: SearchQuery, hits: [SearchHit], groups: [SearchGroup] = [],
        moments: [SearchMoment] = [], diagnostics: SearchDiagnostics
    ) {
        self.query = query
        self.hits = hits
        self.groups = groups
        self.moments = moments
        self.diagnostics = diagnostics
    }

    public var isEmpty: Bool { hits.isEmpty }
}

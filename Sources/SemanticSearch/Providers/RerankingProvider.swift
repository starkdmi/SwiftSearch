import Foundation

public struct RerankingCapabilities: Sendable, Hashable, Codable {
    public var acceptsImages: Bool
    public var acceptsVideo: Bool
    /// Maximum tokens per (query, document) pair. Pairs above it are truncated
    /// document-first and the truncation is reported.
    public var maximumPairTokens: Int
    /// Maximum images per pair, counting query images and document images
    /// together.
    public var maximumImagesPerPair: Int
    public var maximumBatchSize: Int
    /// True only if the provider's scores have been calibrated against a
    /// labelled set, which is the only circumstance under which a threshold on
    /// them means anything. Bundled providers report `false`.
    public var producesCalibratedScores: Bool

    public init(
        acceptsImages: Bool = false,
        acceptsVideo: Bool = false,
        maximumPairTokens: Int = 1024,
        maximumImagesPerPair: Int = 0,
        maximumBatchSize: Int = 4,
        producesCalibratedScores: Bool = false
    ) {
        self.acceptsImages = acceptsImages
        self.acceptsVideo = acceptsVideo
        self.maximumPairTokens = maximumPairTokens
        self.maximumImagesPerPair = maximumImagesPerPair
        self.maximumBatchSize = maximumBatchSize
        self.producesCalibratedScores = producesCalibratedScores
    }
}

public struct RerankerIdentity: Sendable, Hashable, Codable {
    public var modelIdentity: String
    public var weightsVariant: String?
    public var promptVersion: String

    public init(modelIdentity: String, weightsVariant: String? = nil, promptVersion: String) {
        self.modelIdentity = modelIdentity
        self.weightsVariant = weightsVariant
        self.promptVersion = promptVersion
    }

    public var fingerprint: String {
        StableHash.of { hasher in
            hasher.append("reranker.v1")
            hasher.append(modelIdentity)
            hasher.append(weightsVariant ?? "")
            hasher.append(promptVersion)
        }
    }
}

/// One candidate presented to the reranker.
public struct RerankCandidate: Sendable {
    /// Text of the matched representation.
    public var text: String
    /// Bounded context from the parent, already truncated by the engine. Kept
    /// separate from `text` so a provider can drop it first when a pair does
    /// not fit rather than losing the evidence that caused the match.
    public var parentContext: String?
    public var kind: RepresentationKind
    public var images: [ImagePayload]
    /// Short label shown to the model, typically the parent's title.
    public var title: String?

    public init(
        text: String,
        parentContext: String? = nil,
        kind: RepresentationKind = .originalText,
        images: [ImagePayload] = [],
        title: String? = nil
    ) {
        self.text = text
        self.parentContext = parentContext
        self.kind = kind
        self.images = images
        self.title = title
    }
}

public struct RerankQuery: Sendable {
    public var text: String
    public var images: [ImagePayload]
    /// Task description given to the model. Part of the reranker's prompt
    /// version, not of the query.
    public var instruction: String

    public init(text: String, images: [ImagePayload] = [], instruction: String) {
        self.text = text
        self.images = images
        self.instruction = instruction
    }
}

public struct RerankScore: Sendable {
    /// Relevance score, higher is better.
    ///
    /// Not a probability and not a confidence, whatever its range looks like.
    /// See ``RerankingCapabilities/producesCalibratedScores``.
    public var score: Float
    public var wasTruncated: Bool
    /// True when the provider dropped `parentContext` to make the pair fit.
    public var droppedParentContext: Bool
    /// Images the provider could not fit, out of those offered.
    public var droppedImages: Int

    public init(
        score: Float,
        wasTruncated: Bool = false,
        droppedParentContext: Bool = false,
        droppedImages: Int = 0
    ) {
        self.score = score
        self.wasTruncated = wasTruncated
        self.droppedParentContext = droppedParentContext
        self.droppedImages = droppedImages
    }
}

/// Scores (query, candidate) pairs. Implement this to replace the bundled
/// reranker; doing so never requires rebuilding embeddings.
public protocol RerankingProvider: Actor {
    var identity: RerankerIdentity { get }
    /// Must not change with ``isPrepared``. The engine decides what to send —
    /// query images, candidate images — before it makes the reranker resident,
    /// and under sequential residency the reranker is unloaded at that point.
    var capabilities: RerankingCapabilities { get }

    func prepare() async throws
    func release() async
    var isPrepared: Bool { get }

    /// Score every candidate against the query.
    ///
    /// Must honour `Task.isCancelled` between batches and return one score per
    /// candidate, in order.
    func score(query: RerankQuery, candidates: [RerankCandidate]) async throws -> [RerankScore]
}

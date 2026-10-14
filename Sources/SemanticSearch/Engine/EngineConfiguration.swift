import Foundation

/// One embedding space and the provider that fills it.
public struct EmbeddingSpaceConfiguration: Sendable {
    public var id: EmbeddingSpaceID
    public var provider: any EmbeddingProvider
    /// Representation kinds routed into this space. Empty means every kind the
    /// provider accepts, which is the right default for a single-space setup
    /// and the wrong one as soon as a second space exists.
    public var kinds: Set<RepresentationKind>
    public var indexKind: VectorIndexKind
    public var hnsw: HNSWConfiguration
    /// Store vectors at a Matryoshka prefix rather than the provider's native
    /// width. Rejected unless the provider advertises support.
    public var reducedDimensions: Int?
    /// Instruction prefixed to queries against this space, per family. Falls
    /// back to the provider's own default.
    public var queryInstructions: [ChannelFamily: String]

    public init(
        id: EmbeddingSpaceID,
        provider: any EmbeddingProvider,
        kinds: Set<RepresentationKind> = [],
        indexKind: VectorIndexKind = .hnsw,
        hnsw: HNSWConfiguration = .default,
        reducedDimensions: Int? = nil,
        queryInstructions: [ChannelFamily: String] = [:]
    ) {
        self.id = id
        self.provider = provider
        self.kinds = kinds
        self.indexKind = indexKind
        self.hnsw = hnsw
        self.reducedDimensions = reducedDimensions
        self.queryInstructions = queryInstructions
    }
}

/// Bounds on concurrency and memory.
public struct RuntimePolicy: Sendable {
    /// Inference jobs allowed to run at once.
    ///
    /// One, by default and on purpose: MLX models are large, and a second
    /// resident model on an 8 GB device is the difference between a working
    /// index build and a jetsam kill. Raise it only where memory is known.
    public var maximumConcurrentInferenceJobs: Int
    /// Representations embedded per provider call, capped by the provider's own
    /// advertised batch size.
    public var embeddingBatchSize: Int
    /// Release every model after this long with no inference. Zero keeps
    /// models resident until ``SemanticSearchEngine/releaseModels()``.
    ///
    /// The next search or index batch reloads what it needs, so this trades a
    /// cold query after an idle period for not holding gigabytes of weights
    /// while nothing is happening.
    public var modelIdleTimeout: Duration
    /// Keep at most one model resident: every other model is released before
    /// one loads, including the embedder before the reranker and the reranker
    /// before the next query's embedder.
    ///
    /// Sequential residency is the memory policy that makes the 2B multimodal
    /// profile fit on a phone; it costs a reload per query when both models are
    /// needed, which is why it is off by default on macOS.
    public var sequentialModelResidency: Bool

    public init(
        maximumConcurrentInferenceJobs: Int = 1,
        embeddingBatchSize: Int = 8,
        modelIdleTimeout: Duration = .seconds(0),
        sequentialModelResidency: Bool = false
    ) {
        self.maximumConcurrentInferenceJobs = max(1, maximumConcurrentInferenceJobs)
        self.embeddingBatchSize = max(1, embeddingBatchSize)
        self.modelIdleTimeout = modelIdleTimeout
        self.sequentialModelResidency = sequentialModelResidency
    }

    public static let `default` = RuntimePolicy()

    /// Tighter bounds for a device with a few gigabytes of unified memory.
    public static let constrainedDevice = RuntimePolicy(
        maximumConcurrentInferenceJobs: 1,
        embeddingBatchSize: 4,
        modelIdleTimeout: .seconds(60),
        sequentialModelResidency: true)
}

/// Everything the engine needs to be built.
public struct EngineConfiguration: Sendable {
    /// Where the index lives. `nil` for an in-memory index.
    public var storeURL: URL?
    public var spaces: [EmbeddingSpaceConfiguration]
    public var reranker: (any RerankingProvider)?
    public var processor: (any ContentProcessor)?
    public var analyzer: any LexicalAnalyzer
    public var bm25: BM25Parameters
    public var queryTransform: any QueryTransform
    public var runtime: RuntimePolicy
    /// Default reranker instruction.
    public var rerankInstruction: String
    /// Default per-family query instruction, used where a space does not
    /// override it.
    public var queryInstructions: [ChannelFamily: String]
    /// Default for ``SearchOptions/reranksImages``: whether an image-capable
    /// reranker is shown pixels, or judges frames by their captions.
    public var reranksImages: Bool
    /// Describes query images in words before searching, for an index that
    /// matches pictures through their captions. Each image is replaced by its
    /// caption, appended to the query text as `Frame: <caption>` — a caption
    /// alone for an image with no question — so an attached frame can be
    /// searched with text models alone. Use the captioner that captioned the
    /// index, so both sides are described the same way. `nil` searches with
    /// the images themselves.
    public var queryCaptioner: (any ImageCaptioner)?
    /// Whether the engine may read a media file an index refers to: the
    /// picture a reranker is shown, media `reindex` embeds again. `nil` reads
    /// any file that exists. A caller that bounds what is read — a library of
    /// folders an agent searches — passes its boundary here: an indexed file
    /// replaced by a link to one outside it is otherwise read through the
    /// link. A refused file is reported, like a missing one.
    public var mediaAccess: (@Sendable (URL) -> Bool)?

    public init(
        storeURL: URL? = nil,
        spaces: [EmbeddingSpaceConfiguration] = [],
        reranker: (any RerankingProvider)? = nil,
        processor: (any ContentProcessor)? = nil,
        analyzer: any LexicalAnalyzer = LinguisticLexicalAnalyzer(),
        bm25: BM25Parameters = .default,
        queryTransform: any QueryTransform = IdentityQueryTransform(),
        runtime: RuntimePolicy = .default,
        rerankInstruction: String = EngineConfiguration.defaultRerankInstruction,
        queryInstructions: [ChannelFamily: String] = [:],
        reranksImages: Bool = true,
        queryCaptioner: (any ImageCaptioner)? = nil,
        mediaAccess: (@Sendable (URL) -> Bool)? = nil
    ) {
        self.storeURL = storeURL
        self.spaces = spaces
        self.reranker = reranker
        self.processor = processor
        self.analyzer = analyzer
        self.bm25 = bm25
        self.queryTransform = queryTransform
        self.runtime = runtime
        self.rerankInstruction = rerankInstruction
        self.queryInstructions = queryInstructions
        self.reranksImages = reranksImages
        self.queryCaptioner = queryCaptioner
        self.mediaAccess = mediaAccess
    }

    public static let defaultRerankInstruction =
        "Given a search query, judge whether the document answers it."
}

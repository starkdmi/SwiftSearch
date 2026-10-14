import Foundation

/// What is being searched for.
public struct SearchQuery: Sendable {
    public var text: String
    public var images: [ImagePayload]
    /// BCP-47 hint for tokenisation. Detected from the text when omitted.
    public var languageHint: String?

    public init(text: String = "", images: [ImagePayload] = [], languageHint: String? = nil) {
        self.text = text
        self.images = images
        self.languageHint = languageHint
    }

    public init(_ text: String) {
        self.init(text: text)
    }

    public var isEmpty: Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && images.isEmpty
    }

    public var isImageOnly: Bool {
        !images.isEmpty && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

/// How much of the pipeline to run.
///
/// Hybrid is the default for every query including short ones. A short-query
/// fast path that skips dense retrieval and reranking whenever a query is under
/// a few words makes "Tokyo" and "the Tokyo office" take entirely different
/// code paths, so lexical-only is an explicit mode instead, and
/// ``QuickLexicalAnswer`` an opt-in.
public enum SearchMode: String, Sendable, Codable, CaseIterable {
    /// Lexical and every dense channel, fused, then reranked.
    case hybrid
    /// Lexical and dense, fused. No reranker, so no model has to be resident
    /// beyond the embedder.
    case retrievalOnly
    /// BM25 only. Runs with no models loaded at all.
    case lexicalOnly
    /// Dense only. Useful for evaluating a space in isolation.
    case denseOnly
}

/// Per-query tuning.
public struct SearchOptions: Sendable {
    public var mode: SearchMode
    /// Candidates pulled from each retrieval channel.
    public var candidatesPerChannel: Int
    /// Unique units handed to the reranker.
    public var rerankPoolSize: Int
    /// Hits returned.
    public var limit: Int
    public var filter: SearchFilter
    public var fusion: FusionConfiguration
    /// Instruction given to the reranker. Overrides the engine's default.
    public var rerankInstruction: String?
    /// Instruction prefixed to the query for dense retrieval, per family.
    /// Missing families use the engine's configured default.
    public var queryInstructions: [ChannelFamily: String]
    /// Characters of parent text given to the reranker with each candidate.
    /// Bounded so one long parent cannot crowd the evidence out of the pair.
    public var parentContextCharacters: Int
    /// Group hits by parent, keeping the strongest supporting hit for each.
    public var groupByParent: Bool
    /// Group hits into moments: same source, time ranges within this many
    /// seconds of each other. `nil` leaves `moments` empty.
    ///
    /// Moments are drawn from the whole rerank pool, not only the `limit`
    /// returned hits, so a frame of the same moment as a strong transcript
    /// line joins it even when it ranked lower on its own.
    public var momentGap: TimeInterval?
    /// Matryoshka width to query at. Rejected unless every queried space
    /// supports reduction and was built at or above this width.
    public var reducedDimensions: Int?
    /// Order the pool by the reranker alone — retrieve, then rerank, as
    /// monoBERT does (Nogueira & Cho, 2019). `false` fuses the reranker's
    /// rank back with retrieval's instead, which keeps a candidate the
    /// reranker is wrong about but retrieval was sure of.
    ///
    /// On by default. Fused, a hit that several channels agree on — BM25, the
    /// text embedding and a caption — outvotes the reranker's first place, even
    /// one it was near certain of, with pages it judged irrelevant. Always on
    /// when pictures and text share the pool
    /// (``SearchDiagnostics/stratifiedPool``), whose two lists' retrieval scores
    /// cannot be added.
    public var rerankerOrderingOnly: Bool
    /// Show the reranker pixels. `false` has it judge by text alone: a frame
    /// is read through its caption or OCR text, one with neither is left
    /// unscored, and query images are not sent. `nil` takes the engine's
    /// ``EngineConfiguration/reranksImages``.
    ///
    /// The VL reranker costs seconds per image pair and separates videos
    /// better than it separates frames of one scene; a captioned index can
    /// skip that and still rerank its frames.
    public var reranksImages: Bool?
    /// Spaces to query. Empty means every registered space the query's
    /// modalities are compatible with.
    public var spaces: [EmbeddingSpaceID]
    /// Answer a short text query from BM25 alone when it finds anything —
    /// no embedding, no reranking, no model loaded;
    /// ``SearchDiagnostics/answeredLexically`` says it happened, and
    /// the same search with this `nil` searches further. `nil`, the
    /// default, always runs the whole pipeline.
    public var quickLexicalAnswer: QuickLexicalAnswer?

    public init(
        mode: SearchMode = .hybrid,
        candidatesPerChannel: Int = 100,
        rerankPoolSize: Int = 50,
        limit: Int = 10,
        filter: SearchFilter = .all,
        fusion: FusionConfiguration = .default,
        rerankInstruction: String? = nil,
        queryInstructions: [ChannelFamily: String] = [:],
        parentContextCharacters: Int = 600,
        groupByParent: Bool = false,
        momentGap: TimeInterval? = nil,
        reducedDimensions: Int? = nil,
        rerankerOrderingOnly: Bool = true,
        reranksImages: Bool? = nil,
        spaces: [EmbeddingSpaceID] = [],
        quickLexicalAnswer: QuickLexicalAnswer? = nil
    ) {
        self.mode = mode
        self.candidatesPerChannel = candidatesPerChannel
        self.rerankPoolSize = rerankPoolSize
        self.limit = limit
        self.filter = filter
        self.fusion = fusion
        self.rerankInstruction = rerankInstruction
        self.queryInstructions = queryInstructions
        self.parentContextCharacters = parentContextCharacters
        self.groupByParent = groupByParent
        self.momentGap = momentGap
        self.reducedDimensions = reducedDimensions
        self.rerankerOrderingOnly = rerankerOrderingOnly
        self.reranksImages = reranksImages
        self.spaces = spaces
        self.quickLexicalAnswer = quickLexicalAnswer
    }

    public static let `default` = SearchOptions()

    func validated() throws -> SearchOptions {
        guard candidatesPerChannel > 0, rerankPoolSize > 0, limit > 0 else {
            throw SemanticSearchError.configuration(
                "candidatesPerChannel, rerankPoolSize and limit must all be > 0")
        }
        guard fusion.k > 0 else {
            throw SemanticSearchError.configuration("fusion k must be > 0")
        }
        guard parentContextCharacters >= 0 else {
            throw SemanticSearchError.configuration("parentContextCharacters must be >= 0")
        }
        if let momentGap, momentGap < 0 || !momentGap.isFinite {
            throw SemanticSearchError.configuration("momentGap must be a finite value >= 0")
        }
        if let reducedDimensions, reducedDimensions <= 0 {
            throw SemanticSearchError.configuration("reducedDimensions must be > 0 when set")
        }
        if let quick = quickLexicalAnswer,
            quick.maximumWords < 0 || quick.maximumTerms < 0 || quick.minimumHits < 1
        {
            throw SemanticSearchError.configuration(
                "quickLexicalAnswer needs word and term limits >= 0 and minimumHits >= 1")
        }
        return self
    }
}

/// When a query is short enough, and BM25 finds enough, to be answered from
/// BM25 alone.
///
/// A name or a word or two that appears in the index is answered by where it
/// appears, in milliseconds; a caller can offer to search further. A query
/// with a picture never qualifies.
public struct QuickLexicalAnswer: Sendable, Hashable {
    /// Words in the query as typed, split on white space.
    public var maximumWords: Int
    /// Terms left after analysis: pronouns, prepositions, conjunctions,
    /// determiners, particles and interjections dropped, words lemmatised.
    public var maximumTerms: Int
    /// Items BM25 has to find.
    public var minimumHits: Int

    public init(maximumWords: Int = 4, maximumTerms: Int = 2, minimumHits: Int = 1) {
        self.maximumWords = maximumWords
        self.maximumTerms = maximumTerms
        self.minimumHits = minimumHits
    }

    /// Up to four words, two terms, one hit.
    public static let `default` = QuickLexicalAnswer()
}

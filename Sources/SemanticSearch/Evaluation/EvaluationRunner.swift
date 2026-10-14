import Foundation

/// Runs an evaluation corpus against an engine.
///
/// It produces a report and the identity that report was measured under, so a
/// later run can be checked against it rather than merely compared by eye.
public struct EvaluationRunner: Sendable {

    public struct Options: Sendable {
        public var search: SearchOptions
        /// Ranks pulled for the recall measurement. 100 matches the ANN gate,
        /// which is recall@100.
        public var recallDepth: Int
        /// Discarded runs before measurement, to separate cold from warm.
        public var warmupQueries: Int

        public init(
            search: SearchOptions = .default, recallDepth: Int = 100, warmupQueries: Int = 1
        ) {
            self.search = search
            self.recallDepth = recallDepth
            self.warmupQueries = max(0, warmupQueries)
        }

        public static let `default` = Options()
    }

    public let corpus: EvaluationCorpus
    public let options: Options

    public init(corpus: EvaluationCorpus, options: Options = .default) {
        self.corpus = corpus
        self.options = options
    }

    /// Index the corpus into `engine`.
    @discardableResult
    public func index(into engine: SemanticSearchEngine, baseURL: URL? = nil) async throws
        -> IndexingReport
    {
        try await engine.upsert(corpus.items(baseURL: baseURL))
    }

    /// Run every query and build a report.
    public func run(
        against engine: SemanticSearchEngine,
        label: String
    ) async throws -> (report: EvaluationReport, identity: EvaluationIdentity) {
        // Warm-up runs are discarded. The first query of a process pays for
        // lazy Metal pipeline compilation and the first SQLite page reads;
        // folding that into a median makes the median a measure of the warm-up.
        for query in corpus.queries.prefix(options.warmupQueries) {
            _ = try? await engine.search(
                SearchQuery(query.text), options: searchOptions(for: query))
        }

        var results: [(query: EvaluationQuery, ranked: [ItemID], duration: Duration)] = []
        results.reserveCapacity(corpus.queries.count)
        var degraded: [String: Int] = [:]

        for query in corpus.queries {
            try Task.checkCancellation()
            let started = ContinuousClock.now
            let hits = try await engine.search(
                SearchQuery(query.text), options: searchOptions(for: query))
            let elapsed = ContinuousClock.now - started
            var ranked = hits.hits.map(\.itemID)
            if corpus.ignoresIdenticalIDs == true {
                ranked = Array(
                    ranked.filter { $0.rawValue != query.id }.prefix(options.search.limit))
            }
            results.append((query: query, ranked: ranked, duration: elapsed))
            for reason in Self.degradations(hits.diagnostics) {
                degraded[reason, default: 0] += 1
            }
        }

        // `label` is carried on the report, not the identity: two runs of the
        // same configuration under different labels are still comparable.
        let identity = await self.identity(of: engine)
        var report = EvaluationReport.make(label: label, results: results)
        report.degradedQueries = degraded
        return (report, identity)
    }

    /// What a search said it could not do for its query, each reason once:
    /// the engine reports a skipped space once per channel family, and one
    /// empty query vector would otherwise count once for each.
    private static func degradations(_ diagnostics: SearchDiagnostics) -> Set<String> {
        Set(
            diagnostics.skippedSpaces.map { "\($0.space.rawValue) skipped: \($0.reason)" }
                + diagnostics.partiallyAnsweredSpaces.map {
                    "\($0.space.rawValue) answered in part: \($0.reason)"
                })
    }

    /// Recall of the ANN index against exact search.
    ///
    /// Measured on the candidate pool rather than on the final ranking: the
    /// gate is about what the graph *finds*, and a reranker can rescue a poor
    /// candidate list often enough to hide a graph problem in nDCG.
    public func annRecall(
        against engine: SemanticSearchEngine,
        space: EmbeddingSpaceID,
        queryVectors: [[Float]],
        depth: Int? = nil
    ) async throws -> Double {
        // A space below its graph threshold is searched exactly, and its
        // "approximate" neighbours are the exact ones: the measurement would
        // report 1.0 for a graph that was never built.
        guard await engine.usesExactSearch(space: space) == false else {
            throw SemanticSearchError.configuration(
                "space '\(space)' is searched exactly, not through a graph — ANN recall would "
                    + "compare exact search with itself. Index more vectors than the space's "
                    + "minimumVectorsForGraph, or lower it.")
        }
        let k = depth ?? options.recallDepth
        var scores: [Double] = []
        scores.reserveCapacity(queryVectors.count)

        for vector in queryVectors {
            try Task.checkCancellation()
            let approximate = try await engine.approximateNeighbours(
                space: space, query: vector, limit: k)
            let exact = try await engine.exactNeighbours(space: space, query: vector, limit: k)
            scores.append(
                RetrievalMetrics.annRecall(approximate: approximate, exact: exact, k: k))
        }

        return RetrievalMetrics.mean(scores)
    }

    private func searchOptions(for query: EvaluationQuery) -> SearchOptions {
        var options = self.options.search
        let filter = corpus.filter(for: query)
        if case .all = filter {} else {
            options.filter = .and([options.filter, filter])
        }
        // One hit more, so that dropping the query's own document still
        // leaves `limit` to score. The identity keeps the configured limit.
        if corpus.ignoresIdenticalIDs == true { options.limit += 1 }
        return options
    }

    /// The identity a run of this corpus against `engine` is measured under.
    ///
    /// Each space as the engine has it rather than as it was configured: an
    /// index reopened under a space whose fingerprint it already holds keeps
    /// the width and index kind it was created with. Complete once the
    /// engine is open, so a run that cannot be compared with a baseline can
    /// be refused before it indexes anything; only `indexKind` can change
    /// after that, when indexing grows a space past its graph threshold.
    public func identity(of engine: SemanticSearchEngine) async -> EvaluationIdentity {
        let configuration = engine.configuration
        var models: [String] = []
        var spaces: [EvaluationIdentity.Space] = []
        var firstSearchedExactly: Bool?
        for space in configuration.spaces {
            let descriptor = await space.provider.descriptor
            let capabilities = await space.provider.capabilities
            models.append(descriptor.modelIdentity)
            // Every configured space is registered and loaded when the
            // engine opens; the configuration below is a fallback only.
            let searched = await engine.searchedSpace(space.id)

            // In the engine's order (its dense channels): the query's
            // instruction, the space's, the engine's. Only for the families
            // the space embeds queries for: another family's is never read.
            var instructions: [String: String] = [:]
            for family in ChannelFamily.allCases
            where searched?.families.contains(family) ?? true {
                if let instruction = options.search.queryInstructions[family]
                    ?? space.queryInstructions[family]
                    ?? configuration.queryInstructions[family]
                {
                    instructions[family.rawValue] = instruction
                }
            }
            spaces.append(
                EvaluationIdentity.Space(
                    id: space.id.rawValue,
                    fingerprint: searched?.fingerprint ?? descriptor.fingerprint,
                    dimensions: options.search.reducedDimensions
                        ?? searched?.dimensions
                        ?? space.reducedDimensions ?? descriptor.dimensions,
                    maximumInputTokens: capabilities.maximumInputTokens,
                    queryInstructions: instructions))
            if firstSearchedExactly == nil {
                firstSearchedExactly = searched?.searchedExactly ?? (space.indexKind == .exact)
            }
        }
        var rerankerModel: String?
        if let reranker = configuration.reranker {
            rerankerModel = await reranker.identity.modelIdentity
        }

        let indexKind: String
        switch firstSearchedExactly {
        case true?: indexKind = VectorIndexKind.exact.rawValue
        case false?: indexKind = VectorIndexKind.hnsw.rawValue
        case nil: indexKind = "none"
        }

        return EvaluationIdentity(
            corpusName: corpus.name,
            documentCount: corpus.documents.count,
            queryCount: corpus.queries.count,
            embeddingModels: models.sorted(),
            rerankerModel: rerankerModel,
            analyzerVersion: configuration.analyzer.analyzerVersion,
            bm25: configuration.bm25,
            fusionK: options.search.fusion.k,
            candidatesPerChannel: options.search.candidatesPerChannel,
            rerankPoolSize: options.search.rerankPoolSize,
            limit: options.search.limit,
            mode: options.search.mode.rawValue,
            indexKind: indexKind,
            spaces: spaces.sorted { $0.id < $1.id },
            // Written out when off, so that `nil` keeps meaning an identity
            // recorded before the rule was.
            ignoresIdenticalIDs: corpus.ignoresIdenticalIDs ?? false)
    }
}

extension SemanticSearchEngine {
    /// One space as it is stored and searched.
    struct SearchedSpace: Sendable {
        /// Of the stored descriptor, the one its vectors were written under.
        var fingerprint: String
        /// The stored width.
        var dimensions: Int
        /// Families the space embeds queries for.
        var families: Set<ChannelFamily>
        /// By brute force rather than through a graph.
        var searchedExactly: Bool
    }

    /// Registering a space whose fingerprint the store already holds keeps
    /// the stored row, so a reopened space has the width and index kind it
    /// was created with, not the configuration's; and an `hnsw` space below
    /// its graph threshold is searched exactly.
    func searchedSpace(_ id: EmbeddingSpaceID) async -> SearchedSpace? {
        guard let runtime = spaces[id] else { return nil }
        let families = Set(await eligibleKinds(for: runtime).map(\.channelFamily))
        return SearchedSpace(
            fingerprint: runtime.stored.descriptor.fingerprint,
            dimensions: runtime.stored.effectiveDimensions,
            families: families,
            searchedExactly: runtime.index.isExact)
    }
}

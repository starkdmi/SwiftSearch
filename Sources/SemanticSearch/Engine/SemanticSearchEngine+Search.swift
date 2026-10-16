import Foundation
import NaturalLanguage

extension SemanticSearchEngine {

    // MARK: - Public entry points

    /// Search the whole index.
    public func search(
        _ query: SearchQuery,
        options: SearchOptions = .default
    ) async throws -> SearchResults {
        try await execute(query: query, options: try options.validated())
    }

    public func search(
        _ text: String,
        options: SearchOptions = .default
    ) async throws -> SearchResults {
        try await search(SearchQuery(text), options: options)
    }

    /// Search within one item and everything beneath it.
    ///
    /// The scope is applied as a filter on every channel, so a parent-scoped
    /// search retrieves the same number of candidates per channel as an
    /// unscoped one. It is not a post-filter over global results, which would
    /// return a handful of hits for a scope that holds thousands of items.
    public func searchInside(
        _ parent: ItemID,
        query: SearchQuery,
        options: SearchOptions = .default
    ) async throws -> SearchResults {
        var scoped = options
        scoped.filter = .and([options.filter, .withinParent(parent)])
        return try await search(query, options: scoped)
    }

    public func searchInside(
        _ parent: ItemID, _ text: String, options: SearchOptions = .default
    ) async throws -> SearchResults {
        try await searchInside(parent, query: SearchQuery(text), options: options)
    }

    // MARK: - Pipeline

    private func execute(query: SearchQuery, options: SearchOptions) async throws -> SearchResults
    {
        let started = ContinuousClock.now
        guard !query.isEmpty else {
            throw SemanticSearchError.invalidQuery("query must have text or at least one image")
        }
        if query.isImageOnly, options.mode == .lexicalOnly {
            throw SemanticSearchError.invalidQuery(
                "lexicalOnly cannot answer an image-only query")
        }

        var diagnostics = SearchDiagnostics(mode: options.mode)
        // The question's language, not the captions': they are in the
        // captioner's language whatever the user asked in.
        var query = query
        let languageSource = query.isImageOnly ? nil : query.text
        if let captioner = configuration.queryCaptioner, !query.images.isEmpty {
            query = try await describeImages(of: query, with: captioner, diagnostics: &diagnostics)
        }
        diagnostics.detectedQueryLanguage =
            query.languageHint.flatMap(TextNormalization.normalizedLanguageCode)
            ?? Self.detectLanguage(languageSource ?? query.text, indexedLanguages: indexedLanguages)

        // 1. Validate and filter. One resolved slot set, applied identically to
        //    every channel below.
        let allowedSlots = try store.resolveFilter(options.filter)
        if let allowedSlots, allowedSlots.isEmpty {
            diagnostics.totalDuration = ContinuousClock.now - started
            return SearchResults(query: query, hits: [], diagnostics: diagnostics)
        }

        // 2. Query variants. The original is always first and always kept.
        let variants = try await queryVariants(for: query)

        // 3. Retrieval.
        let retrievalStarted = ContinuousClock.now
        var channels: [RetrievalChannelResult] = []
        var answeredLexically = false

        if options.mode != .denseOnly {
            let (lexical, terms, duration) = try lexicalChannel(
                variants: variants, options: options, allowedSlots: allowedSlots,
                language: diagnostics.detectedQueryLanguage)
            diagnostics.queryTerms = terms
            if !lexical.candidates.isEmpty { channels.append(lexical) }
            diagnostics.channels.append(
                SearchDiagnostics.ChannelDiagnostics(
                    channel: .lexical,
                    retrieved: lexical.candidates.count,
                    uniqueUnits: Set(lexical.candidates.map(\.key.item)).count,
                    duration: duration,
                    appliedWeight: 0))
            if let quick = options.quickLexicalAnswer, options.mode != .lexicalOnly,
                query.images.isEmpty,
                query.text.split(whereSeparator: \.isWhitespace).count <= quick.maximumWords,
                terms.count <= quick.maximumTerms,
                Set(lexical.candidates.map(\.key.item)).count >= quick.minimumHits
            {
                answeredLexically = true
                diagnostics.answeredLexically = true
            }
        }

        if options.mode != .lexicalOnly, !answeredLexically {
            let dense = try await denseChannels(
                variants: variants, options: options, allowedSlots: allowedSlots,
                diagnostics: &diagnostics)
            channels.append(contentsOf: dense)
        }
        diagnostics.retrievalDuration = ContinuousClock.now - retrievalStarted

        // 4. Fusion.
        let fused = RankFusion.fuse(channels, configuration: options.fusion)
        diagnostics.fusedCandidates = fused.count

        let appliedWeights = RankFusion.channelWeights(
            for: channels, configuration: options.fusion)
        for index in diagnostics.channels.indices {
            diagnostics.channels[index].appliedWeight =
                appliedWeights[diagnostics.channels[index].channel] ?? 0
        }

        guard !fused.isEmpty else {
            diagnostics.totalDuration = ContinuousClock.now - started
            return SearchResults(query: query, hits: [], diagnostics: diagnostics)
        }

        // Pictures and text found through channels of unequal reach take
        // turns in the pool rather than competing on one fused score.
        let stratified = Self.stratifiedOrder(
            channels, fused: fused, visualFirst: !query.images.isEmpty,
            configuration: options.fusion)
        let ordered = stratified ?? fused
        diagnostics.stratifiedPool = stratified != nil
        var effective = options
        if stratified != nil { effective.rerankerOrderingOnly = true }

        // 5. Rerank.
        let pool = Array(ordered.prefix(options.rerankPoolSize))
        var ranked: [RankedCandidate]
        if options.mode == .hybrid, !answeredLexically, let reranker = configuration.reranker {
            let rerankStarted = ContinuousClock.now
            ranked = try await rerank(
                pool: pool, query: query, options: effective, reranker: reranker,
                diagnostics: &diagnostics)
            diagnostics.rerankDuration = ContinuousClock.now - rerankStarted
        } else {
            if options.mode == .hybrid, !answeredLexically {
                diagnostics.rerankSkippedReason = "no reranker is configured"
            }
            ranked = Self.retrievalOrder(pool)
        }
        let reranked = diagnostics.rerankedCandidates > 0
        if stratified != nil, !reranked {
            // No common judge ran, so the turn-taking order is the ranking.
            // Its scores follow position, which the fused scores of two
            // strata would not.
            for index in ranked.indices {
                ranked[index].finalScore = 1 / (options.fusion.k + Float(index + 1))
            }
        }

        // A limit above the rerank pool is filled from retrieval order, below
        // everything the pool holds. The pool bounds reranking cost, not the
        // number of results a caller can ask for.
        if options.limit > pool.count, ordered.count > pool.count {
            let extra = ordered.dropFirst(pool.count).prefix(options.limit - pool.count)
            for (offset, candidate) in extra.enumerated() {
                var entry = RankedCandidate(
                    candidate: candidate, rerankScore: nil, rerankRank: nil,
                    finalScore: candidate.fusedScore, truncated: false)
                if stratified != nil || (reranked && options.rerankerOrderingOnly) {
                    // Keeps `finalScore` descending down the list: after the
                    // reranked pool, in the positions that follow it.
                    entry.finalScore = 1 / (options.fusion.k + Float(pool.count + offset + 1))
                }
                ranked.append(entry)
            }
        }

        // 6. Results.
        let hits = try await makeHits(
            Array(ranked.prefix(options.limit)), options: options)
        let groups = options.groupByParent ? try groupByParent(hits) : []
        var moments: [SearchMoment] = []
        if let gap = options.momentGap {
            let depth = max(options.limit, pool.count)
            let pooled =
                depth > hits.count
                ? try await makeHits(Array(ranked.prefix(depth)), options: options) : hits
            moments = Array(Self.moments(from: pooled, gap: gap).prefix(options.limit))
        }

        diagnostics.totalDuration = ContinuousClock.now - started
        return SearchResults(
            query: query, hits: hits, groups: groups, moments: moments,
            diagnostics: diagnostics)
    }

    // MARK: - Query captioning

    /// The query with its images replaced by their captions — see
    /// ``EngineConfiguration/queryCaptioner``. An image the captioner has
    /// nothing to say about is kept; a query left with no captions at all is
    /// returned as asked.
    private func describeImages(
        of query: SearchQuery, with captioner: any ImageCaptioner,
        diagnostics: inout SearchDiagnostics
    ) async throws -> SearchQuery {
        let images = query.images
        let captions = try await withInference(priority: .interactive) {
            try await self.makeResident(captioner: captioner)
            var captions: [String?] = []
            for image in images {
                try Task.checkCancellation()
                let caption = try await captioner.caption(image)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                captions.append(caption.isEmpty ? nil : caption)
            }
            return captions
        }
        let written = captions.compactMap { $0 }
        guard !written.isEmpty else { return query }
        diagnostics.queryCaptions = written

        var described = query
        described.images = zip(images, captions).compactMap { $1 == nil ? $0 : nil }
        let question = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        // The caption alone when there is no question, otherwise the
        // question and then `Frame: <caption>`.
        described.text =
            question.isEmpty && written.count == 1
            ? written[0]
            : ([question].filter { !$0.isEmpty } + written.map { "Frame: \($0)" })
                .joined(separator: "\n")
        return described
    }

    // MARK: - Query variants

    private func queryVariants(for query: SearchQuery) async throws -> [QueryVariant] {
        let transform = configuration.queryTransform
        guard transform.maximumAdditionalVariants > 0 else {
            return [QueryVariant(query: query)]
        }
        var produced = try await transform.transform(query)
        // The original is retained whatever the transform returns, and the
        // extras are bounded. An expansion that grows without limit would give
        // its family unbounded say in the fused ranking, which family weight
        // normalisation cannot compensate for if the count is unbounded.
        if !produced.contains(where: { $0.variant == nil }) {
            produced.insert(QueryVariant(query: query), at: 0)
        }
        let extras = produced.filter { $0.variant != nil }
            .prefix(transform.maximumAdditionalVariants)
        return [QueryVariant(query: query)] + extras
    }

    // MARK: - Lexical channel

    private func lexicalChannel(
        variants: [QueryVariant],
        options: SearchOptions,
        allowedSlots: Set<Int>?,
        language: String?
    ) throws -> (RetrievalChannelResult, [String], Duration) {
        let started = ContinuousClock.now

        var terms: [String] = []
        var seen: Set<String> = []
        for variant in variants {
            for term in configuration.analyzer.analyzeQuery(
                variant.query.text, language: language)
            where seen.insert(term).inserted {
                terms.append(term)
            }
        }

        guard !terms.isEmpty else {
            return (
                RetrievalChannelResult(channel: .lexical, candidates: []), terms,
                ContinuousClock.now - started
            )
        }

        let scores = try store.lexicalScores(
            queryTerms: terms, parameters: configuration.bm25, allowedSlots: allowedSlots)
        let top = BM25.top(scores, limit: options.candidatesPerChannel)
        let rows = try store.representations(slots: top.map(\.slot))

        var candidates: [RetrievedCandidate] = []
        candidates.reserveCapacity(top.count)
        for entry in top {
            guard let row = rows[entry.slot] else { continue }
            candidates.append(
                RetrievedCandidate(
                    slot: entry.slot, key: row.key, score: entry.score, weight: row.weight))
        }

        return (
            RetrievalChannelResult(channel: .lexical, candidates: candidates), terms,
            ContinuousClock.now - started
        )
    }

    // MARK: - Dense channels

    private func denseChannels(
        variants: [QueryVariant],
        options: SearchOptions,
        allowedSlots: Set<Int>?,
        diagnostics: inout SearchDiagnostics
    ) async throws -> [RetrievalChannelResult] {
        var results: [RetrievalChannelResult] = []

        let targetSpaces =
            options.spaces.isEmpty
            ? spaces.keys.sorted { $0.rawValue < $1.rawValue }
            : options.spaces

        for spaceID in targetSpaces {
            guard let runtime = spaces[spaceID] else {
                throw SemanticSearchError.unknownEmbeddingSpace(spaceID)
            }

            let provider = runtime.configuration.provider
            let capabilities = await provider.capabilities
            let descriptor = await provider.descriptor

            // Model compatibility, checked on every query rather than assumed
            // from registration: the provider instance can have been swapped
            // since the index was written.
            guard
                descriptor.isCompatible(
                    with: runtime.stored.descriptor,
                    reducedDimensions: options.reducedDimensions
                        ?? runtime.stored.reducedDimensions)
            else {
                throw SemanticSearchError.incompatibleEmbeddingSpace(
                    space: spaceID,
                    expected: runtime.stored.descriptor.fingerprint,
                    actual: descriptor.fingerprint,
                    detail: Self.mismatchDetail(
                        indexed: runtime.stored.descriptor, queried: descriptor))
            }

            if let reduced = options.reducedDimensions {
                guard capabilities.supportsDimensionReduction else {
                    throw SemanticSearchError.dimensionReductionUnsupported(
                        model: descriptor.modelIdentity, requested: reduced,
                        native: descriptor.dimensions)
                }
                guard reduced <= runtime.stored.effectiveDimensions else {
                    throw SemanticSearchError.dimensionReductionUnsupported(
                        model: descriptor.modelIdentity, requested: reduced,
                        native: runtime.stored.effectiveDimensions)
                }
            }

            guard runtime.index.count > 0 else {
                diagnostics.skippedSpaces.append(
                    .init(space: spaceID, reason: "space holds no vectors"))
                continue
            }

            let families = await self.families(for: runtime)
            guard !families.isEmpty else {
                diagnostics.skippedSpaces.append(
                    .init(space: spaceID, reason: "no representation kinds routed to this space"))
                continue
            }

            for variant in variants {
                if variant.query.isImageOnly, !capabilities.acceptsImages {
                    diagnostics.skippedSpaces.append(
                        .init(
                            space: spaceID,
                            reason: "'\(descriptor.modelIdentity)' does not accept image queries"))
                    continue
                }

                // A text-and-image query against a space that cannot take
                // images still has a text half this space can answer, and
                // dropping the space would lose it. What it cannot be is
                // silent: the space answered a narrower query than the one
                // asked, and the diagnostics say so.
                var queryImages = variant.query.images
                if !queryImages.isEmpty, !capabilities.acceptsImages {
                    queryImages = []
                    diagnostics.partiallyAnsweredSpaces.append(
                        .init(
                            space: spaceID,
                            reason:
                                "'\(descriptor.modelIdentity)' does not accept images; "
                                + "only the text of the query was embedded"))
                }

                // Families of one space and variant differ only in their
                // instruction. Without one of their own they ask the same
                // question, and embedding it for each family ran the text
                // profile's model six times a query for one vector.
                var outputs: [String?: EmbeddingOutput] = [:]
                for family in families.sorted(by: { $0.rawValue < $1.rawValue }) {
                    try Task.checkCancellation()
                    let channel = RetrievalChannelID(
                        family: family, space: spaceID, variant: variant.variant)
                    let started = ContinuousClock.now

                    let instruction =
                        options.queryInstructions[family]
                        ?? runtime.configuration.queryInstructions[family]
                        ?? configuration.queryInstructions[family]

                    let output: EmbeddingOutput
                    if let embedded = outputs[instruction] {
                        output = embedded
                    } else {
                        let input = EmbeddingInput(
                            text: variant.query.text,
                            images: queryImages,
                            role: .query(instruction: instruction),
                            requestedDimensions: options.reducedDimensions
                                ?? runtime.stored.reducedDimensions)

                        // Load and embed under one permit: between two, another
                        // job's residency switch could release the model again.
                        output = try await withInference(priority: .interactive) {
                            try await self.makeResident(embedder: provider)
                            return try await provider.embed(input)
                        }
                        outputs[instruction] = output
                    }
                    guard !output.vector.isEmpty else {
                        diagnostics.skippedSpaces.append(
                            .init(
                                space: spaceID,
                                reason: "provider returned an empty query vector"))
                        continue
                    }

                    let comparedDimensions =
                        options.reducedDimensions ?? runtime.stored.effectiveDimensions
                    let queryVector = VectorMath.normalizedPrefix(
                        output.vector, dimensions: comparedDimensions)

                    // Dense candidates are restricted to the kinds that belong
                    // to this family as well as to the caller's filter, so two
                    // families sharing one space do not return each other's
                    // representations. Both tests run inside the index search
                    // rather than after it; filtering afterwards is what leaves
                    // a selective filter with one survivor out of a hundred.
                    let restrictFamily = families.count > 1
                    let kinds = slotKinds
                    let predicate: (@Sendable (Int) -> Bool)?
                    switch (allowedSlots, restrictFamily) {
                    case (nil, false):
                        predicate = nil
                    case (let allowed?, false):
                        predicate = { allowed.contains($0) }
                    case (nil, true):
                        predicate = { kinds[$0]?.channelFamily == family }
                    case (let allowed?, true):
                        predicate = { allowed.contains($0) && kinds[$0]?.channelFamily == family }
                    }

                    let matches = try runtime.index.search(
                        query: queryVector,
                        limit: options.candidatesPerChannel,
                        isAllowed: predicate)

                    let rows = try store.representations(slots: matches.map(\.slot))
                    var candidates: [RetrievedCandidate] = []
                    candidates.reserveCapacity(matches.count)
                    for match in matches {
                        guard let row = rows[match.slot] else { continue }
                        candidates.append(
                            RetrievedCandidate(
                                slot: match.slot, key: row.key, score: match.score,
                                weight: row.weight))
                    }

                    if !candidates.isEmpty {
                        results.append(
                            RetrievalChannelResult(channel: channel, candidates: candidates))
                    }
                    diagnostics.channels.append(
                        SearchDiagnostics.ChannelDiagnostics(
                            channel: channel,
                            retrieved: candidates.count,
                            uniqueUnits: Set(candidates.map(\.key.item)).count,
                            duration: ContinuousClock.now - started,
                            appliedWeight: 0))
                }
            }
        }

        return results
    }

    private func families(for runtime: SpaceRuntime) async -> Set<ChannelFamily> {
        let kinds = await eligibleKinds(for: runtime)
        return Set(kinds.map(\.channelFamily))
    }

    static func mismatchDetail(
        indexed: EmbeddingSpaceDescriptor, queried: EmbeddingSpaceDescriptor
    ) -> String {
        if indexed.modelIdentity != queried.modelIdentity { return "model identity" }
        if indexed.weightsVariant != queried.weightsVariant { return "weights variant" }
        if indexed.tokenizerVersion != queried.tokenizerVersion { return "tokenizer version" }
        if indexed.preprocessingVersion != queried.preprocessingVersion {
            return "preprocessing version"
        }
        if indexed.documentInstruction != queried.documentInstruction {
            return "document instruction"
        }
        if indexed.pooling != queried.pooling { return "pooling" }
        if indexed.normalized != queried.normalized { return "normalization" }
        return "dimensions"
    }

    // MARK: - Reranking

    struct RankedCandidate: Sendable {
        var candidate: FusedCandidate
        var rerankScore: Float?
        var rerankRank: Int?
        var finalScore: Float
        var truncated: Bool
    }

    private func rerank(
        pool: [FusedCandidate],
        query: SearchQuery,
        options: SearchOptions,
        reranker: any RerankingProvider,
        diagnostics: inout SearchDiagnostics
    ) async throws -> [RankedCandidate] {
        guard !pool.isEmpty else { return [] }

        let capabilities = await reranker.capabilities
        let showsImages =
            capabilities.acceptsImages
            && (options.reranksImages ?? configuration.reranksImages)
        if !query.images.isEmpty, !showsImages {
            // An image-only query would reach a text reranker as an empty
            // string, and scores against nothing would still be fused into the
            // ranking with a full channel's weight. Not reranking is the honest
            // outcome; a mixed query can still be judged on its text.
            if query.isImageOnly {
                diagnostics.rerankSkippedReason =
                    capabilities.acceptsImages
                    ? "reranking by text was requested and the query is an image"
                    : "'\(await reranker.identity.modelIdentity)' cannot read an image-only query"
                return Self.retrievalOrder(pool)
            }
            diagnostics.rerankIgnoredQueryImages = true
        }
        let rows = try store.representations(slots: pool.map(\.bestSlot))
        let parents = try parentLookup(
            for: pool.map(\.itemID), contextLimit: options.parentContextCharacters)

        var candidates: [RerankCandidate] = []
        var eligible: [Int] = []
        candidates.reserveCapacity(pool.count)
        // A frame and its caption are one piece of evidence, whichever of the
        // two retrieval reached: a reranker shown the picture also reads what
        // was written about it, and one that is not shown the picture reads
        // the words alone.
        let pairsImages = showsImages && capabilities.maximumImagesPerPair > 0
        let companions = try frameCompanions(
            of: pool.compactMap { rows[$0.bestSlot] }, pairsImages: pairsImages)

        for (offset, fused) in pool.enumerated() {
            guard let row = rows[fused.bestSlot] else { continue }
            let companion = companions[row.key]

            var images: [ImagePayload] = []
            if pairsImages, let visual = row.kind.isVisual ? row : companion?.visual {
                switch visual.content {
                case .imageData(let data, let contentType):
                    images = [.encoded(data, contentType: contentType)]
                case .mediaReference(let url) where url.isFileURL:
                    if let allowed = configuration.mediaAccess, !allowed(url) {
                        diagnostics.unavailableRepresentations.append(
                            .init(key: visual.key, reason: "media may not be read at \(url.path)"))
                    } else if FileManager.default.fileExists(atPath: url.path) {
                        images = [.fileURL(url)]
                    } else {
                        diagnostics.unavailableRepresentations.append(
                            .init(key: visual.key, reason: "media is missing at \(url.path)"))
                    }
                default:
                    break
                }
            }

            var text = row.text ?? ""
            if row.kind.isVisual, let words = companion?.text { text = words }

            // A candidate with no text and no image the reranker can take —
            // a picture in front of a text-only reranker — has nothing to be
            // judged on. Scoring it as an empty string would rank it by how the
            // model feels about emptiness.
            if images.isEmpty, text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                diagnostics.rerankUnscoredCandidates += 1
                continue
            }

            let parent = parents[row.itemID]
            candidates.append(
                RerankCandidate(
                    text: text,
                    parentContext: parent?.text,
                    kind: row.kind,
                    images: images,
                    title: parent?.title))
            eligible.append(offset)
        }

        guard !candidates.isEmpty else { return Self.retrievalOrder(pool) }

        let rerankQuery = RerankQuery(
            text: query.text,
            images: showsImages ? query.images : [],
            instruction: options.rerankInstruction ?? configuration.rerankInstruction)

        // Sequential residency is applied inside the permit, by
        // `makeResident`: the embedder is released before the reranker loads,
        // at the cost of a reload on the next query.
        let batch = candidates
        let scores = try await withInference(priority: .interactive) {
            try await self.makeResident(reranker: reranker)
            return try await reranker.score(query: rerankQuery, candidates: batch)
        }
        guard scores.count == candidates.count else {
            throw SemanticSearchError.providerReturnedNoOutput(
                await reranker.identity.modelIdentity)
        }

        var ranked = pool.map {
            RankedCandidate(
                candidate: $0, rerankScore: nil, rerankRank: nil, finalScore: $0.fusedScore,
                truncated: false)
        }
        for (position, poolIndex) in eligible.enumerated() {
            ranked[poolIndex].rerankScore = scores[position].score
            ranked[poolIndex].truncated = scores[position].wasTruncated
            if scores[position].wasTruncated { diagnostics.rerankTruncatedCandidates += 1 }
            diagnostics.rerankDroppedImages += scores[position].droppedImages
        }
        diagnostics.rerankedCandidates = candidates.count

        // Assign reranker ranks over the scored subset only, then fuse those
        // ranks back with retrieval. Late fusion rather than replacement: when
        // the reranker is wrong about a candidate retrieval was confident in,
        // replacement loses it outright and fusion only demotes it.
        let order = eligible.sorted { lhs, rhs in
            let lhsScore = ranked[lhs].rerankScore ?? -.greatestFiniteMagnitude
            let rhsScore = ranked[rhs].rerankScore ?? -.greatestFiniteMagnitude
            if lhsScore != rhsScore { return lhsScore > rhsScore }
            if ranked[lhs].candidate.fusedScore != ranked[rhs].candidate.fusedScore {
                return ranked[lhs].candidate.fusedScore > ranked[rhs].candidate.fusedScore
            }
            return ranked[lhs].candidate.itemID.rawValue < ranked[rhs].candidate.itemID.rawValue
        }

        // Unscored candidates hold their retrieval position in the reranker's
        // ordering and the scored ones fill the positions around them. The
        // reranker has no opinion on those candidates, so it neither promotes
        // nor demotes them; leaving them out of the ordering would demote
        // every one of them below every scored candidate.
        let scored = Set(eligible)
        var remaining = order.makeIterator()
        let positions = pool.indices.map { scored.contains($0) ? remaining.next()! : $0 }

        for (offset, poolIndex) in positions.enumerated() {
            let rank = offset + 1
            if scored.contains(poolIndex) { ranked[poolIndex].rerankRank = rank }
            let reciprocal = 1 / (options.fusion.k + Float(rank))
            ranked[poolIndex].finalScore =
                options.rerankerOrderingOnly
                ? reciprocal
                : ranked[poolIndex].candidate.fusedScore + reciprocal
        }

        ranked.sort { lhs, rhs in
            if lhs.finalScore != rhs.finalScore { return lhs.finalScore > rhs.finalScore }
            let lhsRerank = lhs.rerankScore ?? -.greatestFiniteMagnitude
            let rhsRerank = rhs.rerankScore ?? -.greatestFiniteMagnitude
            if lhsRerank != rhsRerank { return lhsRerank > rhsRerank }
            if lhs.candidate.fusedScore != rhs.candidate.fusedScore {
                return lhs.candidate.fusedScore > rhs.candidate.fusedScore
            }
            return lhs.candidate.itemID.rawValue < rhs.candidate.itemID.rawValue
        }
        return ranked
    }

    /// What goes with each frame in the pool: for a picture, the words
    /// written about it — its captions, else its OCR text; for a caption, the
    /// picture it was written from, when the reranker will be shown one.
    private func frameCompanions(of rows: [StoredRepresentation], pairsImages: Bool) throws
        -> [RepresentationKey: (text: String?, visual: StoredRepresentation?)]
    {
        var siblings: [ItemID: [StoredRepresentation]] = [:]
        func representations(of itemID: ItemID) throws -> [StoredRepresentation] {
            if let cached = siblings[itemID] { return cached }
            let loaded = try store.representations(forItem: itemID)
            siblings[itemID] = loaded
            return loaded
        }

        var result: [RepresentationKey: (text: String?, visual: StoredRepresentation?)] = [:]
        for row in rows {
            if row.kind.isVisual {
                let all = try representations(of: row.itemID)
                let captions = all.filter { $0.kind == .caption }
                let own = captions.filter { $0.derivedFrom == row.representationID }
                for group in [own, captions, all.filter { $0.kind == .ocrText }] {
                    let text = group
                        .compactMap { $0.text?.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                        .joined(separator: "\n")
                    if !text.isEmpty {
                        result[row.key] = (text, nil)
                        break
                    }
                }
            } else if pairsImages, row.kind == .caption, let source = row.derivedFrom {
                let visual = try representations(of: row.itemID).first {
                    $0.representationID == source && $0.kind.isVisual
                }
                if let visual { result[row.key] = (nil, visual) }
            }
        }
        return result
    }

    /// The pool in retrieval order, unreranked.
    /// Retrieval order for an index where pictures sit beside text, or `nil`
    /// when every channel reaches every candidate kind and plain fusion holds.
    ///
    /// A picture is reached by one channel, visual similarity; text also by
    /// BM25, which cannot score a picture at all. Summed in one RRF, two
    /// middling votes beat one first place, so text crowds pictures out of
    /// the pool whatever the pictures' similarity. Fusion is therefore kept
    /// to channels that can score the same candidates — BM25 with the text
    /// embeddings, the visual embeddings on their own — and the two lists
    /// take turns filling the pool, where the reranker is the first stage to
    /// compare a picture with a line directly. RRF, like CombSUM, assumes
    /// every retriever ranks the same collection (Bruch, Gai & Ingber,
    /// "An Analysis of Fusion Functions for Hybrid Retrieval", 2023); the
    /// split keeps that assumption true within each list.
    ///
    /// The list matching the query goes first — pictures for a query with an
    /// image — which decides only an odd pool's last place and the order
    /// when nothing reranks.
    static func stratifiedOrder(
        _ channels: [RetrievalChannelResult], fused: [FusedCandidate], visualFirst: Bool,
        configuration: FusionConfiguration
    ) -> [FusedCandidate]? {
        let visual = channels.filter { $0.channel.family == .visual }
        let text = channels.filter { $0.channel.family != .visual }
        guard !visual.isEmpty, !text.isEmpty else { return nil }

        // Each item keeps its evidence from every channel; only the order
        // comes from the strata.
        let byItem = Dictionary(fused.map { ($0.itemID, $0) }, uniquingKeysWith: { lhs, _ in lhs })
        var strata = [
            RankFusion.fuse(visual, configuration: configuration),
            RankFusion.fuse(text, configuration: configuration),
        ]
        if !visualFirst { strata.reverse() }

        var ordered: [FusedCandidate] = []
        ordered.reserveCapacity(byItem.count)
        var seen: Set<ItemID> = []
        var cursors = [0, 0]
        while cursors[0] < strata[0].count || cursors[1] < strata[1].count {
            for stratum in 0..<2 {
                // A turn is one new item: one already placed from the other
                // list does not use it up.
                while cursors[stratum] < strata[stratum].count {
                    let itemID = strata[stratum][cursors[stratum]].itemID
                    cursors[stratum] += 1
                    if seen.insert(itemID).inserted, let candidate = byItem[itemID] {
                        ordered.append(candidate)
                        break
                    }
                }
            }
        }
        return ordered
    }

    static func retrievalOrder(_ pool: [FusedCandidate]) -> [RankedCandidate] {
        pool.map {
            RankedCandidate(
                candidate: $0, rerankScore: nil, rerankRank: nil, finalScore: $0.fusedScore,
                truncated: false)
        }
    }

    // MARK: - Result assembly

    private func makeHits(
        _ ranked: [RankedCandidate], options: SearchOptions
    ) async throws -> [SearchHit] {
        guard !ranked.isEmpty else { return [] }

        let rows = try store.representations(slots: ranked.map(\.candidate.bestSlot))
        let itemIDs = ranked.map(\.candidate.itemID)
        let items = try store.items(itemIDs)
        let parents = try parentLookup(
            for: itemIDs, contextLimit: options.parentContextCharacters, items: items)

        var hits: [SearchHit] = []
        hits.reserveCapacity(ranked.count)

        for entry in ranked {
            guard let row = rows[entry.candidate.bestSlot], let item = items[entry.candidate.itemID]
            else { continue }

            let parentContext = parents[item.id].map {
                ParentContext(
                    itemID: $0.item.id,
                    metadata: $0.item.metadata,
                    locator: $0.item.locator,
                    text: $0.text)
            }

            let supporting =
                entry.candidate.evidence
                .map(\.key)
                .filter { $0 != row.key && $0.item == item.id }
                .reduce(into: [RepresentationID]()) { result, key in
                    if !result.contains(key.representation) {
                        result.append(key.representation)
                    }
                }

            hits.append(
                SearchHit(
                    itemID: item.id,
                    representationID: row.representationID,
                    kind: row.kind,
                    text: row.text,
                    language: row.language,
                    locator: row.locator ?? item.locator,
                    metadata: item.metadata,
                    parent: parentContext,
                    provenance: row.provenance,
                    evidence: RankingEvidence(
                        fusedScore: entry.candidate.fusedScore,
                        channels: entry.candidate.evidence,
                        rerankScore: entry.rerankScore,
                        rerankRank: entry.rerankRank,
                        finalScore: entry.finalScore,
                        rerankTruncated: entry.truncated),
                    supportingRepresentations: supporting))
        }

        return hits
    }

    /// Group hits under their parents.
    ///
    /// Ordered by each group's strongest hit, never by how many hits it holds.
    /// A hit-count bonus ranks a long document with many weak matches above a
    /// short one that actually answers the query.
    private func groupByParent(_ hits: [SearchHit]) throws -> [SearchGroup] {
        var order: [ItemID] = []
        var grouped: [ItemID: [SearchHit]] = [:]

        for hit in hits {
            let key = hit.parent?.itemID ?? hit.itemID
            if grouped[key] == nil { order.append(key) }
            grouped[key, default: []].append(hit)
        }

        var groups: [SearchGroup] = []
        for key in order {
            guard let members = grouped[key], let best = members.first else { continue }
            // Taken from any child hit, not only the best: when the parent
            // itself is the strongest hit, its own `parent` is its grandparent
            // or nothing, and the group would lose its context.
            let context = members.lazy.compactMap(\.parent).first { $0.itemID == key }
            groups.append(
                SearchGroup(
                    parentID: key, parent: context, bestHit: best, hits: members))
        }

        groups.sort { lhs, rhs in
            if lhs.bestHit.evidence.finalScore != rhs.bestHit.evidence.finalScore {
                return lhs.bestHit.evidence.finalScore > rhs.bestHit.evidence.finalScore
            }
            return lhs.parentID.rawValue < rhs.parentID.rawValue
        }
        return groups
    }

    /// Group hits by source and time: sorted by start, a hit joins the
    /// current moment when it overlaps it or starts within `gap` seconds of
    /// its end. Hits without a time range
    /// are not moments and are left out.
    static func moments(from hits: [SearchHit], gap: TimeInterval) -> [SearchMoment] {
        var bySource: [String: [(hit: SearchHit, range: SourceLocator.TimeRange)]] = [:]
        for hit in hits {
            guard let range = hit.locator.timeRange else { continue }
            let source =
                hit.locator.source ?? hit.parent?.itemID.rawValue ?? hit.itemID.rawValue
            bySource[source, default: []].append((hit, range))
        }

        func strongerFirst(_ lhs: SearchHit, _ rhs: SearchHit) -> Bool {
            if lhs.evidence.finalScore != rhs.evidence.finalScore {
                return lhs.evidence.finalScore > rhs.evidence.finalScore
            }
            return lhs.itemID.rawValue < rhs.itemID.rawValue
        }

        var moments: [SearchMoment] = []
        for (source, entries) in bySource {
            let ordered = entries.sorted { lhs, rhs in
                lhs.range.start != rhs.range.start
                    ? lhs.range.start < rhs.range.start : lhs.range.end < rhs.range.end
            }
            var members: [SearchHit] = []
            var start: TimeInterval = 0
            var end: TimeInterval = 0
            func close() {
                guard !members.isEmpty else { return }
                let sorted = members.sorted(by: strongerFirst)
                var byKind: [RepresentationKind: SearchHit] = [:]
                for hit in sorted where byKind[hit.kind] == nil { byKind[hit.kind] = hit }
                moments.append(
                    SearchMoment(
                        source: source, timeRange: .init(start: start, end: end),
                        bestHit: sorted[0], bestHitByKind: byKind, hits: sorted))
                members.removeAll()
            }
            for entry in ordered {
                if !members.isEmpty, entry.range.start > end + gap { close() }
                if members.isEmpty {
                    start = entry.range.start
                    end = entry.range.end
                } else {
                    end = max(end, entry.range.end)
                }
                members.append(entry.hit)
            }
            close()
        }
        return moments.sorted { lhs, rhs in
            if lhs.bestHit.evidence.finalScore != rhs.bestHit.evidence.finalScore {
                return lhs.bestHit.evidence.finalScore > rhs.bestHit.evidence.finalScore
            }
            return lhs.timeRange.start < rhs.timeRange.start
        }
    }

    struct ParentInfo {
        var item: StoredItem
        /// Parent text truncated to the context budget; `nil` when the budget
        /// is zero or the parent has no text.
        var text: String?
        var title: String? {
            if case .string(let title)? = item.metadata["title"] { return title }
            return nil
        }
    }

    /// Each item's parent, with its context text, keyed by the *child's* id.
    ///
    /// Batched: one query for the children (unless supplied), one for the
    /// parents, and one per distinct parent for its text. A 50-candidate pool
    /// usually spans a handful of parents, so this is a few queries rather than
    /// three per candidate.
    ///
    /// Context text is bounded because the reranker's pair budget is shared
    /// between evidence and context: an unbounded parent would push the
    /// matched text out of the window the model actually reads.
    func parentLookup(
        for itemIDs: [ItemID], contextLimit: Int, items: [ItemID: StoredItem]? = nil
    ) throws -> [ItemID: ParentInfo] {
        let children = try items ?? store.items(itemIDs)
        var parentOf: [ItemID: ItemID] = [:]
        for id in itemIDs {
            if let parent = children[id]?.parentID { parentOf[id] = parent }
        }
        guard !parentOf.isEmpty else { return [:] }

        let parents = try store.items(Set(parentOf.values))
        var texts: [ItemID: String] = [:]
        if contextLimit > 0 {
            let preferred: [RepresentationKind] = [.summary, .originalText, .caption, .transcript]
            for parentID in Set(parentOf.values).sorted(by: { $0.rawValue < $1.rawValue }) {
                let representations = try store.representations(forItem: parentID)
                for kind in preferred {
                    guard let match = representations.first(where: { $0.kind == kind }),
                        let text = match.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                        !text.isEmpty
                    else { continue }
                    texts[parentID] = String(text.prefix(contextLimit))
                    break
                }
            }
        }

        var result: [ItemID: ParentInfo] = [:]
        for (child, parentID) in parentOf {
            guard let parent = parents[parentID] else { continue }
            result[child] = ParentInfo(item: parent, text: texts[parentID])
        }
        return result
    }

    /// The language to analyse a query in.
    ///
    /// Short queries are where detection is least reliable and matters most:
    /// "espresso martini" is detected as Italian and lemmatised to `esprimere`
    /// and `martino`, which no English document contains, so the lexical
    /// channel returns nothing. A confident detection is taken as it is — a
    /// Russian query stays Russian whatever the corpus holds. Otherwise the
    /// choice is limited to the languages the index actually declares, which
    /// is the only evidence there is about what the query is being matched
    /// against.
    static func detectLanguage(_ text: String, indexedLanguages: Set<String> = []) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 4 else { return nil }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(trimmed)
        let hypotheses = recognizer.languageHypotheses(withMaximum: 1)
        if let (language, probability) = hypotheses.first, probability >= 0.9 {
            return language.rawValue
        }
        guard !indexedLanguages.isEmpty else { return recognizer.dominantLanguage?.rawValue }

        let constrained = NLLanguageRecognizer()
        constrained.languageConstraints = indexedLanguages.sorted().map { NLLanguage(rawValue: $0) }
        constrained.processString(trimmed)
        return constrained.dominantLanguage?.rawValue ?? recognizer.dominantLanguage?.rawValue
    }
}

import Foundation

/// The search engine.
///
/// Owns the store, the vector indexes and the provider lifecycle. Every public
/// operation is one of five — `upsert`, `remove`, `search`, `searchInside`,
/// `reindex` — plus the lifecycle calls that make model residency the caller's
/// decision rather than a side effect of searching.
///
/// The engine never loads a model to answer a lexical query and never
/// substitutes a degraded search for one it cannot run: a space it had to skip
/// is reported in diagnostics, and a missing model is an error.
public actor SemanticSearchEngine {

    /// Runtime state for one embedding space.
    struct SpaceRuntime {
        var stored: StoredSpace
        var configuration: EmbeddingSpaceConfiguration
        var index: any VectorIndex
        /// Generation the in-memory index has caught up to.
        var appliedGeneration: Int
        /// Vectors inserted or removed since the published snapshot, including
        /// the ones replayed on open. Counted in vectors rather than
        /// generations: one generation is a whole embedding batch, so a
        /// generation threshold would mean a different amount of replay work
        /// for every batch size.
        var unsnapshottedChanges = 0
    }

    /// The configuration this engine was built with.
    ///
    /// Public so a benchmark or a diagnostic can record the identity a
    /// measurement was taken under — a number without one cannot be compared to
    /// anything.
    public let configuration: EngineConfiguration
    let store: SearchStore
    private(set) var spaces: [EmbeddingSpaceID: SpaceRuntime] = [:]
    /// Slot → representation kind, kept current on every write. See
    /// ``SearchStore/slotKinds()`` for why it is resident.
    private(set) var slotKinds: [Int: RepresentationKind] = [:]
    /// Base language codes the stored text declares. Grows with upserts and is
    /// not shrunk by removals — a language that is no longer present only
    /// widens the set a query's language is chosen from.
    private(set) var indexedLanguages: Set<String> = []
    /// Serialises inference so a query and a background index build cannot both
    /// hold a model's weights resident. Searches wait in the interactive lane,
    /// indexing in the background one.
    private let inferenceGate: AsyncSemaphore
    /// The gate's permits: releasing models takes all of them.
    private let inferencePermits: Int
    /// One releaser at a time. Two taking the gate's permits one by one could
    /// each end up holding some and wait on the other forever.
    private let releaseGate = AsyncSemaphore(value: 1)
    /// Inference jobs currently holding the gate.
    private var activeInferenceJobs = 0
    /// Processors currently running. They hold no permit — a processor's
    /// captioner or transcriber is not the engine's to schedule — but they
    /// use models, and the idle release that took a captioner shared with
    /// the query side failed the ingest captioning with it.
    private var activeProcessingJobs = 0
    /// Bumped by every inference job, so an idle-release timer can tell whether
    /// anything ran after it was armed.
    private var inferenceEpoch = 0
    private var idleReleaseTask: Task<Void, Never>?

    /// True when opening the index found postings written by a different
    /// lexical analyzer and rebuilt them from the stored text.
    public private(set) var lexicalIndexRebuiltOnOpen = false

    // MARK: - Lifecycle

    public init(configuration: EngineConfiguration) async throws {
        self.configuration = configuration
        _ = try configuration.bm25.validated()

        if let url = configuration.storeURL {
            self.store = try SearchStore(url: url)
        } else {
            self.store = try SearchStore.inMemory()
        }
        // Sequential residency releases models from inside the permit; with a
        // second permit, one job could release the model another is running.
        let permits =
            configuration.runtime.sequentialModelResidency
            ? 1 : configuration.runtime.maximumConcurrentInferenceJobs
        self.inferencePermits = permits
        self.inferenceGate = AsyncSemaphore(value: permits)

        self.lexicalIndexRebuiltOnOpen = try store.ensureLexicalIndex(
            matches: configuration.analyzer)
        try await registerConfiguredSpaces()
        self.slotKinds = try store.slotKinds()
        self.indexedLanguages = try store.indexedLanguages()
        try loadVectorIndexes()
    }

    deinit {
        idleReleaseTask?.cancel()
    }

    /// Register every configured space, rejecting one whose provider disagrees
    /// with what is already stored under that id.
    private func registerConfiguredSpaces() async throws {
        for space in configuration.spaces {
            var descriptor = await space.provider.descriptor
            if descriptor.dimensions <= 0 {
                // Some providers only learn their width from the weights.
                // Registering the space before that would record it as
                // zero-dimensional, and every later query — from a provider
                // that by then knows its width — would be rejected as coming
                // from a different model. Loading here is the only way to get a
                // complete descriptor; providers that know their dimensions up
                // front never pay for it.
                try await withInference(priority: .background) {
                    try await self.makeResident(embedder: space.provider)
                }
                descriptor = await space.provider.descriptor
            }
            let capabilities = await space.provider.capabilities

            if let reduced = space.reducedDimensions {
                guard capabilities.supportsDimensionReduction else {
                    throw SemanticSearchError.dimensionReductionUnsupported(
                        model: descriptor.modelIdentity, requested: reduced,
                        native: descriptor.dimensions)
                }
                guard reduced > 0, reduced <= descriptor.dimensions else {
                    throw SemanticSearchError.dimensionReductionUnsupported(
                        model: descriptor.modelIdentity, requested: reduced,
                        native: descriptor.dimensions)
                }
            }

            try store.registerSpace(
                StoredSpace(
                    id: space.id,
                    descriptor: descriptor,
                    indexKind: space.indexKind,
                    dimensions: descriptor.dimensions,
                    reducedDimensions: space.reducedDimensions,
                    generation: 0,
                    snapshotGeneration: 0,
                    snapshotFile: nil,
                    kinds: space.kinds))
        }
    }

    /// Build or restore the in-memory vector index for every stored space.
    ///
    /// A published snapshot is loaded and then brought forward by replaying the
    /// deltas newer than it, so an interrupted rebuild leaves the index behind
    /// rather than inconsistent. A snapshot that fails to load is discarded and
    /// the space is rebuilt from SQLite, which is authoritative.
    private func loadVectorIndexes() throws {
        for stored in try store.loadSpaces() {
            guard
                let spaceConfiguration = configuration.spaces.first(where: { $0.id == stored.id })
            else {
                // A space in the file that this process was not configured
                // with. Left alone: it belongs to another configuration of the
                // same index, and deleting it would lose work.
                continue
            }

            let dimensions = stored.effectiveDimensions
            let index: any VectorIndex
            var applied = 0

            let vectorCount = try store.vectorCount(space: stored.id)
            let useGraph =
                stored.indexKind == .hnsw
                && vectorCount >= spaceConfiguration.hnsw.minimumVectorsForGraph

            if useGraph {
                let store = self.store
                let spaceID = stored.id
                let graph = try USearchVectorIndex(
                    dimensions: dimensions,
                    configuration: spaceConfiguration.hnsw,
                    fullPrecisionVector: { slot in
                        try? store.vector(space: spaceID, slot: slot)
                    })

                if let file = stored.snapshotFile,
                    let url = store.snapshotURL(for: stored.id),
                    FileManager.default.fileExists(atPath: url.path),
                    url.lastPathComponent == file
                {
                    do {
                        try graph.load(
                            from: url, slots: try store.vectorSlots(space: stored.id))
                        applied = stored.snapshotGeneration
                    } catch {
                        try graph.removeAll()
                        applied = 0
                    }
                }
                index = graph
            } else {
                index = ExactVectorIndex(dimensions: dimensions)
            }

            spaces[stored.id] = SpaceRuntime(
                stored: stored,
                configuration: spaceConfiguration,
                index: index,
                appliedGeneration: applied)

            try replayDeltas(space: stored.id)
        }
    }

    /// Bring a space's in-memory index forward to the store's generation.
    func replayDeltas(space id: EmbeddingSpaceID) throws {
        guard var runtime = spaces[id] else { return }
        let from = runtime.appliedGeneration
        let dimensions = runtime.stored.effectiveDimensions

        // Tombstones first. A slot deleted and re-added within the same window
        // must end up present, and replaying the delete after the insert would
        // leave it absent.
        var replayed = 0
        for slot in try store.tombstones(space: id, newerThan: from) {
            try runtime.index.remove(slot: slot)
            replayed += 1
        }
        var highest = from
        try store.forEachVector(space: id, newerThan: from) { slot, vector in
            try runtime.index.insert(
                slot: slot, vector: VectorMath.normalizedPrefix(vector, dimensions: dimensions))
            replayed += 1
        }
        if let stored = try store.loadSpaces().first(where: { $0.id == id }) {
            highest = stored.generation
            runtime.stored = stored
        }
        runtime.appliedGeneration = highest
        runtime.unsnapshottedChanges += replayed
        spaces[id] = runtime
    }

    /// Load every configured provider's weights.
    ///
    /// Separate from search so preparation and download are the caller's to
    /// schedule, and so a prepared installation works offline.
    ///
    /// Under ``RuntimePolicy/sequentialModelResidency`` each model is loaded in
    /// turn — which is what downloads and validates it — and released when the
    /// next one loads. The embedders go last, so the model a query needs first
    /// is the one left resident.
    public func prepare() async throws {
        if let reranker = configuration.reranker {
            try await withInference(priority: .background) {
                try await self.makeResident(reranker: reranker)
            }
        }
        for space in configuration.spaces {
            try await withInference(priority: .background) {
                try await self.makeResident(embedder: space.provider)
            }
        }
    }

    /// Release every provider's weights. The index stays usable for lexical
    /// search.
    ///
    /// Waits for any inference job in flight, so a model is never released out
    /// from under a batch that is using it.
    public func releaseModels() async {
        await withEveryPermit { await self.releaseAllModels() }
    }

    /// Run `body` with every permit of the gate held, so no inference job is
    /// running or can start. One permit is not that when the gate has
    /// several (`maximumConcurrentInferenceJobs` without sequential
    /// residency): a release through it unloaded a model another job held.
    /// Does nothing if cancelled while waiting.
    private func withEveryPermit(_ body: () async -> Void) async {
        guard (try? await releaseGate.wait(priority: .background)) != nil else { return }
        defer { releaseGate.signal() }
        var held = 0
        defer { for _ in 0..<held { inferenceGate.signal() } }
        for _ in 0..<inferencePermits {
            guard (try? await inferenceGate.wait(priority: .background)) != nil else { return }
            held += 1
        }
        await body()
    }

    private func releaseAllModels() async {
        for space in configuration.spaces { await space.provider.release() }
        await configuration.reranker?.release()
        await configuration.queryCaptioner?.release()
        loadedQueryCaptioner = false
    }

    /// True when the engine loaded the query captioner itself, for a query,
    /// and so unloads it again when sequential residency moves on to another
    /// model. One a processor loaded — the same captioner describing what it
    /// indexes — is the processor's: released at every embed, it would be
    /// loaded again for every file an ingest captions.
    private var loadedQueryCaptioner = false

    /// Make the query captioner ready. Call only while holding the gate.
    func makeResident(captioner: any ImageCaptioner) async throws {
        if configuration.runtime.sequentialModelResidency {
            if let reranker = configuration.reranker, await reranker.isPrepared {
                await reranker.release()
            }
            for space in configuration.spaces {
                if await space.provider.isPrepared { await space.provider.release() }
            }
        }
        if !(await captioner.isPrepared) {
            try await captioner.prepare()
            loadedQueryCaptioner = true
        }
    }

    private func releaseQueryCaptionerIfLoaded() async {
        guard loadedQueryCaptioner, let captioner = configuration.queryCaptioner else { return }
        loadedQueryCaptioner = false
        await captioner.release()
    }

    // MARK: - Residency

    /// Make `provider` ready to embed. Call only while holding the inference
    /// gate.
    ///
    /// Under sequential residency every other model is released *first*, so the
    /// peak is one model's weights rather than two. Release happens inside the
    /// gate for the same reason load does: releasing a model another job is
    /// using would fail that job.
    func makeResident(embedder provider: any EmbeddingProvider) async throws {
        if configuration.runtime.sequentialModelResidency {
            await releaseQueryCaptionerIfLoaded()
            if let reranker = configuration.reranker, await reranker.isPrepared {
                await reranker.release()
            }
            for space in configuration.spaces
            where ObjectIdentifier(space.provider) != ObjectIdentifier(provider) {
                if await space.provider.isPrepared { await space.provider.release() }
            }
        }
        if !(await provider.isPrepared) { try await provider.prepare() }
    }

    /// Make `reranker` ready to score. Call only while holding the inference
    /// gate.
    func makeResident(reranker: any RerankingProvider) async throws {
        if configuration.runtime.sequentialModelResidency {
            await releaseQueryCaptionerIfLoaded()
            for space in configuration.spaces {
                if await space.provider.isPrepared { await space.provider.release() }
            }
        }
        if !(await reranker.isPrepared) { try await reranker.prepare() }
    }

    /// Flush WAL and publish ANN snapshots. Call before terminating if the next
    /// open should be fast.
    public func flush() async throws {
        for id in spaces.keys { try publishSnapshotIfNeeded(space: id) }
        try store.checkpoint()
    }

    public func close() async throws {
        try? await flush()
        store.close()
    }

    // MARK: - Indexing

    /// Insert or replace items.
    ///
    /// Unchanged items are skipped by fingerprint, so re-running an ingest over
    /// a corpus that did not change costs one hash and one row read per item
    /// and no inference at all.
    @discardableResult
    public func upsert(
        _ items: [SearchItem],
        force: Bool = false,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> IndexingReport {
        try await upsert(items, force: force, owner: nil, onProgress: onProgress)
    }

    /// `owner`: the processor whose ingest wrote `items`; see
    /// ``StoredItem/processorFingerprint``.
    func upsert(
        _ items: [SearchItem],
        force: Bool,
        owner: String?,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> IndexingReport {
        var report = IndexingReport()
        var pendingBySpace: [EmbeddingSpaceID: [Int]] = [:]

        for (offset, item) in items.enumerated() {
            try Task.checkCancellation()

            for representation in item.representations {
                if let language = representation.language, !language.isEmpty {
                    indexedLanguages.insert(TextNormalization.baseLanguageCode(language))
                }
            }
            let outcome = try store.upsert(
                item,
                owner: owner,
                analyzer: configuration.analyzer,
                force: force)

            if outcome.didChange {
                report.itemsWritten += 1
                report.representationsRemoved += outcome.removedSlots.count
                report.representationsWritten += outcome.insertedSlots.count

                for slot in outcome.removedSlots { slotKinds.removeValue(forKey: slot) }
                for inserted in outcome.insertedSlots {
                    slotKinds[inserted.slot] = inserted.kind
                }

                try removeFromVectorIndexes(outcome.removedSlots)
                for id in spaces.keys {
                    // Every new slot is offered to every space. Which of them a
                    // space actually embeds is decided in `embedPending`, where
                    // the provider's capabilities are known; deciding here would
                    // duplicate that rule in two places that could drift apart.
                    pendingBySpace[id, default: []].append(
                        contentsOf: outcome.insertedSlots.map(\.slot))
                }
            } else {
                report.itemsUnchanged += 1
            }

            onProgress?(offset + 1, items.count)
        }

        for (id, slots) in pendingBySpace.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            try await embedPending(space: id, slots: slots, report: &report)
        }

        let allInserted = Set(pendingBySpace.values.flatMap { $0 })
        try reportUnreachable(slots: allInserted, report: &report)

        return report
    }

    /// Report representations that ended up retrievable through no channel at
    /// all.
    ///
    /// A visual representation in a text-only index is the case this exists
    /// for: nothing rejected it, nothing embedded it, and without this it would
    /// sit in the store invisible to search with no signal that anything went
    /// wrong. Text is excluded — it is still reachable lexically, so its
    /// absence from a dense space is a routing decision rather than a loss. So
    /// is a picture with text derived from it — a captioned frame in the text
    /// profile is found through its caption, and the pixels stay to be shown.
    private func reportUnreachable(slots: Set<Int>, report: inout IndexingReport) throws {
        guard !slots.isEmpty else { return }
        var alreadyReported = Set(report.skipped.map(\.key))

        let rows = try store.representations(slots: slots)
        for slot in slots.sorted() {
            guard let row = rows[slot], row.weight > 0 else { continue }
            if row.kind.isLexicallySearchable, !(row.text ?? "").isEmpty { continue }
            if spaces.values.contains(where: { $0.index.contains(slot: slot) }) { continue }
            if try store.representations(forItem: row.itemID).contains(where: {
                $0.derivedFrom == row.representationID && $0.weight > 0
                    && $0.kind.isLexicallySearchable && !($0.text ?? "").isEmpty
            }) {
                continue
            }
            guard alreadyReported.insert(row.key).inserted else { continue }
            report.skipped.append(
                IndexingReport.Skip(
                    key: row.key,
                    reason:
                        "no configured embedding space accepts \(row.kind.rawValue) content, "
                        + "and it is not lexically searchable"))
        }
    }

    @discardableResult
    public func upsert(_ item: SearchItem, force: Bool = false) async throws -> IndexingReport {
        try await upsert([item], force: force)
    }

    /// Run raw content through the configured processor and index what it
    /// produces.
    ///
    /// Skipped entirely when the processor has already produced representations
    /// for this content under this configuration — the fingerprint check that
    /// keeps OCR and transcription from being repeated.
    @discardableResult
    public func ingest(
        _ contents: [RawContent],
        onProgress: (@Sendable (ProcessingProgress) -> Void)? = nil
    ) async throws -> IndexingReport {
        guard let processor = configuration.processor else {
            throw SemanticSearchError.configuration(
                "ingest requires a ContentProcessor; configure one or call upsert with "
                    + "prepared items")
        }

        var report = IndexingReport()
        for content in contents {
            try Task.checkCancellation()

            let contentFingerprint = Self.fingerprint(of: content)
            if let state = try store.processingState(
                itemID: content.id, processor: processor.identifier),
                state.content == contentFingerprint,
                state.processor == processor.configurationFingerprint
            {
                report.itemsUnchanged += 1
                continue
            }

            var produced: [SearchItem]
            do {
                activeProcessingJobs += 1
                inferenceEpoch &+= 1
                defer {
                    activeProcessingJobs -= 1
                    armIdleRelease()
                }
                produced = try await processor.process(content, onProgress: onProgress)
            } catch let error as SemanticSearchError {
                throw error
            } catch {
                throw SemanticSearchError.processorFailed(
                    itemID: content.id, processor: processor.identifier, underlying: error)
            }
            // The content's own item is always written, empty if the processor
            // had nothing to say — an emptied file. Without it the old item
            // kept its text searchable, and a first empty file failed to
            // record its state against an item that was never written.
            if !produced.contains(where: { $0.id == content.id }) {
                produced.insert(
                    SearchItem(
                        id: content.id, parentID: content.parentID, metadata: content.metadata,
                        locator: content.locator),
                    at: 0)
            }

            let sub = try await upsert(produced, force: true, owner: processor.identifier)
            report.merge(sub)

            // Children from the previous run that this run did not produce —
            // the tail chunks of a document that got shorter, the utterances
            // of a recording that was re-cut. Left in place they stay
            // searchable with content the source no longer has. Removed after
            // the new items are written, so a processor failure above leaves
            // the old version intact rather than half of each.
            //
            // Only what ingest made is ingest's to take away. A child the
            // caller attached — frames under a video, notes under a document —
            // was upserted by the caller and survives re-ingestion; removing
            // every descendant the new run did not produce would delete it.
            // Ownership is recorded on the item, so what an earlier run made
            // goes even when nothing in this run names its producer: the
            // output shrank to nothing, or a processor was replaced or taken
            // out of a combination.
            let producedIDs = Set(produced.map(\.id))
            // Items written before ownership was recorded: judged, as then, by
            // whether a producer of this run made every representation.
            var producers: Set<String> = [processor.identifier]
            for item in produced {
                for representation in item.representations {
                    producers.insert(representation.provenance.producer)
                }
            }
            producers.remove(Provenance.caller.producer)
            for stale in try store.descendants(of: content.id)
            where !producedIDs.contains(stale) {
                let owned: Bool
                if try store.item(stale)?.processorFingerprint != nil {
                    owned = true
                } else {
                    let representations = try store.representations(forItem: stale)
                    owned =
                        !representations.isEmpty
                        && representations.allSatisfy { producers.contains($0.provenance.producer) }
                }
                guard owned else { continue }
                try await remove(stale, includingDescendants: false)
                report.itemsRemoved += 1
            }

            try store.recordProcessingState(
                itemID: content.id,
                processor: processor.identifier,
                contentFingerprint: contentFingerprint,
                processorFingerprint: processor.configurationFingerprint)
        }
        return report
    }

    /// Remove an item, and optionally everything beneath it.
    public func remove(
        _ itemID: ItemID, includingDescendants: Bool = true
    ) async throws {
        let removed = try store.remove(itemID: itemID, includingDescendants: includingDescendants)
        for slot in removed { slotKinds.removeValue(forKey: slot) }
        try removeFromVectorIndexes(removed)
    }

    private func removeFromVectorIndexes(_ slots: [Int]) throws {
        guard !slots.isEmpty else { return }
        for id in spaces.keys {
            guard var runtime = spaces[id] else { continue }
            for slot in slots where runtime.index.contains(slot: slot) {
                try runtime.index.remove(slot: slot)
                runtime.unsnapshottedChanges += 1
            }
            spaces[id] = runtime
        }
    }

    public func removeAll() async throws {
        for id in try store.allItemIDs() {
            _ = try store.remove(itemID: id, includingDescendants: false)
        }
        slotKinds.removeAll()
        for runtime in spaces.values { try runtime.index.removeAll() }
    }

    /// Re-embed everything a space is missing.
    ///
    /// Also the recovery path: a run interrupted halfway leaves the slots it
    /// did not reach without vectors, and `reindex` finds exactly those.
    @discardableResult
    public func reindex(
        space id: EmbeddingSpaceID? = nil,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> IndexingReport {
        var report = IndexingReport()
        let targets = id.map { [$0] } ?? Array(spaces.keys).sorted { $0.rawValue < $1.rawValue }

        for target in targets {
            guard let runtime = spaces[target] else {
                throw SemanticSearchError.unknownEmbeddingSpace(target)
            }
            let kinds = await eligibleKinds(for: runtime)
            // Empty means "no filter" to the store, not "nothing".
            guard !kinds.isEmpty else { continue }
            let missing = try store.slotsMissingVectors(space: target, kinds: kinds)
            guard !missing.isEmpty else { continue }
            try await embedPending(
                space: target, slots: missing, report: &report, onProgress: onProgress)
        }
        return report
    }

    /// Rebuild a space's graph from scratch and publish it.
    ///
    /// Used after a bulk load, and after enough deletions that the graph holds
    /// more tombstones than live vectors.
    public func rebuildVectorIndex(space id: EmbeddingSpaceID) async throws {
        guard spaces[id] != nil else {
            throw SemanticSearchError.unknownEmbeddingSpace(id)
        }
        let vectorCount = try store.vectorCount(space: id)
        try replaceIndex(space: id, vectorCount: vectorCount)
        try publishSnapshotIfNeeded(space: id, force: true)
    }

    /// Build a fresh index for a space from the store and swap it in.
    ///
    /// Graph or exact by the same rule the open path uses. The swap happens
    /// only once the new index is complete, so a failure part-way leaves the
    /// previous one in place.
    private func replaceIndex(space id: EmbeddingSpaceID, vectorCount: Int) throws {
        guard var runtime = spaces[id] else { return }
        let dimensions = runtime.stored.effectiveDimensions
        let useGraph =
            runtime.stored.indexKind == .hnsw
            && vectorCount >= runtime.configuration.hnsw.minimumVectorsForGraph

        let rebuilt: any VectorIndex
        if useGraph {
            let store = self.store
            rebuilt = try USearchVectorIndex(
                dimensions: dimensions,
                configuration: runtime.configuration.hnsw,
                fullPrecisionVector: { slot in try? store.vector(space: id, slot: slot) })
        } else {
            rebuilt = ExactVectorIndex(dimensions: dimensions)
        }

        try store.forEachVector(space: id, newerThan: 0) { slot, vector in
            try Task.checkCancellation()
            try rebuilt.insert(
                slot: slot, vector: VectorMath.normalizedPrefix(vector, dimensions: dimensions))
        }

        runtime.index = rebuilt
        runtime.unsnapshottedChanges = rebuilt.count
        if let stored = try store.loadSpaces().first(where: { $0.id == id }) {
            runtime.stored = stored
            runtime.appliedGeneration = stored.generation
        }
        spaces[id] = runtime
    }

    /// Switch a space that opened below the graph threshold to its graph once
    /// it has grown past it.
    ///
    /// Without this, a space that starts empty — every new index — would stay
    /// on brute-force search for the whole session however large it grew,
    /// holding every vector in full precision in memory, and would publish no
    /// snapshot, so the next open would build the graph from scratch.
    private func promoteToGraphIfNeeded(space id: EmbeddingSpaceID) throws {
        guard let runtime = spaces[id],
            runtime.stored.indexKind == .hnsw,
            runtime.index is ExactVectorIndex,
            runtime.index.count >= runtime.configuration.hnsw.minimumVectorsForGraph
        else { return }
        try replaceIndex(space: id, vectorCount: runtime.index.count)
    }

    // MARK: - Embedding

    /// Representation kinds a space should hold.
    ///
    /// The explicit configuration when there is one, and everything the
    /// provider accepts otherwise — which is what makes a single-space setup
    /// work without the caller having to enumerate kinds.
    ///
    /// Always limited to what the provider accepts. A configured kind the
    /// provider cannot take — `.audio` in an "every kind" list — would otherwise
    /// cost a query embedding per search for a channel that can never hold a
    /// vector, and would be reported as missing vectors by every `reindex`.
    func eligibleKinds(for runtime: SpaceRuntime) async -> Set<RepresentationKind> {
        let capabilities = await runtime.configuration.provider.capabilities
        let accepted = Set(RepresentationKind.allCases.filter(capabilities.accepts))
        if runtime.configuration.kinds.isEmpty { return accepted }
        return runtime.configuration.kinds.intersection(accepted)
    }

    /// Embed the given slots into a space, in bounded batches.
    private func embedPending(
        space id: EmbeddingSpaceID,
        slots: [Int],
        report: inout IndexingReport,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws {
        guard let runtime = spaces[id], !slots.isEmpty else { return }

        let provider = runtime.configuration.provider
        let capabilities = await provider.capabilities
        let descriptor = await provider.descriptor
        let dimensions = runtime.stored.effectiveDimensions
        let allowedKinds = await eligibleKinds(for: runtime)

        let batchSize = min(
            configuration.runtime.embeddingBatchSize, max(1, capabilities.maximumBatchSize))

        var processed = 0
        for chunk in slots.chunked(into: batchSize) {
            try Task.checkCancellation()

            let rows = try store.representations(slots: chunk)
            var inputs: [EmbeddingInput] = []
            var inputSlots: [Int] = []

            for slot in chunk {
                guard let row = rows[slot] else { continue }
                // Context-only representations are stored but never embedded;
                // see ``ContentRepresentation/weight``.
                if row.weight <= 0 { continue }
                // A kind this space does not hold is not an error on its own —
                // another space may hold it. Whether anything ended up holding
                // it is answered once, globally, by `reportUnreachable`.
                guard allowedKinds.contains(row.kind) else { continue }

                guard
                    let input = try makeEmbeddingInput(
                        row, dimensions: runtime.stored.reducedDimensions, report: &report)
                else { continue }
                inputs.append(input)
                inputSlots.append(slot)
            }

            guard !inputs.isEmpty else {
                processed += chunk.count
                onProgress?(processed, slots.count)
                continue
            }

            let batch = inputs
            let outputs = try await withInference(priority: .background) {
                try await self.makeResident(embedder: provider)
                return try await provider.embed(batch)
            }
            guard outputs.count == inputs.count else {
                throw SemanticSearchError.providerReturnedNoOutput(descriptor.modelIdentity)
            }

            var writes: [(slot: Int, vector: [Float])] = []
            writes.reserveCapacity(outputs.count)
            for (offset, output) in outputs.enumerated() {
                guard !output.vector.isEmpty else {
                    // Reported, not dropped: the representation is stored and
                    // this space can never retrieve it.
                    if let key = rows[inputSlots[offset]]?.key {
                        report.skipped.append(
                            IndexingReport.Skip(
                                key: key,
                                reason: "'\(descriptor.modelIdentity)' returned an empty vector"))
                    }
                    continue
                }
                if output.wasTruncated { report.truncatedRepresentations += 1 }
                let vector = VectorMath.normalizedPrefix(
                    output.vector, dimensions: runtime.stored.reducedDimensions)
                writes.append((slot: inputSlots[offset], vector: vector))
            }

            guard !writes.isEmpty else {
                processed += chunk.count
                onProgress?(processed, slots.count)
                continue
            }

            // The embed above suspended this actor, so an item in the batch can
            // have been removed meanwhile. The store skips those, and only what
            // it wrote goes into the graph.
            let written = Set(try store.writeVectors(space: id, vectors: writes))
            writes.removeAll { !written.contains($0.slot) }
            if var current = spaces[id] {
                for write in writes {
                    try current.index.insert(
                        slot: write.slot,
                        vector: VectorMath.normalizedPrefix(
                            write.vector, dimensions: dimensions))
                }
                current.unsnapshottedChanges += writes.count
                if let stored = try store.loadSpaces().first(where: { $0.id == id }) {
                    current.stored = stored
                    current.appliedGeneration = stored.generation
                }
                spaces[id] = current
                try promoteToGraphIfNeeded(space: id)
            }
            report.vectorsWritten += writes.count

            processed += chunk.count
            onProgress?(processed, slots.count)
        }
    }

    private func makeEmbeddingInput(
        _ row: StoredRepresentation,
        dimensions: Int?,
        report: inout IndexingReport
    ) throws -> EmbeddingInput? {
        switch row.content {
        case .text(let value):
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            return EmbeddingInput(
                text: trimmed, role: .document, requestedDimensions: dimensions)

        case .imageData(let data, let contentType):
            guard !data.isEmpty else { return nil }
            return EmbeddingInput(
                images: [.encoded(data, contentType: contentType)],
                role: .document, requestedDimensions: dimensions)

        case .mediaReference(let url):
            // `EmbeddingInput` has no audio channel; a provider that accepted
            // audio would need one. Until then this is unreachable in practice
            // — `accepts(.audio)` is false for every bundled provider — and
            // reported rather than sent somewhere it does not belong.
            guard row.kind != .audio else {
                report.skipped.append(
                    IndexingReport.Skip(
                        key: row.key, reason: "audio cannot be passed to an embedding provider"))
                return nil
            }
            guard url.isFileURL else {
                // Remote video is streamed by the provider. A remote still has
                // no payload form — `ImagePayload` is a file or bytes — and
                // sending it as a video would embed the wrong thing.
                guard row.kind == .video else {
                    report.skipped.append(
                        IndexingReport.Skip(
                            key: row.key,
                            reason: "remote \(row.kind.rawValue) references are not supported; "
                                + "pass the bytes or a file URL"))
                    return nil
                }
                return EmbeddingInput(
                    videos: [url], role: .document, requestedDimensions: dimensions)
            }
            if let allowed = configuration.mediaAccess, !allowed(url) {
                report.skipped.append(
                    IndexingReport.Skip(
                        key: row.key, reason: "media may not be read at \(url.path)"))
                return nil
            }
            guard FileManager.default.fileExists(atPath: url.path) else {
                // Stale media is reported explicitly, never dropped. The row
                // stays in the index so the caller can re-ingest or remove it
                // deliberately.
                report.skipped.append(
                    IndexingReport.Skip(
                        key: row.key, reason: "media is missing at \(url.path)"))
                return nil
            }
            if row.kind == .video {
                return EmbeddingInput(
                    videos: [url], role: .document, requestedDimensions: dimensions)
            }
            return EmbeddingInput(
                images: [.fileURL(url)], role: .document, requestedDimensions: dimensions)
        }
    }

    // MARK: - Snapshots

    /// Write the current graph to disk and publish it, if it is worth doing.
    ///
    /// Atomic: the file is written beside the destination and renamed, and the
    /// generation pointer is only updated once the bytes are durably in place.
    /// A crash before the rename leaves the previous snapshot. A crash after it
    /// and before the pointer update leaves the *new* graph under the *old*
    /// generation; the next open replays deltas newer than that generation onto
    /// a graph that already has them, which is harmless because replay is
    /// idempotent — inserts replace, and removes of absent keys are no-ops.
    func publishSnapshotIfNeeded(space id: EmbeddingSpaceID, force: Bool = false) throws {
        guard let runtime = spaces[id],
            let graph = runtime.index as? USearchVectorIndex,
            let destination = store.snapshotURL(for: id)
        else { return }

        // A snapshot costs a full serialisation of the graph. Below a few
        // thousand changed vectors, replaying them on the next open is cheaper
        // than writing one now.
        guard force || runtime.unsnapshottedChanges >= 2_000 else { return }
        guard graph.count > 0 else { return }

        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent("." + destination.lastPathComponent + ".tmp")
        try? FileManager.default.removeItem(at: temporary)
        try graph.save(to: temporary)
        _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)

        try store.publishSnapshot(
            space: id,
            generation: runtime.appliedGeneration,
            file: destination.lastPathComponent)

        if var current = spaces[id],
            let stored = try store.loadSpaces().first(where: { $0.id == id })
        {
            current.stored = stored
            current.unsnapshottedChanges = 0
            spaces[id] = current
        }
    }

    // MARK: - Inspection

    public func itemCount() throws -> Int { try store.itemCount() }
    public func representationCount() throws -> Int { try store.representationCount() }
    public func indexSizeBytes() -> Int64 { store.indexSizeBytes() }

    public func vectorCount(space id: EmbeddingSpaceID) throws -> Int {
        try store.vectorCount(space: id)
    }

    /// True when a space is currently searched by brute force rather than by
    /// its graph; `nil` for an unknown space.
    public func usesExactSearch(space id: EmbeddingSpaceID) -> Bool? {
        spaces[id]?.index.isExact
    }

    public func item(_ id: ItemID) throws -> SearchItem? {
        guard let stored = try store.item(id) else { return nil }
        let representations = try store.representations(forItem: id)
        return SearchItem(
            id: stored.id,
            parentID: stored.parentID,
            metadata: stored.metadata,
            locator: stored.locator,
            representations: representations.map {
                ContentRepresentation(
                    id: $0.representationID, kind: $0.kind, content: $0.content,
                    language: $0.language, provenance: $0.provenance, locator: $0.locator,
                    derivedFrom: $0.derivedFrom, weight: $0.weight)
            },
            ordinal: stored.ordinal)
    }

    /// Nearest neighbours as the live index answers them — approximate when the
    /// space uses a graph.
    ///
    /// The other half of the ANN recall measurement; compare with
    /// ``exactNeighbours(space:query:limit:allowedSlots:)``.
    public func approximateNeighbours(
        space id: EmbeddingSpaceID, query: [Float], limit: Int,
        allowedSlots: Set<Int>? = nil
    ) throws -> [VectorMatch] {
        guard let runtime = spaces[id] else {
            throw SemanticSearchError.unknownEmbeddingSpace(id)
        }
        let normalized = VectorMath.normalizedPrefix(
            query, dimensions: runtime.stored.effectiveDimensions)
        var predicate: (@Sendable (Int) -> Bool)?
        if let allowedSlots {
            predicate = { allowedSlots.contains($0) }
        }
        return try runtime.index.search(query: normalized, limit: limit, isAllowed: predicate)
    }

    /// Exact nearest neighbours for a stored vector, bypassing the ANN graph.
    ///
    /// The measurement the ANN recall gate is computed against.
    public func exactNeighbours(
        space id: EmbeddingSpaceID, query: [Float], limit: Int,
        allowedSlots: Set<Int>? = nil
    ) throws -> [VectorMatch] {
        guard let runtime = spaces[id] else {
            throw SemanticSearchError.unknownEmbeddingSpace(id)
        }
        let dimensions = runtime.stored.effectiveDimensions
        let exact = ExactVectorIndex(dimensions: dimensions)
        try store.forEachVector(space: id, newerThan: 0) { slot, vector in
            try exact.insert(
                slot: slot, vector: VectorMath.normalizedPrefix(vector, dimensions: dimensions))
        }
        let normalized = VectorMath.normalizedPrefix(query, dimensions: dimensions)
        var predicate: (@Sendable (Int) -> Bool)?
        if let allowedSlots {
            predicate = { allowedSlots.contains($0) }
        }
        return try exact.search(query: normalized, limit: limit, isAllowed: predicate)
    }

    // MARK: - Helpers

    /// Run one inference job under the concurrency bound.
    ///
    /// `priority` decides the queue, not the job: a search waits in the
    /// interactive lane and is served ahead of queued indexing batches, but a
    /// batch already running finishes first.
    func withInference<T: Sendable>(
        priority: AsyncSemaphore.Priority = .interactive,
        _ body: () async throws -> T
    ) async throws -> T {
        try await inferenceGate.wait(priority: priority)
        activeInferenceJobs += 1
        inferenceEpoch &+= 1
        defer {
            activeInferenceJobs -= 1
            inferenceGate.signal()
            armIdleRelease()
        }
        return try await body()
    }

    // MARK: - Idle release

    /// Release every model once nothing has run for
    /// ``RuntimePolicy/modelIdleTimeout``. Re-armed by every job, so it fires
    /// only after a real idle period.
    private func armIdleRelease() {
        let timeout = configuration.runtime.modelIdleTimeout
        guard timeout > .zero else { return }
        idleReleaseTask?.cancel()
        let epoch = inferenceEpoch
        idleReleaseTask = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled else { return }
            await self?.releaseIfIdle(since: epoch)
        }
    }

    private func releaseIfIdle(since epoch: Int) async {
        guard epoch == inferenceEpoch, activeInferenceJobs == 0, activeProcessingJobs == 0
        else { return }
        // Through the gate, so a job that started between the check above and
        // here is not left holding a released model. Not through
        // `withInference`, which would re-arm the timer it is answering.
        await withEveryPermit {
            guard epoch == self.inferenceEpoch else { return }
            await self.releaseAllModels()
        }
    }

    static func fingerprint(of content: RawContent) -> String {
        StableHash.of { hasher in
            hasher.append("raw.v1")
            hasher.append(content.id.rawValue)
            hasher.append(content.languageHint ?? "")
            switch content.payload {
            case .text(let value):
                hasher.append("t")
                hasher.append(value)
            case .imageData(let data, let contentType):
                hasher.append("i")
                hasher.append(contentType)
                hasher.append(data)
            case .fileURL(let url):
                hasher.append("f")
                hasher.append(url.absoluteString)
                // Size and modification date rather than the bytes: hashing a
                // 4 GB video to decide whether to transcribe it costs more than
                // transcribing it.
                let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
                hasher.append((attributes?[.size] as? Int) ?? -1)
                hasher.append(
                    (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? -1)
            }
        }
    }
}

/// What an indexing call did.
public struct IndexingReport: Sendable {
    public struct Skip: Sendable {
        public var key: RepresentationKey
        public var reason: String

        public init(key: RepresentationKey, reason: String) {
            self.key = key
            self.reason = reason
        }
    }

    public var itemsWritten = 0
    public var itemsUnchanged = 0
    /// Items deleted because re-processing their source no longer produced
    /// them.
    public var itemsRemoved = 0
    public var representationsWritten = 0
    public var representationsRemoved = 0
    public var vectorsWritten = 0
    /// Representations whose text the provider had to truncate.
    public var truncatedRepresentations = 0
    /// Representations that were not embedded, and why.
    public var skipped: [Skip] = []

    public init() {}

    /// Fold another report into this one. Public so a caller that ingests in
    /// several passes can report one total.
    public mutating func merge(_ other: IndexingReport) {
        itemsWritten += other.itemsWritten
        itemsUnchanged += other.itemsUnchanged
        itemsRemoved += other.itemsRemoved
        representationsWritten += other.representationsWritten
        representationsRemoved += other.representationsRemoved
        vectorsWritten += other.vectorsWritten
        truncatedRepresentations += other.truncatedRepresentations
        skipped.append(contentsOf: other.skipped)
    }
}

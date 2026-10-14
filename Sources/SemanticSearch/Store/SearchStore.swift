import Foundation

/// One representation row as the engine sees it.
struct StoredRepresentation: Sendable {
    var slot: Int
    var itemID: ItemID
    var representationID: RepresentationID
    var kind: RepresentationKind
    var language: String?
    var text: String?
    var content: RepresentationContent
    var locator: SourceLocator?
    var provenance: Provenance
    var derivedFrom: RepresentationID?
    var weight: Float

    var key: RepresentationKey {
        RepresentationKey(item: itemID, representation: representationID)
    }
}

struct StoredItem: Sendable {
    var id: ItemID
    var parentID: ItemID?
    var ordinal: Int?
    var locator: SourceLocator
    var metadata: Metadata
    var contentFingerprint: String
    /// The processor whose ingest wrote the item, `nil` for a caller's
    /// upsert — the `processor_fingerprint` column. Ownership, not a
    /// fingerprint: it is what lets re-ingestion remove what an earlier run
    /// made and keep what the caller attached.
    var processorFingerprint: String?
}

/// Persisted description of one embedding space.
struct StoredSpace: Sendable {
    var id: EmbeddingSpaceID
    var descriptor: EmbeddingSpaceDescriptor
    var indexKind: VectorIndexKind
    var dimensions: Int
    /// Matryoshka width the space is stored at, when it is narrower than the
    /// provider's native output.
    var reducedDimensions: Int?
    var generation: Int
    var snapshotGeneration: Int
    var snapshotFile: String?
    /// Representation kinds routed into this space. Empty means all kinds the
    /// provider accepts.
    var kinds: Set<RepresentationKind>

    var effectiveDimensions: Int { reducedDimensions ?? dimensions }
}

/// What changed during an upsert, so the caller can do exactly the follow-up
/// work the change implies.
struct UpsertOutcome: Sendable {
    var itemID: ItemID
    /// False when the content fingerprint matched what was already stored and
    /// nothing was rewritten.
    var didChange: Bool
    /// Slots that now need vectors, with the kind each one holds. The kind
    /// travels with the slot so the engine can keep its slot→kind map current
    /// without a second read.
    var insertedSlots: [(slot: Int, kind: RepresentationKind)]
    /// Slots whose rows are gone. Their vectors and postings went with them.
    var removedSlots: [Int]
}

/// Writes one representation's lexical postings.
///
/// Shared by upsert and by the full rebuild that runs when the analyzer
/// changes, so there is exactly one definition of what gets indexed.
struct LexicalWriter {
    private let terms: SQLiteStatement
    private let documents: SQLiteStatement

    init(database: SQLiteDatabase) throws {
        terms = try database.prepare(
            "INSERT INTO lexical_terms(slot, term, term_frequency) VALUES(?, ?, ?)")
        documents = try database.prepare(
            "INSERT INTO lexical_documents(slot, token_count) VALUES(?, ?)")
    }

    func index(
        slot: Int, kind: RepresentationKind, weight: Float, text: String?, language: String?,
        analyzer: any LexicalAnalyzer
    ) throws {
        // Weight <= 0 means context-only; see ``ContentRepresentation/weight``.
        guard weight > 0, kind.isLexicallySearchable, let text, !text.isEmpty else { return }

        let analyzed = analyzer.analyzeDocument(text, language: language)
        guard !analyzed.terms.isEmpty else { return }

        var frequencies: [String: Int] = [:]
        for term in analyzed.terms { frequencies[term, default: 0] += 1 }
        for (term, frequency) in frequencies {
            terms.bind(1, slot)
            terms.bind(2, term)
            terms.bind(3, frequency)
            try terms.run()
        }
        documents.bind(1, slot)
        documents.bind(2, max(1, analyzed.tokenCount))
        try documents.run()
    }
}

/// SQLite-backed storage.
///
/// Owned exclusively by ``SemanticSearchEngine``, which provides the
/// serialisation this class does not. Kept out of the actor so the engine can
/// hand it to a detached rebuild task without crossing isolation on every row.
final class SearchStore: @unchecked Sendable {

    let databaseURL: URL?
    /// Directory holding published ANN snapshots. `nil` for in-memory stores,
    /// which rebuild their graphs on every open by definition.
    let snapshotDirectory: URL?

    private let database: SQLiteDatabase
    private let lock = NSLock()
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    // MARK: - Lifecycle

    /// Open or create a store at `url`. The parent directory is created if
    /// needed; snapshots live next to the database in `<name>.snapshots`.
    init(url: URL) throws {
        self.databaseURL = url
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.snapshotDirectory = directory.appendingPathComponent(
            url.deletingPathExtension().lastPathComponent + ".snapshots", isDirectory: true)
        self.database = try SQLiteDatabase(path: url.path)
        try FileManager.default.createDirectory(
            at: snapshotDirectory!, withIntermediateDirectories: true)
        try migrate()
    }

    /// A private, temporary store. Used by tests and by evaluation runs that
    /// must not touch a real index.
    static func inMemory() throws -> SearchStore {
        try SearchStore(inMemoryMarker: ())
    }

    private init(inMemoryMarker: Void) throws {
        self.databaseURL = nil
        self.snapshotDirectory = nil
        self.database = try SQLiteDatabase(path: ":memory:")
        try migrate()
    }

    private func migrate() throws {
        try database.transaction {
            for statement in Schema.statements { try database.execute(statement) }

            let existing = try readMetaLocked("schema_version").flatMap(Int.init)
            if let existing {
                guard existing <= Schema.version else {
                    throw SemanticSearchError.schemaVersionUnsupported(
                        found: existing, supported: Schema.version)
                }
            } else {
                try writeMetaLocked("schema_version", String(Schema.version))
            }

            // Tombstones with no published snapshot to subtract from, left by
            // builds that recorded them unconditionally.
            try database.execute(
                """
                DELETE FROM vector_tombstones WHERE space NOT IN
                    (SELECT id FROM embedding_spaces WHERE snapshot_file IS NOT NULL)
                """)
        }
    }

    func close() {
        database.close()
    }

    func checkpoint() throws {
        lock.lock()
        defer { lock.unlock() }
        try database.checkpoint()
    }

    /// Prepare, use and reset one statement under the store's lock.
    func withStatement<T>(_ sql: String, _ body: (SQLiteStatement) throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(sql)
        defer { statement.reset() }
        return try body(statement)
    }

    // MARK: - Meta

    func readMeta(_ key: String) throws -> String? {
        lock.lock()
        defer { lock.unlock() }
        return try readMetaLocked(key)
    }

    func writeMeta(_ key: String, _ value: String) throws {
        lock.lock()
        defer { lock.unlock() }
        try writeMetaLocked(key, value)
    }

    private func readMetaLocked(_ key: String) throws -> String? {
        let statement = try database.prepare("SELECT value FROM meta WHERE key = ?")
        statement.bind(1, key)
        defer { statement.reset() }
        guard try statement.step() else { return nil }
        return statement.string(0)
    }

    private func writeMetaLocked(_ key: String, _ value: String) throws {
        let statement = try database.prepare(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
                + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        statement.bind(1, key)
        statement.bind(2, value)
        try statement.run()
    }

    // MARK: - Spaces

    func registerSpace(_ space: StoredSpace) throws {
        // A zero-width space would accept vectors of any width, which is the
        // compatibility check failing open.
        guard space.dimensions > 0 else {
            throw SemanticSearchError.configuration(
                "embedding space '\(space.id.rawValue)' has no dimensions; the provider must "
                    + "report them before the space is registered")
        }
        lock.lock()
        defer { lock.unlock() }
        try database.transaction {
            if let existing = try loadSpaceLocked(space.id) {
                // A descriptor change means every vector in the space was
                // produced by different rules. Silently keeping them would make
                // the space internally inconsistent, which no amount of
                // rechecking at query time can detect.
                guard existing.descriptor.fingerprint == space.descriptor.fingerprint else {
                    throw SemanticSearchError.incompatibleEmbeddingSpace(
                        space: space.id,
                        expected: existing.descriptor.fingerprint,
                        actual: space.descriptor.fingerprint,
                        detail:
                            "a space with this id already exists with different model identity; "
                            + "build a replacement space under a new id and switch after it "
                            + "completes")
                }
                return
            }

            let statement = try database.prepare(
                """
                INSERT INTO embedding_spaces
                    (id, descriptor, index_kind, dimensions, reduced_dimensions,
                     generation, snapshot_generation, snapshot_file, kinds)
                VALUES (?, ?, ?, ?, ?, 0, 0, NULL, ?)
                """)
            statement.bind(1, space.id.rawValue)
            statement.bind(2, try encoder.encode(space.descriptor))
            statement.bind(3, space.indexKind.rawValue)
            statement.bind(4, space.dimensions)
            statement.bind(5, space.reducedDimensions)
            statement.bind(6, try encoder.encode(space.kinds.map(\.rawValue).sorted()))
            try statement.run()
        }
    }

    func loadSpaces() throws -> [StoredSpace] {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            """
            SELECT id, descriptor, index_kind, dimensions, reduced_dimensions,
                   generation, snapshot_generation, snapshot_file, kinds
            FROM embedding_spaces ORDER BY id
            """)
        var spaces: [StoredSpace] = []
        try statement.forEachRow { row in
            if let space = try decodeSpace(row) { spaces.append(space) }
        }
        return spaces
    }

    private func loadSpaceLocked(_ id: EmbeddingSpaceID) throws -> StoredSpace? {
        let statement = try database.prepare(
            """
            SELECT id, descriptor, index_kind, dimensions, reduced_dimensions,
                   generation, snapshot_generation, snapshot_file, kinds
            FROM embedding_spaces WHERE id = ?
            """)
        statement.bind(1, id.rawValue)
        defer { statement.reset() }
        guard try statement.step() else { return nil }
        return try decodeSpace(statement)
    }

    private func decodeSpace(_ row: SQLiteStatement) throws -> StoredSpace? {
        guard let id = row.string(0), let descriptorData = row.data(1),
            let indexKindRaw = row.string(2),
            let indexKind = VectorIndexKind(rawValue: indexKindRaw)
        else { return nil }

        let kinds: Set<RepresentationKind>
        if let kindsData = row.data(8),
            let raw = try? decoder.decode([String].self, from: kindsData)
        {
            kinds = Set(raw.compactMap(RepresentationKind.init(rawValue:)))
        } else {
            kinds = []
        }

        return StoredSpace(
            id: EmbeddingSpaceID(id),
            descriptor: try decoder.decode(EmbeddingSpaceDescriptor.self, from: descriptorData),
            indexKind: indexKind,
            dimensions: row.int(3),
            reducedDimensions: row.optionalInt(4),
            generation: row.int(5),
            snapshotGeneration: row.int(6),
            snapshotFile: row.string(7),
            kinds: kinds)
    }

    func removeSpace(_ id: EmbeddingSpaceID) throws {
        lock.lock()
        defer { lock.unlock() }
        try database.transaction {
            for sql in [
                "DELETE FROM vectors WHERE space = ?",
                "DELETE FROM vector_tombstones WHERE space = ?",
                "DELETE FROM embedding_spaces WHERE id = ?",
            ] {
                let statement = try database.prepare(sql)
                statement.bind(1, id.rawValue)
                try statement.run()
            }
        }
        if let snapshotDirectory {
            try? FileManager.default.removeItem(
                at: snapshotDirectory.appendingPathComponent(Self.snapshotName(for: id)))
        }
    }

    // MARK: - Items

    /// Insert or replace an item.
    ///
    /// The fingerprint check is what makes re-ingesting an unchanged corpus
    /// nearly free: an item whose content and processor both match what is
    /// stored is left exactly as it is, vectors included.
    /// `owner` is the processor whose ingest writes the item; `nil` — a
    /// caller's upsert — keeps whatever owner the item had.
    func upsert(
        _ item: SearchItem,
        owner: String?,
        analyzer: any LexicalAnalyzer,
        force: Bool = false
    ) throws -> UpsertOutcome {
        lock.lock()
        defer { lock.unlock() }

        return try database.transaction {
            let fingerprint = item.contentFingerprint
            if !force, let existing = try loadItemRowLocked(item.id),
                existing.contentFingerprint == fingerprint
            {
                // Metadata is not part of the content fingerprint — it changes
                // without invalidating a single vector — so it is refreshed
                // even on the unchanged path. The same goes for each
                // representation's locator, provenance and lineage: a
                // corrected bounding box or time range must land, and it
                // changes nothing an embedding or a posting depends on.
                try updateItemMetadataLocked(item)
                try updateRepresentationDescriptionsLocked(item)
                return UpsertOutcome(
                    itemID: item.id, didChange: false, insertedSlots: [], removedSlots: [])
            }

            let removedSlots = try slotsForItemLocked(item.id)
            try deleteRepresentationsLocked(itemID: item.id)

            let itemStatement = try database.prepare(
                """
                INSERT INTO items
                    (id, parent_id, ordinal, locator, metadata,
                     content_fingerprint, processor_fingerprint, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    parent_id = excluded.parent_id,
                    ordinal = excluded.ordinal,
                    locator = excluded.locator,
                    metadata = excluded.metadata,
                    content_fingerprint = excluded.content_fingerprint,
                    processor_fingerprint =
                        COALESCE(excluded.processor_fingerprint, items.processor_fingerprint),
                    updated_at = excluded.updated_at
                """)
            itemStatement.bind(1, item.id.rawValue)
            itemStatement.bind(2, item.parentID?.rawValue)
            itemStatement.bind(3, item.ordinal)
            itemStatement.bind(4, try encoder.encode(item.locator))
            itemStatement.bind(5, try encoder.encode(item.metadata))
            itemStatement.bind(6, fingerprint)
            itemStatement.bind(7, owner)
            itemStatement.bind(8, Date().timeIntervalSince1970)
            try itemStatement.run()

            var insertedSlots: [(slot: Int, kind: RepresentationKind)] = []
            let representationStatement = try database.prepare(
                """
                INSERT INTO representations
                    (item_id, rep_id, kind, language, text, content_kind, content_type,
                     blob, media_url, locator, provenance, derived_from, weight, slot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """)
            let lexical = try LexicalWriter(database: database)
            var nextSlot = try nextSlotLocked()

            for representation in item.representations {
                representationStatement.bind(1, item.id.rawValue)
                representationStatement.bind(2, representation.id.rawValue)
                representationStatement.bind(3, representation.kind.rawValue)
                representationStatement.bind(4, representation.language)

                switch representation.content {
                case .text(let value):
                    representationStatement.bind(5, value)
                    representationStatement.bind(6, 0)
                    representationStatement.bind(7, nil as String?)
                    representationStatement.bind(8, nil as Data?)
                    representationStatement.bind(9, nil as String?)
                case .imageData(let data, let contentType):
                    representationStatement.bind(5, nil as String?)
                    representationStatement.bind(6, 1)
                    representationStatement.bind(7, contentType)
                    representationStatement.bind(8, data)
                    representationStatement.bind(9, nil as String?)
                case .mediaReference(let url):
                    representationStatement.bind(5, nil as String?)
                    representationStatement.bind(6, 2)
                    representationStatement.bind(7, nil as String?)
                    representationStatement.bind(8, nil as Data?)
                    representationStatement.bind(9, url.absoluteString)
                }

                representationStatement.bind(
                    10, try representation.locator.map { try encoder.encode($0) })
                representationStatement.bind(11, try encoder.encode(representation.provenance))
                representationStatement.bind(12, representation.derivedFrom?.rawValue)
                representationStatement.bind(13, Double(representation.weight))
                let slot = nextSlot
                nextSlot += 1
                representationStatement.bind(14, slot)
                try representationStatement.run()

                insertedSlots.append((slot: slot, kind: representation.kind))

                try lexical.index(
                    slot: slot, kind: representation.kind, weight: representation.weight,
                    text: representation.content.text, language: representation.language,
                    analyzer: analyzer)
            }
            try writeMetaLocked(Self.nextSlotKey, String(nextSlot))

            return UpsertOutcome(
                itemID: item.id, didChange: true, insertedSlots: insertedSlots,
                removedSlots: removedSlots)
        }
    }

    /// Delete an item and, optionally, everything beneath it.
    @discardableResult
    func remove(itemID: ItemID, includingDescendants: Bool) throws -> [Int] {
        lock.lock()
        defer { lock.unlock() }

        return try database.transaction {
            var targets: [ItemID] = [itemID]
            if includingDescendants {
                targets.append(contentsOf: try descendantsLocked(of: itemID))
            }

            var removed: [Int] = []
            for target in targets {
                removed.append(contentsOf: try slotsForItemLocked(target))
                try deleteRepresentationsLocked(itemID: target)
                let statement = try database.prepare("DELETE FROM items WHERE id = ?")
                statement.bind(1, target.rawValue)
                try statement.run()
            }
            return removed
        }
    }

    /// Give an item, and everything beneath it, another id — a file that
    /// moved. Children named after it (`<id>#…`) are renamed with it.
    ///
    /// Slots stay, so nothing is embedded or indexed again: the vectors, the
    /// postings and the graphs are keyed by slot. The content fingerprint
    /// covers the id and the parent's, so it is computed again from the
    /// stored representations; left alone, the next upsert of the moved
    /// items would find every one changed and embed it anew.
    func renameTree(from old: ItemID, to new: ItemID) throws {
        lock.lock()
        defer { lock.unlock() }

        try database.transaction {
            guard try loadItemRowLocked(new) == nil else {
                throw SemanticSearchError.configuration(
                    "cannot rename \(old) to \(new): an item already has that id")
            }
            let ids = [old] + (try descendantsLocked(of: old))
            let renamed = Set(ids)
            func mapped(_ id: ItemID) -> ItemID {
                if id == old { return new }
                guard id.rawValue.hasPrefix(old.rawValue + "#") else { return id }
                return ItemID(new.rawValue + id.rawValue.dropFirst(old.rawValue.count))
            }

            for id in ids {
                guard let item = try loadItemRowLocked(id) else { continue }
                let newID = mapped(id)
                let parent = item.parentID.map { renamed.contains($0) ? mapped($0) : $0 }
                let fingerprint = SearchItem(
                    id: newID, parentID: parent, metadata: [:], locator: .unspecified,
                    representations: try representationsLocked(forItem: id).map {
                        ContentRepresentation(
                            id: $0.representationID, kind: $0.kind, content: $0.content,
                            language: $0.language, provenance: $0.provenance,
                            locator: $0.locator, derivedFrom: $0.derivedFrom, weight: $0.weight)
                    }
                ).contentFingerprint

                if newID == id {
                    let statement = try database.prepare(
                        "UPDATE items SET parent_id = ?, content_fingerprint = ? WHERE id = ?")
                    statement.bind(1, parent?.rawValue)
                    statement.bind(2, fingerprint)
                    statement.bind(3, id.rawValue)
                    try statement.run()
                    continue
                }
                // A new row, then everything that points at the old one, then
                // the old row: representations and processing state cascade
                // on delete, so they have to have moved first.
                let insert = try database.prepare(
                    """
                    INSERT INTO items
                        (id, parent_id, ordinal, locator, metadata,
                         content_fingerprint, processor_fingerprint, updated_at)
                    SELECT ?, ?, ordinal, locator, metadata, ?, processor_fingerprint, ?
                    FROM items WHERE id = ?
                    """)
                insert.bind(1, newID.rawValue)
                insert.bind(2, parent?.rawValue)
                insert.bind(3, fingerprint)
                insert.bind(4, Date().timeIntervalSince1970)
                insert.bind(5, id.rawValue)
                try insert.run()
                for sql in [
                    "UPDATE representations SET item_id = ? WHERE item_id = ?",
                    "UPDATE processing_state SET item_id = ? WHERE item_id = ?",
                ] {
                    let statement = try database.prepare(sql)
                    statement.bind(1, newID.rawValue)
                    statement.bind(2, id.rawValue)
                    try statement.run()
                }
                let delete = try database.prepare("DELETE FROM items WHERE id = ?")
                delete.bind(1, id.rawValue)
                try delete.run()
            }
        }
    }

    /// Every item beneath `itemID`, at any depth.
    func descendants(of itemID: ItemID) throws -> [ItemID] {
        lock.lock()
        defer { lock.unlock() }
        return try descendantsLocked(of: itemID)
    }

    // MARK: - Slot allocation

    static let nextSlotKey = "next_slot"

    /// The next slot number, which is never one that has been used before.
    ///
    /// Slots are the key into the vector graphs, their snapshots and the
    /// tombstone table, so a reused slot lets a stale graph entry for deleted
    /// content resolve to whatever representation took the number next. SQLite
    /// hands out `MAX(rowid) + 1`, which reuses the number of the most recently
    /// deleted row; the high-water mark in `meta` is what prevents that. The
    /// maxima of the live and tombstoned slots cover a file written before the
    /// mark existed.
    private func nextSlotLocked() throws -> Int {
        let recorded = try readMetaLocked(Self.nextSlotKey).flatMap(Int.init) ?? 1
        let statement = try database.prepare(
            """
            SELECT MAX(
                COALESCE((SELECT MAX(slot) FROM representations), 0),
                COALESCE((SELECT MAX(slot) FROM vector_tombstones), 0)) + 1
            """)
        defer { statement.reset() }
        guard try statement.step() else { return recorded }
        return max(recorded, statement.int(0))
    }

    // MARK: - Lexical analyzer identity

    static let analyzerVersionKey = "analyzer_version"

    /// Make the stored postings agree with `analyzer`.
    ///
    /// Postings are the analyzer's output baked into the file. Opening an index
    /// with a different analyzer and querying it would tokenise queries one way
    /// and documents another, and nothing would fail — results would just get
    /// quietly worse. So the version is recorded, and on a mismatch every
    /// posting is rebuilt from the representation text SQLite already holds.
    /// No model is involved, so this is always possible.
    ///
    /// - Returns: `true` if the postings were rebuilt.
    @discardableResult
    func ensureLexicalIndex(matches analyzer: any LexicalAnalyzer) throws -> Bool {
        lock.lock()
        defer { lock.unlock() }

        return try database.transaction {
            let recorded = try readMetaLocked(Self.analyzerVersionKey)
            if recorded == analyzer.analyzerVersion { return false }

            let hasPostings = try database.prepare("SELECT 1 FROM lexical_documents LIMIT 1")
            defer { hasPostings.reset() }
            // A file with postings and no recorded version predates the check;
            // which analyzer wrote it is unknowable, so it is rebuilt too.
            let rebuild = try hasPostings.step()
            if rebuild { try rebuildLexicalIndexLocked(analyzer: analyzer) }

            try writeMetaLocked(Self.analyzerVersionKey, analyzer.analyzerVersion)
            return rebuild
        }
    }

    private func rebuildLexicalIndexLocked(analyzer: any LexicalAnalyzer) throws {
        try database.execute("DELETE FROM lexical_terms")
        try database.execute("DELETE FROM lexical_documents")

        let lexical = try LexicalWriter(database: database)
        let rows = try database.prepare(
            """
            SELECT slot, kind, language, text, weight FROM representations
            WHERE text IS NOT NULL AND weight > 0 ORDER BY slot
            """)
        try rows.forEachRow { row in
            guard let kind = row.string(1).flatMap(RepresentationKind.init(rawValue:)) else {
                return
            }
            try lexical.index(
                slot: row.int(0), kind: kind, weight: Float(row.double(4)),
                text: row.string(3), language: row.string(2), analyzer: analyzer)
        }
    }

    private func descendantsLocked(of itemID: ItemID) throws -> [ItemID] {
        let statement = try database.prepare(
            """
            WITH RECURSIVE tree(id) AS (
                SELECT id FROM items WHERE parent_id = ?
                UNION ALL
                SELECT items.id FROM items JOIN tree ON items.parent_id = tree.id
            )
            SELECT id FROM tree
            """)
        statement.bind(1, itemID.rawValue)
        var result: [ItemID] = []
        try statement.forEachRow { row in
            if let id = row.string(0) { result.append(ItemID(id)) }
        }
        return result
    }

    private func slotsForItemLocked(_ itemID: ItemID) throws -> [Int] {
        let statement = try database.prepare("SELECT slot FROM representations WHERE item_id = ?")
        statement.bind(1, itemID.rawValue)
        var slots: [Int] = []
        try statement.forEachRow { slots.append($0.int(0)) }
        return slots
    }

    /// Remove an item's representations and everything derived from them.
    ///
    /// The tombstone rows are written before the vectors are deleted: a
    /// snapshot taken earlier still holds those keys, and the tombstone is the
    /// only record that they must be subtracted on load.
    private func deleteRepresentationsLocked(itemID: ItemID) throws {
        let slots = try slotsForItemLocked(itemID)
        guard !slots.isEmpty else { return }

        for space in try loadSpacesLocked() {
            // A tombstone only matters to a published snapshot that still holds
            // the key. With none, the next open builds the index from the
            // vectors table, which no longer lists the slot — and an exact
            // space never publishes one, so its tombstones were never pruned.
            guard space.snapshotFile != nil else { continue }
            let generation = space.generation + 1
            let tombstone = try database.prepare(
                "INSERT INTO vector_tombstones(space, slot, generation) VALUES(?, ?, ?) "
                    + "ON CONFLICT(space, slot) DO UPDATE SET generation = excluded.generation")
            var wrote = false
            for slot in slots {
                guard try hasVectorLocked(space: space.id, slot: slot) else { continue }
                tombstone.bind(1, space.id.rawValue)
                tombstone.bind(2, slot)
                tombstone.bind(3, generation)
                try tombstone.run()
                wrote = true
            }
            if wrote { try bumpGenerationLocked(space: space.id, to: generation) }
        }

        let statement = try database.prepare("DELETE FROM representations WHERE item_id = ?")
        statement.bind(1, itemID.rawValue)
        try statement.run()
    }

    private func updateItemMetadataLocked(_ item: SearchItem) throws {
        let statement = try database.prepare(
            "UPDATE items SET metadata = ?, ordinal = ?, locator = ?, parent_id = ? WHERE id = ?")
        statement.bind(1, try encoder.encode(item.metadata))
        statement.bind(2, item.ordinal)
        statement.bind(3, try encoder.encode(item.locator))
        statement.bind(4, item.parentID?.rawValue)
        statement.bind(5, item.id.rawValue)
        try statement.run()
    }

    /// Rewrite the fields of each representation that describe it without
    /// being embedded or indexed. Only valid when the content fingerprint
    /// matched, which guarantees the same representation ids.
    private func updateRepresentationDescriptionsLocked(_ item: SearchItem) throws {
        let statement = try database.prepare(
            """
            UPDATE representations SET locator = ?, provenance = ?, derived_from = ?
            WHERE item_id = ? AND rep_id = ?
            """)
        for representation in item.representations {
            statement.bind(1, try representation.locator.map { try encoder.encode($0) })
            statement.bind(2, try encoder.encode(representation.provenance))
            statement.bind(3, representation.derivedFrom?.rawValue)
            statement.bind(4, item.id.rawValue)
            statement.bind(5, representation.id.rawValue)
            try statement.run()
        }
    }

    private func loadItemRowLocked(_ id: ItemID) throws -> StoredItem? {
        let statement = try database.prepare(
            """
            SELECT id, parent_id, ordinal, locator, metadata,
                   content_fingerprint, processor_fingerprint
            FROM items WHERE id = ?
            """)
        statement.bind(1, id.rawValue)
        defer { statement.reset() }
        guard try statement.step() else { return nil }
        return try decodeItem(statement)
    }

    func decodeItem(_ row: SQLiteStatement) throws -> StoredItem? {
        guard let id = row.string(0), let fingerprint = row.string(5) else { return nil }
        let locator =
            row.data(3).flatMap { try? decoder.decode(SourceLocator.self, from: $0) }
            ?? .unspecified
        let metadata =
            row.data(4).flatMap { try? decoder.decode(Metadata.self, from: $0) } ?? [:]
        return StoredItem(
            id: ItemID(id),
            parentID: row.string(1).map { ItemID($0) },
            ordinal: row.optionalInt(2),
            locator: locator,
            metadata: metadata,
            contentFingerprint: fingerprint,
            processorFingerprint: row.string(6))
    }

    func item(_ id: ItemID) throws -> StoredItem? {
        lock.lock()
        defer { lock.unlock() }
        return try loadItemRowLocked(id)
    }

    func items(_ ids: some Collection<ItemID>) throws -> [ItemID: StoredItem] {
        guard !ids.isEmpty else { return [:] }
        lock.lock()
        defer { lock.unlock() }

        var result: [ItemID: StoredItem] = [:]
        result.reserveCapacity(ids.count)
        // Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER, which is 999 on the
        // system SQLite that ships in older OS versions.
        for chunk in Array(ids).chunked(into: 500) {
            let placeholders = Array(repeating: "?", count: chunk.count).joined(separator: ",")
            let statement = try database.prepare(
                """
                SELECT id, parent_id, ordinal, locator, metadata,
                       content_fingerprint, processor_fingerprint
                FROM items WHERE id IN (\(placeholders))
                """)
            for (offset, id) in chunk.enumerated() {
                statement.bind(Int32(offset + 1), id.rawValue)
            }
            try statement.forEachRow { row in
                if let item = try decodeItem(row) { result[item.id] = item }
            }
        }
        return result
    }

    func childItems(of parent: ItemID, limit: Int) throws -> [StoredItem] {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            """
            SELECT id, parent_id, ordinal, locator, metadata,
                   content_fingerprint, processor_fingerprint
            FROM items WHERE parent_id = ?
            ORDER BY COALESCE(ordinal, 0), id
            LIMIT ?
            """)
        statement.bind(1, parent.rawValue)
        statement.bind(2, limit)
        var result: [StoredItem] = []
        try statement.forEachRow { row in
            if let item = try decodeItem(row) { result.append(item) }
        }
        return result
    }

    func itemCount() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare("SELECT COUNT(*) FROM items")
        defer { statement.reset() }
        guard try statement.step() else { return 0 }
        return statement.int(0)
    }

    func representationCount() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare("SELECT COUNT(*) FROM representations")
        defer { statement.reset() }
        guard try statement.step() else { return 0 }
        return statement.int(0)
    }

    func allItemIDs() throws -> [ItemID] {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare("SELECT id FROM items ORDER BY id")
        var result: [ItemID] = []
        try statement.forEachRow { row in
            if let id = row.string(0) { result.append(ItemID(id)) }
        }
        return result
    }

    // MARK: - Representations

    func representations(slots: some Collection<Int>) throws -> [Int: StoredRepresentation] {
        guard !slots.isEmpty else { return [:] }
        lock.lock()
        defer { lock.unlock() }
        return try representationsLocked(slots: slots)
    }

    private func representationsLocked(slots: some Collection<Int>) throws
        -> [Int: StoredRepresentation]
    {
        var result: [Int: StoredRepresentation] = [:]
        result.reserveCapacity(slots.count)
        for chunk in Array(slots).chunked(into: 500) {
            let placeholders = Array(repeating: "?", count: chunk.count).joined(separator: ",")
            let statement = try database.prepare(
                """
                SELECT slot, item_id, rep_id, kind, language, text, content_kind,
                       content_type, blob, media_url, locator, provenance,
                       derived_from, weight
                FROM representations WHERE slot IN (\(placeholders))
                """)
            for (offset, slot) in chunk.enumerated() {
                statement.bind(Int32(offset + 1), slot)
            }
            try statement.forEachRow { row in
                if let representation = try decodeRepresentation(row) {
                    result[representation.slot] = representation
                }
            }
        }
        return result
    }

    /// Slot → kind for every representation in the store.
    ///
    /// Held in memory by the engine so a per-family candidate restriction is an
    /// O(1) predicate rather than a query. Two columns and no blobs: at 300k
    /// representations the map is single-digit megabytes, which is the cheapest
    /// way to keep two families sharing one embedding space from returning each
    /// other's representations.
    func slotKinds() throws -> [Int: RepresentationKind] {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare("SELECT slot, kind FROM representations")
        var result: [Int: RepresentationKind] = [:]
        try statement.forEachRow { row in
            guard let raw = row.string(1), let kind = RepresentationKind(rawValue: raw) else {
                return
            }
            result[row.int(0)] = kind
        }
        return result
    }

    /// Base language codes of every lexically indexed representation that
    /// declares one. Read once on open; see ``SemanticSearchEngine`` for why
    /// query language detection needs it.
    func indexedLanguages() throws -> Set<String> {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            "SELECT DISTINCT language FROM representations WHERE language IS NOT NULL")
        var result: Set<String> = []
        try statement.forEachRow { row in
            if let language = row.string(0), !language.isEmpty {
                result.insert(TextNormalization.baseLanguageCode(language))
            }
        }
        return result
    }

    func representations(forItem itemID: ItemID) throws -> [StoredRepresentation] {
        lock.lock()
        defer { lock.unlock() }
        return try representationsLocked(forItem: itemID)
    }

    private func representationsLocked(forItem itemID: ItemID) throws -> [StoredRepresentation] {
        let statement = try database.prepare(
            """
            SELECT slot, item_id, rep_id, kind, language, text, content_kind,
                   content_type, blob, media_url, locator, provenance,
                   derived_from, weight
            FROM representations WHERE item_id = ? ORDER BY slot
            """)
        statement.bind(1, itemID.rawValue)
        var result: [StoredRepresentation] = []
        try statement.forEachRow { row in
            if let representation = try decodeRepresentation(row) { result.append(representation) }
        }
        return result
    }

    /// Stream every representation, oldest slot first.
    ///
    /// Used by reindex and by BM25 rebuild. A callback rather than an array:
    /// materialising 100k representations with their blobs is hundreds of
    /// megabytes, and both callers consume one row at a time.
    func forEachRepresentation(
        kinds: Set<RepresentationKind>? = nil,
        _ body: (StoredRepresentation) throws -> Void
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let filter: String
        if let kinds, !kinds.isEmpty {
            let list = kinds.map { "'\($0.rawValue)'" }.sorted().joined(separator: ",")
            filter = "WHERE kind IN (\(list))"
        } else {
            filter = ""
        }
        let statement = try database.prepare(
            """
            SELECT slot, item_id, rep_id, kind, language, text, content_kind,
                   content_type, blob, media_url, locator, provenance,
                   derived_from, weight
            FROM representations \(filter) ORDER BY slot
            """)
        try statement.forEachRow { row in
            if let representation = try decodeRepresentation(row) { try body(representation) }
        }
    }

    private func decodeRepresentation(_ row: SQLiteStatement) throws -> StoredRepresentation? {
        guard let itemID = row.string(1), let representationID = row.string(2),
            let kindRaw = row.string(3), let kind = RepresentationKind(rawValue: kindRaw)
        else { return nil }

        let text = row.string(5)
        let content: RepresentationContent
        switch row.int(6) {
        case 1:
            content = .imageData(row.data(8) ?? Data(), contentType: row.string(7) ?? "")
        case 2:
            guard let urlString = row.string(9), let url = URL(string: urlString) else {
                return nil
            }
            content = .mediaReference(url)
        default:
            content = .text(text ?? "")
        }

        let provenance =
            row.data(11).flatMap { try? decoder.decode(Provenance.self, from: $0) }
            ?? .caller

        return StoredRepresentation(
            slot: row.int(0),
            itemID: ItemID(itemID),
            representationID: RepresentationID(representationID),
            kind: kind,
            language: row.string(4),
            text: text,
            content: content,
            locator: row.data(10).flatMap { try? decoder.decode(SourceLocator.self, from: $0) },
            provenance: provenance,
            derivedFrom: row.string(12).map { RepresentationID($0) },
            weight: Float(row.double(13)))
    }

    // MARK: - Lexical

    /// Every posting in the store, for rebuilding the BM25 index on open.
    func forEachLexicalDocument(
        _ body: (_ slot: Int, _ tokenCount: Int, _ terms: [(String, Int)]) throws -> Void
    ) throws {
        lock.lock()
        defer { lock.unlock() }

        // One ordered scan of each table, merged by slot. Two sorted streams
        // beat a correlated subquery per document by a wide margin at 100k rows.
        let documents = try database.prepare(
            "SELECT slot, token_count FROM lexical_documents ORDER BY slot")
        let terms = try database.prepare(
            "SELECT slot, term, term_frequency FROM lexical_terms ORDER BY slot")

        var pendingSlot: Int?
        var pendingTerms: [(String, Int)] = []
        var hasTermRow = try terms.step()

        defer {
            documents.reset()
            terms.reset()
        }

        while try documents.step() {
            let slot = documents.int(0)
            let tokenCount = documents.int(1)

            while hasTermRow, terms.int(0) < slot { hasTermRow = try terms.step() }
            pendingTerms.removeAll(keepingCapacity: true)
            while hasTermRow, terms.int(0) == slot {
                if let term = terms.string(1) { pendingTerms.append((term, terms.int(2))) }
                hasTermRow = try terms.step()
            }
            pendingSlot = slot
            try body(slot, tokenCount, pendingTerms)
        }
        _ = pendingSlot
    }

    // MARK: - Vectors

    private func loadSpacesLocked() throws -> [StoredSpace] {
        let statement = try database.prepare(
            """
            SELECT id, descriptor, index_kind, dimensions, reduced_dimensions,
                   generation, snapshot_generation, snapshot_file, kinds
            FROM embedding_spaces ORDER BY id
            """)
        var spaces: [StoredSpace] = []
        try statement.forEachRow { row in
            if let space = try decodeSpace(row) { spaces.append(space) }
        }
        return spaces
    }

    private func hasVectorLocked(space: EmbeddingSpaceID, slot: Int) throws -> Bool {
        let statement = try database.prepare(
            "SELECT 1 FROM vectors WHERE space = ? AND slot = ? LIMIT 1")
        statement.bind(1, space.rawValue)
        statement.bind(2, slot)
        defer { statement.reset() }
        return try statement.step()
    }

    private func bumpGenerationLocked(space: EmbeddingSpaceID, to generation: Int) throws {
        let statement = try database.prepare(
            "UPDATE embedding_spaces SET generation = MAX(generation, ?) WHERE id = ?")
        statement.bind(1, generation)
        statement.bind(2, space.rawValue)
        try statement.run()
    }

    /// Write a batch of vectors under one generation, in one transaction.
    ///
    /// Batched rather than per-vector because the generation has to be uniform:
    /// a snapshot published at generation *g* must contain either all of a
    /// batch or none of it, and per-vector generations make "all of it" a range
    /// the reader has to guess at.
    ///
    /// A slot whose representation no longer exists is skipped. The engine
    /// embeds outside any lock, so an item can be removed while its vectors are
    /// being computed; writing them anyway would either violate the foreign key
    /// or resurrect a vector nothing refers to.
    ///
    /// - Returns: The slots actually written, in input order.
    @discardableResult
    func writeVectors(
        space: EmbeddingSpaceID, vectors: [(slot: Int, vector: [Float])]
    ) throws -> [Int] {
        guard !vectors.isEmpty else { return [] }
        lock.lock()
        defer { lock.unlock() }

        return try database.transaction {
            guard let stored = try loadSpaceLocked(space) else {
                throw SemanticSearchError.unknownEmbeddingSpace(space)
            }
            let generation = stored.generation + 1

            let exists = try database.prepare(
                "SELECT 1 FROM representations WHERE slot = ? LIMIT 1")
            let insert = try database.prepare(
                """
                INSERT INTO vectors(space, slot, dimensions, data, generation)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(space, slot) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    data = excluded.data,
                    generation = excluded.generation
                """)
            let clearTombstone = try database.prepare(
                "DELETE FROM vector_tombstones WHERE space = ? AND slot = ?")

            var written: [Int] = []
            written.reserveCapacity(vectors.count)
            for (slot, vector) in vectors {
                exists.bind(1, slot)
                let present = try exists.step()
                exists.reset()
                guard present else { continue }

                insert.bind(1, space.rawValue)
                insert.bind(2, slot)
                insert.bind(3, vector.count)
                insert.bind(4, vector.vectorBlob)
                insert.bind(5, generation)
                try insert.run()

                // A slot that was deleted and re-added must not keep a
                // tombstone: replaying it would subtract the vector that was
                // just written.
                clearTombstone.bind(1, space.rawValue)
                clearTombstone.bind(2, slot)
                try clearTombstone.run()
                written.append(slot)
            }

            if !written.isEmpty { try bumpGenerationLocked(space: space, to: generation) }
            return written
        }
    }

    func vector(space: EmbeddingSpaceID, slot: Int) throws -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            "SELECT data FROM vectors WHERE space = ? AND slot = ?")
        statement.bind(1, space.rawValue)
        statement.bind(2, slot)
        defer { statement.reset() }
        guard try statement.step() else { return nil }
        return statement.floatVector(0)
    }

    /// Vectors written strictly after `generation`, oldest first.
    func forEachVector(
        space: EmbeddingSpaceID,
        newerThan generation: Int,
        _ body: (_ slot: Int, _ vector: [Float]) throws -> Void
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            "SELECT slot, data FROM vectors WHERE space = ? AND generation > ? ORDER BY generation, slot")
        statement.bind(1, space.rawValue)
        statement.bind(2, generation)
        try statement.forEachRow { row in
            guard let vector = row.floatVector(1) else { return }
            try body(row.int(0), vector)
        }
    }

    func tombstones(space: EmbeddingSpaceID, newerThan generation: Int) throws -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            "SELECT slot FROM vector_tombstones WHERE space = ? AND generation > ? ORDER BY slot")
        statement.bind(1, space.rawValue)
        statement.bind(2, generation)
        var slots: [Int] = []
        try statement.forEachRow { slots.append($0.int(0)) }
        return slots
    }

    func vectorSlots(space: EmbeddingSpaceID) throws -> Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare("SELECT slot FROM vectors WHERE space = ?")
        statement.bind(1, space.rawValue)
        var slots: Set<Int> = []
        try statement.forEachRow { slots.insert($0.int(0)) }
        return slots
    }

    func vectorCount(space: EmbeddingSpaceID) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare("SELECT COUNT(*) FROM vectors WHERE space = ?")
        statement.bind(1, space.rawValue)
        defer { statement.reset() }
        guard try statement.step() else { return 0 }
        return statement.int(0)
    }

    /// Slots that have a representation this space should cover but no vector
    /// yet. Drives incremental embedding and recovery after an interrupted run.
    func slotsMissingVectors(space: EmbeddingSpaceID, kinds: Set<RepresentationKind>) throws
        -> [Int]
    {
        lock.lock()
        defer { lock.unlock() }
        let kindFilter: String
        if kinds.isEmpty {
            kindFilter = ""
        } else {
            let list = kinds.map { "'\($0.rawValue)'" }.sorted().joined(separator: ",")
            kindFilter = "AND r.kind IN (\(list))"
        }
        let statement = try database.prepare(
            """
            SELECT r.slot FROM representations r
            LEFT JOIN vectors v ON v.slot = r.slot AND v.space = ?
            WHERE v.slot IS NULL AND r.weight > 0 \(kindFilter)
            ORDER BY r.slot
            """)
        statement.bind(1, space.rawValue)
        var slots: [Int] = []
        try statement.forEachRow { slots.append($0.int(0)) }
        return slots
    }

    // MARK: - Snapshots

    static func snapshotName(for space: EmbeddingSpaceID) -> String {
        // Space ids are caller-supplied and can contain anything; the file name
        // is derived from a hash so it is always a valid single path component.
        let digest = StableHash.of { $0.append(space.rawValue) }
        return "space-\(digest).usearch"
    }

    func snapshotURL(for space: EmbeddingSpaceID) -> URL? {
        snapshotDirectory?.appendingPathComponent(Self.snapshotName(for: space))
    }

    /// Record that a snapshot covering `generation` is now the published one.
    ///
    /// Called only after the file is durably in place. Publishing the pointer
    /// before the bytes are there is the one ordering that can lose data, since
    /// a crash in between leaves a generation marker for a file that does not
    /// exist.
    func publishSnapshot(space: EmbeddingSpaceID, generation: Int, file: String) throws {
        lock.lock()
        defer { lock.unlock() }
        try database.transaction {
            let statement = try database.prepare(
                "UPDATE embedding_spaces SET snapshot_generation = ?, snapshot_file = ? WHERE id = ?"
            )
            statement.bind(1, generation)
            statement.bind(2, file)
            statement.bind(3, space.rawValue)
            try statement.run()

            // Tombstones the snapshot already accounts for are no longer needed.
            let prune = try database.prepare(
                "DELETE FROM vector_tombstones WHERE space = ? AND generation <= ?")
            prune.bind(1, space.rawValue)
            prune.bind(2, generation)
            try prune.run()
        }
    }

    // MARK: - Processing state

    func processingState(itemID: ItemID, processor: String) throws -> (content: String, processor: String)? {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            """
            SELECT content_fingerprint, processor_fingerprint
            FROM processing_state WHERE item_id = ? AND processor = ?
            """)
        statement.bind(1, itemID.rawValue)
        statement.bind(2, processor)
        defer { statement.reset() }
        guard try statement.step(),
            let content = statement.string(0), let processorFingerprint = statement.string(1)
        else { return nil }
        return (content, processorFingerprint)
    }

    func recordProcessingState(
        itemID: ItemID, processor: String, contentFingerprint: String,
        processorFingerprint: String
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let statement = try database.prepare(
            """
            INSERT INTO processing_state
                (item_id, processor, content_fingerprint, processor_fingerprint, completed_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(item_id, processor) DO UPDATE SET
                content_fingerprint = excluded.content_fingerprint,
                processor_fingerprint = excluded.processor_fingerprint,
                completed_at = excluded.completed_at
            """)
        statement.bind(1, itemID.rawValue)
        statement.bind(2, processor)
        statement.bind(3, contentFingerprint)
        statement.bind(4, processorFingerprint)
        statement.bind(5, Date().timeIntervalSince1970)
        try statement.run()
    }

    // MARK: - Diagnostics

    /// Bytes on disk: the database, its WAL, and every published snapshot.
    func indexSizeBytes() -> Int64 {
        var total: Int64 = 0
        let manager = FileManager.default
        if let databaseURL {
            for suffix in ["", "-wal", "-shm"] {
                let url = URL(fileURLWithPath: databaseURL.path + suffix)
                if let size = try? manager.attributesOfItem(atPath: url.path)[.size] as? Int64 {
                    total += size
                }
            }
        }
        if let snapshotDirectory,
            let contents = try? manager.contentsOfDirectory(
                at: snapshotDirectory, includingPropertiesForKeys: [.fileSizeKey])
        {
            for url in contents {
                if let size = try? manager.attributesOfItem(atPath: url.path)[.size] as? Int64 {
                    total += size
                }
            }
        }
        return total
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        guard size > 0, count > size else { return isEmpty ? [] : [self] }
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

import Foundation

/// The on-disk schema.
///
/// SQLite is authoritative for everything: items, representations, lexical
/// postings, full-precision vectors and indexing state. The HNSW graphs on disk
/// are derived artefacts — deleting them costs a rebuild and never costs data.
///
/// ### Generations
///
/// Every vector write stamps a monotonically increasing generation on its row.
/// A published ANN snapshot records the generation it covers. On open the engine
/// loads the snapshot and replays every vector and tombstone newer than that
/// generation, so an interrupted rebuild can only ever leave the index *behind*,
/// never inconsistent: never a half-written graph that exposes deleted or
/// mismatched content.
enum Schema {

    /// Bumped for any change that makes an older build misread the file.
    static let version = 1

    static let statements: [String] = [
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS items (
            id                     TEXT PRIMARY KEY,
            parent_id              TEXT,
            ordinal                INTEGER,
            locator                BLOB,
            metadata               BLOB,
            content_fingerprint    TEXT NOT NULL,
            processor_fingerprint  TEXT,
            updated_at             REAL NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS items_parent ON items(parent_id)",

        // `slot` is the join key for everything downstream: posting lists,
        // vectors, ANN keys. Assigned explicitly from the persisted `next_slot`
        // high-water mark, never by SQLite's `MAX(rowid) + 1`, so a slot is
        // never reused — a reused slot would let a stale snapshot key resolve
        // to unrelated content.
        """
        CREATE TABLE IF NOT EXISTS representations (
            slot          INTEGER PRIMARY KEY,
            item_id       TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            rep_id        TEXT NOT NULL,
            kind          TEXT NOT NULL,
            language      TEXT,
            text          TEXT,
            content_kind  INTEGER NOT NULL,
            content_type  TEXT,
            blob          BLOB,
            media_url     TEXT,
            locator       BLOB,
            provenance    BLOB,
            derived_from  TEXT,
            weight        REAL NOT NULL DEFAULT 1,
            UNIQUE(item_id, rep_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS representations_item ON representations(item_id)",
        "CREATE INDEX IF NOT EXISTS representations_kind ON representations(kind)",

        // Postings are stored per slot rather than as inverted lists, so
        // deleting an item is one `DELETE ... WHERE slot = ?` instead of a
        // read-modify-write of every list the item appears in. Queries read
        // them through the term index below; nothing is held in memory.
        """
        CREATE TABLE IF NOT EXISTS lexical_terms (
            slot            INTEGER NOT NULL REFERENCES representations(slot) ON DELETE CASCADE,
            term            TEXT NOT NULL,
            term_frequency  INTEGER NOT NULL,
            PRIMARY KEY(slot, term)
        ) WITHOUT ROWID
        """,
        // The inverted direction. `lexical_terms` is keyed by slot so a delete
        // is one range scan; scoring needs term → slots, which is what this
        // index provides.
        "CREATE INDEX IF NOT EXISTS lexical_terms_term ON lexical_terms(term)",

        """
        CREATE TABLE IF NOT EXISTS lexical_documents (
            slot         INTEGER PRIMARY KEY REFERENCES representations(slot) ON DELETE CASCADE,
            token_count  INTEGER NOT NULL
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS embedding_spaces (
            id                   TEXT PRIMARY KEY,
            descriptor           BLOB NOT NULL,
            index_kind           TEXT NOT NULL,
            dimensions           INTEGER NOT NULL,
            reduced_dimensions   INTEGER,
            generation           INTEGER NOT NULL DEFAULT 0,
            snapshot_generation  INTEGER NOT NULL DEFAULT 0,
            snapshot_file        TEXT,
            kinds                BLOB
        )
        """,

        // Full precision, always. The ANN graph holds a quantised copy; this is
        // what candidates are rescored against and what a rebuild reads.
        """
        CREATE TABLE IF NOT EXISTS vectors (
            space       TEXT NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
            slot        INTEGER NOT NULL REFERENCES representations(slot) ON DELETE CASCADE,
            dimensions  INTEGER NOT NULL,
            data        BLOB NOT NULL,
            generation  INTEGER NOT NULL,
            PRIMARY KEY(space, slot)
        ) WITHOUT ROWID
        """,
        "CREATE INDEX IF NOT EXISTS vectors_generation ON vectors(space, generation)",

        // Survives the cascade that removes the vector row, which is the point:
        // a snapshot taken before the deletion still contains the key, and the
        // engine needs to know to subtract it on load. Written only while a
        // snapshot is published, and pruned when the next one covers it.
        """
        CREATE TABLE IF NOT EXISTS vector_tombstones (
            space       TEXT NOT NULL,
            slot        INTEGER NOT NULL,
            generation  INTEGER NOT NULL,
            PRIMARY KEY(space, slot)
        ) WITHOUT ROWID
        """,
        "CREATE INDEX IF NOT EXISTS vector_tombstones_generation "
            + "ON vector_tombstones(space, generation)",

        // Records what a processor has already produced for an item, so an
        // upsert of unchanged content skips OCR, transcription and embedding.
        """
        CREATE TABLE IF NOT EXISTS processing_state (
            item_id                TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            processor              TEXT NOT NULL,
            content_fingerprint    TEXT NOT NULL,
            processor_fingerprint  TEXT NOT NULL,
            completed_at           REAL NOT NULL,
            PRIMARY KEY(item_id, processor)
        ) WITHOUT ROWID
        """,
    ]
}

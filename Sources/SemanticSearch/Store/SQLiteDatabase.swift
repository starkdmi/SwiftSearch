import Foundation
import SQLite3

private let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// A thin, synchronous wrapper over the system SQLite.
///
/// No dependency: SQLite ships in the OS on every platform this package
/// targets, and a package wrapper would add a build dependency for the sake of
/// syntax over an API this uses about a dozen calls from.
///
/// Not thread-safe by itself. The store owns one connection and serialises
/// access through its own actor isolation; WAL then lets a reader connection —
/// a benchmark, an external tool — read concurrently with the writer.
final class SQLiteDatabase {

    private var handle: OpaquePointer?
    let path: String

    init(path: String, readOnly: Bool = false) throws {
        self.path = path
        var handle: OpaquePointer?
        let flags =
            readOnly
            ? SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX
            : SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX
        let result = sqlite3_open_v2(path, &handle, flags, nil)
        guard result == SQLITE_OK, let handle else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close_v2(handle)
            throw SemanticSearchError.storage("cannot open \(path): \(message)", code: result)
        }
        self.handle = handle

        // Busy handler before anything else: a WAL checkpoint from another
        // connection can block the first statement this one runs.
        sqlite3_busy_timeout(handle, 5_000)
        if !readOnly {
            try execute("PRAGMA journal_mode = WAL")
            try execute("PRAGMA synchronous = NORMAL")
            try execute("PRAGMA foreign_keys = ON")
            // 64 MB page cache. The BM25 rebuild on open reads every posting
            // row, and the default 2 MB turns that into a scan of the file.
            try execute("PRAGMA cache_size = -65536")
        }
    }

    deinit {
        if let handle { sqlite3_close_v2(handle) }
    }

    func close() {
        if let handle { sqlite3_close_v2(handle) }
        handle = nil
    }

    // MARK: - Statements

    func execute(_ sql: String) throws {
        guard let handle else { throw SemanticSearchError.storage("database is closed", code: nil) }
        var errorMessage: UnsafeMutablePointer<CChar>?
        let result = sqlite3_exec(handle, sql, nil, nil, &errorMessage)
        guard result == SQLITE_OK else {
            let message = errorMessage.map { String(cString: $0) } ?? "unknown error"
            sqlite3_free(errorMessage)
            throw SemanticSearchError.storage("\(message) — while running: \(sql)", code: result)
        }
    }

    func prepare(_ sql: String) throws -> SQLiteStatement {
        guard let handle else { throw SemanticSearchError.storage("database is closed", code: nil) }
        var statement: OpaquePointer?
        let result = sqlite3_prepare_v2(handle, sql, -1, &statement, nil)
        guard result == SQLITE_OK, let statement else {
            let message = String(cString: sqlite3_errmsg(handle))
            throw SemanticSearchError.storage("\(message) — while preparing: \(sql)", code: result)
        }
        return SQLiteStatement(statement: statement, database: handle)
    }

    var lastInsertedRowID: Int64 {
        guard let handle else { return 0 }
        return sqlite3_last_insert_rowid(handle)
    }

    var changes: Int {
        guard let handle else { return 0 }
        return Int(sqlite3_changes(handle))
    }

    // MARK: - Transactions

    /// Run `body` in a transaction, rolling back if it throws.
    ///
    /// `IMMEDIATE` rather than `DEFERRED`: an upsert reads before it writes, and
    /// a deferred transaction that upgrades to a write lock mid-way can fail
    /// with `SQLITE_BUSY` after the read has already succeeded, which is the
    /// hardest kind of conflict to retry correctly.
    func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE")
        do {
            let value = try body()
            try execute("COMMIT")
            return value
        } catch {
            // A rollback failure here would mask the real error, so it is
            // swallowed; the connection is closed by the caller in that case.
            try? execute("ROLLBACK")
            throw error
        }
    }

    /// Force a WAL checkpoint. Used after a bulk build so the `-wal` file does
    /// not outgrow the database itself.
    func checkpoint() throws {
        try execute("PRAGMA wal_checkpoint(TRUNCATE)")
    }
}

/// A prepared statement. Reset and reused rather than re-prepared: a 100k-item
/// build runs the same insert 100k times, and preparation is the expensive part.
final class SQLiteStatement {
    private let statement: OpaquePointer
    private let database: OpaquePointer

    init(statement: OpaquePointer, database: OpaquePointer) {
        self.statement = statement
        self.database = database
    }

    deinit {
        sqlite3_finalize(statement)
    }

    // MARK: - Binding

    func bind(_ index: Int32, _ value: String?) {
        guard let value else {
            sqlite3_bind_null(statement, index)
            return
        }
        sqlite3_bind_text(statement, index, value, -1, sqliteTransient)
    }

    func bind(_ index: Int32, _ value: Int?) {
        guard let value else {
            sqlite3_bind_null(statement, index)
            return
        }
        sqlite3_bind_int64(statement, index, Int64(value))
    }

    func bind(_ index: Int32, _ value: Int64?) {
        guard let value else {
            sqlite3_bind_null(statement, index)
            return
        }
        sqlite3_bind_int64(statement, index, value)
    }

    func bind(_ index: Int32, _ value: Double?) {
        guard let value else {
            sqlite3_bind_null(statement, index)
            return
        }
        sqlite3_bind_double(statement, index, value)
    }

    func bind(_ index: Int32, _ value: Data?) {
        guard let value else {
            sqlite3_bind_null(statement, index)
            return
        }
        if value.isEmpty {
            sqlite3_bind_zeroblob(statement, index, 0)
            return
        }
        _ = value.withUnsafeBytes { buffer in
            sqlite3_bind_blob(statement, index, buffer.baseAddress, Int32(buffer.count),
                sqliteTransient)
        }
    }

    // MARK: - Stepping

    /// True while a row is available.
    @discardableResult
    func step() throws -> Bool {
        let result = sqlite3_step(statement)
        switch result {
        case SQLITE_ROW: return true
        case SQLITE_DONE: return false
        default:
            let message = String(cString: sqlite3_errmsg(database))
            throw SemanticSearchError.storage(message, code: result)
        }
    }

    func run() throws {
        defer { reset() }
        while try step() {}
    }

    func reset() {
        sqlite3_reset(statement)
        sqlite3_clear_bindings(statement)
    }

    /// Iterate every row, resetting afterwards.
    func forEachRow(_ body: (SQLiteStatement) throws -> Void) throws {
        defer { reset() }
        while try step() { try body(self) }
    }

    // MARK: - Reading

    func string(_ index: Int32) -> String? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL,
            let pointer = sqlite3_column_text(statement, index)
        else { return nil }
        return String(cString: pointer)
    }

    func int(_ index: Int32) -> Int {
        Int(sqlite3_column_int64(statement, index))
    }

    func optionalInt(_ index: Int32) -> Int? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else { return nil }
        return Int(sqlite3_column_int64(statement, index))
    }

    func double(_ index: Int32) -> Double {
        sqlite3_column_double(statement, index)
    }

    func data(_ index: Int32) -> Data? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else { return nil }
        let count = Int(sqlite3_column_bytes(statement, index))
        guard count > 0, let pointer = sqlite3_column_blob(statement, index) else {
            return Data()
        }
        return Data(bytes: pointer, count: count)
    }

    /// Read a blob as a float vector without a bounce through `[UInt8]`.
    func floatVector(_ index: Int32) -> [Float]? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else { return nil }
        let byteCount = Int(sqlite3_column_bytes(statement, index))
        guard byteCount > 0, let pointer = sqlite3_column_blob(statement, index) else {
            return []
        }
        let count = byteCount / MemoryLayout<Float>.size
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            memcpy(buffer.baseAddress!, pointer, count * MemoryLayout<Float>.size)
            initialized = count
        }
    }
}

extension Array where Element == Float {
    /// Little-endian f32 blob. Both platforms this package targets are
    /// little-endian, and the format is documented in the schema notes so a
    /// future big-endian port knows what it is looking at.
    var vectorBlob: Data {
        withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

import AVFoundation
import Foundation
import ImageIO
import PDFKit
import SemanticSearch
import UniformTypeIdentifiers

public enum LibraryError: Error, CustomStringConvertible {
    case notInLibrary(String)
    case notAFolder(String)
    case unreadable(String, reason: String)
    case unsupported(String)
    case redirected(String, to: String)

    public var description: String {
        switch self {
        case .unsupported(let reason): reason
        case .redirected(let folder, let target):
            "\(folder) now leads to \(target); remove it and add the folder again"
        case .notInLibrary(let path):
            "\(path) is not an indexed file in the library's folders"
        case .notAFolder(let path): "\(path) is not a folder"
        case .unreadable(let path, let reason): "cannot read \(path): \(reason)"
        }
    }
}

/// The library's folders, shared with the engine's `mediaAccess` and kept
/// current as folders are added and removed.
final class FolderBoundary: @unchecked Sendable {
    private let lock = NSLock()
    private var folders: [String]

    init(_ folders: [String]) { self.folders = folders }

    func set(_ folders: [String]) { lock.withLock { self.folders = folders } }

    func allows(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        return Library.isReadable(url.path, in: lock.withLock { folders })
    }
}

// MARK: - Answers

/// Where in a file a search matched.
public struct FileMatch: Codable, Sendable, Equatable {
    /// What matched: "text", "speech", "picture", "frame", "sound", …
    public var kind: String
    /// Seconds, for recordings and videos: the span the match stands for.
    public var start: Double?
    public var end: Double?
    /// The second the match is at — a frame's own time, a sentence's
    /// start. `read_file` with this `time` shows the frame a description was
    /// written from; the span's `end` is already the next shot.
    public var time: Double?
    /// 1-based, for PDFs.
    public var page: Int?
    /// The matched text, cut to the library's snippet length.
    public var text: String?
}

public struct FileHit: Codable, Sendable, Equatable {
    public var path: String
    public var type: String?
    public var title: String?
    /// The reranker's score when it ran, the fused retrieval score otherwise.
    public var score: Float
    public var matches: [FileMatch]
}

public struct SearchAnswer: Codable, Sendable {
    public var query: String
    public var files: [FileHit]
    /// Anything the caller should know about how complete the answer is.
    public var notes: [String]
    /// Where the time went. Not encoded: an agent reading the answer has no
    /// use for it.
    public var timings: SearchTimings? = nil

    enum CodingKeys: String, CodingKey { case query, files, notes }
}

/// Where a search's time went, summed over its rounds, for a caller that
/// shows or logs it.
public struct SearchTimings: Sendable {
    /// Waiting for the models to be prepared: the launch-time sync loads them.
    public var preparing: Duration = .zero
    /// Searches run: one unless a long document crowded the first.
    public var rounds = 0
    /// Query embedding and retrieval, the model calls' waits and loads
    /// included.
    public var retrieval: Duration = .zero
    /// The reranker, its wait and load included.
    public var rerank: Duration = .zero
    /// Waiting for the inference permit, held by another search or an ingest.
    public var inferenceWait: Duration = .zero
    /// Loading models a release had unloaded.
    public var modelLoad: Duration = .zero
    public var rerankedCandidates = 0
    public var total: Duration = .zero
}

public struct FileEntry: Codable, Sendable, Equatable {
    public var path: String
    public var type: String?
    public var size: Int64?
    /// When the photo or video was taken, or the document written.
    public var created: Date?
    public var modified: Date?
    public var title: String?
    public var duration: Double?
    public var place: String?
}

public struct FileListing: Codable, Sendable {
    public var total: Int
    public var offset: Int
    public var files: [FileEntry]
}

public struct FileInfo: Codable, Sendable {
    public var path: String
    /// Every metadata value, as text.
    public var metadata: [String: String]
    /// What was extracted, by kind: `{"speech": 120, "frame": 32}`.
    public var parts: [String: Int]
}

public enum FileContent: Sendable {
    case text(String, truncated: Bool)
    case image(Data, mimeType: String)
}

public struct SyncSummary: Codable, Sendable {
    public var started: Date
    public var finished: Date?
    public var added = 0
    public var updated = 0
    public var unchanged = 0
    /// Files found at a new path with the bytes of one that left: their
    /// index moved with them, nothing processed again.
    public var moved = 0
    public var removed = 0
    /// Files whose place an earlier lookup could not name, named now.
    public var placesNamed = 0
    public var unsupported = 0
    public var failed: [String] = []
    public var unreadable: [String] = []
    /// Files done and to do, while a sync runs.
    public var completed = 0
    public var total = 0
    public var current: String?
    public var error: String?
}

public struct LibraryStatus: Codable, Sendable {
    public var home: String
    public var profile: Profile
    public var folders: [String]
    public var files: Int
    public var byType: [String: Int]
    public var indexBytes: Int64
    public var sync: SyncSummary?
}

/// What a listing or a search is limited to.
public struct FileQuery: Sendable, Equatable {
    /// `image`, `video`, `audio`, `pdf`, `text`, or a UTI.
    public var type: String?
    /// An absolute folder path.
    public var folder: String?
    public var after: Date?
    public var before: Date?
    /// Which date `after` and `before` test.
    public var dateField: DateField

    public enum DateField: String, Sendable, CaseIterable {
        /// When the content was made: EXIF, QuickTime or PDF creation date.
        case created
        /// The file's modification date.
        case modified
    }

    public init(
        type: String? = nil, folder: String? = nil, after: Date? = nil, before: Date? = nil,
        dateField: DateField = .created
    ) {
        self.type = type
        self.folder = folder
        self.after = after
        self.before = before
        self.dateField = dateField
    }

    static let typeAliases: [String: String] = [
        "image": UTType.image.identifier, "photo": UTType.image.identifier,
        "picture": UTType.image.identifier, "video": UTType.movie.identifier,
        "movie": UTType.movie.identifier, "audio": UTType.audio.identifier,
        "pdf": UTType.pdf.identifier, "text": UTType.text.identifier,
    ]

    var filter: SearchFilter {
        var clauses: [SearchFilter] = []
        if let type, !type.isEmpty {
            let identifier = Self.typeAliases[type.lowercased()] ?? type
            clauses.append(.equals(key: "contentType", value: .string(identifier)))
        }
        if let folder, !folder.isEmpty {
            // Resolved, as the library's folders are stored: files are
            // indexed under the paths a resolved folder's walk gives them.
            clauses.append(.equals(key: "folders", value: .string(Library.resolved(folder))))
        }
        let key = dateField == .created ? "contentCreated" : "fileModified"
        if let after { clauses.append(.greaterThan(key: key, value: .date(after))) }
        if let before { clauses.append(.lessThan(key: key, value: .date(before))) }
        return clauses.isEmpty ? .all : .and(clauses)
    }
}

public enum ListOrder: String, Sendable, CaseIterable {
    case path, newest, oldest
}

// MARK: - Library

/// One library: its folders, its index, the models its profile uses.
///
/// Open in one process at a time (``IndexLock``). Files are indexed under
/// their `file://` URLs, so a path is all a caller needs to name one.
public actor Library {
    public let home: Home
    public private(set) var settings: Settings
    private let lock: IndexLock
    private let engine: SemanticSearchEngine
    private let releaseProcessorModels: @Sendable () async -> Void
    private var preparing: Task<Void, any Error>?
    private var syncTask: Task<SyncSummary, any Error>?
    /// Which sync `syncTask` is, so a finished one clears only itself.
    private var syncGeneration = 0
    private let progress = SyncProgressBox()

    public static func open(home: Home = .default()) async throws -> Library {
        try home.createPrivately()
        let lock = try IndexLock(home.lockURL)
        let settings = try home.loadSettings()
        if let directory = settings.modelDirectory {
            setenv("SEMANTIC_SEARCH_MODEL_DIRECTORY", directory, 1)
        }
        let boundary = FolderBoundary(settings.folders)
        let built = await Recipe.build(
            settings: settings, storeURL: home.indexURL(for: settings.profile),
            boundary: boundary)
        let engine = try await SemanticSearchEngine(configuration: built.configuration)
        return Library(
            home: home, settings: settings, lock: lock, engine: engine, boundary: boundary,
            releaseProcessorModels: built.releaseProcessorModels,
            namePendingPlaces: built.namePendingPlaces)
    }

    private let namePendingPlaces: @Sendable (SemanticSearchEngine) async throws -> Int

    /// The folders, as the engine reads them: it opens the pictures it shows
    /// the reranker itself.
    private let boundary: FolderBoundary

    private init(
        home: Home, settings: Settings, lock: IndexLock, engine: SemanticSearchEngine,
        boundary: FolderBoundary,
        releaseProcessorModels: @escaping @Sendable () async -> Void,
        namePendingPlaces: @escaping @Sendable (SemanticSearchEngine) async throws -> Int
    ) {
        self.home = home
        self.settings = settings
        self.lock = lock
        self.engine = engine
        self.boundary = boundary
        self.releaseProcessorModels = releaseProcessorModels
        self.namePendingPlaces = namePendingPlaces
    }

    public func close() async throws {
        // Released here, not when the actor is freed: a caller that keeps a
        // closed library around must not keep everyone else out.
        defer { lock.release() }
        syncTask?.cancel()
        // Preparation runs in a task of its own, so a sync's cancellation
        // does not reach it: an agent that disconnects during start-up must
        // not leave a download running with the library locked.
        preparing?.cancel()
        _ = try? await syncTask?.value
        _ = try? await preparing?.value
        try await engine.close()
    }

    /// Download and check the profile's models. Done on first use; call it to
    /// pay the cost up front.
    public func prepare() async throws {
        // One load, however many callers: the actor is re-entered while it
        // waits, and a search during the start-up sync would load every model
        // a second time.
        if preparing == nil {
            let engine = engine
            preparing = Task { try await engine.prepare() }
        }
        do {
            try await preparing?.value
        } catch {
            preparing = nil
            throw error
        }
    }

    /// Unload every model until it is needed again: the engine's embedders
    /// and reranker, and the captioner and Parakeet the processors hold.
    public func releaseModels() async {
        await engine.releaseModels()
        await releaseProcessorModels()
    }

    // MARK: Folders

    public func addFolder(_ path: String) throws -> String {
        // Stored resolved: a link to `/` passed the check below as written,
        // and a sync then walked the whole disk under the link's name.
        let folder = Self.resolved(path)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { throw LibraryError.notAFolder(folder) }
        // The engine's `folders` leave `/` out — every file is under it — so
        // a search bounded to it would find nothing; and a library of the
        // whole disk would hand an agent everything it exists to keep back.
        guard folder != "/" else {
            throw LibraryError.unsupported("add the folders to index, not the whole disk")
        }
        if !settings.folders.contains(folder) {
            settings.folders.append(folder)
            try home.save(settings)
            boundary.set(settings.folders)
        }
        return folder
    }

    /// Stop indexing a folder and remove what was indexed from it.
    public func removeFolder(_ path: String) async throws -> Int {
        // As added — resolved — or as an older version stored it.
        let folder =
            settings.folders.contains(Self.standardized(path))
            ? Self.standardized(path) : Self.resolved(path)
        // A running sync walks the folders it read when it began; stopped
        // first, or it would index this one again after its files are gone.
        if let syncTask {
            syncTask.cancel()
            _ = try? await syncTask.value
        }
        settings.folders.removeAll { $0 == folder }
        try home.save(settings)
        boundary.set(settings.folders)
        // Files another folder still covers — a subfolder also configured —
        // stay.
        let remaining = settings.folders
        return try await removeItems { path in
            Self.isInside(path, folder) && !remaining.contains { Self.isInside(path, $0) }
        }
    }

    static func isInside(_ path: String, _ folder: String) -> Bool {
        path == folder || path.hasPrefix(folder + "/")
    }

    /// Folders are stored resolved, so one that resolves elsewhere now has
    /// had itself or a folder above it replaced by a link. Neither read nor
    /// synced: a sync would index the link's target under the folder's name.
    static func checkNotRedirected(_ folder: String) throws {
        let now = resolved(folder)
        if now != folder { throw LibraryError.redirected(folder, to: now) }
    }

    /// Only files under the library's folders, whatever else the index
    /// holds. Removing a folder clears it from this profile's index; another
    /// profile's still has it until that profile syncs, so every search and
    /// listing is bounded here, not by sync.
    private func bounded(_ filter: SearchFilter) -> SearchFilter {
        .and([
            filter, .anyOf(key: "folders", values: settings.folders.map { .string($0) }),
        ])
    }

    private func removeItems(where isRemoved: (String) -> Bool) async throws -> Int {
        var doomed: [ItemID] = []
        var offset = 0
        while true {
            let page = try await engine.listItems(limit: 1_000, offset: offset)
            for item in page.items {
                if let path = Self.path(of: item.id), isRemoved(path) { doomed.append(item.id) }
            }
            offset += page.items.count
            if page.items.isEmpty || offset >= page.total { break }
        }
        for id in doomed { try await engine.remove(id, includingDescendants: true) }
        return doomed.count
    }

    // MARK: Sync

    /// Index every folder: new and changed files in, deleted ones out, and
    /// anything outside the folders — a folder no longer listed — removed.
    /// One sync at a time; a second call waits for the running one.
    public func sync() async throws -> SyncSummary {
        try await startSync().value
    }

    /// Start a sync without waiting for it, or join the one running.
    @discardableResult
    public func startSync() -> Task<SyncSummary, any Error> {
        if let syncTask { return syncTask }
        syncGeneration += 1
        let generation = syncGeneration
        let task = Task {
            // The task runs on the actor, so this clears the handle before
            // anyone awaiting it resumes.
            defer { self.syncFinished(generation) }
            return try await self.runSync()
        }
        syncTask = task
        return task
    }

    private func syncFinished(_ generation: Int) {
        if generation == syncGeneration { syncTask = nil }
    }

    /// The running or last sync, cheaply.
    public func syncProgress() -> SyncSummary? { progress.snapshot() }

    public var isSyncing: Bool { syncTask != nil }

    private func runSync() async throws -> SyncSummary {
        progress.start()
        do {
            try await prepare()
            let folders = settings.folders
            _ = try await removeItems { path in
                !folders.contains { Self.isInside(path, $0) }
            }
            // A folder inside another is walked with it. Walked again on its
            // own, its files' words — the folders below the synced one —
            // would change back and forth between the two on every sync.
            for folder in folders
            where !folders.contains(where: { $0 != folder && Self.isInside(folder, $0) }) {
                // One missing folder — an unplugged drive — must not keep the
                // others from syncing; the engine removes nothing from it.
                do {
                    try Self.checkNotRedirected(folder)
                    let report = try await engine.sync(
                        folder: URL(fileURLWithPath: folder, isDirectory: true)
                    ) { [progress] update in
                        progress.update(update)
                    }
                    progress.finishFolder(report)
                } catch {
                    if Task.isCancelled { throw CancellationError() }
                    progress.folderFailed(folder, error)
                }
            }
            // Places a refused or offline lookup left out, in this sync or an
            // earlier one. Failing again is the next sync's to retry.
            do {
                progress.placesNamed(try await namePendingPlaces(engine))
            } catch {
                if Task.isCancelled { throw CancellationError() }
            }
            // The captioner and Parakeet are only needed to index; the engine
            // has already moved on to its own models.
            await releaseProcessorModels()
        } catch {
            progress.fail(error)
            throw error
        }
        return progress.finish()
    }

    // MARK: Search

    /// Files that match, best first, each with up to three places it
    /// matched — or, `within` one file, up to `limit` places in it: the
    /// moments of a long recording, the pages of a PDF.
    ///
    /// `image` has to be an indexed picture in the library unless
    /// `imageFromAnywhere`: captioning or embedding a file is a way of
    /// reading it, and an agent that `read_file` refuses must not learn what
    /// an unindexed photo shows by searching with it. The CLI, run by the
    /// user, takes any picture.
    ///
    /// `reranks: false` returns retrieval's order without running the
    /// reranker — a fraction of the time, for a list shown while typing that
    /// a reranked search then replaces.
    public func search(
        _ text: String, image: String? = nil, imageFromAnywhere: Bool = false,
        in query: FileQuery = FileQuery(), within file: String? = nil, limit: Int = 10,
        reranks: Bool = true
    ) async throws -> SearchAnswer {
        let started = ContinuousClock.now
        var timings = SearchTimings()
        var scope: ItemID?
        if let file { scope = try await indexed(file).0 }
        var images: [ImagePayload] = []
        if let image {
            images.append(.fileURL(try await queryPicture(image, fromAnywhere: imageFromAnywhere)))
        }
        let preparing = ContinuousClock.now
        try await prepare()
        timings.preparing = ContinuousClock.now - preparing
        let searchQuery = SearchQuery(text: text, images: images)
        let matchesPerFile = scope == nil ? 3 : limit
        let filter = bounded(query.filter)

        var files: [FileHit] = []
        var index: [String: Int] = [:]
        var found: [ItemID] = []
        var notes: [String] = []
        /// Each file's first hit's reranker score, in `files` order.
        var rerankScores: [Float?] = []
        // The engine ranks passages, not files: one long document can fill
        // every channel's candidates and leave one file where ten were asked
        // for. When a channel stopped at its cap and files are short, search
        // again without the files already found.
        for _ in 0..<(scope == nil ? 4 : 1) {
            var options = SearchOptions(
                mode: Recipe.searchMode(settings.profile, reranks: reranks), limit: 200,
                filter: found.isEmpty
                    ? filter : .and([filter, .not(.or(found.map { .withinParent($0) }))]))
            options.rerankPoolSize = 50
            let results =
                if let scope {
                    try await engine.searchInside(scope, query: searchQuery, options: options)
                } else {
                    try await engine.search(searchQuery, options: options)
                }
            let diagnostics = results.diagnostics
            timings.rounds += 1
            timings.retrieval += diagnostics.retrievalDuration
            timings.rerank += diagnostics.rerankDuration
            timings.inferenceWait += diagnostics.inferenceWaitDuration
            timings.modelLoad += diagnostics.modelLoadDuration
            timings.rerankedCandidates += diagnostics.rerankedCandidates
            for note in self.notes(from: diagnostics) where !notes.contains(note) {
                notes.append(note)
            }
            let before = files.count
            for hit in results.hits {
                guard let path = Self.path(of: hit.itemID) else { continue }
                // A file's details are about the whole file, not the span its
                // item happens to cover.
                let range = hit.kind == .metadata ? nil : hit.locator.timeRange
                let match = FileMatch(
                    kind: Self.kindName(hit), start: range?.start, end: range?.end,
                    time: hit.kind == .metadata ? nil : hit.locator.focusTime,
                    page: hit.locator.pageIndex.map { $0 + 1 }, text: snippet(hit.text))
                if let position = index[path] {
                    if files[position].matches.count < matchesPerFile {
                        files[position].matches.append(match)
                    }
                    continue
                }
                guard files.count < limit else { continue }
                let metadata = hit.parent?.metadata ?? hit.metadata
                index[path] = files.count
                rerankScores.append(hit.evidence.rerankScore)
                found.append(Self.fileID(of: hit.itemID))
                files.append(
                    FileHit(
                        path: path, type: Self.typeName(metadata),
                        title: metadata["title"]?.stringValue,
                        score: hit.evidence.rerankScore ?? hit.evidence.finalScore,
                        matches: [match]))
            }
            let crowded = results.diagnostics.channels.contains {
                $0.retrieved >= options.candidatesPerChannel
            }
            guard files.count < limit, crowded, files.count > before else { break }
        }
        files = Self.merged(files, rerankScores: rerankScores)

        if files.contains(where: { $0.matches.contains { $0.kind == "frame description" } }) {
            // Seen in a Claude Code run: asked what was on screen, the agent
            // answered from captions alone and never looked.
            notes.append(
                "frame descriptions are machine-written captions and can be wrong; "
                    + "read_file with the path and a time returns the frame itself")
        }
        if isSyncing { notes.append("indexing is still running; results may be incomplete") }
        if settings.folders.isEmpty { notes.append("the library has no folders yet") }
        timings.total = ContinuousClock.now - started
        return SearchAnswer(query: text, files: files, notes: notes, timings: timings)
    }

    /// One ranking over every round: a later round's file the reranker
    /// scored higher goes ahead of an earlier one. The reranker's score is
    /// a judgement of the pair alone, so it compares across rounds; the
    /// fused retrieval score is rank-based — every round's first hit gets
    /// the same one — and does not. So reranked files are ordered by their
    /// score, and the rest keep their order after them, as the engine puts
    /// its reranked pool ahead of the tail.
    static func merged(_ files: [FileHit], rerankScores: [Float?]) -> [FileHit] {
        let reranked = files.indices.filter { rerankScores[$0] != nil }.sorted {
            let (a, b) = (rerankScores[$0]!, rerankScores[$1]!)
            return a != b ? a > b : $0 < $1
        }
        let rest = files.indices.filter { rerankScores[$0] == nil }
        return (reranked + rest).map { files[$0] }
    }

    /// What the engine could not do for a search, in words an agent can act
    /// on. The engine reports a degraded search rather than hiding it; an
    /// answer that dropped the report would look complete when it is not.
    func notes(from diagnostics: SearchDiagnostics) -> [String] {
        var notes: [String] = []
        if let reason = diagnostics.rerankSkippedReason { notes.append("not reranked: \(reason)") }
        if diagnostics.rerankIgnoredQueryImages {
            notes.append("the reranker cannot see pictures; it judged by the query's words alone")
        }
        if diagnostics.rerankUnscoredCandidates > 0 {
            notes.append(
                "\(diagnostics.rerankUnscoredCandidates) candidates had nothing the reranker "
                    + "could read and keep their retrieval order")
        }
        if diagnostics.rerankDroppedImages > 0 {
            notes.append(
                "\(diagnostics.rerankDroppedImages) candidate pictures did not fit the "
                    + "reranker's image budget and were judged without them")
        }
        for caption in diagnostics.queryCaptions {
            notes.append("the query picture was searched as its description: \(caption)")
        }
        for space in diagnostics.skippedSpaces {
            notes.append("not searched by \(space.space.rawValue): \(space.reason)")
        }
        for space in diagnostics.partiallyAnsweredSpaces {
            notes.append("searched only in part by \(space.space.rawValue): \(space.reason)")
        }
        // Only files in the library's folders: the answer is bounded to them.
        let missing = Dictionary(
            diagnostics.unavailableRepresentations.compactMap { entry in
                Self.path(of: entry.key.item).flatMap { path in
                    settings.folders.contains { Self.isInside(path, $0) }
                        ? (path, entry.reason) : nil
                }
            }, uniquingKeysWith: { first, _ in first })
        for (path, reason) in missing.sorted(by: { $0.key < $1.key }).prefix(5) {
            notes.append("could not evaluate part of \(path): \(reason)")
        }
        if missing.count > 5 { notes.append("… and \(missing.count - 5) more files") }
        return notes
    }

    /// A picture a search can be asked with: one the profile can see, that
    /// decodes, and — unless `fromAnywhere` — an indexed picture in the
    /// library, checked as `read_file` checks it.
    private func queryPicture(_ path: String, fromAnywhere: Bool) async throws -> URL {
        guard settings.profile != .lexical else {
            throw LibraryError.unsupported(
                "the lexical profile has no model that sees pictures; "
                    + "search with words, or switch to the fast or multimodal profile")
        }
        let url = URL(fileURLWithPath: Self.standardized(path))
        if !fromAnywhere {
            let (_, item) = try await indexed(path)
            guard Self.typeName(item.metadata) == "image" else {
                throw LibraryError.unsupported("\(url.path) is not a picture")
            }
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            CGImageSourceGetCount(source) > 0
        else { throw LibraryError.unreadable(url.path, reason: "not a decodable image") }
        return url
    }

    private func snippet(_ text: String?) -> String? {
        guard settings.snippetCharacters > 0, let text, !text.isEmpty else { return nil }
        let flat = text.split(whereSeparator: \.isNewline).joined(separator: " ")
        guard flat.count > settings.snippetCharacters else { return flat }
        return String(flat.prefix(settings.snippetCharacters)) + "…"
    }

    // MARK: List and inspect

    public func list(
        _ query: FileQuery = FileQuery(), order: ListOrder = .path, limit: Int = 50,
        offset: Int = 0
    ) async throws -> FileListing {
        let ordering: ItemListing.Order =
            switch order {
            case .path: .id
            case .newest: .metadata(Self.dateKey(query), descending: true)
            case .oldest: .metadata(Self.dateKey(query), descending: false)
            }
        let listing = try await engine.listItems(
            matching: bounded(query.filter), order: ordering, limit: limit, offset: offset)
        return FileListing(
            total: listing.total, offset: offset,
            files: listing.items.compactMap { item in
                guard let path = Self.path(of: item.id) else { return nil }
                let metadata = item.metadata
                func date(_ key: String) -> Date? {
                    if case .date(let date)? = metadata[key] { return date }
                    return nil
                }
                var size: Int64?
                if case .integer(let bytes)? = metadata["fileSize"] { size = bytes }
                var duration: Double?
                if case .double(let seconds)? = metadata["duration"] { duration = seconds }
                return FileEntry(
                    path: path, type: Self.typeName(metadata), size: size,
                    created: date("contentCreated"), modified: date("fileModified"),
                    title: metadata["title"]?.stringValue, duration: duration,
                    place: metadata["place"]?.stringValue)
            })
    }

    public func info(_ path: String) async throws -> FileInfo {
        let (id, item) = try await indexed(path)
        var parts: [String: Int] = [:]
        let children = try await engine.listItems(
            matching: .withinParent(id), topLevelOnly: false, limit: 100_000,
            includesRepresentations: true)
        for child in children.items {
            for representation in child.representations where representation.weight > 0 {
                parts[Self.kindName(representation.kind, timed: child.locator.timeRange != nil),
                    default: 0] += 1
            }
        }
        var metadata: [String: String] = [:]
        // The hash is how a moved file is recognised; to an agent it is
        // sixty-four characters of nothing.
        for (key, value) in item.metadata where key != SemanticSearchEngine.contentHashKey {
            metadata[key] = value.description
        }
        return FileInfo(path: Self.standardized(path), metadata: metadata, parts: parts)
    }

    // MARK: Read

    /// A file, or the part of it asked for, in a form an agent can take:
    /// a picture as a JPEG no larger than `maximumPixels` on its long side,
    /// a video frame at `time`, a PDF's pages, a recording's transcript from
    /// the index, a text file's characters.
    public func read(
        _ path: String, page: Int? = nil, time: Double? = nil, start: Double? = nil,
        end: Double? = nil, offset: Int = 0, maximumCharacters: Int = 20_000,
        maximumPixels: Int = 1_568
    ) async throws -> FileContent {
        let (id, item) = try await indexed(path)
        let url = URL(fileURLWithPath: Self.standardized(path))
        let types = item.metadata["contentType"].map { value -> [String] in
            if case .stringList(let list) = value { return list }
            return [value.description]
        } ?? []

        if types.contains(UTType.image.identifier) {
            return .image(try Self.jpeg(url, maximumPixels: maximumPixels), mimeType: "image/jpeg")
        }
        if types.contains(UTType.movie.identifier), let time {
            return .image(
                try await Self.frame(url, at: time, maximumPixels: maximumPixels),
                mimeType: "image/jpeg")
        }
        if types.contains(UTType.audiovisualContent.identifier) {
            return try await transcript(
                of: id, start: start, end: end, offset: offset, limit: maximumCharacters)
        }
        if types.contains(UTType.pdf.identifier) {
            return try await pdfText(
                url, id: id, page: page, offset: offset, limit: maximumCharacters)
        }
        if types.contains(UTType.text.identifier) {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                throw LibraryError.unreadable(url.path, reason: "not UTF-8 text")
            }
            let rest = text.dropFirst(max(0, offset))
            return .text(String(rest.prefix(maximumCharacters)), truncated: rest.count > maximumCharacters)
        }
        throw LibraryError.unreadable(
            url.path, reason: "no readable form for \(types.first ?? "this type")")
    }

    /// `offset` continues a transcript the last answer cut short, as it does
    /// a text file: `start` cannot, since the sentence cut in two began
    /// before the cut and is returned again from its start.
    private func transcript(
        of id: ItemID, start: Double?, end: Double?, offset: Int, limit: Int
    ) async throws -> FileContent {
        let children = try await engine.listItems(
            matching: .and([.withinParent(id), .hasKind([.transcript, .soundEvent])]),
            topLevelOnly: false, limit: 100_000, includesRepresentations: true)
        var lines: [(Double, String)] = []
        for child in children.items where child.id != id {
            let representations = child.representations
            // Each utterance is indexed as a window of sentences around it and
            // kept alone, unindexed, as `utterance`; read one sentence a line
            // or the windows repeat each other.
            let line: (ContentRepresentation, String)? =
                if let sentence = representations.first(where: { $0.id == "utterance" })
                    ?? representations.first(where: { $0.kind == .transcript }),
                    let text = sentence.content.text
                {
                    (sentence, text)
                } else if let sound = representations.first(where: { $0.kind == .soundEvent }),
                    let text = sound.content.text
                {
                    (sound, "(sound: \(text))")
                } else {
                    nil
                }
            guard let (representation, text) = line,
                let range = representation.locator?.timeRange ?? child.locator.timeRange
            else { continue }
            if let start, range.end < start { continue }
            if let end, range.start > end { continue }
            lines.append((range.start, "[\(Self.clock(range.start))] \(text)"))
        }
        let text = lines.sorted { $0.0 < $1.0 }.map(\.1).joined(separator: "\n")
        if text.isEmpty {
            return .text(
                start != nil || end != nil
                    ? "(nothing was heard between those times)"
                    : "(nothing was heard in this recording)", truncated: false)
        }
        let rest = text.dropFirst(max(0, offset))
        return .text(String(rest.prefix(limit)), truncated: rest.count > limit)
    }

    // MARK: Status

    public func status() async throws -> LibraryStatus {
        var byType: [String: Int] = [:]
        for name in ["image", "video", "audio", "pdf", "text"] {
            let count = try await engine.listItems(
                matching: bounded(FileQuery(type: name).filter), limit: 0
            ).total
            if count > 0 { byType[name] = count }
        }
        return LibraryStatus(
            home: home.url.path, profile: settings.profile, folders: settings.folders,
            files: try await engine.listItems(matching: bounded(.all), limit: 0).total,
            byType: byType,
            indexBytes: await engine.indexSizeBytes(), sync: progress.snapshot())
    }

    // MARK: Helpers

    /// The indexed file at `path`, if it is inside the library's folders.
    private func indexed(_ path: String) async throws -> (ItemID, SearchItem) {
        let standardized = Self.standardized(path)
        guard Self.isReadable(standardized, in: settings.folders) else {
            throw LibraryError.notInLibrary(standardized)
        }
        let id = SemanticSearchEngine.itemID(forFile: URL(fileURLWithPath: standardized))
        guard let item = try await engine.item(id) else {
            throw LibraryError.notInLibrary(standardized)
        }
        return (id, item)
    }

    /// Whether `path` may be read: as written and as resolved — a file in the
    /// folder replaced by a link to one outside it must not be read through
    /// the link — and against the folder as it was added, never as it
    /// resolves now: a folder replaced by a link to another one would carry
    /// the boundary with it, and a path indexed under the old folder would
    /// read whatever the new one holds there.
    static func isReadable(_ path: String, in folders: [String]) -> Bool {
        let standardized = standardized(path)
        let resolved = URL(fileURLWithPath: standardized).resolvingSymlinksInPath().path
        let inside = { (path: String) in
            folders.contains { folder in
                path.hasPrefix(folder + "/") && (try? checkNotRedirected(folder)) != nil
            }
        }
        return inside(standardized) && inside(resolved)
    }

    static func standardized(_ path: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL.path
    }

    /// Standardized, then with every symbolic link followed.
    static func resolved(_ path: String) -> String {
        URL(fileURLWithPath: standardized(path)).resolvingSymlinksInPath().path
    }

    /// The file a hit or item belongs to: processors name children
    /// `<file URL>#…`, and a URL's own `#` is percent-encoded.
    static func path(of id: ItemID) -> String? {
        guard let url = URL(string: fileID(of: id).rawValue), url.isFileURL else { return nil }
        return url.path
    }

    static func fileID(of id: ItemID) -> ItemID {
        ItemID(id.rawValue.split(separator: "#", maxSplits: 1).first.map(String.init) ?? id.rawValue)
    }

    static func typeName(_ metadata: Metadata) -> String? {
        guard case .stringList(let types)? = metadata["contentType"] else { return nil }
        for (name, identifier) in [
            ("image", UTType.image), ("video", UTType.movie), ("audio", UTType.audio),
            ("pdf", UTType.pdf), ("text", UTType.text),
        ] where types.contains(identifier.identifier) {
            return name
        }
        return types.first
    }

    static func dateKey(_ query: FileQuery) -> String {
        query.dateField == .created ? "contentCreated" : "fileModified"
    }

    static func kindName(_ hit: SearchHit) -> String {
        kindName(hit.kind, timed: hit.locator.timeRange != nil)
    }

    static func kindName(_ kind: RepresentationKind, timed: Bool) -> String {
        switch kind {
        case .originalText, .summary, .translation: "text"
        case .ocrText: timed ? "text on screen" : "text in picture"
        case .transcript: "speech"
        case .image: timed ? "frame" : "picture"
        case .caption: timed ? "frame description" : "picture description"
        case .video: "video"
        case .audio: "audio"
        case .soundEvent: "sound"
        case .metadata: "file details"
        }
    }

    static func clock(_ seconds: Double) -> String {
        let total = Int(seconds)
        return total >= 3600
            ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
            : String(format: "%d:%02d", total / 60, total % 60)
    }

    static func jpeg(_ url: URL, maximumPixels: Int) throws -> Data {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateThumbnailAtIndex(
                source, 0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: maximumPixels,
                ] as CFDictionary)
        else { throw LibraryError.unreadable(url.path, reason: "not a decodable image") }
        return try encodeJPEG(image, path: url.path)
    }

    static func frame(_ url: URL, at seconds: Double, maximumPixels: Int) async throws -> Data {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: maximumPixels, height: maximumPixels)
        let (image, _) = try await generator.image(
            at: CMTime(seconds: max(0, seconds), preferredTimescale: 600))
        return try encodeJPEG(image, path: url.path)
    }

    private static func encodeJPEG(_ image: CGImage, path: String) throws -> Data {
        let data = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                data, UTType.jpeg.identifier as CFString, 1, nil)
        else { throw LibraryError.unreadable(path, reason: "cannot encode JPEG") }
        CGImageDestinationAddImage(
            destination, image, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw LibraryError.unreadable(path, reason: "cannot encode JPEG")
        }
        return data as Data
    }

    /// A PDF's pages: the OCR the index holds for a page that was read as a
    /// scan, the text layer for the others.
    func pdfText(
        _ url: URL, id: ItemID, page: Int?, offset: Int, limit: Int
    ) async throws -> FileContent {
        guard let document = PDFDocument(url: url) else {
            throw LibraryError.unreadable(url.path, reason: "not a readable PDF")
        }
        let pages: Range<Int>
        if let page {
            guard (1...max(1, document.pageCount)).contains(page) else {
                throw LibraryError.unreadable(
                    url.path, reason: "page \(page) of \(document.pageCount)")
            }
            pages = (page - 1)..<page
        } else {
            pages = 0..<document.pageCount
        }
        // Saturating: `offset` comes from the caller, and an overflow traps.
        let offset = max(0, offset)
        let (end, overflow) = offset.addingReportingOverflow(limit)
        let enough = overflow ? Int.max : end
        var text = ""
        var characters = 0
        for index in pages {
            // The page as it was indexed: a scan with a selectable page number
            // was read by OCR (`PDFProcessor` routes sparse or broken layers
            // there), and its one-line layer is not what the page says.
            var body = try await indexedText(
                of: ItemID("\(id.rawValue)#page-\(index + 1)"), kinds: [.ocrText])
            if body.isEmpty { body = document.page(at: index)?.string ?? "" }
            let part = "--- page \(index + 1) ---\n" + body + "\n"
            text += part
            characters += part.count
            if characters > enough { break }
        }
        // `offset` continues a page, or the pages, the last answer cut short.
        let rest = text.dropFirst(offset)
        return .text(String(rest.prefix(limit)), truncated: rest.count > limit)
    }

    /// The text indexed for an item and the items under it, in order.
    private func indexedText(
        of id: ItemID, kinds: Set<RepresentationKind>
    ) async throws -> String {
        let items = try await engine.listItems(
            matching: .withinParent(id), topLevelOnly: false, limit: 10_000,
            includesRepresentations: true)
        let texts: [String] = items.items
            .sorted { ($0.id == id ? -1 : $0.ordinal ?? 0) < ($1.id == id ? -1 : $1.ordinal ?? 0) }
            .flatMap { item in
                item.representations.filter {
                    $0.weight > 0 && kinds.contains($0.kind)
                }.compactMap(\.content.text)
            }
        return texts.joined(separator: "\n")
    }
}

/// Sync progress, written from the engine's callback and read by `status`.
final class SyncProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var summary: SyncSummary?

    func start() {
        lock.withLock { summary = SyncSummary(started: Date()) }
    }

    func update(_ update: FolderSyncProgress) {
        lock.withLock {
            summary?.completed = update.completed
            summary?.total = update.total
            summary?.current = update.file.path
        }
    }

    func finishFolder(_ report: FolderSyncReport) {
        lock.withLock {
            summary?.added += report.added.count
            summary?.updated += report.updated.count
            summary?.unchanged += report.unchanged
            summary?.moved += report.moved.count
            summary?.removed += report.removed.count
            summary?.unsupported += report.unsupported.count
            summary?.failed += report.failed.map { "\($0.file.path): \($0.reason)" }
            summary?.unreadable += report.unreadable.map(\.path)
        }
    }

    func placesNamed(_ count: Int) {
        lock.withLock { summary?.placesNamed += count }
    }

    func folderFailed(_ folder: String, _ error: any Error) {
        lock.withLock { summary?.failed.append("\(folder): \(error)") }
    }

    func fail(_ error: any Error) {
        lock.withLock {
            summary?.error = String(describing: error)
            summary?.finished = Date()
            summary?.current = nil
        }
    }

    func finish() -> SyncSummary {
        lock.withLock {
            summary?.finished = Date()
            summary?.current = nil
            return summary ?? SyncSummary(started: Date())
        }
    }

    func snapshot() -> SyncSummary? {
        lock.withLock { summary }
    }
}

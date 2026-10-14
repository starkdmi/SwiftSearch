import Foundation

#if canImport(CryptoKit)
    import CryptoKit
#endif

/// A page of items a filter admits, with how many there are in all.
public struct ItemListing: Sendable {
    public enum Order: Sendable, Hashable {
        /// By item id — for synced files, their URLs, so folder by folder.
        case id
        /// By a metadata value: dates, numbers or strings. Items without the
        /// key come last in either direction.
        case metadata(String, descending: Bool = false)
    }

    public var items: [SearchItem]
    /// Every item the filter admits, not just this page.
    public var total: Int
}

/// What ``SemanticSearchEngine/sync(folder:options:onProgress:)`` walks.
public struct FolderSyncOptions: Sendable, Hashable {
    /// Files and folders whose names start with a dot, or marked hidden.
    public var includesHiddenFiles: Bool
    /// Inside bundles — `.app`, `.photoslibrary`, `.rtfd`. Off, a package is
    /// passed over whole: its insides are one application's storage, not the
    /// user's files.
    public var descendsIntoPackages: Bool
    /// File extensions to take, lowercased, without the dot; `nil` for
    /// whatever the processor accepts.
    public var extensions: Set<String>?
    /// Remove what a previous sync of this folder indexed from files that are
    /// no longer there — or no longer taken: hidden, of another extension, or
    /// not accepted by the processor.
    public var removesMissingFiles: Bool
    /// Recognise a file that was moved or renamed inside the folder by its
    /// bytes, and move what was indexed from it instead of processing it
    /// again. Costs one read of every file that is processed, to record its
    /// SHA-256 (`contentHash`).
    public var detectsMoves: Bool

    public init(
        includesHiddenFiles: Bool = false,
        descendsIntoPackages: Bool = false,
        extensions: Set<String>? = nil,
        removesMissingFiles: Bool = true,
        detectsMoves: Bool = true
    ) {
        self.includesHiddenFiles = includesHiddenFiles
        self.descendsIntoPackages = descendsIntoPackages
        self.extensions = extensions.map { Set($0.map { $0.lowercased() }) }
        self.removesMissingFiles = removesMissingFiles
        self.detectsMoves = detectsMoves
    }
}

public struct FolderSyncProgress: Sendable {
    public var file: URL
    /// Files finished before this one.
    public var completed: Int
    public var total: Int
    /// The processor's own progress on this file, when it reports any.
    public var stage: ProcessingProgress?
}

public struct FolderSyncReport: Sendable {
    public struct Failure: Sendable {
        public var file: URL
        public var reason: String
    }

    public struct Move: Sendable, Hashable {
        public var from: ItemID
        public var to: ItemID
    }

    /// Files indexed for the first time.
    public var added: [ItemID] = []
    /// Files indexed again because they, or the processor, changed.
    public var updated: [ItemID] = []
    /// Files skipped because nothing changed.
    public var unchanged = 0
    /// Files found at a new path with the bytes of one that left, whose
    /// index entries moved with them.
    public var moved: [Move] = []
    /// Files gone from the folder, and removed from the index.
    public var removed: [ItemID] = []
    /// Files no processor accepts.
    public var unsupported: [URL] = []
    /// Folders and files the walk could not read — no permission, or gone
    /// mid-walk. Nothing indexed from under them is removed: a folder that
    /// cannot be listed is not a folder whose files were deleted.
    public var unreadable: [URL] = []
    /// Files a processor failed on. What was indexed from them before stays,
    /// and the next sync tries them again.
    public var failed: [Failure] = []
    public var indexing = IndexingReport()
}

extension SemanticSearchEngine {

    /// The id a synced file is indexed under: its file URL.
    ///
    /// An agent or an app that finds a hit can open it without a lookup, and
    /// the files of one folder share a prefix — which is how a sync finds the
    /// ones that are gone.
    public static func itemID(forFile url: URL) -> ItemID {
        ItemID(url.standardizedFileURL.absoluteString)
    }

    /// Items a filter admits, without a query.
    ///
    /// "Every PDF", "photos taken in June, newest first", "what is in this
    /// folder" — the filter is the one search takes, compiled the same way.
    /// An item is admitted when the filter holds for it with any of its
    /// representations, as search would find it through any of them.
    /// `topLevelOnly` lists files rather than their chunks, frames and
    /// utterances; representations are left out unless asked for, since a
    /// file's can hold its whole text.
    public func listItems(
        matching filter: SearchFilter = .all,
        topLevelOnly: Bool = true,
        order: ItemListing.Order = .id,
        limit: Int = 100,
        offset: Int = 0,
        includesRepresentations: Bool = false
    ) throws -> ItemListing {
        let (stored, total) = try store.listItems(
            filter: filter, topLevelOnly: topLevelOnly, order: order, limit: limit,
            offset: offset)
        let items = try stored.map { item in
            SearchItem(
                id: item.id, parentID: item.parentID, metadata: item.metadata,
                locator: item.locator,
                representations: includesRepresentations
                    ? try store.representations(forItem: item.id).map {
                        ContentRepresentation(
                            id: $0.representationID, kind: $0.kind, content: $0.content,
                            language: $0.language, provenance: $0.provenance,
                            locator: $0.locator, derivedFrom: $0.derivedFrom, weight: $0.weight)
                    } : [],
                ordinal: item.ordinal)
        }
        return ItemListing(items: items, total: total)
    }

    /// Bring the index in line with a folder: ingest what is new or changed,
    /// remove what is gone.
    ///
    /// Each file is ingested on its own under ``itemID(forFile:)``, so one
    /// unreadable file is reported in ``FolderSyncReport/failed`` rather than
    /// ending the sync, and an unchanged file costs a `stat` — ingest's
    /// fingerprint is the file's size and modification date. Every item a
    /// file produces carries `folders`: every folder the file is in, as
    /// absolute paths, so `.equals("folders", "/Users/me/Photos/Trips")` lists
    /// a subfolder. They depend on the file's path alone, not on the folder
    /// synced — syncing `Photos/Trips` and later `Photos` writes the same
    /// value, which an unchanged file, skipped by ingest, could not update.
    /// `folderPath` — the folders between `folder` and the file — does depend
    /// on it, so an unchanged file whose recorded path differs from the walk's
    /// has only its path-dependent parts redone (``ContentProcessor/relocate``).
    /// Sync each file under one folder: a file synced under two, a folder and
    /// its parent, is redone on every sync of either.
    ///
    /// Removal happens only after the whole folder was walked, and never
    /// under what the walk could not read (``FolderSyncReport/unreadable``);
    /// a folder that is not there throws instead. An unmounted drive or a
    /// folder without permission must not look like one whose files were
    /// all deleted.
    ///
    /// A file moved or renamed inside the folder keeps what was indexed from
    /// it (``FolderSyncOptions/detectsMoves``): a file at a new path whose
    /// SHA-256 is that of one the walk no longer finds takes over its items —
    /// slots, vectors and all — and nothing is processed or embedded again
    /// but what depends on the path. Only files indexed with a hash, and only
    /// within one synced folder: a move between two folders synced apart is
    /// a removal from one and an addition to the other.
    @discardableResult
    public func sync(
        folder: URL,
        options: FolderSyncOptions = FolderSyncOptions(),
        onProgress: (@Sendable (FolderSyncProgress) -> Void)? = nil
    ) async throws -> FolderSyncReport {
        guard let processor = configuration.processor else {
            throw SemanticSearchError.configuration(
                "sync requires a ContentProcessor to turn files into items")
        }
        var isDirectory: ObjCBool = false
        let root = URL(fileURLWithPath: folder.standardizedFileURL.path, isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw SemanticSearchError.configuration(
                "\(root.path) is not a folder that exists; nothing was removed")
        }

        let walk = Self.walk(root, options: options, excluding: storeFiles())
        let files = walk.files
        var report = FolderSyncReport()
        report.unreadable = walk.unreadable
        // A file URL, or a folder's without its trailing slash.
        let unreadable = walk.unreadable.map { url in
            let string = url.absoluteString
            return string.hasSuffix("/") ? String(string.dropLast()) : string
        }
        func underUnreadable(_ id: ItemID) -> Bool {
            unreadable.contains { id.rawValue == $0 || id.rawValue.hasPrefix($0 + "/") }
        }

        var contents: [RawContent] = []
        for file in files {
            let content = RawContent(
                id: Self.itemID(forFile: file), payload: .fileURL(file),
                metadata: [
                    "folders": .stringList(Self.folders(of: file)),
                    Self.folderPathKey: .string(Self.folderPath(of: file, in: root)),
                ])
            if processor.canProcess(content) {
                contents.append(content)
            } else {
                report.unsupported.append(file)
            }
        }
        let present = Set(contents.map(\.id))

        // Where a moved file may have come from: files indexed under this
        // folder that the walk did not find, by the hash of their bytes —
        // and only those processed as the processor is configured now, or
        // the move would carry old output under a fingerprint that says it
        // is current.
        var departed: [String: [ItemID]] = [:]
        if options.detectsMoves {
            for id in try store.topLevelItemIDs(withPrefix: root.absoluteString)
            where !present.contains(id) && !underUnreadable(id) {
                guard case .string(let hash)? = try store.item(id)?.metadata[Self.contentHashKey],
                    let state = try store.processingState(
                        itemID: id, processor: processor.identifier),
                    state.processor == processor.configurationFingerprint
                else { continue }
                departed[hash, default: []].append(id)
            }
        }

        for (index, original) in contents.enumerated() {
            try Task.checkCancellation()
            var content = original
            let id = content.id
            guard case .fileURL(let file) = content.payload else { continue }
            let stored = try store.item(id)
            let existed = stored != nil
            let total = contents.count
            onProgress?(FolderSyncProgress(file: file, completed: index, total: total))
            do {
                let current = try isCurrent(content, processor: processor)
                // Unchanged, but seen from another folder — a parent added
                // to the library — or indexed before the path was recorded:
                // only what depends on the path is redone, as for a move.
                if current, let stored,
                    stored.metadata[Self.folderPathKey] != content.metadata[Self.folderPathKey]
                {
                    report.indexing.merge(
                        try await relocate(id, to: content, processor: processor))
                    report.unchanged += 1
                    continue
                }
                // Hashed only when it will be processed: an unchanged file
                // still costs a `stat` and nothing more.
                if options.detectsMoves, !current, let hash = try Self.contentHash(of: file) {
                    content.metadata[Self.contentHashKey] = .string(hash)
                    if !existed, let from = departed[hash]?.first {
                        departed[hash]?.removeFirst()
                        report.indexing.merge(
                            try await relocate(from, to: content, processor: processor))
                        report.moved.append(FolderSyncReport.Move(from: from, to: id))
                        continue
                    }
                }
                let indexed = try await ingest([content]) { stage in
                    onProgress?(
                        FolderSyncProgress(
                            file: file, completed: index, total: total, stage: stage))
                }
                report.indexing.merge(indexed)
                if indexed.itemsUnchanged > 0 {
                    report.unchanged += 1
                } else if existed {
                    report.updated.append(id)
                } else {
                    report.added.append(id)
                }
            } catch {
                // A processor may wrap the cancellation it was stopped by.
                if Task.isCancelled { throw CancellationError() }
                report.failed.append(
                    FolderSyncReport.Failure(file: file, reason: String(describing: error)))
            }
        }

        if options.removesMissingFiles {
            try Task.checkCancellation()
            // A moved file's old id is gone already: renamed, not removed.
            for id in try store.topLevelItemIDs(withPrefix: root.absoluteString)
            where !present.contains(id) && !underUnreadable(id) {
                try await remove(id, includingDescendants: true)
                report.removed.append(id)
            }
        }
        return report
    }

    /// The SHA-256 of a file's bytes, on every item of a synced file.
    public static let contentHashKey = "contentHash"

    /// The folders between the synced folder and the file — `Trips/Fethiye
    /// 2022` — on every item of a synced file; empty at the top.
    public static let folderPathKey = "folderPath"

    static func folderPath(of file: URL, in root: URL) -> String {
        let parent = file.deletingLastPathComponent().standardizedFileURL.path
        let top = root.standardizedFileURL.path
        guard parent.hasPrefix(top + "/") else { return "" }
        return String(parent.dropFirst(top.count + 1))
    }

    /// Whether ingest would skip this content: processed before, from the
    /// same file, under the same processor configuration.
    func isCurrent(_ content: RawContent, processor: any ContentProcessor) throws -> Bool {
        guard
            let state = try store.processingState(
                itemID: content.id, processor: processor.identifier)
        else { return false }
        return state.content == Self.fingerprint(of: content)
            && state.processor == processor.configurationFingerprint
    }

    /// Move what was indexed from `old` to `content`'s id: renamed in the
    /// store, slots and vectors kept; then the processor updates what
    /// depends on the path, and only a representation whose content that
    /// changed — the name among a file's words — is embedded again.
    func relocate(
        _ old: ItemID, to content: RawContent, processor: any ContentProcessor
    ) async throws -> IndexingReport {
        guard let oldURL = URL(string: old.rawValue) else {
            throw SemanticSearchError.configuration("\(old) is not a file URL")
        }
        if old != content.id { try store.renameTree(from: old, to: content.id) }
        let items = try listItems(
            matching: .withinParent(content.id), topLevelOnly: false, limit: 1_000_000,
            includesRepresentations: true
        ).items
        let report = try await upsert(
            try await processor.relocate(items, from: oldURL, to: content))
        try store.recordProcessingState(
            itemID: content.id, processor: processor.identifier,
            contentFingerprint: Self.fingerprint(of: content),
            processorFingerprint: processor.configurationFingerprint)
        return report
    }

    /// SHA-256, read a megabyte at a time into one buffer: `FileHandle`'s
    /// `Data` chunks are autoreleased, and a 4 GB video would hold them all
    /// until the sync returned.
    static func contentHash(of url: URL) throws -> String? {
        #if canImport(CryptoKit)
            // Darwin's: the engine has a `close` of its own.
            let descriptor = Darwin.open(url.path, O_RDONLY)
            guard descriptor >= 0 else {
                throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
            }
            defer { Darwin.close(descriptor) }
            var hasher = SHA256()
            var buffer = [UInt8](repeating: 0, count: 1 << 20)
            while true {
                let count = buffer.withUnsafeMutableBytes {
                    Darwin.read(descriptor, $0.baseAddress, $0.count)
                }
                guard count >= 0 else {
                    throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
                }
                if count == 0 { break }
                buffer.withUnsafeBytes {
                    hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0.prefix(count)))
                }
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        #else
            return nil
        #endif
    }

    /// The index's own files, which may sit inside the folder being synced.
    private func storeFiles() -> (paths: Set<String>, directory: String?) {
        var paths: Set<String> = []
        if let database = store.databaseURL {
            for suffix in ["", "-wal", "-shm", "-journal"] {
                paths.insert(URL(fileURLWithPath: database.path + suffix).resolvingSymlinksInPath().path)
            }
        }
        return (paths, store.snapshotDirectory?.resolvingSymlinksInPath().path)
    }

    /// Regular files under `root`, sorted by path, as URLs built on `root` —
    /// so their ids share its prefix whatever the file system resolves — and
    /// what could not be read.
    ///
    /// Walked by hand rather than with `FileManager`'s enumerators, which pass
    /// over a folder they cannot open without saying so; a sync that believed
    /// them would remove everything indexed from it.
    static func walk(
        _ root: URL, options: FolderSyncOptions,
        excluding store: (paths: Set<String>, directory: String?)
    ) -> (files: [URL], unreadable: [URL]) {
        var files: [URL] = []
        var unreadable: [URL] = []
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey, .isDirectoryKey, .isPackageKey, .isHiddenKey,
        ]
        var pending = [root]
        while let directory = pending.popLast() {
            let names: [String]
            do {
                names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            } catch {
                unreadable.append(directory)
                continue
            }
            for name in names {
                if !options.includesHiddenFiles, name.hasPrefix(".") { continue }
                let entry = directory.appendingPathComponent(name, isDirectory: false)
                let values: URLResourceValues
                do {
                    values = try entry.resourceValues(forKeys: keys)
                } catch {
                    unreadable.append(entry)
                    continue
                }
                if !options.includesHiddenFiles, values.isHidden == true { continue }
                if values.isDirectory == true {
                    if values.isPackage == true, !options.descendsIntoPackages { continue }
                    pending.append(directory.appendingPathComponent(name, isDirectory: true))
                    continue
                }
                guard values.isRegularFile == true else { continue }
                if let extensions = options.extensions,
                    !extensions.contains(entry.pathExtension.lowercased())
                {
                    continue
                }
                let resolved = entry.resolvingSymlinksInPath().path
                if store.paths.contains(resolved) { continue }
                if let directory = store.directory, resolved.hasPrefix(directory + "/") {
                    continue
                }
                files.append(entry)
            }
        }
        return (files.sorted { $0.path < $1.path }, unreadable.sorted { $0.path < $1.path })
    }

    /// Every folder the file is in, from the top down; `/` itself left out.
    static func folders(of file: URL) -> [String] {
        var folders: [String] = []
        var current = ""
        for component in file.deletingLastPathComponent().path.split(separator: "/") {
            current += "/" + component
            folders.append(current)
        }
        return folders
    }
}

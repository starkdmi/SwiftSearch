import Foundation
import Hub
import MLXLMCommon
import Tokenizers

// mlx-swift-lm 3 loads through a `Downloader` and a `TokenizerLoader` instead
// of calling swift-transformers itself. These adapt the same `HubApi` and
// tokenizer 2.x used, so downloads stay where they were —
// `<downloadBase>/models/<org>/<name>` — and nothing already on disk is
// fetched again. Its own `MLXHuggingFace` macros download through
// swift-huggingface's `HubClient` into the Hugging Face cache layout instead.

extension ModelReference {
    /// The runtime's configuration: the directory, or the repository at its
    /// pinned revision.
    var runtimeConfiguration: MLXLMCommon.ModelConfiguration {
        if let url = directoryURL { return MLXLMCommon.ModelConfiguration(directory: url) }
        return MLXLMCommon.ModelConfiguration(id: hubID ?? identity, revision: revision ?? "main")
    }
}

/// Hub downloads through `HubApi.snapshot`. A pinned commit already complete
/// on disk is used without asking the Hub; anything else is checked online,
/// and the copy on disk is used when the Hub cannot be reached.
///
/// The `HubApi` must be made with `useOfflineMode: false`: its own offline
/// mode returns whatever copy it has, so a snapshot that returned could not
/// be told from one that fetched, and a copy recorded complete could be one
/// an interrupted download left.
struct HubDownloader: MLXLMCommon.Downloader {
    let hub: HubApi

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let repo = Hub.Repo(id: id)
        let local = hub.localRepoLocation(repo)
        // A commit cannot change, so a copy recorded complete at it needs no
        // network — not even a request that would time out offline.
        if let revision, Self.isCommit(revision),
            Self.isComplete(revision, at: local, matching: patterns)
        {
            return local
        }
        let directory: URL
        var fetched = false
        do {
            directory = try await hub.snapshot(
                from: repo, revision: revision ?? "main", matching: patterns,
                progressHandler: progressHandler)
            fetched = true
        } catch {
            guard let cached = Self.cachedFallback(after: error, at: local) else { throw error }
            directory = cached
        }
        // Every return, not only the fallback.
        guard Self.holds(revision, at: directory, matching: patterns) else {
            throw RevisionNotOnDisk(id: id, revision: revision ?? "main")
        }
        if fetched, let revision, Self.isCommit(revision) {
            try? Self.recordComplete(revision, at: directory, matching: patterns)
        }
        return directory
    }

    static func isCommit(_ revision: String) -> Bool {
        revision.count == 40 && revision.allSatisfy(\.isHexDigit)
    }

    /// Where a fetched copy is recorded complete: the files a snapshot at
    /// `revision` with `patterns` left, which each still have to hold.
    static func completionRecord(_ revision: String, at directory: URL) -> URL {
        directory.appendingPathComponent(".cache/huggingface/complete-\(revision).json")
    }

    struct Completion: Codable {
        var patterns: [String]
        var files: [String]
    }

    static func recordComplete(_ revision: String, at directory: URL, matching patterns: [String])
        throws
    {
        let files = Self.files(in: directory).filter { path in
            patterns.contains { fnmatch($0, path, 0) == 0 }
        }
        let record = Completion(patterns: patterns.sorted(), files: files.sorted())
        try JSONEncoder().encode(record).write(
            to: completionRecord(revision, at: directory), options: .atomic)
    }

    /// A copy recorded complete at `revision` for these patterns, every file
    /// of it still there and still recorded at the commit.
    static func isComplete(_ revision: String, at directory: URL, matching patterns: [String])
        -> Bool
    {
        guard
            let data = try? Data(contentsOf: completionRecord(revision, at: directory)),
            let record = try? JSONDecoder().decode(Completion.self, from: data),
            record.patterns == patterns.sorted(), !record.files.isEmpty
        else { return false }
        let present = Set(Self.files(in: directory))
        return record.files.allSatisfy(present.contains)
            && holds(revision, at: directory, matching: patterns)
    }

    struct RevisionNotOnDisk: LocalizedError {
        let id: String
        let revision: String
        var errorDescription: String? {
            "\(id) on disk is not at \(revision), and the Hub could not be asked for it"
        }
    }

    /// True when every file the loader reads was downloaded at `revision`,
    /// or when no commit was pinned. A copy of another commit is not the
    /// model the reference names: loading it would put its vectors in the
    /// pinned space. HubApi records each file's commit in the first line of
    /// `.cache/huggingface/download/<file>.metadata`, and a file whose bytes
    /// did not change is recorded again at the new commit by the next online
    /// run. A file without a record proves nothing, so it fails the check.
    static func holds(_ revision: String?, at directory: URL, matching patterns: [String]) -> Bool {
        guard let revision, isCommit(revision) else { return true }
        let records = directory.appendingPathComponent(".cache/huggingface/download")
        let files = Self.files(in: directory).filter { path in
            patterns.contains { fnmatch($0, path, 0) == 0 }
        }
        return !files.isEmpty
            && files.allSatisfy { path in
                let record = records.appendingPathComponent(path + ".metadata")
                let commit = (try? String(contentsOf: record, encoding: .utf8))?
                    .split(separator: "\n", maxSplits: 1).first.map(String.init)
                return commit == revision
            }
    }

    /// Every regular file under `directory`, relative to it — the names
    /// HubApi matches its patterns against — outside its `.cache`.
    static func files(in directory: URL) -> [String] {
        guard let walk = FileManager.default.enumerator(atPath: directory.path) else { return [] }
        var files: [String] = []
        while let path = walk.nextObject() as? String {
            if path == ".cache" { walk.skipDescendants(); continue }
            if walk.fileAttributes?[.type] as? FileAttributeType == .typeRegular {
                files.append(path)
            }
        }
        return files
    }

    /// The copy already on disk, when the Hub refused or could not be
    /// reached. 2.x did the same: a repository made private, renamed or
    /// gated after it was downloaded answers 401/403, and a model that
    /// loaded yesterday must not fail today for that. Only a directory with
    /// files in it; otherwise the Hub's error says more than a missing
    /// `config.json` would. The caller still checks the copy's commit.
    static func cachedFallback(after error: any Error, at directory: URL) -> URL? {
        switch error {
        case Hub.HubClientError.authorizationRequired:
            return existing(directory)
        case let error as URLError where unreachable.contains(error.code):
            return existing(directory)
        case Hub.HubClientError.networkError(let error) where unreachable.contains(error.code):
            return existing(directory)
        default:
            return nil
        }
    }

    /// The Hub could not be asked: no network, no name, no answer. Not an
    /// answer it gave, such as a file that does not exist.
    static let unreachable: Set<URLError.Code> = [
        .notConnectedToInternet, .cannotFindHost, .dnsLookupFailed, .cannotConnectToHost,
        .timedOut, .networkConnectionLost, .dataNotAllowed, .internationalRoamingOff,
    ]

    private static func existing(_ directory: URL) -> URL? {
        let contents = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
        return contents?.isEmpty == false ? directory : nil
    }
}

/// swift-transformers' `PreTrainedTokenizer`, read from the model's directory —
/// the tokenizer mlx-swift-lm 2.x built.
struct TransformersTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        try await TokenizerCache.shared.tokenizer(in: directory)
    }
}

/// Tokenizers by model directory, kept when a provider releases its weights.
/// Under sequential residency every query loads its models again, and each
/// load parsed `tokenizer.json` again — Qwen3's 151k tokens and merges. One
/// instance can serve every load: swift-transformers never changes a built
/// tokenizer but for its chat-template cache, which it locks. The key carries
/// the size and date of the directory's files, so a local conversion
/// rewritten in place is read again.
actor TokenizerCache {
    static let shared = TokenizerCache()

    private struct Key: Hashable {
        let directory: String
        let files: [String]
    }

    private var loads: [Key: Task<TransformersTokenizer, any Error>] = [:]

    func tokenizer(in directory: URL) async throws -> TransformersTokenizer {
        let key = Key(directory: directory.standardizedFileURL.path, files: Self.stamps(directory))
        if let load = loads[key] { return try await load.value }
        // The directory's older tokenizer is replaced, not kept beside it.
        loads = loads.filter { $0.key.directory != key.directory }
        let load = Task {
            TransformersTokenizer(try await AutoTokenizer.from(modelFolder: directory))
        }
        loads[key] = load
        do {
            return try await load.value
        } catch {
            if loads[key] == load { loads[key] = nil }
            throw error
        }
    }

    /// Name, size and date of each file directly in `directory`.
    private static func stamps(_ directory: URL) -> [String] {
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: keys, options: .skipsHiddenFiles))
            ?? []
        return files.map { file in
            let values = try? file.resolvingSymlinksInPath().resourceValues(forKeys: Set(keys))
            let date = values?.contentModificationDate?.timeIntervalSinceReferenceDate ?? 0
            return "\(file.lastPathComponent):\(values?.fileSize ?? -1):\(date)"
        }.sorted()
    }
}

struct TransformersTokenizer: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

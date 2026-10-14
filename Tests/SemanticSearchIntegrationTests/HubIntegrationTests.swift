import Foundation
import Hub
import Network
import XCTest

@testable import SemanticSearchMLX

/// The downloader adapter against a Hub that refuses every request. Pure;
/// runs without a gate — the Hub is a listener on localhost.
final class HubIntegrationTests: XCTestCase {

    func testRefusedRepositoryLoadsFromItsDownloadedCopy() async throws {
        let hub = try await RefusingHub.start()
        defer { hub.stop() }
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let api = HubApi(
            downloadBase: base, cache: nil, endpoint: hub.endpoint, useOfflineMode: false)
        let downloader = HubDownloader(hub: api)
        func download() async throws -> URL {
            try await downloader.download(
                id: "org/model", revision: nil, matching: ["*.json"], useLatest: false,
                progressHandler: { _ in })
        }

        // Nothing on disk: the refusal is the error.
        do {
            _ = try await download()
            XCTFail("a refused repository with no local copy loaded")
        } catch Hub.HubClientError.authorizationRequired {
        }

        // Downloaded before the repository was made private: the copy loads.
        let local = api.localRepoLocation(Hub.Repo(id: "org/model"))
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: local.appendingPathComponent("config.json"))
        let resolved = try await download()
        XCTAssertEqual(resolved.standardizedFileURL, local.standardizedFileURL)
    }

    func testOnlyRefusalAndNoConnectionFallBack() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let offline = URLError(.notConnectedToInternet)

        // An empty directory is no copy.
        XCTAssertNil(HubDownloader.cachedFallback(after: offline, at: directory))

        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        XCTAssertEqual(HubDownloader.cachedFallback(after: offline, at: directory), directory)
        XCTAssertEqual(
            HubDownloader.cachedFallback(
                after: Hub.HubClientError.authorizationRequired, at: directory),
            directory)
        // No answer is no network: the copy, whose commit the caller checks.
        XCTAssertEqual(
            HubDownloader.cachedFallback(after: URLError(.timedOut), at: directory), directory)
        XCTAssertEqual(
            HubDownloader.cachedFallback(after: URLError(.cannotFindHost), at: directory),
            directory)
        // An answer is not: a file the Hub says does not exist.
        XCTAssertNil(
            HubDownloader.cachedFallback(after: URLError(.badServerResponse), at: directory))
        XCTAssertNil(
            HubDownloader.cachedFallback(after: Hub.HubClientError.fileNotFound("x"), at: directory))
    }

    func testAPinnedReferenceLoadsOnlyItsOwnCommit() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let pinned = String(repeating: "a", count: 40)
        let other = String(repeating: "b", count: 40)
        let patterns = ["*.json", "*.safetensors"]
        try Self.copy(at: directory, files: ["config.json": pinned, "model.safetensors": nil])
        func holds(_ revision: String?) -> Bool {
            HubDownloader.holds(revision, at: directory, matching: patterns)
        }

        XCTAssertTrue(holds(nil))
        XCTAssertTrue(holds("main"))
        // Weights without a record: config.json's alone proves nothing.
        XCTAssertFalse(holds(pinned))

        try Self.copy(at: directory, files: ["model.safetensors": other])
        XCTAssertFalse(holds(pinned))
        try Self.copy(at: directory, files: ["model.safetensors": pinned])
        XCTAssertTrue(holds(pinned))
        // A file the loader does not read is not asked about.
        try Self.copy(at: directory, files: ["README.md": nil])
        XCTAssertTrue(holds(pinned))
    }

    /// Offline, HubApi returns the copy it has without asking which commit
    /// it is; the downloader must not.
    func testOfflineCopyOfAnotherCommitIsRefused() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let api = HubApi(downloadBase: base, cache: nil, useOfflineMode: true)
        let downloader = HubDownloader(hub: api)
        let local = api.localRepoLocation(Hub.Repo(id: "org/model"))
        let pinned = String(repeating: "a", count: 40)
        let other = String(repeating: "b", count: 40)
        func download() async throws -> URL {
            try await downloader.download(
                id: "org/model", revision: pinned, matching: ["*.json", "*.safetensors"],
                useLatest: false, progressHandler: { _ in })
        }

        try Self.copy(at: local, files: ["config.json": other, "model.safetensors": other])
        do {
            _ = try await download()
            XCTFail("an offline copy of another commit loaded")
        } catch is HubDownloader.RevisionNotOnDisk {
        }

        try Self.copy(at: local, files: ["config.json": pinned, "model.safetensors": pinned])
        let resolved = try await download()
        XCTAssertEqual(resolved.standardizedFileURL, local.standardizedFileURL)
    }

    /// A pinned commit recorded complete loads without a request; one with a
    /// file gone asks the Hub.
    func testACompletePinnedCopyNeedsNoNetwork() async throws {
        let hub = try await RefusingHub.start()
        defer { hub.stop() }
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let api = HubApi(
            downloadBase: base, cache: nil, endpoint: hub.endpoint, useOfflineMode: false)
        let downloader = HubDownloader(hub: api)
        let local = api.localRepoLocation(Hub.Repo(id: "org/model"))
        let pinned = String(repeating: "a", count: 40)
        let patterns = ["*.json", "*.safetensors"]
        func download() async throws -> URL {
            try await downloader.download(
                id: "org/model", revision: pinned, matching: patterns, useLatest: false,
                progressHandler: { _ in })
        }

        try Self.copy(at: local, files: ["config.json": pinned, "model.safetensors": pinned])
        try HubDownloader.recordComplete(pinned, at: local, matching: patterns)
        let resolved = try await download()
        XCTAssertEqual(resolved.standardizedFileURL, local.standardizedFileURL)
        XCTAssertEqual(hub.requests, 0)

        // Other patterns are another download.
        XCTAssertFalse(HubDownloader.isComplete(pinned, at: local, matching: ["*.json"]))
        try FileManager.default.removeItem(at: local.appendingPathComponent("model.safetensors"))
        XCTAssertFalse(HubDownloader.isComplete(pinned, at: local, matching: patterns))
        _ = try? await download()
        XCTAssertGreaterThan(hub.requests, 0)
    }

    /// Files in a downloaded copy, each with HubApi's record of the commit it
    /// came from, or none.
    static func copy(at directory: URL, files: [String: String?]) throws {
        let records = directory.appendingPathComponent(".cache/huggingface/download")
        try FileManager.default.createDirectory(at: records, withIntermediateDirectories: true)
        for (name, commit) in files {
            try Data("{}".utf8).write(to: directory.appendingPathComponent(name))
            let record = records.appendingPathComponent("\(name).metadata")
            if let commit {
                try Data("\(commit)\netag\n0\n".utf8).write(to: record)
            } else {
                try? FileManager.default.removeItem(at: record)
            }
        }
    }

    func testAReferenceNamesItsRevision() {
        let pinned = ModelReference("org/model@0123abc")
        XCTAssertEqual(pinned.hubID, "org/model")
        XCTAssertEqual(pinned.revision, "0123abc")
        XCTAssertEqual(pinned.identity, "org/model@0123abc")
        XCTAssertNotEqual(pinned.identity, ModelReference("org/model").identity)

        let unpinned = ModelReference("org/model")
        XCTAssertEqual(unpinned.hubID, "org/model")
        XCTAssertNil(unpinned.revision)
        XCTAssertEqual(unpinned.runtimeConfiguration.name, "org/model")

        for model in [
            ModelProfile.Text.embeddingModel, ModelProfile.Text.rerankerModel,
            ModelProfile.Multimodal.embeddingModel, ModelProfile.Multimodal.rerankerModel,
            ModelProfile.Captioning.model, ModelProfile.EmbeddingGemma.embeddingModel,
        ] {
            XCTAssertEqual(model.revision?.count, 40, "\(model.identity) is not pinned")
        }
    }
}

/// Answers every request with 401, as the Hub does for a private repository.
private final class RefusingHub: @unchecked Sendable {
    private let listener: NWListener
    private let counter: Counter
    let endpoint: String

    /// Connections it has answered.
    var requests: Int { counter.value }

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        var value: Int { lock.withLock { count } }
        func increment() { lock.withLock { count += 1 } }
    }

    private init(listener: NWListener, port: UInt16, counter: Counter) {
        self.listener = listener
        self.counter = counter
        endpoint = "http://127.0.0.1:\(port)"
    }

    static func start() async throws -> RefusingHub {
        let listener = try NWListener(using: .tcp, on: .any)
        let counter = Counter()
        listener.newConnectionHandler = { connection in
            counter.increment()
            connection.start(queue: .global())
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { _, _, _, _ in
                let response = "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                connection.send(
                    content: Data(response.utf8),
                    completion: .contentProcessed { _ in connection.cancel() })
            }
        }
        let port: UInt16 = try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    listener.stateUpdateHandler = nil
                    continuation.resume(returning: listener.port?.rawValue ?? 0)
                case .failed(let error):
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                default:
                    break
                }
            }
            listener.start(queue: .global())
        }
        return RefusingHub(listener: listener, port: port, counter: counter)
    }

    func stop() { listener.cancel() }
}

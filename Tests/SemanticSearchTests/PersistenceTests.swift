import Foundation
import Testing

@testable import SemanticSearch

@Suite("Persistence")
struct PersistenceTests {

    @Test("An index survives close and reopen")
    func reopenPreservesEverything() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
            try await engine.upsert([
                textItem("a", "hydrothermal vents on the ocean floor",
                    metadata: ["topic": .string("geology")]),
                textItem("b", "the history of the printing press"),
            ])
            try await engine.close()
        }

        let reopened = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
        #expect(try await reopened.itemCount() == 2)
        #expect(try await reopened.vectorCount(space: "test") == 2)

        let results = try await reopened.search("hydrothermal vents")
        #expect(results.hits.first?.itemID == "a")

        // Metadata and filters survive too, not just the text.
        let filtered = try await reopened.search(
            "hydrothermal",
            options: SearchOptions(filter: .equals(key: "topic", value: .string("geology"))))
        #expect(filtered.hits.count == 1)
        try await reopened.close()
    }

    @Test("A deleted item stays deleted after reopen")
    func deletionsSurviveReopen() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
            try await engine.upsert([textItem("a", "alpha"), textItem("b", "beta")])
            try await engine.remove("a")
            try await engine.close()
        }

        let reopened = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
        #expect(try await reopened.itemCount() == 1)
        #expect(try await reopened.vectorCount(space: "test") == 1)
        try await reopened.close()
    }

    @Test("A tombstone survives a snapshot taken before the deletion")
    func tombstonesSubtractFromAStaleSnapshot() async throws {
        // The failure this prevents: a graph snapshot published at generation
        // 5 still holds a key that was deleted at generation 6. Replaying the
        // tombstone on load is the only record that it must be subtracted.
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(
                storeURL: storeURL, indexKind: .hnsw)
            try await engine.upsert(
                (0..<50).map { textItem("item-\($0)", "document number \($0)") })
            // Publish a snapshot that includes every item.
            try await engine.rebuildVectorIndex(space: "test")
            try await engine.remove("item-7")
            try await engine.close()
        }

        let reopened = try await SemanticSearchEngine.testEngine(
            storeURL: storeURL, indexKind: .hnsw)
        let results = try await reopened.search(
            "document number 7", options: SearchOptions(limit: 50))
        #expect(!results.hits.contains { $0.itemID == "item-7" })
        try await reopened.close()
    }

    @Test("An interrupted index build is completed by reindex")
    func reindexRecoversAnInterruptedBuild() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        // Simulate the interruption by writing the rows without vectors, which
        // is the state a process killed mid-embed leaves behind.
        let store = try SearchStore(url: storeURL)
        let analyzer = LinguisticLexicalAnalyzer()
        let descriptor = await HashingEmbeddingProvider(dimensions: 64).descriptor
        try store.registerSpace(
            StoredSpace(
                id: "test", descriptor: descriptor, indexKind: .exact, dimensions: 64,
                reducedDimensions: nil, generation: 0, snapshotGeneration: 0,
                snapshotFile: nil, kinds: []))
        for index in 0..<20 {
            _ = try store.upsert(
                textItem("item-\(index)", "recovered document number \(index)"),
                owner: nil, analyzer: analyzer)
        }
        store.close()

        let engine = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
        #expect(try await engine.vectorCount(space: "test") == 0)

        let report = try await engine.reindex()
        #expect(report.vectorsWritten == 20)
        #expect(try await engine.vectorCount(space: "test") == 20)

        let results = try await engine.search(
            "recovered document", options: SearchOptions(mode: .denseOnly))
        #expect(!results.hits.isEmpty)
        try await engine.close()
    }

    @Test("A corrupt snapshot is discarded and the space rebuilds from SQLite")
    func corruptSnapshotFallsBackToTheDatabase() async throws {
        // SQLite is authoritative; a graph on disk is a derived artefact and
        // deleting or breaking it must only ever cost a rebuild.
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(
                storeURL: storeURL, indexKind: .hnsw)
            try await engine.upsert(
                (0..<40).map { textItem("item-\($0)", "document number \($0) about kelp") })
            try await engine.rebuildVectorIndex(space: "test")
            try await engine.close()
        }

        let snapshotDirectory = directory.url.appendingPathComponent("index.snapshots")
        let snapshots = try FileManager.default.contentsOfDirectory(
            at: snapshotDirectory, includingPropertiesForKeys: nil)
        #expect(!snapshots.isEmpty)
        for file in snapshots {
            try Data("not a usearch index".utf8).write(to: file)
        }

        let reopened = try await SemanticSearchEngine.testEngine(
            storeURL: storeURL, indexKind: .hnsw)
        #expect(try await reopened.vectorCount(space: "test") == 40)
        let results = try await reopened.search(
            "document about kelp", options: SearchOptions(mode: .denseOnly))
        #expect(!results.hits.isEmpty)
        try await reopened.close()
    }

    @Test("Deleting the snapshot directory costs a rebuild and no data")
    func missingSnapshotIsRebuilt() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(
                storeURL: storeURL, indexKind: .hnsw)
            try await engine.upsert(
                (0..<30).map { textItem("item-\($0)", "document \($0)") })
            try await engine.close()
        }

        try? FileManager.default.removeItem(
            at: directory.url.appendingPathComponent("index.snapshots"))

        let reopened = try await SemanticSearchEngine.testEngine(
            storeURL: storeURL, indexKind: .hnsw)
        #expect(try await reopened.itemCount() == 30)
        try await reopened.close()
    }

    @Test("Vector blobs round-trip bit for bit")
    func vectorBlobsRoundTrip() throws {
        let store = try SearchStore.inMemory()
        let descriptor = EmbeddingSpaceDescriptor(
            modelIdentity: "m", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "last", normalized: true, dimensions: 4)
        try store.registerSpace(
            StoredSpace(
                id: "s", descriptor: descriptor, indexKind: .exact, dimensions: 4,
                reducedDimensions: nil, generation: 0, snapshotGeneration: 0,
                snapshotFile: nil, kinds: []))
        _ = try store.upsert(
            textItem("a", "text"), owner: nil,
            analyzer: LinguisticLexicalAnalyzer())

        let original: [Float] = [0.5, -0.25, 1e-8, -3.402_823e38]
        _ = try store.writeVectors(space: "s", vectors: [(slot: 1, vector: original)])
        #expect(try store.vector(space: "s", slot: 1) == original)
    }

    @Test("An index size is reported")
    func indexSizeIsReported() async throws {
        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine.testEngine(storeURL: directory.storeURL)
        try await engine.upsert((0..<100).map { textItem("item-\($0)", "document \($0)") })
        try await engine.flush()
        #expect(await engine.indexSizeBytes() > 0)
        try await engine.close()
    }

    @Test("An in-memory index needs no files")
    func inMemoryStore() async throws {
        let engine = try await SemanticSearchEngine.testEngine(storeURL: nil)
        try await engine.upsert([textItem("a", "in memory only")])
        #expect(try await engine.search("in memory").hits.count == 1)
        #expect(await engine.indexSizeBytes() == 0)
    }
}

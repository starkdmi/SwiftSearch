import Foundation
import Testing

@testable import SemanticSearch
@testable import SemanticSearchShots

/// The segmenter around the model, without one: what it is called, when its
/// segments count as new, and where it looks for its files.
@Suite("OmniShotCut segmenter")
struct OmniShotCutSegmenterTests {

    private func fingerprint(_ configuration: OmniShotCutSegmenter.Configuration) -> String {
        OmniShotCutSegmenter(configuration: configuration).configurationFingerprint
    }

    @Test("Every setting that changes the segments changes the fingerprint")
    func fingerprintCoversTheSegments() {
        let local = URL(fileURLWithPath: "/tmp/omnishotcut-models", isDirectory: true)
        let configurations: [OmniShotCutSegmenter.Configuration] = [
            .init(),
            .init(mode: .fp32),
            .init(mode: .exact()),
            .init(mode: .exact(threshold: 0.25)),
            .init(fastPrediction: false),
            .init(overlap: 20),
            .init(modelDirectory: local),
            .init(modelDirectory: local.appendingPathComponent("other")),
        ]
        let fingerprints = configurations.map(fingerprint)
        #expect(Set(fingerprints).count == configurations.count)
        #expect(fingerprint(.init()) == fingerprint(.init()))
        // Where downloads land is not what they are.
        #expect(
            fingerprint(.init(downloadBase: URL(fileURLWithPath: "/tmp/elsewhere")))
                == fingerprint(.init()))
    }

    @Test("The identity is the repository run on every window, or the local directory's path")
    func identity() {
        #expect(OmniShotCutSegmenter().identifier == "starkdmi/OmniShotCut-v1.5-coreml-fp16")
        #expect(
            OmniShotCutSegmenter(configuration: .init(mode: .exact())).identifier
                == "starkdmi/OmniShotCut-v1.5-coreml-fp16")
        #expect(
            OmniShotCutSegmenter(configuration: .init(mode: .fp32)).identifier
                == "starkdmi/OmniShotCut-v1.5-coreml-fp32")
        let local = URL(fileURLWithPath: "/tmp/models/../omnishotcut", isDirectory: true)
        #expect(
            OmniShotCutSegmenter(configuration: .init(modelDirectory: local)).identifier
                == "/tmp/omnishotcut")
    }

    @Test("Downloads land under models/<org>/<name>, as every other model's")
    func downloadDirectory() {
        let base = URL(fileURLWithPath: "/tmp/downloads", isDirectory: true)
        #expect(
            OmniShotCutModelStore.directory(.fp16, base: base).path
                == "/tmp/downloads/models/starkdmi/OmniShotCut-v1.5-coreml-fp16")
        #expect(
            OmniShotCutModelStore.directory(.fp32, base: base).path
                == "/tmp/downloads/models/starkdmi/OmniShotCut-v1.5-coreml-fp32")
    }

    @Test("A local directory's compiled model is preferred to its package")
    func localLookup() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnishotcut-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in [
            "OmniShotCut-v1.5-fp16.mlpackage", "OmniShotCut-v1.5-fp16.mlmodelc",
            "OmniShotCut-v1.5-fp32.mlpackage",
        ] {
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        let configuration = OmniShotCutSegmenter.Configuration(modelDirectory: directory)
        let fp16 = try await OmniShotCutSegmenter.model(.fp16, configuration: configuration)
        let fp32 = try await OmniShotCutSegmenter.model(.fp32, configuration: configuration)
        #expect(fp16.lastPathComponent == "OmniShotCut-v1.5-fp16.mlmodelc")
        #expect(fp32.lastPathComponent == "OmniShotCut-v1.5-fp32.mlpackage")
    }

    @Test("A directory holding what `hf download` writes is read too")
    func downloadedLayout() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnishotcut-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in [
            "OmniShotCut-v1.5-coreml-fp16/OmniShotCut-v1.5-fp16.mlpackage",
            "OmniShotCut-v1.5-coreml-fp32/OmniShotCut-v1.5-fp32.mlpackage",
        ] {
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        let configuration = OmniShotCutSegmenter.Configuration(modelDirectory: directory)
        let fp16 = try await OmniShotCutSegmenter.model(.fp16, configuration: configuration)
        let fp32 = try await OmniShotCutSegmenter.model(.fp32, configuration: configuration)
        #expect(
            fp16.deletingLastPathComponent().lastPathComponent
                == "OmniShotCut-v1.5-coreml-fp16")
        #expect(fp32.lastPathComponent == "OmniShotCut-v1.5-fp32.mlpackage")
    }

    @Test("A local directory without the model fails to prepare, and downloads nothing")
    func missingLocalModel() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnishotcut-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let segmenter = OmniShotCutSegmenter(
            configuration: .init(modelDirectory: directory, mode: .exact()))
        await #expect {
            try await segmenter.prepare()
        } throws: { error in
            guard case SemanticSearchError.modelUnavailable(let model, _) = error else {
                return false
            }
            return model == directory.standardizedFileURL.path
        }
        #expect(await !segmenter.isPrepared)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    @Test("A missing video is reported as missing media before any model loads")
    func missingVideo() async {
        let segmenter = OmniShotCutSegmenter(
            configuration: .init(modelDirectory: URL(fileURLWithPath: "/nonexistent")))
        let video = URL(fileURLWithPath: "/nonexistent/video.mp4")
        await #expect {
            _ = try await segmenter.segments(in: video)
        } throws: { error in
            guard case SemanticSearchError.mediaUnavailable(let url, _) = error else {
                return false
            }
            return url == video
        }
        #expect(await !segmenter.isPrepared)
    }

    @Test("A model compiled from the pinned revision is used as it is")
    func compiledRevisionIsReused() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnishotcut-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let compiled = OmniShotCutModelStore.directory(.fp16, base: base)
            .appendingPathComponent("OmniShotCut-v1.5-fp16.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        try OmniShotCutModelStore.Conversion.fp16.revision.write(
            to: compiled.appendingPathComponent(OmniShotCutModelStore.stampName),
            atomically: true, encoding: .utf8)
        // No package beside it: reaching for one would download or fail.
        let found = try await OmniShotCutModelStore.compiledModel(.fp16, base: base)
        #expect(found.standardizedFileURL == compiled.standardizedFileURL)
    }
}

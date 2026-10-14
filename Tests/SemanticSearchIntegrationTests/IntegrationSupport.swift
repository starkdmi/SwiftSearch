import Foundation
import SemanticSearch
import SemanticSearchMLX
import XCTest

/// Shared setup for the model-backed suites.
///
/// These are XCTest rather than swift-testing on purpose: they are run through
/// `xcodebuild build-for-testing` and `xcrun xctest`, which is the only path
/// that compiles MLX's Metal kernels. `swift test` produces a bundle that fails
/// on the first MLX call with "Failed to load the default metallib".
///
/// Every test here is opt-in. They download gigabytes of weights and take
/// minutes, so an unset environment skips them rather than failing a machine
/// that never asked to run them.
enum Integration {

    /// Gate for tests that load a text model.
    static var runsModelTests: Bool {
        flag("SEMANTIC_SEARCH_RUN_INTEGRATION_TESTS")
    }

    /// Gate for tests that load the 2B multimodal pair — several gigabytes
    /// resident, separate from the text gate so a machine can run one and not
    /// the other.
    static var runsMultimodalTests: Bool {
        flag("SEMANTIC_SEARCH_RUN_MULTIMODAL_TESTS")
    }

    /// Gate for tests that download and run Parakeet.
    static var runsASRTests: Bool {
        flag("SEMANTIC_SEARCH_RUN_ASR_TESTS")
    }

    /// Gate for tests that run the SmolVLM2 captioner (~500 MB).
    static var runsCaptionTests: Bool {
        flag("SEMANTIC_SEARCH_RUN_CAPTION_TESTS")
    }

    private static func flag(_ name: String) -> Bool {
        ["1", "true", "yes", "on"].contains(
            ProcessInfo.processInfo.environment[name]?.lowercased() ?? "")
    }

    /// Gate for tests that load EmbeddingGemma 2 on MLX (at bf16 ~0.54 GB
    /// text-only, ~0.88 GB with vision, ~1.5 GB with every tower).
    static var runsEmbeddingGemmaTests: Bool {
        flag("SEMANTIC_SEARCH_RUN_EMBEDDINGGEMMA_TESTS")
    }

    /// Gate for tests that run the Core ML conversion of EmbeddingGemma 2,
    /// read from ``coreMLEmbeddingGemmaDirectory``.
    static var runsCoreMLEmbeddingGemmaTests: Bool {
        flag("SEMANTIC_SEARCH_RUN_COREML_EMBEDDINGGEMMA_TESTS")
    }

    /// Model overrides, so a machine with local conversions does not have to
    /// re-download them.
    static var textEmbeddingModel: ModelReference {
        reference("SEMANTIC_SEARCH_TEXT_EMBEDDING_MODEL")
            ?? ModelProfile.Text.embeddingModel
    }

    static var textRerankerModel: ModelReference {
        reference("SEMANTIC_SEARCH_TEXT_RERANKER_MODEL") ?? ModelProfile.Text.rerankerModel
    }

    static var multimodalEmbeddingModel: ModelReference {
        reference("SEMANTIC_SEARCH_VL_EMBEDDING_MODEL")
            ?? ModelProfile.Multimodal.embeddingModel
    }

    static var multimodalRerankerModel: ModelReference {
        reference("SEMANTIC_SEARCH_VL_RERANKER_MODEL") ?? ModelProfile.Multimodal.rerankerModel
    }

    static var captionModel: ModelReference {
        reference("SEMANTIC_SEARCH_CAPTION_MODEL") ?? ModelProfile.Captioning.model
    }

    static var embeddingGemmaModel: ModelReference {
        reference("SEMANTIC_SEARCH_EMBEDDINGGEMMA_MODEL")
            ?? ModelProfile.EmbeddingGemma.embeddingModel
    }

    /// A local copy of `FluidInference/embeddinggemma-2-coreml`, from
    /// `SEMANTIC_SEARCH_COREML_EMBEDDINGGEMMA_DIRECTORY`; `nil` when unset.
    static var coreMLEmbeddingGemmaDirectory: URL? {
        guard
            let raw = ProcessInfo.processInfo
                .environment["SEMANTIC_SEARCH_COREML_EMBEDDINGGEMMA_DIRECTORY"]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty
        else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
    }

    private static func reference(_ name: String) -> ModelReference? {
        guard
            let raw = ProcessInfo.processInfo.environment[name]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty
        else { return nil }
        return ModelReference(raw)
    }

    /// A baseline to gate against, when one has been recorded.
    ///
    /// Point `SEMANTIC_SEARCH_BASELINE` at an `EvaluationBaseline` an earlier
    /// run wrote with `write(to:)`. Without one the suite still reports its
    /// numbers; it just has nothing to compare them to.
    static var baselineURL: URL? {
        guard let raw = ProcessInfo.processInfo.environment["SEMANTIC_SEARCH_BASELINE"],
            !raw.isEmpty
        else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
    }

    static func fixture(_ name: String, extension ext: String) throws -> URL {
        // `Bundle.module` is the generated accessor for the target's resource
        // bundle. `Bundle(for:)` finds the `.xctest` bundle instead, which is
        // its container and does not hold the resources — under `xcrun xctest`
        // that lookup silently misses and every fixture-backed test skips.
        if let url = Bundle.module.url(
            forResource: name, withExtension: ext, subdirectory: "Fixtures")
        {
            return url
        }
        if let url = Bundle.module.url(forResource: name, withExtension: ext) {
            return url
        }
        for candidate in Bundle.allBundles + [Bundle.main] {
            if let url = candidate.url(
                forResource: name, withExtension: ext, subdirectory: "Fixtures")
            {
                return url
            }
        }
        throw XCTSkip("fixture \(name).\(ext) is not in the test bundle")
    }

    static func multilingualCorpus() throws -> EvaluationCorpus {
        try EvaluationCorpus.load(from: try fixture("multilingual-corpus", extension: "json"))
    }

    /// A file of `Fixtures/library`, by its full name.
    static func library(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(
                forResource: (name as NSString).deletingPathExtension,
                withExtension: (name as NSString).pathExtension,
                subdirectory: "Fixtures/library")
        else { throw XCTSkip("fixture library/\(name) is not in the test bundle") }
        return url
    }

    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("semanticsearch-integration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A few seconds of sound written at test time, in a directory of its
    /// own: 16 kHz mono 16-bit PCM, five rising tones, then a falling chirp,
    /// each faded in and out.
    ///
    /// The audio towers are tested on whether a recording is embedded or
    /// refused, not on what it is of, so any well-formed sound will do — and
    /// one made here ships no recording of unknown licence. The fades keep it
    /// free of clicks.
    static func synthesisedAudio(seconds: Double = 5) throws -> URL {
        let sampleRate = 16_000
        let count = Int(seconds * Double(sampleRate))
        let toneCount = 5
        let tonesEnd = count * 3 / 4
        let toneLength = tonesEnd / toneCount
        var samples = [Int16](repeating: 0, count: count)
        var phase = 0.0
        for index in 0..<count {
            let frequency: Double
            let position: Double
            if index < tonesEnd {
                let tone = min(index / toneLength, toneCount - 1)
                frequency = 330 * pow(2, Double(tone) / 4)
                position = Double(index - tone * toneLength) / Double(toneLength)
            } else {
                position = Double(index - tonesEnd) / Double(count - tonesEnd)
                frequency = 1_200 - 900 * position
            }
            phase += 2 * Double.pi * frequency / Double(sampleRate)
            let envelope = sin(Double.pi * min(max(position, 0), 1))
            samples[index] = Int16(0.4 * envelope * sin(phase) * Double(Int16.max))
        }

        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        let payload = count * MemoryLayout<Int16>.size
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + payload))
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        append(UInt32(16))  // format chunk size
        append(UInt16(1))  // PCM
        append(UInt16(1))  // mono
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))  // bytes a second
        append(UInt16(2))  // bytes a frame
        append(UInt16(16))  // bits a sample
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(payload))
        for sample in samples { append(sample) }

        let url = try temporaryDirectory().appendingPathComponent("tones.wav")
        try data.write(to: url)
        return url
    }

    /// Print a labelled measurement: cold and warm latency, indexing
    /// throughput, index size and peak memory on a real machine.
    static func measure(_ label: String, _ values: [String: String]) {
        let body = values.keys.sorted().map { "\($0)=\(values[$0]!)" }.joined(separator: " ")
        print("[measure] \(label) \(body)")
    }

    static func peakFootprintMegabytes() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        // The high-water mark, not `phys_footprint`: that is the footprint at
        // the moment of the call, after models may already have been released.
        return Double(info.ledger_phys_footprint_peak) / 1_048_576
    }

    static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1e15
    }
}

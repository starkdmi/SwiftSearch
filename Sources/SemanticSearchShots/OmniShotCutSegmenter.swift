import CoreML
import Foundation
import OmniShotCutCoreML
import SemanticSearch

/// Shots and transitions with OmniShotCut v1.5 on Core ML, from the
/// [OmniShotCutCoreML](https://github.com/starkdmi/OmniShotCutCoreML) package.
///
/// Every frame is read, at the file's own rate: the model was trained on
/// native-rate clips, and at 12 fps it already misses cuts, more of them the
/// faster the edit. The package decodes as ffmpeg does, byte for byte — the
/// model moves cuts on a one-level brightness offset — and windows and stitches
/// as the official `engine.py` does; its own tests hold it to the PyTorch code
/// on three open movies. This type adds what the engine needs around it: the
/// model directory, residency and errors.
///
/// On an M4 the fp16 conversion on the Neural Engine is 3.6× faster a window
/// than the MLX port's default, in an eighth of the memory (the package's
/// README).
public actor OmniShotCutSegmenter: ShotSegmenter {

    /// Which conversion runs every window, and where.
    public enum Mode: Sendable, Hashable {
        /// The fp16 conversion on the Neural Engine: 218 of PyTorch's 219
        /// reference cuts on the same frame with the same labels on an M4, 216
        /// and 2 added on an M1 Pro. fp16 arithmetic moves logits by tenths and
        /// differs between chips, so which near-ties it decides the other way
        /// does too; every difference is a decision PyTorch in fp64 takes by
        /// less than 0.05, in a window ``exact(threshold:)`` runs again. 20 ms
        /// a window, 167 MB with the Neural Engine's program (M4).
        case fp16
        /// The fp32 conversion on the GPU: all 219, at 55 ms a window and
        /// ~490 MB.
        case fp32
        /// fp16 on the Neural Engine for every window, and fp32 on the GPU
        /// again for the windows where a decision the segments depend on is
        /// closer than `threshold` logits: all 219, at 27 ms a window with
        /// both models resident (~630 MB).
        case exact(threshold: Float = OmniShotCut.defaultThreshold)
    }

    public struct Configuration: Sendable, Hashable {
        /// A local conversion: a directory holding `OmniShotCut-v1.5-fp16`
        /// and `OmniShotCut-v1.5-fp32` (each `.mlmodelc` or `.mlpackage`, as
        /// the package's `scripts/convert.py` writes them), or only the one
        /// ``mode`` runs, directly or in the directories `hf download` writes
        /// (`OmniShotCut-v1.5-coreml-fp16`, `-fp32`). Used as it is and never
        /// downloaded into; its path is the model's identity. nil downloads
        /// the pinned revisions of `starkdmi/OmniShotCut-v1.5-coreml-fp16`
        /// (103 MB) and, for ``Mode/fp32`` and ``Mode/exact(threshold:)``,
        /// `-fp32` (198 MB) into `<downloadBase>/models/starkdmi/<repository>`.
        public var modelDirectory: URL?
        /// Base for downloads, as `SEMANTIC_SEARCH_MODEL_DIRECTORY` and
        /// `MLXRuntime.Configuration.modelDirectory`. nil reads
        /// `SEMANTIC_SEARCH_MODEL_DIRECTORY`, else `~/Library/Caches`, so
        /// `~/Library/Caches/models/…` as the MLX loaders' default.
        public var downloadBase: URL?
        public var mode: Mode
        /// Loads with `MLOptimizationHints.specializationStrategy`
        /// `.fastPrediction` on macOS 15 / iOS 18 and later: 20.3 ms a window
        /// on the Neural Engine against 25.4, a first load of 7.7 s against
        /// 3.9 (M4). Its outputs were bit-identical there; it is in the
        /// fingerprint all the same, as it compiles the Neural Engine program
        /// differently.
        public var fastPrediction: Bool
        /// Frames shared by neighbouring windows. The official default.
        public var overlap: Int

        public init(
            modelDirectory: URL? = nil,
            downloadBase: URL? = nil,
            mode: Mode = .fp16,
            fastPrediction: Bool = true,
            overlap: Int = 10
        ) {
            self.modelDirectory = modelDirectory
            self.downloadBase = downloadBase
            self.mode = mode
            self.fastPrediction = fastPrediction
            self.overlap = max(0, overlap)
        }
    }

    public let configuration: Configuration
    private var detector: OmniShotCut?

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    /// The local directory, or the repository of the conversion that runs
    /// every window.
    public nonisolated var identifier: String {
        configuration.modelDirectory?.standardizedFileURL.path
            ?? conversions[0].repository
    }

    /// The conversions ``Configuration/mode`` loads, the one run on every
    /// window first.
    nonisolated var conversions: [OmniShotCutModelStore.Conversion] {
        switch configuration.mode {
        case .fp16: return [.fp16]
        case .fp32: return [.fp32]
        case .exact: return [.fp16, .fp32]
        }
    }

    public nonisolated var configurationFingerprint: String {
        StableHash.of { hasher in
            // coreml.v1: the Core ML conversions.
            hasher.append("omnishotcut.coreml.v1")
            hasher.append(identifier)
            if configuration.modelDirectory == nil {
                for conversion in conversions { hasher.append(conversion.revision) }
            }
            switch configuration.mode {
            case .fp16: hasher.append("fp16")
            case .fp32: hasher.append("fp32")
            case .exact(let threshold):
                hasher.append("exact")
                hasher.append(Double(threshold))
            }
            hasher.append(configuration.fastPrediction)
            hasher.append(configuration.overlap)
        }
    }

    public var isPrepared: Bool { detector != nil }

    public func prepare() async throws {
        guard detector == nil else { return }
        let configuration = self.configuration
        do {
            func model(_ conversion: OmniShotCutModelStore.Conversion) async throws -> URL {
                try await Self.model(conversion, configuration: configuration)
            }
            switch configuration.mode {
            case .fp16:
                detector = try await OmniShotCut(
                    model: model(.fp16), computeUnits: .cpuAndNeuralEngine,
                    fastPrediction: configuration.fastPrediction,
                    overlap: configuration.overlap)
            case .fp32:
                detector = try await OmniShotCut(
                    model: model(.fp32), computeUnits: .cpuAndGPU,
                    fastPrediction: configuration.fastPrediction,
                    overlap: configuration.overlap)
            case .exact(let threshold):
                detector = try await OmniShotCut(
                    model: model(.fp16), computeUnits: .cpuAndNeuralEngine,
                    verifier: model(.fp32), threshold: threshold,
                    fastPrediction: configuration.fastPrediction,
                    overlap: configuration.overlap)
            }
        } catch {
            throw SemanticSearchError.modelUnavailable(identifier, underlying: error)
        }
    }

    public func release() async {
        detector = nil
    }

    public func segments(in video: URL) async throws -> [SemanticSearch.ShotSegment] {
        guard FileManager.default.fileExists(atPath: video.path) else {
            throw SemanticSearchError.mediaUnavailable(url: video, itemID: nil)
        }
        if detector == nil { try await prepare() }
        guard let detector else {
            throw SemanticSearchError.providerNotPrepared(identifier)
        }
        let segments: [OmniShotCutCoreML.ShotSegment]
        do {
            segments = try await detector.segments(in: video)
        } catch let error as OmniShotCutError {
            throw translate(error, video: video)
        }
        // The raw values are OmniShotCut's label ids on both sides.
        return segments.map { segment in
            SemanticSearch.ShotSegment(
                start: segment.start, end: segment.end,
                kind: .init(rawValue: segment.kind.rawValue) ?? .shot,
                boundary: .init(rawValue: segment.boundary.rawValue) ?? .hardCut)
        }
    }

    /// The model file for `conversion`: in the local directory, or
    /// downloaded and compiled into the model directory.
    static func model(
        _ conversion: OmniShotCutModelStore.Conversion, configuration: Configuration
    ) async throws -> URL {
        guard let directory = configuration.modelDirectory else {
            let base = configuration.downloadBase ?? environmentBase ?? URL.cachesDirectory
            return try await OmniShotCutModelStore.compiledModel(conversion, base: base)
        }
        // The compiled form first, as the package's own lookup prefers it.
        let repository = (conversion.repository as NSString).lastPathComponent
        for folder in [directory, directory.appendingPathComponent(repository)] {
            for form in ["mlmodelc", "mlpackage"] {
                let url = folder.appendingPathComponent("\(conversion.rawValue).\(form)")
                if FileManager.default.fileExists(atPath: url.path) { return url }
            }
        }
        throw OmniShotCutError.modelUnavailable(
            directory.path, "holds neither \(conversion.rawValue).mlmodelc nor .mlpackage")
    }

    private static var environmentBase: URL? {
        guard let raw = ProcessInfo.processInfo.environment["SEMANTIC_SEARCH_MODEL_DIRECTORY"],
            !raw.isEmpty
        else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
    }

    private func translate(_ error: OmniShotCutError, video: URL) -> SemanticSearchError {
        func unsupported(_ detail: String) -> SemanticSearchError {
            .unsupportedContent(itemID: ItemID(video.lastPathComponent), detail: detail)
        }
        switch error {
        case .videoUnavailable: return .mediaUnavailable(url: video, itemID: nil)
        case .noVideoTrack: return unsupported("no video track")
        case .noFrames: return unsupported("video has no readable frames")
        case .decodingFailed(_, let detail): return unsupported(detail)
        case .unsupportedPixelFormat: return unsupported("decoder did not return 4:2:0 YUV")
        // The segmenter cannot run at all: no Metal for the frame scaler, or
        // a model that fails or returns the wrong shapes.
        case .modelUnavailable, .predictionFailed, .metalUnavailable:
            return .modelUnavailable(identifier, underlying: error)
        }
    }
}

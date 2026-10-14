import AVFoundation
import Foundation
import SemanticSearch
import SoundAnalysis

/// Apple's built-in sound classifier (`SNClassifierIdentifier.version1`):
/// 303 classes — applause, dog bark, glass clink, music, sirens, typing —
/// on device, nothing to download, ~3 s for eleven minutes of audio.
///
/// Windows are the classifier's own unless set: 3 s, overlapping by half.
/// Apple suggests one second or longer; shorter windows place a short sound
/// more precisely, longer ones hear a siren's rise and fall. Its scores are
/// independent per class and move with the window length, so a threshold
/// chosen for one window length does not carry to another.
public actor AppleSoundClassifier: SoundClassifier {

    public struct Configuration: Sendable, Hashable {
        /// Seconds per window, 0.5–15. `nil` keeps the classifier's default.
        public var windowDuration: TimeInterval?
        /// Fraction of each window shared with the next, 0..<1. `nil` keeps
        /// the classifier's default.
        public var overlapFactor: Double?

        public init(windowDuration: TimeInterval? = nil, overlapFactor: Double? = nil) {
            self.windowDuration = windowDuration
            self.overlapFactor = overlapFactor
        }

        public static let `default` = Configuration()
    }

    public let configuration: Configuration

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    public var identifier: String { "apple.soundanalysis.version1" }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("apple.soundanalysis.version1")
            hasher.append(configuration.windowDuration ?? -1)
            hasher.append(configuration.overlapFactor ?? -1)
        }
    }

    public var isPrepared: Bool { true }
    public func prepare() async throws {}
    public func release() async {}

    /// The classes the classifier knows.
    public nonisolated static func knownClasses() throws -> [String] {
        try SNClassifySoundRequest(classifierIdentifier: .version1).knownClassifications
    }

    public func classify(_ url: URL) async throws -> [SoundWindow] {
        // No sound, no windows (`AudioExtractor.hasAudioTrack`).
        guard try await AudioExtractor.hasAudioTrack(url) else { return [] }
        let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
        if let seconds = configuration.windowDuration {
            request.windowDuration = CMTime(seconds: seconds, preferredTimescale: 48_000)
        }
        if let overlap = configuration.overlapFactor {
            request.overlapFactor = overlap
        }
        let analyzer = try SNAudioFileAnalyzer(url: url)
        let observer = Observer()
        try analyzer.add(request, withObserver: observer)
        let box = AnalyzerBox(analyzer)

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<[SoundWindow], Error>) in
                // Asynchronous, so seven minutes of audio do not hold this
                // actor's thread; the handler runs once the observer has seen
                // every window.
                box.analyzer.analyze { reachedEnd in
                    continuation.resume(with: observer.result(reachedEnd: reachedEnd))
                }
            }
        } onCancel: {
            box.analyzer.cancelAnalysis()
        }
    }

    /// `SNAudioFileAnalyzer` is not `Sendable`; it is only started and
    /// cancelled, each once.
    private final class AnalyzerBox: @unchecked Sendable {
        let analyzer: SNAudioFileAnalyzer
        init(_ analyzer: SNAudioFileAnalyzer) { self.analyzer = analyzer }
    }

    /// Collects results on the analyzer's thread.
    private final class Observer: NSObject, SNResultsObserving, @unchecked Sendable {
        private let lock = NSLock()
        private var windows: [SoundWindow] = []
        private var failure: (any Error)?

        func request(_ request: SNRequest, didProduce result: SNResult) {
            guard let result = result as? SNClassificationResult else { return }
            var scores: [String: Float] = [:]
            for classification in result.classifications where classification.confidence > 0 {
                scores[classification.identifier] = Float(classification.confidence)
            }
            let window = SoundWindow(
                start: result.timeRange.start.seconds,
                end: result.timeRange.end.seconds, scores: scores)
            lock.withLock { windows.append(window) }
        }

        func request(_ request: SNRequest, didFailWithError error: Error) {
            lock.withLock { failure = error }
        }

        func result(reachedEnd: Bool) -> Result<[SoundWindow], any Error> {
            lock.withLock {
                if let failure { return .failure(failure) }
                guard reachedEnd else { return .failure(CancellationError()) }
                return .success(windows.sorted { $0.start < $1.start })
            }
        }
    }
}

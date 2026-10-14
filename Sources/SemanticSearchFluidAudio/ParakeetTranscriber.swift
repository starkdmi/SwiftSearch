import Foundation
import SemanticSearch
import SemanticSearchApple

@preconcurrency import CoreML
@preconcurrency import FluidAudio

/// Which Parakeet weights to run.
public enum ParakeetVersion: String, Sendable, CaseIterable {
    /// English only.
    case v2
    /// Multilingual.
    case v3

    /// Languages the weights are actually trained on.
    ///
    /// Advertised rather than assumed. "Multilingual" on a model card is not a
    /// promise about Japanese, and a caller indexing a Japanese recording with
    /// v3 should get an explicit answer rather than a transcript of plausible
    /// European words.
    public var supportedLanguages: [String] {
        switch self {
        case .v2:
            return ["en"]
        case .v3:
            // NVIDIA parakeet-tdt-0.6b-v3: 25 European languages.
            return [
                "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
                "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
            ]
        }
    }

    public func supports(language: String?) -> Bool {
        guard let language else { return true }
        return supportedLanguages.contains(TextNormalization.baseLanguageCode(language))
    }

    var repositoryID: String {
        switch self {
        case .v2: return "FluidInference/parakeet-tdt-0.6b-v2-coreml"
        case .v3: return "FluidInference/parakeet-tdt-0.6b-v3-coreml"
        }
    }

    var fluidVersion: AsrModelVersion {
        switch self {
        case .v2: return .v2
        case .v3: return .v3
        }
    }
}

/// Parakeet speech recognition, as a content processor.
///
/// Keeps the model's word timing. What it produces is a parent item for the
/// recording and one child per utterance, each carrying its own time range — so
/// a transcript hit resolves to a moment in the audio rather than to the file.
///
/// Transcript evidence stays separate from visual evidence. A video indexed
/// with both this and ``VideoFrameExtractor`` gets two independent sets of
/// children under one parent, and neither requires the other to exist.
public actor ParakeetTranscriber {

    public struct Configuration: Sendable {
        public var version: ParakeetVersion
        /// Restrict Core ML to the CPU. Slower, and the only way to run while
        /// another process holds the Neural Engine.
        public var cpuOnly: Bool
        /// Seconds of silence that end an utterance. Word timings come out of
        /// the model; grouping them into utterances is this adapter's choice,
        /// and the gap is the only knob it exposes.
        public var utteranceGap: TimeInterval
        /// Longest utterance before it is split regardless of pauses. Keeps one
        /// unbroken monologue from becoming a single 40-minute segment that no
        /// embedding can represent.
        public var maximumUtteranceSeconds: TimeInterval
        /// End an utterance at the end of a sentence as well as at a pause.
        /// Sentences are the unit people search for and cite; pauses alone
        /// left utterances several sentences long.
        public var splitsAtSentenceEnds: Bool
        /// Directory to download weights into. `nil` uses FluidAudio's cache.
        public var modelDirectory: URL?

        public init(
            version: ParakeetVersion = .v3,
            cpuOnly: Bool = false,
            utteranceGap: TimeInterval = 0.6,
            maximumUtteranceSeconds: TimeInterval = 30,
            splitsAtSentenceEnds: Bool = true,
            modelDirectory: URL? = nil
        ) {
            self.version = version
            self.cpuOnly = cpuOnly
            self.utteranceGap = max(0.05, utteranceGap)
            self.maximumUtteranceSeconds = max(1, maximumUtteranceSeconds)
            self.splitsAtSentenceEnds = splitsAtSentenceEnds
            self.modelDirectory = modelDirectory
        }

        public static let `default` = Configuration()
    }

    private let configuration: Configuration
    private var manager: AsrManager?

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    public var isPrepared: Bool { manager != nil }

    public func prepare() async throws {
        guard manager == nil else { return }

        var modelConfiguration: MLModelConfiguration?
        if configuration.cpuOnly {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            config.allowLowPrecisionAccumulationOnGPU = false
            modelConfiguration = config
        }

        do {
            // FluidAudio loads `<parent>/<repository name>`, whatever the
            // directory it is given is called; the pinned copy goes there.
            let version = configuration.version
            let directory =
                (configuration.modelDirectory
                ?? AsrModels.defaultCacheDirectory(for: version.fluidVersion))
                .deletingLastPathComponent()
                .appendingPathComponent(
                    String(version.repositoryID.split(separator: "/").last ?? ""),
                    isDirectory: true)
            try await ParakeetModelStore.ensure(version, in: directory)
            let models = try await AsrModels.load(
                from: directory, configuration: modelConfiguration,
                version: version.fluidVersion)
            try Task.checkCancellation()
            let manager = AsrManager(config: .default)
            try await manager.initialize(models: models)
            try Task.checkCancellation()
            self.manager = manager
        } catch is CancellationError {
            throw SemanticSearchError.cancelled
        } catch {
            throw SemanticSearchError.modelUnavailable(
                configuration.version.repositoryID, underlying: error)
        }
    }

    public func release() async {
        if let manager { await manager.cleanup() }
        manager = nil
    }

    /// Transcribe a file.
    ///
    /// `languageHint` is checked against what the weights actually support. A
    /// language the model was not trained on is an error, not a silent attempt:
    /// Parakeet will happily emit fluent-looking Latin-script output for
    /// Japanese audio, and an index full of that is worse than an index with a
    /// gap in it.
    public func transcribe(
        url: URL,
        languageHint: String? = nil,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Transcript {
        guard configuration.version.supports(language: languageHint) else {
            throw SemanticSearchError.unsupportedModality(
                itemID: ItemID(url.lastPathComponent), kind: .transcript,
                provider:
                    "parakeet-\(configuration.version.rawValue) supports "
                    + configuration.version.supportedLanguages.joined(separator: ", "))
        }

        if let silent = try await Transcript.ofSilentFile(url) { return silent }
        if manager == nil { try await prepare() }
        guard let manager else {
            throw SemanticSearchError.providerNotPrepared(configuration.version.repositoryID)
        }

        onProgress?(0.05)
        try Task.checkCancellation()

        let extractor = AudioExtractor()
        let samples = try await extractor.samples(from: url)
        try Task.checkCancellation()
        onProgress?(0.2)

        let result: ASRResult
        do {
            result = try await manager.transcribe(samples)
        } catch is CancellationError {
            throw SemanticSearchError.cancelled
        } catch {
            throw SemanticSearchError.processorFailed(
                itemID: ItemID(url.lastPathComponent), processor: "parakeet",
                underlying: error)
        }
        try Task.checkCancellation()
        onProgress?(0.9)

        let words = Self.words(from: result.tokenTimings)
        let segments = Self.utterances(
            from: words,
            gap: configuration.utteranceGap,
            maximumSeconds: configuration.maximumUtteranceSeconds,
            sentenceBreaks: configuration.splitsAtSentenceEnds)

        // From the audio, not from the last word: an utterance's end is trimmed
        // to where speech stops, so the final segment ends before the recording
        // does whenever it trails off into silence.
        let duration = Double(samples.count) / AudioExtractor.targetSampleRate

        onProgress?(1)
        return Transcript(
            text: result.text.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: segments,
            duration: duration,
            language: languageHint)
    }

    // MARK: - Token assembly

    struct Word: Sendable {
        var text: String
        var start: TimeInterval
        var end: TimeInterval
        var confidence: Float
    }

    /// Merge BPE tokens into words.
    ///
    /// Parakeet marks a word boundary by a leading space on the token, so the
    /// rule is "a token that starts with whitespace begins a new word". A
    /// leading token with no space still begins one, which is the first word of
    /// the recording.
    static func words(from timings: [TokenTiming]?) -> [Word] {
        guard let timings, !timings.isEmpty else { return [] }

        var words: [Word] = []
        var current = ""
        var start: TimeInterval?
        var end: TimeInterval = 0
        var confidences: [Float] = []

        func flush() {
            guard !current.isEmpty, let start else { return }
            words.append(
                Word(
                    text: current, start: start, end: end,
                    confidence: confidences.isEmpty
                        ? 0 : confidences.reduce(0, +) / Float(confidences.count)))
        }

        for timing in timings {
            let token = timing.token
            if token.hasPrefix(" ") || token.hasPrefix("\n") || token.hasPrefix("\t") {
                flush()
                current = token.trimmingCharacters(in: .whitespacesAndNewlines)
                start = timing.startTime
                end = timing.endTime
                confidences = [timing.confidence]
            } else {
                if start == nil { start = timing.startTime }
                current += token
                end = timing.endTime
                confidences.append(timing.confidence)
            }
        }
        flush()

        return words.filter { !$0.text.isEmpty }
    }

    /// Seconds of speech a word of this length plausibly occupies.
    ///
    /// Rough on purpose, and only ever used as an upper bound (see
    /// ``trailingSilence(after:)``). The rate is tuned for alphabetic scripts,
    /// which is all Parakeet v2 and v3 are trained on — see
    /// ``ParakeetVersion/supportedLanguages``. A syllable-aware estimate would
    /// be better and is not worth its cost for the one decision it informs.
    static func plausibleSpokenDuration(of text: String) -> TimeInterval {
        max(0.18, 0.075 * Double(text.count))
    }

    /// Silence at the end of a word's reported span.
    ///
    /// Parakeet reports a start time per token and ends each token where the
    /// next one begins, so a pause in the audio does not appear as a gap
    /// between words — it is swallowed by the span of the word before it: a
    /// quarter-second word at the end of a sentence followed by a second of
    /// silence is reported as one 1.28-second word.
    ///
    /// Taking `word.start - previous.end` as the pause therefore always yields
    /// zero, which made ``ParakeetTranscriber/Configuration/utteranceGap`` a
    /// knob that did nothing and turned every recording into one utterance up
    /// to the length cap. The excess of a word's span over its plausible spoken
    /// duration is the pause, and that is what this returns.
    static func trailingSilence(after word: Word) -> TimeInterval {
        let span = word.end - word.start
        return max(0, span - min(span, plausibleSpokenDuration(of: word.text)))
    }

    /// True when a word closes a sentence: it ends in `.`, `?` or `!` and is
    /// not an abbreviation like "Dr." or "St." — one or two letters starting
    /// with a capital.
    static func endsSentence(_ word: String) -> Bool {
        let stripped = word.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = stripped.last, ".?!".contains(last) else { return false }
        if last == "." {
            let base = stripped.replacingOccurrences(
                of: #"\.+$"#, with: "", options: .regularExpression)
            if base.count > 0, base.count <= 2, base.first?.isUppercase == true { return false }
        }
        return true
    }

    /// Group words into utterances on pauses, sentence ends when asked, and a
    /// hard length cap.
    static func utterances(
        from words: [Word], gap: TimeInterval, maximumSeconds: TimeInterval,
        sentenceBreaks: Bool = false
    ) -> [TranscriptSegment] {
        guard !words.isEmpty else { return [] }

        var segments: [TranscriptSegment] = []
        var buffer: [Word] = [words[0]]

        func flush() {
            guard let first = buffer.first, let last = buffer.last else { return }
            // Trim the trailing silence off the last word, so the segment ends
            // where speech ends. A segment that runs a second past the last
            // syllable seeks a player to the wrong place and reads as a bug.
            segments.append(
                TranscriptSegment(
                    text: buffer.map(\.text).joined(separator: " "),
                    start: first.start,
                    end: last.end - trailingSilence(after: last),
                    confidence: buffer.map(\.confidence).reduce(0, +) / Float(buffer.count)))
            buffer.removeAll(keepingCapacity: true)
        }

        for word in words.dropFirst() {
            guard let previous = buffer.last, let segmentStart = buffer.first?.start else {
                buffer.append(word)
                continue
            }
            // Both terms, so this still works if a future backend reports real
            // gaps between words rather than folding them into the span.
            let pause = max(0, word.start - previous.end) + trailingSilence(after: previous)
            let length = word.end - segmentStart

            if pause >= gap || length >= maximumSeconds
                || (sentenceBreaks && Self.endsSentence(previous.text))
            {
                flush()
            }
            buffer.append(word)
        }
        flush()

        return segments
    }
}

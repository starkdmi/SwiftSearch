import AVFoundation
import Foundation
import NaturalLanguage
import SemanticSearch

#if canImport(Speech) && compiler(>=6.2)
    import Speech
#endif

/// On-device speech recognition through Apple's SpeechAnalyzer.
///
/// The second speech backend beside Parakeet, for the languages Parakeet v3
/// is not trained on: its 25 are European, so Japanese,
/// Chinese, Korean and Arabic recordings need this one. Requires macOS 26 /
/// iOS 26; on earlier systems ``prepare()`` reports it unavailable rather than
/// failing later.
///
/// `prepare()` installs the locale's speech assets through `AssetInventory` —
/// the one step that needs a network — so a prepared installation transcribes
/// offline, as the rest of the package does.
public actor AppleSpeechTranscriber {

    public struct Configuration: Sendable, Hashable {
        /// Locale used when a recording has no language hint. SpeechAnalyzer
        /// transcribes one locale and does not detect the language itself.
        public var defaultLocale: String
        /// Seconds of silence between words that end an utterance, in addition
        /// to sentence-ending punctuation.
        public var utteranceGap: TimeInterval

        public init(defaultLocale: String = "en-US", utteranceGap: TimeInterval = 0.8) {
            self.defaultLocale = defaultLocale
            self.utteranceGap = max(0.05, utteranceGap)
        }

        public static let `default` = Configuration()
    }

    public let configuration: Configuration
    private var prepared: Set<String> = []

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    /// True when this OS has SpeechAnalyzer at all.
    public static var isAvailable: Bool {
        #if canImport(Speech) && compiler(>=6.2)
            if #available(macOS 26.0, iOS 26.0, *) { return SpeechTranscriber.isAvailable }
        #endif
        return false
    }

    /// Locales SpeechAnalyzer can transcribe on this device, as BCP-47 tags.
    public static func supportedLanguages() async -> [String] {
        #if canImport(Speech) && compiler(>=6.2)
            if #available(macOS 26.0, iOS 26.0, *) {
                return await SpeechTranscriber.supportedLocales.map { $0.identifier(.bcp47) }
                    .sorted()
            }
        #endif
        return []
    }

    /// Make the locale's assets available, downloading them if needed.
    public func prepare(language: String? = nil) async throws {
        #if canImport(Speech) && compiler(>=6.2)
            if #available(macOS 26.0, iOS 26.0, *) {
                let locale = try await Self.resolvedLocale(
                    language ?? configuration.defaultLocale)
                let key = locale.identifier(.bcp47)
                guard !prepared.contains(key) else { return }
                let transcriber = Self.makeTranscriber(locale: locale)
                if let request = try await AssetInventory.assetInstallationRequest(
                    supporting: [transcriber])
                {
                    try await request.downloadAndInstall()
                }
                prepared.insert(key)
                return
            }
        #endif
        throw SemanticSearchError.modelUnavailable(Self.unavailable, underlying: nil)
    }

    /// Transcribe a file — audio, or the audio track of a video.
    public func transcribe(
        url: URL, languageHint: String? = nil,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Transcript {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        if let silent = try await Transcript.ofSilentFile(url) {
            return Transcript(
                text: "", segments: [], duration: silent.duration, language: languageHint)
        }
        #if canImport(Speech) && compiler(>=6.2)
            if #available(macOS 26.0, iOS 26.0, *) {
                let locale = try await Self.resolvedLocale(
                    languageHint ?? configuration.defaultLocale)
                try await prepare(language: locale.identifier(.bcp47))
                onProgress?(0.05)

                // SpeechAnalyzer reads an audio file; a video's track is
                // extracted first, to the 16 kHz mono it would convert to anyway.
                let samples = try await AudioExtractor().samples(from: url)
                let duration = Double(samples.count) / AudioExtractor.targetSampleRate
                let staging = FileManager.default.temporaryDirectory
                    .appendingPathComponent("speech-\(UUID().uuidString).wav")
                defer { try? FileManager.default.removeItem(at: staging) }
                try Self.writeWAV(samples, to: staging)
                onProgress?(0.15)

                let transcriber = Self.makeTranscriber(locale: locale)
                let analyzer = SpeechAnalyzer(modules: [transcriber])
                let collector = Collector()
                let reader = Task {
                    for try await result in transcriber.results where result.isFinal {
                        await collector.append(result.text)
                    }
                }
                do {
                    _ = try await analyzer.analyzeSequence(
                        from: try AVAudioFile(forReading: staging))
                    try await analyzer.finalizeAndFinishThroughEndOfInput()
                    try await reader.value
                } catch {
                    reader.cancel()
                    if Task.isCancelled || error is CancellationError {
                        throw SemanticSearchError.cancelled
                    }
                    throw SemanticSearchError.processorFailed(
                        itemID: ItemID(url.lastPathComponent), processor: "apple.speech",
                        underlying: error)
                }
                try Task.checkCancellation()

                let words = await collector.words
                let segments = Self.utterances(from: words, gap: configuration.utteranceGap)
                onProgress?(1)
                return Transcript(
                    text: segments.map(\.text).joined(separator: " "),
                    segments: segments,
                    duration: duration,
                    language: languageHint.map(TextNormalization.baseLanguageCode)
                        ?? locale.language.languageCode?.identifier)
            }
        #endif
        throw SemanticSearchError.modelUnavailable(Self.unavailable, underlying: nil)
    }

    // MARK: - Grouping

    struct Word: Sendable {
        var text: String
        var start: TimeInterval
        var end: TimeInterval
    }

    /// Group timed runs into utterances at sentence ends and pauses — the
    /// rule the Parakeet transcriber uses, with the full-width punctuation
    /// Japanese and Chinese end sentences with.
    static func utterances(from words: [Word], gap: TimeInterval) -> [TranscriptSegment] {
        var segments: [TranscriptSegment] = []
        var buffer: [Word] = []
        func flush() {
            let text = buffer.map(\.text).joined()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let first = buffer.first, let last = buffer.last, !text.isEmpty {
                segments.append(TranscriptSegment(text: text, start: first.start, end: last.end))
            }
            buffer.removeAll()
        }
        for word in words {
            if let previous = buffer.last, word.start - previous.end >= gap { flush() }
            buffer.append(word)
            let trimmed = word.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if let last = trimmed.last, ".?!。？！".contains(last) { flush() }
        }
        flush()
        return segments
    }

    // MARK: - Helpers

    private static let unavailable =
        "Apple SpeechAnalyzer requires macOS 26 / iOS 26 and a Swift 6.2 toolchain"

    #if canImport(Speech) && compiler(>=6.2)
        @available(macOS 26.0, iOS 26.0, *)
        private static func makeTranscriber(locale: Locale) -> SpeechTranscriber {
            // Final results only, each run carrying its own audio time range —
            // what word-level grouping needs.
            SpeechTranscriber(
                locale: locale, transcriptionOptions: [], reportingOptions: [],
                attributeOptions: [.audioTimeRange])
        }

        @available(macOS 26.0, iOS 26.0, *)
        private static func resolvedLocale(_ tag: String) async throws -> Locale {
            guard let locale = await SpeechTranscriber.supportedLocale(
                equivalentTo: Locale(identifier: tag))
            else {
                throw SemanticSearchError.unsupportedModality(
                    itemID: "speech", kind: .transcript,
                    provider: "Apple SpeechAnalyzer does not support \(tag) on this device")
            }
            return locale
        }

        @available(macOS 26.0, iOS 26.0, *)
        private actor Collector {
            var words: [Word] = []

            func append(_ text: AttributedString) {
                for run in text.runs {
                    guard let range = run.audioTimeRange else { continue }
                    let start = range.start.seconds
                    let end = range.end.seconds
                    guard start.isFinite, end.isFinite, end >= start else { continue }
                    words.append(
                        Word(text: String(text[run.range].characters), start: start, end: end))
                }
            }
        }
    #endif

    private static func writeWAV(_ samples: [Float], to url: URL) throws {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: AudioExtractor.targetSampleRate,
                channels: 1, interleaved: false),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "cannot stage audio for speech")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
}

/// ``AppleSpeechTranscriber`` as a ``ContentProcessor``, producing the same
/// parent-and-windowed-utterances shape as the Parakeet processor.
public struct AppleSpeechTranscriptionProcessor: ContentProcessor {

    public let transcriberConfiguration: AppleSpeechTranscriber.Configuration
    public let layout: TranscriptLayout
    private let transcriber: AppleSpeechTranscriber

    public init(
        configuration: AppleSpeechTranscriber.Configuration = .default,
        layout: TranscriptLayout = .default
    ) {
        self.transcriberConfiguration = configuration
        self.layout = layout
        self.transcriber = AppleSpeechTranscriber(configuration: configuration)
    }

    public var identifier: String { "apple.speech" }
    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsAudio: true, acceptsVideo: true,
            contentTypes: ["public.audio", "public.movie"],
            producedKinds: layout.retainsMediaReference
                ? [.transcript, .video, .audio] : [.transcript])
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("apple.speech.v1")
            hasher.append(transcriberConfiguration.defaultLocale)
            hasher.append(transcriberConfiguration.utteranceGap)
            layout.appendFingerprint(to: &hasher)
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        guard AppleSpeechTranscriber.isAvailable,
            case .fileURL(let url) = content.payload
        else { return false }
        return Self.mediaExtensions.contains(url.pathExtension.lowercased())
    }

    public func prepare(language: String? = nil) async throws {
        try await transcriber.prepare(language: language)
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .fileURL(let url) = content.payload else {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "AppleSpeechTranscriptionProcessor needs a file URL")
        }
        let itemID = content.id
        let transcript = try await transcriber.transcribe(
            url: url, languageHint: content.languageHint,
            onProgress: { fraction in
                onProgress?(
                    ProcessingProgress(itemID: itemID, stage: "transcribe", fraction: fraction))
            })
        let provenance = Provenance(
            producer: identifier, version: version,
            configurationFingerprint: configurationFingerprint)
        let items = try layout.items(
            for: content, url: url, transcript: transcript, provenance: provenance)
        onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
        return items
    }

    static let mediaExtensions: Set<String> = TranscriptLayout.videoExtensions.union([
        "wav", "mp3", "m4a", "aac", "flac", "aiff", "aif", "caf",
    ])
}

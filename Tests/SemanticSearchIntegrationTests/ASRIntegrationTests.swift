import AVFoundation
import FluidAudio
import Foundation
import SemanticSearch
import SemanticSearchApple
import XCTest

@testable import SemanticSearchFluidAudio

/// Parakeet transcription, end to end.
///
/// The token-assembly and language-advertisement tests run everywhere: they are
/// pure functions and the answers they check are contractual. Everything that
/// loads weights is behind `SEMANTIC_SEARCH_RUN_ASR_TESTS`.
///
/// Speech is synthesised rather than committed as a fixture. A checked-in WAV
/// of real speech is both large and a licensing question, and `AVSpeechSynthesizer`
/// produces clean audio with pauses this suite controls — which is what makes
/// utterance segmentation testable at all.
final class ASRIntegrationTests: XCTestCase {

    // MARK: - Token assembly (no model)

    func testWordAssemblyMergesSubwordTokens() {
        // Parakeet marks a word boundary with a leading space on the token.
        let timings = [
            TokenTiming(token: "The", tokenId: 1, startTime: 0.0, endTime: 0.2, confidence: 0.9),
            TokenTiming(token: " quick", tokenId: 2, startTime: 0.2, endTime: 0.5, confidence: 0.8),
            TokenTiming(token: " bro", tokenId: 3, startTime: 0.5, endTime: 0.7, confidence: 0.7),
            TokenTiming(token: "wn", tokenId: 4, startTime: 0.7, endTime: 0.9, confidence: 0.9),
            TokenTiming(token: " fox", tokenId: 5, startTime: 0.9, endTime: 1.2, confidence: 1.0),
        ]

        let words = ParakeetTranscriber.words(from: timings)

        XCTAssertEqual(words.map(\.text), ["The", "quick", "brown", "fox"])
        // The merged word spans both of its tokens.
        XCTAssertEqual(words[2].start, 0.5, accuracy: 1e-9)
        XCTAssertEqual(words[2].end, 0.9, accuracy: 1e-9)
        // and averages their confidences rather than taking the last.
        XCTAssertEqual(words[2].confidence, 0.8, accuracy: 1e-5)
    }

    func testWordAssemblyToleratesMissingTimings() {
        XCTAssertTrue(ParakeetTranscriber.words(from: nil).isEmpty)
        XCTAssertTrue(ParakeetTranscriber.words(from: []).isEmpty)
    }

    func testUtterancesSplitOnPauses() {
        // Spans tight against each word's plausible spoken duration, so the only
        // pause here is the literal gap between 0.47 and 1.9.
        let words = [
            ParakeetTranscriber.Word(text: "one", start: 0.0, end: 0.22, confidence: 1),
            ParakeetTranscriber.Word(text: "two", start: 0.25, end: 0.47, confidence: 1),
            ParakeetTranscriber.Word(text: "three", start: 1.9, end: 2.27, confidence: 1),
            ParakeetTranscriber.Word(text: "four", start: 2.3, end: 2.6, confidence: 1),
        ]

        let segments = ParakeetTranscriber.utterances(
            from: words, gap: 0.6, maximumSeconds: 30)

        XCTAssertEqual(segments.map(\.text), ["one two", "three four"])
        XCTAssertEqual(segments[0].start, 0.0, accuracy: 1e-9)
        XCTAssertEqual(segments[0].end, 0.47, accuracy: 1e-9)
        XCTAssertEqual(segments[1].start, 1.9, accuracy: 1e-9)
        XCTAssertEqual(segments[1].end, 2.6, accuracy: 1e-9)
    }

    /// Sentences end utterances even without a pause — but "Dr." and "St."
    /// do not end a sentence.
    func testUtterancesSplitAtSentenceEnds() {
        func word(_ text: String, _ start: Double) -> ParakeetTranscriber.Word {
            ParakeetTranscriber.Word(text: text, start: start, end: start + 0.2, confidence: 1)
        }
        let words = [
            word("Meet", 0), word("Dr.", 0.2), word("Smith", 0.4), word("here.", 0.6),
            word("Is", 0.8), word("it", 1.0), word("late?", 1.2), word("No", 1.4),
        ]
        let segments = ParakeetTranscriber.utterances(
            from: words, gap: 0.6, maximumSeconds: 30, sentenceBreaks: true)
        XCTAssertEqual(segments.map(\.text), ["Meet Dr. Smith here.", "Is it late?", "No"])

        let unsplit = ParakeetTranscriber.utterances(
            from: words, gap: 0.6, maximumSeconds: 30, sentenceBreaks: false)
        XCTAssertEqual(unsplit.count, 1)

        XCTAssertTrue(ParakeetTranscriber.endsSentence("done!"))
        XCTAssertFalse(ParakeetTranscriber.endsSentence("St."))
        XCTAssertFalse(ParakeetTranscriber.endsSentence("and"))
    }

    /// The case that matters, because it is the only one Parakeet produces.
    ///
    /// Word timings from the real model are contiguous — each word ends where
    /// the next begins — so a pause shows up as an implausibly long word rather
    /// than as a gap. These numbers are taken from an actual transcript: the
    /// 1.28 s "arrived." is a short word plus a second of silence.
    func testUtterancesSplitOnSilenceSwallowedByAWordSpan() {
        let words = [
            ParakeetTranscriber.Word(text: "the", start: 1.52, end: 1.76, confidence: 1),
            ParakeetTranscriber.Word(text: "storm", start: 1.76, end: 2.24, confidence: 1),
            ParakeetTranscriber.Word(text: "arrived.", start: 2.24, end: 3.52, confidence: 1),
            ParakeetTranscriber.Word(text: "A", start: 3.52, end: 3.84, confidence: 1),
            ParakeetTranscriber.Word(text: "lighthouse", start: 3.84, end: 4.56, confidence: 1),
        ]

        let segments = ParakeetTranscriber.utterances(
            from: words, gap: 0.6, maximumSeconds: 30)

        XCTAssertEqual(segments.map(\.text), ["the storm arrived.", "A lighthouse"])
        // The first segment ends where speech ends, not where the silence does.
        XCTAssertEqual(
            segments[0].end,
            2.24 + ParakeetTranscriber.plausibleSpokenDuration(of: "arrived."),
            accuracy: 1e-9)
        XCTAssertLessThan(segments[0].end, 3.52)
        XCTAssertEqual(segments[1].start, 3.52, accuracy: 1e-9)

        // And no split where the speech is continuous.
        XCTAssertEqual(
            ParakeetTranscriber.utterances(
                from: Array(words.prefix(2)), gap: 0.6, maximumSeconds: 30
            ).count, 1)
    }

    func testUtterancesSplitOnLengthWithoutAPause() {
        // Continuous speech, no pause anywhere: every span is shorter than its
        // word plausibly takes to say. Without the length cap this is one
        // segment no embedding can represent.
        let words = (0..<100).map { index in
            ParakeetTranscriber.Word(
                text: "word\(index)", start: Double(index) * 0.3,
                end: Double(index) * 0.3 + 0.28, confidence: 1)
        }

        XCTAssertEqual(
            ParakeetTranscriber.utterances(from: words, gap: 0.6, maximumSeconds: 600).count, 1,
            "no pause means no split until the cap applies")

        let segments = ParakeetTranscriber.utterances(
            from: words, gap: 0.6, maximumSeconds: 10)

        XCTAssertGreaterThan(segments.count, 1)
        for segment in segments {
            XCTAssertLessThanOrEqual(segment.end - segment.start, 11)
        }
        // Splitting must not drop or duplicate words.
        XCTAssertEqual(
            segments.flatMap { $0.text.split(separator: " ").map(String.init) },
            words.map(\.text))
    }

    // MARK: - Advertised languages (no model)

    func testVersionsAdvertiseTheirActualLanguages() {
        XCTAssertEqual(ParakeetVersion.v2.supportedLanguages, ["en"])

        let v3 = Set(ParakeetVersion.v3.supportedLanguages)
        XCTAssertEqual(v3.count, 25)
        // Multilingual, but specifically 25 European languages.
        for supported in ["en", "ru", "de", "fr", "uk", "pl"] {
            XCTAssertTrue(v3.contains(supported), "v3 should cover \(supported)")
        }
        for unsupported in ["ja", "zh", "ar", "tr", "ko", "hi"] {
            XCTAssertFalse(v3.contains(unsupported), "v3 must not claim \(unsupported)")
        }
    }

    func testLanguageSupportChecksTheBaseSubtag() {
        XCTAssertTrue(ParakeetVersion.v3.supports(language: "ru-RU"))
        XCTAssertTrue(ParakeetVersion.v3.supports(language: "en_GB"))
        // No hint means no claim to check.
        XCTAssertTrue(ParakeetVersion.v3.supports(language: nil))
        XCTAssertFalse(ParakeetVersion.v3.supports(language: "ja-JP"))
        XCTAssertFalse(ParakeetVersion.v2.supports(language: "ru"))
    }

    /// An unsupported hint is refused before a single byte of weights is
    /// downloaded, which is why this test needs no gate.
    func testUnsupportedLanguageHintIsRefusedWithoutLoadingTheModel() async throws {
        let transcriber = ParakeetTranscriber(
            configuration: .init(version: .v3))
        let url = URL(fileURLWithPath: "/nonexistent/japanese-lecture.m4a")

        do {
            _ = try await transcriber.transcribe(url: url, languageHint: "ja")
            XCTFail("expected an unsupported-modality error")
        } catch let error as SemanticSearchError {
            guard case .unsupportedModality(_, let kind, let provider) = error else {
                return XCTFail("expected .unsupportedModality, got \(error)")
            }
            XCTAssertEqual(kind, .transcript)
            XCTAssertTrue(provider.contains("parakeet-v3"))
            // The error names what the model does cover, so the caller can act.
            XCTAssertTrue(provider.contains("en"))
        }
        let prepared = await transcriber.isPrepared
        XCTAssertFalse(prepared, "a rejected hint must not have loaded the model")
    }

    func testProcessorAdvertisesSupportedLanguagesAndMediaTypes() {
        let processor = TranscriptionProcessor()

        XCTAssertEqual(processor.supportedLanguages, ParakeetVersion.v3.supportedLanguages)
        XCTAssertTrue(processor.capabilities.acceptsAudio)
        XCTAssertTrue(processor.capabilities.acceptsVideo)
        XCTAssertTrue(processor.capabilities.producedKinds.contains(.transcript))

        XCTAssertTrue(
            processor.canProcess(
                RawContent(id: "a", payload: .fileURL(URL(fileURLWithPath: "/tmp/talk.m4a")))))
        XCTAssertTrue(
            processor.canProcess(
                RawContent(id: "b", payload: .fileURL(URL(fileURLWithPath: "/tmp/talk.MOV")))))
        XCTAssertFalse(
            processor.canProcess(
                RawContent(id: "c", payload: .fileURL(URL(fileURLWithPath: "/tmp/notes.pdf")))))
        XCTAssertFalse(processor.canProcess(RawContent(id: "d", payload: .text("hello"))))
    }

    /// Changing the utterance grouping must change the fingerprint, or a
    /// re-ingest after a configuration change is silently skipped.
    func testConfigurationFingerprintTracksGrouping() {
        let base = TranscriptionProcessor()
        let widerGap = TranscriptionProcessor(
            configuration: .init(transcriber: .init(utteranceGap: 1.5)))
        let shorterCap = TranscriptionProcessor(
            configuration: .init(transcriber: .init(maximumUtteranceSeconds: 15)))
        let v2 = TranscriptionProcessor(configuration: .init(transcriber: .init(version: .v2)))

        let fingerprints = Set(
            [base, widerGap, shorterCap, v2].map(\.configurationFingerprint))
        XCTAssertEqual(fingerprints.count, 4)
        XCTAssertEqual(base.configurationFingerprint, TranscriptionProcessor().configurationFingerprint)
    }

    func testMissingMediaIsReportedAsMissingMedia() async throws {
        let processor = TranscriptionProcessor()
        let missing = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString).wav")

        do {
            _ = try await processor.process(
                RawContent(id: "gone", payload: .fileURL(missing)), onProgress: nil)
            XCTFail("expected a mediaUnavailable error")
        } catch let error as SemanticSearchError {
            guard case .mediaUnavailable(let url, let itemID) = error else {
                return XCTFail("expected .mediaUnavailable, got \(error)")
            }
            XCTAssertEqual(url, missing)
            XCTAssertEqual(itemID, "gone")
        }
    }

    // MARK: - Model-backed

    func testTranscribesSpeechWithTimestamps() async throws {
        try XCTSkipUnless(Integration.runsASRTests, "set SEMANTIC_SEARCH_RUN_ASR_TESTS=1")

        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let audio = try await SpeechFixture.write(Self.script, to: directory)

        let transcriber = ParakeetTranscriber(configuration: .init(version: .v3))
        defer { Task { await transcriber.release() } }

        let progress = ProgressRecorder()
        let start = ContinuousClock.now
        let transcript = try await transcriber.transcribe(
            url: audio, languageHint: "en",
            onProgress: { value in progress.record(value) })
        let elapsed = ContinuousClock.now - start

        print("[transcript] \(transcript.text)")
        XCTAssertFalse(transcript.text.isEmpty)
        XCTAssertEqual(transcript.language, "en")
        XCTAssertFalse(transcript.segments.isEmpty, "expected timed utterances, not just text")

        // Timings are ordered, non-degenerate and inside the recording.
        var previousEnd = -1.0
        for segment in transcript.segments {
            XCTAssertGreaterThanOrEqual(segment.start, previousEnd - 0.001)
            XCTAssertGreaterThan(segment.end, segment.start)
            XCTAssertLessThanOrEqual(segment.end, transcript.duration + 0.5)
            XCTAssertFalse(segment.text.isEmpty)
            previousEnd = segment.end
        }
        XCTAssertGreaterThan(transcript.duration, 1)

        // The pauses in the script should have produced more than one utterance.
        XCTAssertGreaterThan(transcript.segments.count, 1)

        // Loose on wording, strict on content: TTS is clean audio, so most of
        // the distinctive words should survive.
        let spoken = Set(
            transcript.text.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty })
        let expected = ["harbour", "lighthouse", "ferry", "storm", "captain"]
        let found = expected.filter { spoken.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 3,
            "recognised \(found) of \(expected) in: \(transcript.text)")

        // The progress callback is the only thing a UI has to go on.
        let values = progress.values
        XCTAssertEqual(values.first, 0.05)
        XCTAssertEqual(values.last, 1)
        XCTAssertEqual(values, values.sorted(), "progress must be monotonic")

        Integration.measure(
            "asr.transcribe",
            [
                "audio_seconds": String(format: "%.1f", transcript.duration),
                "wall_ms": String(format: "%.0f", Integration.milliseconds(elapsed)),
                "segments": String(transcript.segments.count),
                "peak_mb": String(format: "%.0f", Integration.peakFootprintMegabytes()),
            ])
    }

    func testCancellationStopsTranscription() async throws {
        try XCTSkipUnless(Integration.runsASRTests, "set SEMANTIC_SEARCH_RUN_ASR_TESTS=1")

        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let audio = try await SpeechFixture.write(Self.script, to: directory)

        let transcriber = ParakeetTranscriber(configuration: .init(version: .v3))
        defer { Task { await transcriber.release() } }
        // Pay the download and load cost outside the cancelled task, so what is
        // being cancelled is transcription and not a model fetch.
        try await transcriber.prepare()

        let started = StartSignal()
        let task = Task {
            try await transcriber.transcribe(
                url: audio,
                onProgress: { _ in started.signal() })
        }
        await started.wait()
        task.cancel()

        do {
            _ = try await task.value
            // Short audio can finish before cancellation is observed. That is a
            // pass: the contract is that cancellation is honoured between
            // stages, not that it always wins the race.
            print("[asr] transcription completed before cancellation took effect")
        } catch let error as SemanticSearchError {
            guard case .cancelled = error else {
                return XCTFail("expected .cancelled, got \(error)")
            }
        } catch is CancellationError {
            // Also acceptable: the checkCancellation inside the extractor.
        }

        // The transcriber survives a cancelled call and still works.
        let recovered = try await transcriber.transcribe(url: audio, languageHint: "en")
        XCTAssertFalse(recovered.text.isEmpty)
    }

    func testRecordingsWithoutPictureAreNotVideo() {
        XCTAssertTrue(TranscriptionProcessor.audioExtensions.isDisjoint(
            with: TranscriptionProcessor.videoExtensions))
        for ext in ["wav", "m4a", "caf", "mp3"] {
            XCTAssertTrue(TranscriptionProcessor.audioExtensions.contains(ext), ext)
        }
        for ext in ["mov", "mp4", "m4v"] {
            XCTAssertTrue(TranscriptionProcessor.videoExtensions.contains(ext), ext)
        }
        XCTAssertEqual(
            TranscriptionProcessor.mediaExtensions,
            TranscriptionProcessor.audioExtensions.union(TranscriptionProcessor.videoExtensions))
    }

    func testProcessorProducesParentAndTimedChildren() async throws {
        try XCTSkipUnless(Integration.runsASRTests, "set SEMANTIC_SEARCH_RUN_ASR_TESTS=1")

        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let audio = try await SpeechFixture.write(Self.script, to: directory)

        let processor = TranscriptionProcessor()
        defer { Task { await processor.release() } }

        let stages = ProgressRecorder()
        let items = try await processor.process(
            RawContent(
                id: "talk", payload: .fileURL(audio), metadata: ["source": .string("fixture")],
                languageHint: "en"),
            onProgress: { progress in stages.recordStage(progress.stage) })

        XCTAssertGreaterThan(items.count, 1, "expected a parent and at least one utterance")

        let parent = try XCTUnwrap(items.first)
        XCTAssertEqual(parent.id, "talk")
        XCTAssertNil(parent.parentID)

        // The recording is kept as a reference. It has no picture, so it is
        // audio, not video — a multimodal embedder must never be handed it as
        // frames — and it is context-only, since nothing embeds sound yet.
        XCTAssertFalse(parent.representations.contains { $0.kind == .video })
        let media = try XCTUnwrap(parent.representations.first { $0.kind == .audio })
        guard case .mediaReference(let referenced) = media.content else {
            return XCTFail("parent media representation should carry a file reference")
        }
        XCTAssertEqual(referenced, audio)
        XCTAssertEqual(media.weight, 0)

        // The full transcript is kept as context, with weight 0: it must not
        // compete in retrieval against the utterance that actually matches.
        let whole = try XCTUnwrap(parent.representations.first { $0.kind == .transcript })
        XCTAssertEqual(whole.weight, 0)
        XCTAssertEqual(whole.locator?.timeRange?.start, 0)

        let children = Array(items.dropFirst())
        for (index, child) in children.enumerated() {
            XCTAssertEqual(child.parentID, "talk")
            XCTAssertEqual(child.ordinal, index)
            XCTAssertEqual(child.metadata["source"], .string("fixture"))

            let representation = try XCTUnwrap(child.representations.first)
            XCTAssertEqual(representation.kind, .transcript)
            XCTAssertEqual(representation.language, "en")
            XCTAssertGreaterThan(representation.weight, 0)

            // The searchable text is a window around the utterance; its range
            // covers the utterance's own, and playback seeks to the utterance.
            let range = try XCTUnwrap(representation.locator?.timeRange)
            XCTAssertGreaterThan(range.end, range.start)
            let own = try XCTUnwrap(
                child.representations.first { $0.id == "utterance" }?.locator?.timeRange)
            XCTAssertLessThanOrEqual(range.start, own.start)
            XCTAssertGreaterThanOrEqual(range.end, own.end)
            XCTAssertEqual(child.locator.focusTime, own.start)
            // Provenance carries the model's own confidence, reported and never
            // used for ranking.
            XCTAssertNotNil(representation.provenance.confidence)
            XCTAssertEqual(representation.provenance.producer, "parakeet.v3")
        }

        XCTAssertTrue(stages.stages.contains("transcribe"))
        XCTAssertEqual(stages.stages.last, "done")
    }

    /// The whole point of timestamps: a transcript hit resolves to a moment.
    func testTranscriptSearchReturnsTheMatchingMoment() async throws {
        try XCTSkipUnless(Integration.runsASRTests, "set SEMANTIC_SEARCH_RUN_ASR_TESTS=1")

        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let audio = try await SpeechFixture.write(Self.script, to: directory)

        let processor = TranscriptionProcessor()
        defer { Task { await processor.release() } }

        // Lexical-only: no embedding model has to be resident for this to work,
        // which is the core package's contract.
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: directory.appendingPathComponent("index.sqlite"),
                processor: processor))
        defer { Task { try? await engine.close() } }

        let report = try await engine.ingest([
            RawContent(id: "talk", payload: .fileURL(audio), languageHint: "en")
        ])
        XCTAssertGreaterThan(report.itemsWritten, 1)

        // Re-ingesting the same file under the same configuration does no work.
        let again = try await engine.ingest([
            RawContent(id: "talk", payload: .fileURL(audio), languageHint: "en")
        ])
        XCTAssertEqual(again.itemsUnchanged, 1)
        XCTAssertEqual(again.itemsWritten, 0)

        // Pick a term from a real utterance rather than assuming what was
        // recognised, then check the hit points back at that utterance.
        let items = try await engine.allTranscriptChildren()
        let target = try XCTUnwrap(items.last(where: { $0.words.count >= 2 }) ?? items.last)
        let term = try XCTUnwrap(target.words.max(by: { $0.count < $1.count }))

        let results = try await engine.search(
            SearchQuery(term), options: SearchOptions(mode: .lexicalOnly, limit: 5))
        XCTAssertFalse(results.hits.isEmpty, "no hit for '\(term)'")

        let hit = try XCTUnwrap(results.hits.first { $0.itemID == target.id })
        XCTAssertEqual(hit.kind, .transcript)
        let range = try XCTUnwrap(hit.locator.timeRange)
        XCTAssertEqual(range.start, target.start, accuracy: 0.001)
        XCTAssertEqual(hit.parent?.itemID, "talk")
        XCTAssertFalse(results.diagnostics.channels.isEmpty)
    }

    // MARK: - Script

    /// Deliberately plain vocabulary with distinctive nouns, and sentence-level
    /// pauses long enough to segment on.
    static let script: [SpeechFixture.Line] = [
        .init(text: "The harbour was quiet before the storm arrived.", pause: 1.2),
        .init(text: "A lighthouse marked the entrance to the bay.", pause: 1.2),
        .init(text: "The captain delayed the evening ferry until morning.", pause: 0.8),
    ]
}

// MARK: - Helpers

/// Collects callback values from whatever task the provider calls back on.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Double] = []
    private var stageStorage: [String] = []

    func record(_ value: Double) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }

    func recordStage(_ stage: String) {
        lock.lock()
        stageStorage.append(stage)
        lock.unlock()
    }

    var values: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    var stages: [String] {
        lock.lock()
        defer { lock.unlock() }
        return stageStorage
    }
}

/// Lets a test wait until work has actually begun before cancelling it, rather
/// than sleeping and hoping.
private final class StartSignal: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var signalled = false

    func signal() {
        lock.lock()
        let first = !signalled
        signalled = true
        lock.unlock()
        if first { semaphore.signal() }
    }

    func wait() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                // Bounded: if the callback never fires the test should cancel
                // anyway rather than hang until the harness times out.
                _ = self.semaphore.wait(timeout: .now() + 30)
                continuation.resume()
            }
        }
    }
}

/// Renders a short script to a file with `AVSpeechSynthesizer`.
///
/// Each line is synthesised separately and joined with explicit silence, which
/// is the only way to get pauses the utterance grouping can be tested against —
/// the synthesiser's own inter-sentence pause is far shorter than the default
/// 0.6 s gap.
enum SpeechFixture {

    struct Line: Sendable {
        var text: String
        /// Silence appended after the line, in seconds.
        var pause: TimeInterval

        init(text: String, pause: TimeInterval) {
            self.text = text
            self.pause = pause
        }
    }

    @MainActor
    static func write(_ lines: [Line], to directory: URL, voice language: String = "en-US")
        async throws -> URL
    {
        let url = directory.appendingPathComponent("speech-\(UUID().uuidString).caf")
        var file: AVAudioFile?

        for line in lines {
            let buffers = try await render(line.text, language: language)
            guard let format = buffers.first?.format else {
                throw XCTSkip("AVSpeechSynthesizer produced no audio on this machine")
            }
            if file == nil {
                file = try AVAudioFile(
                    forWriting: url, settings: format.settings,
                    commonFormat: format.commonFormat, interleaved: format.isInterleaved)
            }
            for buffer in buffers where buffer.frameLength > 0 {
                try file?.write(from: buffer)
            }
            if line.pause > 0, let silence = silence(seconds: line.pause, format: format) {
                try file?.write(from: silence)
            }
        }

        guard file != nil else {
            throw XCTSkip("no speech was synthesised")
        }
        // Close the file before anything reads it.
        file = nil
        return url
    }

    @MainActor
    private static func render(_ text: String, language: String) async throws
        -> [AVAudioPCMBuffer]
    {
        guard let voice = AVSpeechSynthesisVoice(language: language) else {
            throw XCTSkip("no \(language) system voice is installed")
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        // Slightly slower than default: clearer audio, and it keeps the fixture
        // long enough that transcription is a meaningful measurement.
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9

        let synthesizer = AVSpeechSynthesizer()
        var collected: [AVAudioPCMBuffer] = []

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            var finished = false
            synthesizer.write(utterance) { buffer in
                guard let pcm = buffer as? AVAudioPCMBuffer else { return }
                if pcm.frameLength == 0 {
                    // A zero-length buffer is how `write` says it is done.
                    if !finished {
                        finished = true
                        continuation.resume()
                    }
                    return
                }
                collected.append(pcm)
            }
        }

        withExtendedLifetime(synthesizer) {}
        return collected
    }

    private static func silence(seconds: TimeInterval, format: AVAudioFormat)
        -> AVAudioPCMBuffer?
    {
        let frames = AVAudioFrameCount(seconds * format.sampleRate)
        guard frames > 0,
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
        else { return nil }
        buffer.frameLength = frames

        // A freshly allocated buffer is not guaranteed to be zeroed, and
        // uninitialised memory written as audio is noise the model will try to
        // transcribe.
        let channels = Int(format.channelCount)
        let bytesPerChannel = Int(frames) * Int(format.streamDescription.pointee.mBytesPerFrame)
        if let int16 = buffer.int16ChannelData {
            for channel in 0..<channels {
                memset(int16[channel], 0, bytesPerChannel)
            }
        } else if let floats = buffer.floatChannelData {
            for channel in 0..<channels {
                memset(floats[channel], 0, bytesPerChannel)
            }
        } else if let int32 = buffer.int32ChannelData {
            for channel in 0..<channels {
                memset(int32[channel], 0, bytesPerChannel)
            }
        }
        return buffer
    }
}

// MARK: - Engine convenience

extension SemanticSearchEngine {

    fileprivate struct TranscriptChild {
        var id: ItemID
        var text: String
        var start: TimeInterval
        var words: [String]
    }

    /// Every transcript utterance in the index, in time order.
    fileprivate func allTranscriptChildren() async throws -> [TranscriptChild] {
        // Search is the only public read path that enumerates representations,
        // and a match-everything filter with a large limit is how a test walks
        // what was indexed.
        let results = try await search(
            SearchQuery("the a of to and was"),
            options: SearchOptions(mode: .lexicalOnly, limit: 200))
        return
            results.hits
            .filter { $0.kind == .transcript && $0.locator.timeRange != nil }
            .map { hit in
                let text = hit.text ?? ""
                return TranscriptChild(
                    id: hit.itemID,
                    text: text,
                    start: hit.locator.timeRange?.start ?? 0,
                    words: text.lowercased()
                        .components(separatedBy: CharacterSet.alphanumerics.inverted)
                        .filter { $0.count > 3 })
            }
            .sorted { $0.start < $1.start }
    }
}

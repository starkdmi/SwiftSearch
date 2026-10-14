import Foundation
import SemanticSearch
import SemanticSearchApple

/// Wraps ``ParakeetTranscriber`` as a ``ContentProcessor``.
///
/// Produces a parent for the recording and one child per utterance. Each child
/// carries its own `timeRange`, which is what lets a transcript hit be played
/// from the right second instead of from the start of the file.
public struct TranscriptionProcessor: ContentProcessor {

    public struct Configuration: Sendable {
        public var transcriber: ParakeetTranscriber.Configuration
        /// Keep a reference to the media on the parent, so re-ingestion and a
        /// player can find it.
        public var retainsMediaReference: Bool
        /// Also embed a video file as a whole in a multimodal space. Off by
        /// default: the utterances (and shot frames, when a shot processor runs
        /// alongside) already give every moment its own evidence, and one
        /// vector for a whole recording costs minutes of inference and
        /// gigabytes of memory on anything longer than a clip.
        public var embedsVideo: Bool
        /// Utterances on each side included in what an utterance is searched
        /// by. A sentence out of context is often too short to match anything;
        /// with its neighbours it carries the topic. The window's text is
        /// embedded and indexed and its time range is the hit's; the utterance
        /// alone is kept beside it as `utterance`.
        public var contextUtterances: Int
        /// Keep the full transcript on the parent as context-only text. See
        /// ``ContentRepresentation/weight`` for what context-only means.
        public var retainsFullTranscript: Bool

        public init(
            transcriber: ParakeetTranscriber.Configuration = .default,
            retainsMediaReference: Bool = true,
            retainsFullTranscript: Bool = true,
            embedsVideo: Bool = false,
            contextUtterances: Int = 1
        ) {
            self.transcriber = transcriber
            self.retainsMediaReference = retainsMediaReference
            self.retainsFullTranscript = retainsFullTranscript
            self.embedsVideo = embedsVideo
            self.contextUtterances = max(0, contextUtterances)
        }

        public static let `default` = Configuration()

        /// The shared parent-and-utterances shape these settings describe.
        public var layout: TranscriptLayout {
            TranscriptLayout(
                retainsMediaReference: retainsMediaReference,
                retainsFullTranscript: retainsFullTranscript,
                embedsVideo: embedsVideo, contextUtterances: contextUtterances)
        }
    }

    public let configuration: Configuration
    private let transcriber: ParakeetTranscriber

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.transcriber = ParakeetTranscriber(configuration: configuration.transcriber)
    }

    public var identifier: String { "parakeet.\(configuration.transcriber.version.rawValue)" }

    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsAudio: true,
            acceptsVideo: true,
            contentTypes: ["public.audio", "public.movie"],
            producedKinds: configuration.retainsMediaReference
                ? [.transcript, .video, .audio] : [.transcript])
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("parakeet.v3")
            hasher.append(configuration.transcriber.version.rawValue)
            // The weights' commit: other weights write other transcripts.
            hasher.append(configuration.transcriber.version.revision)
            hasher.append(configuration.transcriber.cpuOnly)
            hasher.append(configuration.transcriber.utteranceGap)
            hasher.append(configuration.transcriber.maximumUtteranceSeconds)
            hasher.append(configuration.transcriber.splitsAtSentenceEnds)
            configuration.layout.appendFingerprint(to: &hasher)
        }
    }

    /// Languages the configured weights actually cover.
    public var supportedLanguages: [String] {
        configuration.transcriber.version.supportedLanguages
    }

    public func canProcess(_ content: RawContent) -> Bool {
        guard case .fileURL(let url) = content.payload else { return false }
        return Self.mediaExtensions.contains(url.pathExtension.lowercased())
    }

    public func prepare() async throws {
        try await transcriber.prepare()
    }

    public func release() async {
        await transcriber.release()
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .fileURL(let url) = content.payload else {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "TranscriptionProcessor needs a media file URL")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: content.id)
        }

        let itemID = content.id
        let transcript = try await transcriber.transcribe(
            url: url,
            languageHint: content.languageHint,
            onProgress: { fraction in
                onProgress?(
                    ProcessingProgress(itemID: itemID, stage: "transcribe", fraction: fraction))
            })

        let provenance = Provenance(
            producer: identifier,
            version: version,
            configurationFingerprint: configurationFingerprint)
        let items = try configuration.layout.items(
            for: content, url: url, transcript: transcript, provenance: provenance)

        onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
        return items
    }

    static let audioExtensions: Set<String> = [
        "wav", "mp3", "m4a", "aac", "flac", "aiff", "aif", "caf", "ogg",
    ]
    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "mkv", "avi"]
    static let mediaExtensions = audioExtensions.union(videoExtensions)
}

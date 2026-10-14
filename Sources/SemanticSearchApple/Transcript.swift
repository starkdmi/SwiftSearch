import AVFoundation
import Foundation
import SemanticSearch

/// One timed span of transcribed speech.
public struct TranscriptSegment: Sendable, Hashable {
    public var text: String
    public var start: TimeInterval
    public var end: TimeInterval
    /// The model's own confidence, averaged over the segment, when the backend
    /// reports one. Reported, never used for ranking — it is not comparable
    /// with anything else in the pipeline.
    public var confidence: Float?

    public init(text: String, start: TimeInterval, end: TimeInterval, confidence: Float? = nil) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

public struct Transcript: Sendable {
    public var text: String
    public var segments: [TranscriptSegment]
    public var duration: TimeInterval
    /// BCP-47 tag, when the backend reports or the caller supplies one.
    public var language: String?

    public init(
        text: String, segments: [TranscriptSegment], duration: TimeInterval,
        language: String? = nil
    ) {
        self.text = text
        self.segments = segments
        self.duration = duration
        self.language = language
    }
}

extension Transcript {
    /// The transcript of a file without an audio track: no words, the
    /// file's length. `nil` when it has one. Checked before a model loads,
    /// so a silent video costs a model nothing.
    public static func ofSilentFile(_ url: URL) async throws -> Transcript? {
        guard try await !AudioExtractor.hasAudioTrack(url) else { return nil }
        let duration = try await AVURLAsset(url: url).load(.duration).seconds
        return Transcript(text: "", segments: [], duration: duration.isFinite ? duration : 0)
    }
}

/// How a transcript becomes items: a parent for the recording and one child
/// per utterance. Shared by every speech backend so that a Parakeet index and
/// an Apple Speech index have the same shape.
public struct TranscriptLayout: Sendable, Hashable {
    /// Keep a reference to the media on the parent, so re-ingestion and a
    /// player can find it.
    public var retainsMediaReference: Bool
    /// Keep the full transcript on the parent as context-only text. See
    /// ``ContentRepresentation/weight`` for what context-only means.
    public var retainsFullTranscript: Bool
    /// Also embed a video file as a whole in a multimodal space. Off by
    /// default: the utterances (and shot frames, when a shot processor runs
    /// alongside) already give every moment its own evidence, and one vector
    /// for a whole recording costs minutes of inference and gigabytes of memory
    /// on anything longer than a clip.
    public var embedsVideo: Bool
    /// Utterances on each side included in what an utterance is searched by. A
    /// sentence out of context is often too short to match anything; with its
    /// neighbours it carries the topic. The window's text is embedded and
    /// indexed and its time range is the hit's; the utterance alone is kept
    /// beside it as `utterance`.
    public var contextUtterances: Int

    public init(
        retainsMediaReference: Bool = true,
        retainsFullTranscript: Bool = true,
        embedsVideo: Bool = false,
        contextUtterances: Int = 1
    ) {
        self.retainsMediaReference = retainsMediaReference
        self.retainsFullTranscript = retainsFullTranscript
        self.embedsVideo = embedsVideo
        self.contextUtterances = max(0, contextUtterances)
    }

    public static let `default` = TranscriptLayout()

    /// Everything here changes the items produced, so all of it belongs in
    /// the owning processor's configuration fingerprint.
    public func appendFingerprint(to hasher: inout StableHash) {
        hasher.append(retainsMediaReference)
        hasher.append(retainsFullTranscript)
        hasher.append(embedsVideo)
        hasher.append(contextUtterances)
    }

    public static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "mkv", "avi"]

    /// The parent and its utterance children.
    public func items(
        for content: RawContent, url: URL, transcript: Transcript, provenance: Provenance
    ) throws -> [SearchItem] {
        var parentRepresentations: [ContentRepresentation] = []
        if retainsMediaReference {
            // A recording is only a video if it has a picture. Labelling an
            // `.m4a` as video would hand it to a multimodal embedder as frames.
            let hasPicture = Self.videoExtensions.contains(url.pathExtension.lowercased())
            parentRepresentations.append(
                ContentRepresentation(
                    id: "media", kind: hasPicture ? .video : .audio,
                    content: .mediaReference(url),
                    locator: SourceLocator(source: url.absoluteString),
                    // A pointer, not embedded, unless asked: no bundled
                    // provider takes sound, and a whole video is one blurred
                    // vector at a high price. The utterances below are how a
                    // recording is searched.
                    weight: hasPicture && embedsVideo ? 1 : 0))
        }
        if retainsFullTranscript, !transcript.text.isEmpty {
            parentRepresentations.append(
                ContentRepresentation(
                    id: "transcript", kind: .transcript, content: .text(transcript.text),
                    language: transcript.language, provenance: provenance,
                    locator: SourceLocator(
                        source: url.absoluteString,
                        timeRange: SourceLocator.TimeRange(start: 0, end: transcript.duration)),
                    // Context only. The utterances below carry the same words;
                    // indexing both would give every recording a second vote
                    // in retrieval and put the whole transcript in competition
                    // with the moment inside it that actually matches.
                    weight: 0))
        }

        var items: [SearchItem] = [
            SearchItem(
                id: content.id,
                parentID: content.parentID,
                metadata: content.metadata,
                locator: SourceLocator(
                    source: url.absoluteString,
                    timeRange: SourceLocator.TimeRange(start: 0, end: transcript.duration)),
                representations: parentRepresentations)
        ]

        let segments = transcript.segments
        let context = contextUtterances
        for (index, segment) in segments.enumerated() {
            try Task.checkCancellation()
            let window = segments[
                max(0, index - context)...min(segments.count - 1, index + context)]
            let windowText = window.map(\.text).joined(separator: " ")
            let locator = SourceLocator(
                source: url.absoluteString,
                timeRange: SourceLocator.TimeRange(
                    start: window.first!.start, end: window.last!.end),
                focusTime: segment.start)
            let segmentProvenance = Provenance(
                producer: provenance.producer,
                version: provenance.version,
                configurationFingerprint: provenance.configurationFingerprint,
                confidence: segment.confidence)
            var representations = [
                ContentRepresentation(
                    id: "transcript", kind: .transcript,
                    content: .text(context > 0 ? windowText : segment.text),
                    language: transcript.language,
                    provenance: segmentProvenance,
                    locator: locator)
            ]
            if context > 0 {
                // The utterance on its own, for display and quoting. Context
                // only: indexing it as well would give every sentence a second
                // vote beside the window that already contains it.
                representations.append(
                    ContentRepresentation(
                        id: "utterance", kind: .transcript, content: .text(segment.text),
                        language: transcript.language, provenance: segmentProvenance,
                        locator: SourceLocator(
                            source: url.absoluteString,
                            timeRange: SourceLocator.TimeRange(
                                start: segment.start, end: segment.end),
                            focusTime: segment.start),
                        weight: 0))
            }
            items.append(
                SearchItem(
                    id: ItemID("\(content.id.rawValue)#utterance-\(index)"),
                    parentID: content.id,
                    metadata: content.metadata,
                    locator: locator,
                    representations: representations,
                    ordinal: index))
        }
        return items
    }
}

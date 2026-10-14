import Foundation

/// Scores what a recording sounds like, window by window.
///
/// Multi-label: every window has an independent score per class, and several
/// classes can be present at once — music under a glass clink. The scores do
/// not sum to one and are not comparable across classes.
public protocol SoundClassifier: Actor {
    /// Model and version, e.g. `"apple.soundanalysis.version1"`. Recorded as
    /// the events' ``Provenance/producer``.
    var identifier: String { get }
    /// Changes whenever the scores would: model, window, hop. Part of the
    /// processor fingerprint.
    var configurationFingerprint: String { get }

    func prepare() async throws
    func release() async
    var isPrepared: Bool { get }

    /// Every window of the recording's sound, in time order.
    func classify(_ url: URL) async throws -> [SoundWindow]
}

/// One analysis window and its per-class scores, 0...1.
public struct SoundWindow: Sendable, Hashable {
    public var start: TimeInterval
    public var end: TimeInterval
    /// Class identifier → score. Classes left out scored zero.
    public var scores: [String: Float]

    public init(start: TimeInterval, end: TimeInterval, scores: [String: Float]) {
        self.start = start
        self.end = end
        self.scores = scores
    }
}

/// Turns a recording's sound into timed words: one child per sound event.
///
/// A class is present in a window when its score reaches the decision
/// threshold, and consecutive windows where it is present are one event —
/// frame-level output turned into events as DCASE's sound event detection
/// evaluates it (Mesaros, Heittola & Virtanen, "Metrics for Polyphonic Sound
/// Event Detection", 2016). The threshold is 0.5, the binarisation of the
/// DCASE Task 4 baselines for independent per-class scores. Apple's guidance
/// for its classifier is that the right threshold is a trade-off to be set
/// per sound; `thresholds` takes those per class once they are measured.
/// The baselines also median-filter frames over ~0.45 s; windows here are
/// seconds long, so that filter would not change anything and is not applied.
///
/// Each event is a `.soundEvent` child — `#sound-3`, the class name as text
/// ("glass clink"), its time range, and its peak score as the provenance's
/// confidence — so a sound is found by BM25, by any text embedder, and read
/// by the reranker like any other line, and grouped into moments with the
/// transcript and frames around it. Combine it with a transcriber and a
/// frame source through ``CombinedContentProcessor``.
public struct SoundEventProcessor: ContentProcessor {

    public struct Configuration: Sendable, Hashable {
        /// Score at which a class counts as present in a window.
        public var threshold: Float
        /// Per-class thresholds, overriding ``threshold``.
        public var thresholds: [String: Float]
        /// Classes never emitted — e.g. `"speech"` when a transcriber already
        /// covers what was said. Empty by default.
        public var excludedClasses: Set<String>
        /// Keep a reference to the recording on the parent, as the
        /// transcription processors do.
        public var retainsMediaReference: Bool

        public init(
            threshold: Float = 0.5,
            thresholds: [String: Float] = [:],
            excludedClasses: Set<String> = [],
            retainsMediaReference: Bool = true
        ) {
            self.threshold = threshold
            self.thresholds = thresholds
            self.excludedClasses = excludedClasses
            self.retainsMediaReference = retainsMediaReference
        }

        public static let `default` = Configuration()
    }

    public let classifier: any SoundClassifier
    public let configuration: Configuration
    private let classifierIdentifier: String
    private let classifierFingerprint: String

    /// The classifier's identity is read once, here, so the processor's
    /// fingerprint is a plain property like every other processor's.
    public init(classifier: any SoundClassifier, configuration: Configuration = .default) async {
        self.classifier = classifier
        self.configuration = configuration
        self.classifierIdentifier = await classifier.identifier
        self.classifierFingerprint = await classifier.configurationFingerprint
    }

    public var identifier: String { classifierIdentifier }

    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsAudio: true,
            acceptsVideo: true,
            contentTypes: ["public.audio", "public.movie"],
            producedKinds: configuration.retainsMediaReference
                ? [.soundEvent, .audio, .video] : [.soundEvent])
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("sound-events.v1")
            hasher.append(classifierIdentifier)
            hasher.append(classifierFingerprint)
            hasher.append(Double(configuration.threshold))
            for (label, value) in configuration.thresholds.sorted(by: { $0.key < $1.key }) {
                hasher.append(label)
                hasher.append(Double(value))
            }
            for label in configuration.excludedClasses.sorted() { hasher.append(label) }
            hasher.append(configuration.retainsMediaReference)
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        guard case .fileURL(let url) = content.payload else { return false }
        return Self.mediaExtensions.contains(url.pathExtension.lowercased())
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .fileURL(let url) = content.payload else {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "SoundEventProcessor needs a media file URL")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: content.id)
        }
        onProgress?(ProcessingProgress(itemID: content.id, stage: "classify sounds"))
        let windows: [SoundWindow]
        do {
            windows = try await classifier.classify(url)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw SemanticSearchError.processorFailed(
                itemID: content.id, processor: identifier, underlying: error)
        }
        let duration = windows.map(\.end).max() ?? 0

        var parentRepresentations: [ContentRepresentation] = []
        if configuration.retainsMediaReference {
            let hasPicture = Self.videoExtensions.contains(url.pathExtension.lowercased())
            parentRepresentations.append(
                ContentRepresentation(
                    id: "media", kind: hasPicture ? .video : .audio,
                    content: .mediaReference(url),
                    locator: SourceLocator(source: url.absoluteString),
                    // A pointer: the events below are how the sound is found.
                    weight: 0))
        }
        var items = [
            SearchItem(
                id: content.id, parentID: content.parentID, metadata: content.metadata,
                locator: SourceLocator(
                    source: url.absoluteString,
                    timeRange: SourceLocator.TimeRange(start: 0, end: duration)),
                representations: parentRepresentations)
        ]

        for (index, event) in events(in: windows).enumerated() {
            try Task.checkCancellation()
            let locator = SourceLocator(
                source: url.absoluteString,
                timeRange: SourceLocator.TimeRange(start: event.start, end: event.end))
            items.append(
                SearchItem(
                    id: ItemID("\(content.id.rawValue)#sound-\(index)"),
                    parentID: content.id,
                    metadata: content.metadata,
                    locator: locator,
                    representations: [
                        ContentRepresentation(
                            id: "sound", kind: .soundEvent,
                            content: .text(Self.displayName(event.label)),
                            // Class names are English whatever was recorded.
                            language: "en",
                            provenance: Provenance(
                                producer: classifierIdentifier, version: version,
                                configurationFingerprint: configurationFingerprint,
                                confidence: event.peak),
                            locator: locator)
                    ],
                    ordinal: index))
        }
        onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
        return items
    }

    struct Event: Equatable {
        var label: String
        var start: TimeInterval
        var end: TimeInterval
        var peak: Float
    }

    /// Runs of windows in which a class reached its threshold, in order of
    /// start time, then label.
    func events(in windows: [SoundWindow]) -> [Event] {
        var open: [String: Event] = [:]
        var closed: [Event] = []
        for window in windows.sorted(by: { $0.start < $1.start }) {
            var present: Set<String> = []
            for (label, score) in window.scores
            where !configuration.excludedClasses.contains(label)
                && score >= (configuration.thresholds[label] ?? configuration.threshold)
            {
                present.insert(label)
                if var event = open[label] {
                    event.end = max(event.end, window.end)
                    event.peak = max(event.peak, score)
                    open[label] = event
                } else {
                    open[label] = Event(
                        label: label, start: window.start, end: window.end, peak: score)
                }
            }
            // A window without the class ends its event.
            for label in open.keys where !present.contains(label) {
                closed.append(open.removeValue(forKey: label)!)
            }
        }
        closed += open.values
        return closed.sorted { ($0.start, $0.label) < ($1.start, $1.label) }
    }

    /// `"dog_bow_wow"` → `"dog bow wow"`.
    static func displayName(_ identifier: String) -> String {
        identifier.replacingOccurrences(of: "_", with: " ")
    }

    static let audioExtensions: Set<String> = [
        "wav", "mp3", "m4a", "aac", "flac", "aiff", "aif", "caf", "ogg",
    ]
    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "mkv", "avi"]
    static let mediaExtensions = audioExtensions.union(videoExtensions)
}

import AVFoundation
import CoreGraphics
import Foundation
import SemanticSearch

/// One extracted frame.
public struct ExtractedFrame: Sendable {
    public var image: CGImage
    /// Time the frame was actually taken from, which is the nearest keyframe to
    /// the requested time unless exact seeking was asked for.
    public var time: TimeInterval
    public var index: Int

    public init(image: CGImage, time: TimeInterval, index: Int) {
        self.image = image
        self.time = time
        self.index = index
    }
}

/// Pulls frames out of a video.
///
/// Bounded on purpose: `maximumFrames` is a hard cap, not a hint. A helper that
/// sampled a two-hour film every second would produce 7,200 frames and a
/// multimodal index of them would take a day to build. This is a *basic
/// bounded* clip and frame helper, and the bound is the point.
public struct VideoFrameExtractor: Sendable {

    public struct Configuration: Sendable, Hashable {
        /// Seconds between sampled frames.
        public var interval: TimeInterval
        /// Hard cap. When the interval would produce more than this, the
        /// interval is widened to spread the cap across the whole asset rather
        /// than truncating at the start.
        public var maximumFrames: Int
        /// Downscale so neither side exceeds this.
        public var maximumDimension: Int
        /// Seek exactly rather than to the nearest keyframe.
        ///
        /// On by default, and worth the several-times slowdown. Long-GOP
        /// content — which is most encoded video — can have keyframes seconds
        /// apart, and nearest-keyframe seeking then returns *the same frame* for
        /// every requested time. The result is a dozen identical vectors with a
        /// dozen different timestamps: an index that looks full and answers
        /// every visual query with the same moment.
        public var exactSeeking: Bool
        /// Also embed the whole video on the parent item. Off by default: the
        /// sampled frames already carry the visual evidence with timestamps,
        /// and a whole-video embedding of a long file costs minutes and
        /// gigabytes for one blurred vector.
        public var embedsWholeVideo: Bool

        public init(
            interval: TimeInterval = 5,
            maximumFrames: Int = 64,
            maximumDimension: Int = 1_024,
            exactSeeking: Bool = true,
            embedsWholeVideo: Bool = false
        ) {
            self.interval = max(0.1, interval)
            self.maximumFrames = max(1, maximumFrames)
            self.maximumDimension = max(64, maximumDimension)
            self.exactSeeking = exactSeeking
            self.embedsWholeVideo = embedsWholeVideo
        }

        public static let `default` = Configuration()
    }

    public let configuration: Configuration

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    /// Sample frames across the whole asset.
    public func frames(from url: URL) async throws -> [ExtractedFrame] {
        let duration = try await duration(of: url)
        return try await frames(from: url, at: sampleTimes(duration: duration))
    }

    /// Times to sample, spread across `duration` within the frame cap.
    func sampleTimes(duration: TimeInterval) -> [TimeInterval] {
        // Widen rather than truncate: a cap applied by stopping early indexes
        // the first five minutes of a film and none of the rest.
        var interval = configuration.interval
        let wanted = Int(duration / interval) + 1
        if wanted > configuration.maximumFrames {
            interval = duration / Double(configuration.maximumFrames)
        }

        var times: [TimeInterval] = []
        var time: TimeInterval = 0
        while time < duration, times.count < configuration.maximumFrames {
            times.append(time)
            time += interval
        }
        return times
    }

    private func duration(of url: URL) async throws -> TimeInterval {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        let duration = try await AVURLAsset(url: url).load(.duration).seconds
        guard duration.isFinite, duration > 0 else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "video has no usable duration")
        }
        return duration
    }

    /// The span each frame stands for: from its own time to the next frame's,
    /// and the last one to the end of the asset.
    ///
    /// Not `time + configuration.interval`. The interval is widened on long
    /// assets to respect the frame cap, and a range computed from the
    /// configured value would then cover a fraction of what the frame
    /// represents — and the last one would run past the end of the file.
    static func timeRanges(for frames: [ExtractedFrame], duration: TimeInterval)
        -> [SourceLocator.TimeRange]
    {
        frames.indices.map { position in
            let start = frames[position].time
            let next = position + 1 < frames.count ? frames[position + 1].time : duration
            return SourceLocator.TimeRange(start: start, end: max(start, min(next, duration)))
        }
    }

    /// Extract frames at specific times.
    public func frames(from url: URL, at times: [TimeInterval]) async throws -> [ExtractedFrame] {
        guard !times.isEmpty else { return [] }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }

        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(
            width: configuration.maximumDimension, height: configuration.maximumDimension)
        if configuration.exactSeeking {
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero
        }

        var frames: [ExtractedFrame] = []
        frames.reserveCapacity(times.count)

        for (index, seconds) in times.prefix(configuration.maximumFrames).enumerated() {
            try Task.checkCancellation()
            let requested = CMTime(seconds: seconds, preferredTimescale: 600)
            do {
                let (image, actual) = try await generator.image(at: requested)
                frames.append(
                    ExtractedFrame(image: image, time: actual.seconds, index: index))
            } catch {
                // One unreadable frame — a corrupt GOP, a seek past a truncated
                // tail — should not abandon the other sixty-three.
                continue
            }
        }

        guard !frames.isEmpty else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "no frames could be extracted")
        }
        return frames
    }

    /// Turn a video into items: one parent for the asset and one child per
    /// sampled frame, each carrying the frame image and its timestamp.
    ///
    /// Visual evidence and transcript evidence stay separately addressable —
    /// this produces the visual half, and ``SemanticSearchFluidAudio`` produces
    /// the other. Neither requires the other to exist.
    public func items(
        from url: URL,
        id: ItemID,
        parentID: ItemID? = nil,
        metadata: Metadata = [:]
    ) async throws -> [SearchItem] {
        let duration = try await duration(of: url)
        let extracted = try await frames(from: url, at: sampleTimes(duration: duration))
        let ranges = Self.timeRanges(for: extracted, duration: duration)
        let provenance = Provenance(
            producer: "avfoundation.frames",
            version: "1",
            configurationFingerprint: fingerprint)

        var items: [SearchItem] = [
            SearchItem(
                id: id, parentID: parentID, metadata: metadata,
                locator: SourceLocator(source: url.absoluteString),
                representations: [
                    ContentRepresentation(
                        id: "video", kind: .video, content: .mediaReference(url),
                        provenance: provenance,
                        locator: SourceLocator(source: url.absoluteString),
                        weight: configuration.embedsWholeVideo ? 1 : 0)
                ])
        ]

        for (frame, range) in zip(extracted, ranges) {
            let data = try ImageDecoding.encode(frame.image)
            let locator = SourceLocator(
                source: url.absoluteString,
                frameIndex: frame.index,
                timeRange: range,
                focusTime: frame.time)
            items.append(
                SearchItem(
                    id: ItemID("\(id.rawValue)#frame-\(frame.index)"),
                    parentID: id,
                    metadata: metadata,
                    locator: locator,
                    representations: [
                        ContentRepresentation(
                            id: "frame", kind: .image,
                            content: .imageData(data, contentType: "public.png"),
                            provenance: provenance, locator: locator)
                    ],
                    ordinal: frame.index))
        }
        return items
    }

    public var fingerprint: String {
        StableHash.of { hasher in
            hasher.append("avfoundation.frames.v3")
            hasher.append(configuration.interval)
            hasher.append(configuration.maximumFrames)
            hasher.append(configuration.maximumDimension)
            hasher.append(configuration.exactSeeking)
            hasher.append(configuration.embedsWholeVideo)
        }
    }
}

import AVFoundation
import Accelerate
import CoreGraphics
import Foundation
import SemanticSearch
import Vision

/// A fixed number of frames that together represent a whole video.
///
/// Frame count follows the caller, not the edit. One frame per shot gives a
/// programme cut between two cameras a frame for every cut, most of them the
/// same wide shot; a video is searched as one of many, and a set of frames
/// each unlike the others is what tells it apart.
///
/// No published rule derives the count from duration, shots or content, so
/// the default of 32 is a budget. Frames picked for diversity keep most of
/// the retrieval quality with far fewer of them (six do on 15-second clips
/// nearly as well as sixteen; Wu et al., 2023), and adding frames pays less
/// past one a second (Bain et al., 2022). At six this selector showed mostly
/// title cards and logos. The count is the same for every video on purpose:
/// when a video is matched by its best frame, a video with more frames gets
/// more chances.
///
/// Every step is from the literature rather than tuned here:
///
/// - Candidates: with a ``ShotSegmenter``, the middle frame of every shot,
///   skipping transitions, whose middle is two pictures at once (a dissolve)
///   or one half faded; without one, a frame a second, as VSUMM samples (de Avila et
///   al., 2011). Black candidates are dropped (a mean-luma test), unless
///   every candidate is black.
/// - Descriptors: shots are compared by Vision's image feature print, which
///   keeps one title card apart from the next on the same background;
///   one-a-second candidates by STIMO's 256-bin HSV histogram (16 hue × 4
///   saturation × 4 value; Furini et al., 2010), which folds the seconds of
///   one camera angle together where the feature print tells them apart.
/// - Selection: farthest-point-first, as STIMO clusters (Gonzalez, 1985):
///   the most central candidate, then repeatedly the one farthest from every
///   frame chosen so far. A camera angle that recurs fifty times is near the
///   frame already chosen for it and is not picked again. Greedy facility
///   location rewards what is frequent instead: it gave one recurring wide
///   shot half of the places.
///
/// Each kept frame's time range is the stretch of candidates around it that
/// are nearer to it than to any other kept frame, so a hit still points at a
/// place in the video.
public struct KeyframeExtractor: Sendable {

    public struct Configuration: Sendable, Hashable {
        /// Frames kept per video.
        public var frameCount: Int
        /// Seconds between candidates when there is no segmenter.
        public var candidateInterval: TimeInterval
        /// Cost bound, not a quality setting: past it the candidate interval
        /// widens across the whole video, as ``VideoFrameExtractor`` does.
        /// The default covers 20 minutes at one per second.
        public var maximumCandidates: Int
        /// Downscale so neither side of a kept frame exceeds this.
        public var maximumDimension: Int
        /// Mean luma below which a candidate counts as black.
        public var blackLuma: Double
        /// Also embed the whole video on the parent item. See
        /// ``VideoFrameExtractor/Configuration/embedsWholeVideo``.
        public var embedsWholeVideo: Bool

        public init(
            frameCount: Int = 32,
            candidateInterval: TimeInterval = 1,
            maximumCandidates: Int = 1_200,
            maximumDimension: Int = 1_024,
            blackLuma: Double = 0.05,
            embedsWholeVideo: Bool = false
        ) {
            self.frameCount = max(1, frameCount)
            self.candidateInterval = max(0.1, candidateInterval)
            self.maximumCandidates = max(1, maximumCandidates)
            self.maximumDimension = max(64, maximumDimension)
            self.blackLuma = blackLuma
            self.embedsWholeVideo = embedsWholeVideo
        }

        public static let `default` = Configuration()

        public var fingerprint: String {
            StableHash.of { hasher in
                // v3: transparent frames are judged on white, not black.
                hasher.append("vision.keyframes.v3")
                hasher.append(frameCount)
                hasher.append(candidateInterval)
                hasher.append(maximumCandidates)
                hasher.append(maximumDimension)
                hasher.append(blackLuma)
                hasher.append(embedsWholeVideo)
            }
        }
    }

    /// A kept frame and the stretch of video it stands for.
    public struct Keyframe: Sendable {
        public var image: CGImage
        public var time: TimeInterval
        public var range: SourceLocator.TimeRange
    }

    /// A frame that could be kept: when to take it, and the stretch of video
    /// it stands for on its own — its shot, or its second.
    struct Candidate: Sendable, Hashable {
        var time: TimeInterval
        var start: TimeInterval
        var end: TimeInterval
    }

    public let configuration: Configuration
    /// Where shots come from. `nil` samples a candidate a second.
    public let segmenter: (any ShotSegmenter)?
    private let segmenterFingerprint: String?

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.segmenter = nil
        self.segmenterFingerprint = nil
    }

    public init(configuration: Configuration = .default, segmenter: any ShotSegmenter) async {
        self.configuration = configuration
        self.segmenter = segmenter
        self.segmenterFingerprint = await segmenter.configurationFingerprint
    }

    /// Changes whenever the kept frames would: the configuration and, when
    /// there is one, the segmenter.
    public var fingerprint: String {
        guard let segmenterFingerprint else { return configuration.fingerprint }
        return StableHash.of { hasher in
            hasher.append(configuration.fingerprint)
            hasher.append(segmenterFingerprint)
        }
    }

    /// Candidates are decoded this small: the descriptors scale their input
    /// down further, and only the kept frames are decoded at full size.
    private static let candidateDimension = 360

    public func frames(from url: URL) async throws -> [Keyframe] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        let duration = try await AVURLAsset(url: url).load(.duration).seconds
        guard duration.isFinite, duration > 0 else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "video has no usable duration")
        }

        var candidates: [Candidate] = []
        if let segmenter {
            candidates = Self.candidates(from: try await segmenter.segments(in: url))
        }
        // A video that is one long transition has no shot to take a frame
        // from; seconds still do.
        let shots = !candidates.isEmpty
        if !shots {
            let times = VideoFrameExtractor(
                configuration: .init(
                    interval: configuration.candidateInterval,
                    maximumFrames: configuration.maximumCandidates,
                    maximumDimension: Self.candidateDimension)
            ).sampleTimes(duration: duration)
            candidates = times.enumerated().map { index, time in
                Candidate(
                    time: time, start: time,
                    end: index + 1 < times.count ? times[index + 1] : duration)
            }
        }

        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(
            width: Self.candidateDimension, height: Self.candidateDimension)
        // Exact seeking: long-GOP video returns the same keyframe for every
        // nearby request otherwise. See VideoFrameExtractor.
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        // Only the descriptors are kept: at full size, eleven minutes of
        // candidates would hold gigabytes of pixels to use a few dozen.
        var usable: [Candidate] = []
        var descriptors: [[Float]] = []
        var dark: [Candidate] = []
        var darkDescriptors: [[Float]] = []
        for candidate in candidates {
            try Task.checkCancellation()
            guard
                let (image, _) = try? await generator.image(
                    at: CMTime(seconds: candidate.time, preferredTimescale: 600)),
                let descriptor =
                    shots ? try Self.featurePrint(of: image) : Self.hsvHistogram(of: image)
            else { continue }
            if Self.meanLuma(of: image) >= configuration.blackLuma {
                usable.append(candidate)
                descriptors.append(descriptor)
            } else {
                dark.append(candidate)
                darkDescriptors.append(descriptor)
            }
        }
        // The black test drops fades and slates between pictures; a video
        // dark from end to end — a night sky, fireworks, a podcast uploaded
        // with a black screen, a ProRes 4444 clip whose alpha the image
        // generator drops, leaving a ball on black — is what it is. Thrown,
        // it failed the whole video, speech included. Identical black
        // frames collapse to one in the selection.
        if descriptors.isEmpty {
            usable = dark
            descriptors = darkDescriptors
        }
        guard !descriptors.isEmpty else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "no usable frames: every candidate was unreadable")
        }

        let distances =
            shots ? Self.cosineDistances(descriptors) : Self.manhattanDistances(descriptors)
        let selection = Self.select(distances: distances, count: configuration.frameCount)
        let ranges = Self.ranges(
            selected: selection.selected, assignment: selection.assignment,
            candidates: usable, duration: duration)

        // Asked for at the candidate's own time: a frame's decoded timestamp
        // rounds to a hair before it at 1/600 s and would return the frame
        // before.
        let kept = try await VideoFrameExtractor(
            configuration: .init(
                maximumFrames: selection.selected.count,
                maximumDimension: configuration.maximumDimension)
        ).frames(from: url, at: selection.selected.map { usable[$0].time })
        // `index` is the position in the requested times, so a frame that
        // failed to decode cannot shift the others onto the wrong range.
        return kept.map { frame in
            let candidate = selection.selected[frame.index]
            return Keyframe(image: frame.image, time: frame.time, range: ranges[candidate]!)
        }
    }

    /// The middle of every shot, with the shot as its stretch.
    static func candidates(from segments: [ShotSegment]) -> [Candidate] {
        segments.filter { !$0.kind.isTransition && $0.end > $0.start }.map { segment in
            Candidate(
                time: (segment.start + segment.end) / 2, start: segment.start,
                end: segment.end)
        }
    }

    // MARK: - Selection

    /// Farthest-point-first over a row-major distance matrix (Gonzalez,
    /// 1985): the medoid, then repeatedly the candidate farthest from every
    /// chosen one.
    ///
    /// Returns the chosen candidates in time order, and for every candidate
    /// the chosen one nearest to it. Ties go to the earlier candidate, so the
    /// result depends on nothing but the input.
    static func select(distances: [Float], count: Int) -> (selected: [Int], assignment: [Int]) {
        let total = Int(Double(distances.count).squareRoot().rounded())
        guard total > 0 else { return ([], []) }

        var medoid = 0
        var smallest = Float.infinity
        distances.withUnsafeBufferPointer { matrix in
            for candidate in 0..<total {
                var sum: Float = 0
                vDSP_sve(matrix.baseAddress! + candidate * total, 1, &sum, vDSP_Length(total))
                if sum < smallest {
                    smallest = sum
                    medoid = candidate
                }
            }
        }

        var chosen = [medoid]
        var nearest = Array(distances[(medoid * total)..<((medoid + 1) * total)])
        var owner = [Int](repeating: medoid, count: total)
        while chosen.count < min(count, total) {
            var farthest = 0
            var largest: Float = -1
            for candidate in 0..<total where nearest[candidate] > largest {
                largest = nearest[candidate]
                farthest = candidate
            }
            // Everything left is a copy of something chosen.
            guard largest > 0 else { break }
            chosen.append(farthest)
            for other in 0..<total where distances[farthest * total + other] < nearest[other] {
                nearest[other] = distances[farthest * total + other]
                owner[other] = farthest
            }
        }
        return (chosen.sorted(), owner)
    }

    /// 1 − cos for every pair, row-major.
    static func cosineDistances(_ descriptors: [[Float]]) -> [Float] {
        let total = descriptors.count
        let dimensions = descriptors[0].count
        var matrix = [Float](repeating: 0, count: total * dimensions)
        for (index, descriptor) in descriptors.enumerated() {
            var norm: Float = 0
            vDSP_svesq(descriptor, 1, &norm, vDSP_Length(dimensions))
            let scale = norm > 0 ? 1 / norm.squareRoot() : 0
            for dimension in 0..<min(dimensions, descriptor.count) {
                matrix[index * dimensions + dimension] = descriptor[dimension] * scale
            }
        }
        var cosine = [Float](repeating: 0, count: total * total)
        var transposed = [Float](repeating: 0, count: total * dimensions)
        vDSP_mtrans(matrix, 1, &transposed, 1, vDSP_Length(dimensions), vDSP_Length(total))
        vDSP_mmul(
            matrix, 1, transposed, 1, &cosine, 1,
            vDSP_Length(total), vDSP_Length(total), vDSP_Length(dimensions))
        return cosine.map { max(0, 1 - $0) }
    }

    /// Sum of absolute differences for every pair, row-major: the L1
    /// distance of normalized histograms.
    static func manhattanDistances(_ descriptors: [[Float]]) -> [Float] {
        let total = descriptors.count
        var result = [Float](repeating: 0, count: total * total)
        var difference = [Float](repeating: 0, count: descriptors[0].count)
        for row in 0..<total {
            for column in (row + 1)..<max(row + 1, total) {
                let length = vDSP_Length(
                    min(descriptors[row].count, descriptors[column].count))
                vDSP_vsub(descriptors[column], 1, descriptors[row], 1, &difference, 1, length)
                var sum: Float = 0
                vDSP_svemg(difference, 1, &sum, length)
                result[row * total + column] = sum
                result[column * total + row] = sum
            }
        }
        return result
    }

    /// For each chosen candidate, the unbroken run of candidates around it
    /// that are nearest to it, from the run's first start to its last end.
    static func ranges(
        selected: [Int], assignment: [Int], candidates: [Candidate], duration: TimeInterval
    ) -> [Int: SourceLocator.TimeRange] {
        var result: [Int: SourceLocator.TimeRange] = [:]
        for candidate in selected {
            var first = candidate
            while first > 0, assignment[first - 1] == candidate { first -= 1 }
            var last = candidate
            while last + 1 < candidates.count, assignment[last + 1] == candidate { last += 1 }
            let start = candidates[first].start
            result[candidate] = SourceLocator.TimeRange(
                start: start, end: max(start, min(candidates[last].end, duration)))
        }
        return result
    }

    // MARK: - Descriptors

    static func featurePrint(of image: CGImage) throws -> [Float]? {
        let request = VNGenerateImageFeaturePrintRequest()
        // Pinned: a new revision is a different vector space, and a
        // selection should not change under an OS update.
        request.revision = VNGenerateImageFeaturePrintRequestRevision2
        try VNImageRequestHandler(cgImage: image).perform([request])
        guard let observation = request.results?.first,
            observation.elementType == .float
        else { return nil }
        return observation.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// STIMO's frame descriptor: a 256-bin HSV histogram, 16 hue × 4
    /// saturation × 4 value, normalized to sum to one. Drawn at 160×90:
    /// a histogram needs no more pixels than that.
    static func hsvHistogram(of image: CGImage) -> [Float]? {
        let width = 160
        let height = 90
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            Self.drawOnWhite(image, in: context, width: width, height: height)
            return true
        }
        guard drawn else { return nil }
        var histogram = [Float](repeating: 0, count: 256)
        for pixel in 0..<(width * height) {
            let r = Float(pixels[pixel * 4]) / 255
            let g = Float(pixels[pixel * 4 + 1]) / 255
            let b = Float(pixels[pixel * 4 + 2]) / 255
            let value = max(r, g, b)
            let chroma = value - min(r, g, b)
            var hue: Float = 0
            if chroma > 0 {
                if value == r {
                    hue = ((g - b) / chroma).truncatingRemainder(dividingBy: 6)
                    if hue < 0 { hue += 6 }
                } else if value == g {
                    hue = (b - r) / chroma + 2
                } else {
                    hue = (r - g) / chroma + 4
                }
                hue /= 6
            }
            let saturation = value > 0 ? chroma / value : 0
            let bin =
                min(Int(hue * 16), 15) * 16 + min(Int(saturation * 4), 3) * 4
                + min(Int(value * 4), 3)
            histogram[bin] += 1
        }
        let count = Float(width * height)
        return histogram.map { $0 / count }
    }

    /// A frame as it is seen: a video with an alpha channel (HEVC with
    /// alpha, ProRes 4444) is shown over something, and on the zeroed canvas
    /// its transparent background read as black — both clips of a ball on a
    /// transparent background were dropped as black frames and failed to
    /// index. White, as `PDFProcessor` renders a page.
    static func drawOnWhite(_ image: CGImage, in context: CGContext, width: Int, height: Int) {
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(bounds)
        context.draw(image, in: bounds)
    }

    /// Mean luma, 0…1, from a 32×18 greyscale thumbnail.
    static func meanLuma(of image: CGImage) -> Double {
        let width = 32
        let height = 18
        var pixels = [UInt8](repeating: 0, count: width * height)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return false }
            Self.drawOnWhite(image, in: context, width: width, height: height)
            return true
        }
        guard drawn else { return 0 }
        return Double(pixels.reduce(0) { $0 + Int($1) }) / Double(pixels.count) / 255
    }

    // MARK: - Items

    /// A parent for the video and one child per kept frame.
    public func items(
        from url: URL,
        id: ItemID,
        parentID: ItemID? = nil,
        metadata: Metadata = [:]
    ) async throws -> [SearchItem] {
        let frames = try await frames(from: url)
        let provenance = Provenance(
            producer: KeyframeProcessor.producer, version: "2",
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
        for (index, frame) in frames.enumerated() {
            let data = try ImageDecoding.encode(frame.image)
            let locator = SourceLocator(
                source: url.absoluteString,
                frameIndex: index,
                timeRange: frame.range,
                focusTime: frame.time)
            items.append(
                SearchItem(
                    id: ItemID("\(id.rawValue)#keyframe-\(index)"),
                    parentID: id,
                    metadata: metadata,
                    locator: locator,
                    representations: [
                        ContentRepresentation(
                            id: "frame", kind: .image,
                            content: .imageData(data, contentType: "public.png"),
                            provenance: provenance, locator: locator)
                    ],
                    ordinal: index))
        }
        return items
    }
}

/// ``KeyframeExtractor`` as a ``ContentProcessor`` for video files.
///
/// Combine it with a transcription processor through
/// ``CombinedContentProcessor`` to index a video's pictures and speech under
/// one parent, or wrap it in `CaptioningProcessor` for the text profile.
public struct KeyframeProcessor: ContentProcessor {

    static let producer = "vision.keyframes"

    public let extractor: KeyframeExtractor

    public init(configuration: KeyframeExtractor.Configuration = .default) {
        self.extractor = KeyframeExtractor(configuration: configuration)
    }

    /// Keyframes taken from shots: the middle of each, never of a transition.
    public init(
        configuration: KeyframeExtractor.Configuration = .default, segmenter: any ShotSegmenter
    ) async {
        self.extractor = await KeyframeExtractor(
            configuration: configuration, segmenter: segmenter)
    }

    public var identifier: String { Self.producer }
    public var version: String { "2" }

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsVideo: true, contentTypes: ["public.movie"],
            producedKinds: [.image, .video])
    }

    public var configurationFingerprint: String { extractor.fingerprint }

    public func canProcess(_ content: RawContent) -> Bool {
        guard case .fileURL(let url) = content.payload else { return false }
        return Self.videoExtensions.contains(url.pathExtension.lowercased())
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .fileURL(let url) = content.payload else {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "KeyframeProcessor needs a video file URL")
        }
        onProgress?(ProcessingProgress(itemID: content.id, stage: "keyframes", fraction: 0))
        let items = try await extractor.items(
            from: url, id: content.id, parentID: content.parentID,
            metadata: content.metadata)
        onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
        return items
    }

    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "mkv", "avi"]
}

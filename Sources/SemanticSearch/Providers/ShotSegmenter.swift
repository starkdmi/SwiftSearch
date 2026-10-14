import Foundation

/// One stretch of a video between two boundaries: a shot, or the transition
/// between two shots.
///
/// The labels are OmniShotCut's. ``kind`` says what the stretch itself is — a
/// shot, or a dissolve, wipe or fade on its way to the next one — and
/// ``boundary`` says how it began. A dissolve's middle frame is two pictures
/// at once and a fade's is half black, which is why a frame picker takes
/// frames from shots only.
public struct ShotSegment: Sendable, Hashable, Codable {

    /// What the segment is. Raw values are OmniShotCut's intra-relation ids.
    public enum Kind: Int, Sendable, Hashable, Codable {
        case shot = 0
        case dissolve = 1
        case wipe = 2
        case push = 3
        case slide = 4
        case zoom = 5
        /// Fade to or from black or white, including a dip between shots.
        case fade = 6
        case doorway = 7
        /// The model's padding class; it marks frames past a window's end and
        /// should not survive stitching.
        case padding = 8

        /// Anything but a shot is on its way between pictures: a dissolve's
        /// middle frame is two of them, a fade's is one half gone to black —
        /// a studio logo, dim, beside the same logo in full.
        public var isTransition: Bool { self != .shot }
    }

    /// How the segment starts. Raw values are OmniShotCut's inter-relation ids.
    public enum Boundary: Int, Sendable, Hashable, Codable {
        case videoStart = 0
        case hardCut = 1
        case transitionSource = 2
        case transition = 3
        /// A jump within one camera setup — a cut in an interview.
        case suddenJump = 4
        case padding = 5
    }

    public var start: TimeInterval
    public var end: TimeInterval
    public var kind: Kind
    public var boundary: Boundary

    public init(start: TimeInterval, end: TimeInterval, kind: Kind, boundary: Boundary) {
        self.start = start
        self.end = end
        self.kind = kind
        self.boundary = boundary
    }

    public var duration: TimeInterval { end - start }
}

/// Splits a video into shots and transitions. Implement this to use a
/// different detector; the bundled one is OmniShotCut on Core ML
/// (`OmniShotCutSegmenter` in `SemanticSearchShots`).
public protocol ShotSegmenter: Actor {
    /// Model identity: a Hub id, or the path of a local conversion.
    var identifier: String { get }
    /// Changes whenever the segments would: model, weights, windowing.
    var configurationFingerprint: String { get }

    func prepare() async throws
    func release() async
    var isPrepared: Bool { get }

    /// Segments covering the whole video, in order, without gaps.
    func segments(in video: URL) async throws -> [ShotSegment]
}

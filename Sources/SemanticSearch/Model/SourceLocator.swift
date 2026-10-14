import Foundation

/// Where a representation came from, precisely enough to take a reader back to
/// it.
///
/// A hit is only useful if it can be opened. The engine carries the locator
/// through retrieval untouched and hands it back on the result, so "page 4,
/// characters 120–480" survives chunking, fusion and reranking without the
/// caller having to re-derive it from text offsets.
public struct SourceLocator: Sendable, Hashable, Codable {

    /// Time span within an audio or video source, in seconds.
    public struct TimeRange: Sendable, Hashable, Codable {
        public var start: TimeInterval
        public var end: TimeInterval

        public init(start: TimeInterval, end: TimeInterval) {
            self.start = max(0, start)
            // A zero-width range breaks interval arithmetic downstream (cluster
            // merging, overlap tests), so the end is nudged rather than clamped
            // to the start.
            self.end = end > self.start ? end : self.start + 0.000_001
        }

        public var duration: TimeInterval { end - start }

        public func overlaps(_ other: TimeRange) -> Bool {
            start < other.end && other.start < end
        }
    }

    /// Half-open range of UTF-16 offsets into the representation's own text.
    ///
    /// UTF-16 rather than `String.Index` because the range has to survive a
    /// round trip through SQLite and JSON, and rather than byte offsets because
    /// every Apple text API (`NSRange`, `NSLinguisticTagger`, `NSAttributedString`)
    /// already speaks UTF-16.
    public struct CharacterRange: Sendable, Hashable, Codable {
        public var location: Int
        public var length: Int

        public init(location: Int, length: Int) {
            self.location = max(0, location)
            self.length = max(0, length)
        }

        public var upperBound: Int { location + length }

        public func range(in text: String) -> Range<String.Index>? {
            let utf16 = text.utf16
            guard
                let lower = utf16.index(
                    utf16.startIndex, offsetBy: location, limitedBy: utf16.endIndex),
                let upper = utf16.index(lower, offsetBy: length, limitedBy: utf16.endIndex),
                let lowerIndex = lower.samePosition(in: text),
                let upperIndex = upper.samePosition(in: text)
            else { return nil }
            return lowerIndex..<upperIndex
        }
    }

    /// Normalised rectangle, origin top-left, in the unit square of a page or
    /// frame. Normalised rather than in points so a locator stays valid when
    /// the same page is rasterised at a different scale.
    public struct BoundingBox: Sendable, Hashable, Codable {
        public var x: Double
        public var y: Double
        public var width: Double
        public var height: Double

        public init(x: Double, y: Double, width: Double, height: Double) {
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        }

        public var minX: Double { x }
        public var minY: Double { y }
        public var maxX: Double { x + width }
        public var maxY: Double { y + height }

        public func union(_ other: BoundingBox) -> BoundingBox {
            let minX = Swift.min(self.minX, other.minX)
            let minY = Swift.min(self.minY, other.minY)
            let maxX = Swift.max(self.maxX, other.maxX)
            let maxY = Swift.max(self.maxY, other.maxY)
            return BoundingBox(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        }
    }

    /// The file, URL or opaque address the content was read from.
    public var source: String?
    /// Page or slide, zero-based.
    public var pageIndex: Int?
    /// Frame index within a video source.
    public var frameIndex: Int?
    public var timeRange: TimeRange?
    public var characterRange: CharacterRange?
    public var boundingBoxes: [BoundingBox]
    /// Point within `timeRange` that best represents it — the frame a video hit
    /// should be shown at, rather than the start of the window it was found in.
    public var focusTime: TimeInterval?

    public init(
        source: String? = nil,
        pageIndex: Int? = nil,
        frameIndex: Int? = nil,
        timeRange: TimeRange? = nil,
        characterRange: CharacterRange? = nil,
        boundingBoxes: [BoundingBox] = [],
        focusTime: TimeInterval? = nil
    ) {
        self.source = source
        self.pageIndex = pageIndex
        self.frameIndex = frameIndex
        self.timeRange = timeRange
        self.characterRange = characterRange
        self.boundingBoxes = boundingBoxes
        self.focusTime = focusTime
    }

    public static let unspecified = SourceLocator()

    public var isEmpty: Bool {
        source == nil && pageIndex == nil && frameIndex == nil && timeRange == nil
            && characterRange == nil && boundingBoxes.isEmpty && focusTime == nil
    }

    /// Narrow this locator to a sub-span of the representation's text.
    ///
    /// Used by the chunker: each chunk keeps its parent's page and source but
    /// carries its own character range, so a hit on chunk 3 of a page reports
    /// where on the page it actually is.
    public func narrowed(toCharacters range: CharacterRange) -> SourceLocator {
        var copy = self
        if let existing = characterRange {
            copy.characterRange = CharacterRange(
                location: existing.location + range.location, length: range.length)
        } else {
            copy.characterRange = range
        }
        return copy
    }

    /// Narrow this locator to a sub-span of the representation's time range.
    public func narrowed(toTime range: TimeRange) -> SourceLocator {
        var copy = self
        copy.timeRange = range
        return copy
    }
}

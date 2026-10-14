import Foundation
import Testing

@testable import SemanticSearch
@testable import SemanticSearchApple

/// Fixed-count keyframes: farthest-point-first over candidate descriptors,
/// shots as candidates, and the extractor end to end on a generated movie.
@Suite("Keyframes")
struct KeyframeTests {

    /// `count` copies of a unit vector along `axis`, nudged so no two are
    /// identical.
    private static func cluster(axis: Int, count: Int, dimensions: Int = 8) -> [[Float]] {
        (0..<count).map { copy in
            var vector = [Float](repeating: 0, count: dimensions)
            vector[axis] = 1
            vector[(axis + 1) % dimensions] = Float(copy) * 0.001
            return vector
        }
    }

    private static func select(_ descriptors: [[Float]], count: Int)
        -> (selected: [Int], assignment: [Int])
    {
        KeyframeExtractor.select(
            distances: KeyframeExtractor.cosineDistances(descriptors), count: count)
    }

    @Test("A camera angle that recurs twenty times earns one frame, not the budget")
    func repeatsEarnOneFrame() {
        let descriptors =
            Self.cluster(axis: 0, count: 20) + Self.cluster(axis: 2, count: 1)
            + Self.cluster(axis: 4, count: 1)
        let selection = Self.select(descriptors, count: 3)
        #expect(selection.selected.count == 3)
        #expect(selection.selected.filter { $0 < 20 }.count == 1)
        #expect(selection.selected.contains(20))
        #expect(selection.selected.contains(21))
    }

    @Test("Every candidate is assigned to the chosen frame of its own cluster")
    func assignmentFollowsClusters() {
        let descriptors =
            Self.cluster(axis: 0, count: 5) + Self.cluster(axis: 3, count: 5)
        let selection = Self.select(descriptors, count: 2)
        let first = selection.assignment[0]
        let second = selection.assignment[5]
        #expect(first != second)
        #expect(selection.assignment[0..<5].allSatisfy { $0 == first })
        #expect(selection.assignment[5..<10].allSatisfy { $0 == second })
    }

    @Test("Fewer candidates than the budget keeps them all, in order")
    func shortVideoKeepsAll() {
        let descriptors = Self.cluster(axis: 0, count: 1) + Self.cluster(axis: 5, count: 1)
        #expect(Self.select(descriptors, count: 6).selected == [0, 1])
    }

    @Test("Identical candidates are one frame, the earliest")
    func copiesAreOneFrame() {
        let same = [[Float]](repeating: [1, 0, 0], count: 4)
        #expect(Self.select(same, count: 3).selected == [0])
    }

    @Test("Histograms are compared by L1")
    func manhattan() {
        let distances = KeyframeExtractor.manhattanDistances([[0.5, 0.5, 0], [0, 0.5, 0.5]])
        #expect(distances == [0, 1, 1, 0])
    }

    @Test("A frame's range runs from the first stretch it represents to the last")
    func rangesAreRuns() {
        // Candidates 0-2 belong to 1, 3-4 to 4, 5 to 1 again.
        let assignment = [1, 1, 1, 4, 4, 1]
        let candidates = (0..<6).map {
            KeyframeExtractor.Candidate(
                time: Double($0) + 0.5, start: Double($0), end: Double($0) + 1)
        }
        let ranges = KeyframeExtractor.ranges(
            selected: [1, 4], assignment: assignment, candidates: candidates, duration: 5.5)
        #expect(ranges[1]?.start == 0)
        #expect(ranges[1]?.end == 3)
        #expect(ranges[4]?.start == 3)
        #expect(ranges[4]?.end == 5)
    }

    @Test("Shots give their middle frame; a dissolve or a fade gives none")
    func candidatesSkipTransitions() {
        let segments = [
            ShotSegment(start: 0, end: 4, kind: .shot, boundary: .videoStart),
            ShotSegment(start: 4, end: 5, kind: .dissolve, boundary: .transitionSource),
            ShotSegment(start: 5, end: 9, kind: .shot, boundary: .transition),
            ShotSegment(start: 9, end: 11, kind: .fade, boundary: .transitionSource),
        ]
        let candidates = KeyframeExtractor.candidates(from: segments)
        #expect(candidates.map(\.time) == [2, 7])
        #expect(candidates.map(\.start) == [0, 5])
    }

    @Test("A movie becomes a parent and the requested number of timed frames")
    func extractorEndToEnd() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("clip.mov")
        try await AppleHelpersTests.makeMovie(at: movie, seconds: 6)

        let items = try await KeyframeExtractor(
            configuration: .init(frameCount: 3, maximumDimension: 128)
        ).items(from: movie, id: "video")

        let parent = try #require(items.first)
        #expect(parent.id == "video")
        let frames = Array(items.dropFirst())
        #expect(frames.count == 3)
        #expect(frames.allSatisfy { $0.parentID == "video" })
        #expect(frames.allSatisfy { $0.representations.first?.kind == .image })
        let focus = frames.compactMap(\.locator.focusTime)
        #expect(focus == focus.sorted())
        for frame in frames {
            let range = try #require(frame.locator.timeRange)
            let time = try #require(frame.locator.focusTime)
            #expect(range.start <= time + 0.001 && time <= range.end + 0.001)
            #expect(range.end <= 6.001)
        }
    }

    /// Segments as a detector would return them, without a model.
    actor ScriptedSegmenter: ShotSegmenter {
        let result: [ShotSegment]
        init(_ result: [ShotSegment]) { self.result = result }
        var identifier: String { "scripted" }
        var configurationFingerprint: String { "scripted.v1" }
        var isPrepared: Bool { true }
        func prepare() async throws {}
        func release() async {}
        func segments(in video: URL) async throws -> [ShotSegment] { result }
    }

    @Test("With a segmenter, frames come from shots and never from a dissolve")
    func framesFromShots() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("clip.mov")
        try await AppleHelpersTests.makeMovie(at: movie, seconds: 6)
        let segmenter = ScriptedSegmenter([
            ShotSegment(start: 0, end: 2, kind: .shot, boundary: .videoStart),
            ShotSegment(start: 2, end: 4, kind: .dissolve, boundary: .transitionSource),
            ShotSegment(start: 4, end: 6, kind: .shot, boundary: .transition),
        ])
        let extractor = await KeyframeExtractor(
            configuration: .init(frameCount: 3, maximumDimension: 128), segmenter: segmenter)
        let frames = try await extractor.frames(from: movie)
        #expect(!frames.isEmpty && frames.count <= 2)
        #expect(frames.allSatisfy { !(2..<4).contains($0.time) })
        #expect(extractor.fingerprint != KeyframeExtractor().fingerprint)
    }

    @Test("A missing video is reported as unavailable media")
    func missingVideo() async {
        let missing = URL(fileURLWithPath: "/tmp/no-such-video-\(UUID().uuidString).mov")
        await #expect(throws: SemanticSearchError.self) {
            _ = try await KeyframeExtractor().frames(from: missing)
        }
    }
}

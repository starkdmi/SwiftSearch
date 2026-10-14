import CoreGraphics
import Foundation
import SemanticSearch
import SemanticSearchApple
import XCTest

@testable import SemanticSearchMLX

/// The SmolVLM2 captioner on its own.
final class CaptionIntegrationTests: XCTestCase {

    /// Pure; runs without the gate. A picture is brought down to a 1024 long
    /// side, and further when that leaves its short side just over a tile.
    func testCaptionImageSizeFitsTheVisionTiles() {
        func size(_ width: Int, _ height: Int) -> [Int]? {
            MLXImageCaptioner.preferredSize(width: width, height: height).map {
                [$0.width, $0.height]
            }
        }
        // 16:9 at a 1024 long side leaves the short side at 576 — just over
        // one 512 tile — so it is brought down to one tile row.
        XCTAssertEqual(size(1_920, 1_080), [896, 504])
        XCTAssertEqual(size(1_280, 720), [896, 504])
        XCTAssertEqual(size(1_080, 1_920), [504, 896])
        // 4:3 at 1024 is 768 short: two tile rows either way, so keep 1024.
        XCTAssertEqual(size(4_000, 3_000), [1_024, 768])
        // Never upscales.
        XCTAssertNil(size(800, 600))
        XCTAssertNil(size(512, 512))
    }

    func testCaptionsARenderedScene() async throws {
        try XCTSkipUnless(
            Integration.runsCaptionTests,
            "set SEMANTIC_SEARCH_RUN_CAPTION_TESTS=1 to run the SmolVLM2 captioner")

        let size = 640
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        context.setFillColor(CGColor(red: 0.9, green: 0.05, blue: 0.05, alpha: 1))
        context.fillEllipse(in: CGRect(x: 160, y: 160, width: 320, height: 320))
        let data = try ImageDecoding.encode(try XCTUnwrap(context.makeImage()))

        let captioner = ModelProfile.Captioning.captioner(model: Integration.captionModel)
        let loadStarted = ContinuousClock.now
        try await captioner.prepare()
        let loaded = ContinuousClock.now - loadStarted

        let started = ContinuousClock.now
        let first = try await captioner.caption(.encoded(data, contentType: "public.png"))
        let elapsed = ContinuousClock.now - started
        let second = try await captioner.caption(.encoded(data, contentType: "public.png"))
        await captioner.release()

        Integration.measure(
            "caption.rendered",
            [
                "load_ms": String(format: "%.0f", Integration.milliseconds(loaded)),
                "caption_ms": String(format: "%.0f", Integration.milliseconds(elapsed)),
                "caption": first,
                "peak_mb": String(format: "%.0f", Integration.peakFootprintMegabytes()),
            ])
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(first, second, "greedy decoding should be deterministic")
        let words = Set(first.lowercased().split { !$0.isLetter }.map(String.init))
        XCTAssertFalse(
            words.isDisjoint(with: ["red", "circle", "ball", "dot", "sphere", "round"]),
            "caption should describe a red circle: \(first)")
    }
}

import CoreGraphics
import Foundation
import SemanticSearch
import SemanticSearchApple
import XCTest

@testable import SemanticSearchMLX

/// The captioner on its own.
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

    /// Pure. Each model gets its own prompt unless one is given; any but
    /// LFM2-VL keeps the one every model had before.
    func testPromptsPerModel() async {
        typealias Configuration = MLXImageCaptioner.Configuration
        XCTAssertEqual(Configuration().prompt, "Provide a caption for an image.")
        XCTAssertEqual(
            Configuration(model: "mlx-community/Qwen2-VL-2B-Instruct-4bit").prompt,
            "Provide a caption for an image.")
        XCTAssertEqual(
            Configuration(model: MLXImageCaptioner.lfm25VL450M).prompt,
            "Describe this image in one sentence.")
        XCTAssertEqual(
            Configuration(model: MLXImageCaptioner.lfm25VL450M, prompt: "Describe this image.")
                .prompt,
            "Describe this image.")
        // Changing the model changes a prompt nobody chose, and only that.
        var configuration = Configuration()
        configuration.model = MLXImageCaptioner.lfm25VL450M
        XCTAssertEqual(configuration.prompt, "Describe this image in one sentence.")
        configuration.prompt = "Describe this image."
        configuration.model = MLXImageCaptioner.smolVLM2
        XCTAssertEqual(configuration.prompt, "Describe this image.")
        let captioner = ModelProfile.Captioning.captioner(model: MLXImageCaptioner.lfm25VL450M)
        let prompt = await captioner.configuration.prompt
        XCTAssertEqual(prompt, "Describe this image in one sentence.")
    }

    func testCaptionsARenderedScene() async throws {
        try XCTSkipUnless(
            Integration.runsCaptionTests,
            "set SEMANTIC_SEARCH_RUN_CAPTION_TESTS=1 to run the captioner")

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

        let captioner = ModelProfile.Captioning.captioner(
            model: Integration.captionModel, prompt: Integration.captionPrompt)
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

    /// The library's two pictures, a recipe card and a timetable: what the
    /// captioner says about each and how long it takes. Unscored but for
    /// saying something; the numbers in `MLXImageCaptioner`'s documentation
    /// come from here.
    func testCaptionsTheLibraryPictures() async throws {
        try XCTSkipUnless(
            Integration.runsCaptionTests,
            "set SEMANTIC_SEARCH_RUN_CAPTION_TESTS=1 to run the captioner")

        let captioner = ModelProfile.Captioning.captioner(
            model: Integration.captionModel, prompt: Integration.captionPrompt)
        let loadStarted = ContinuousClock.now
        try await captioner.prepare()
        let loaded = ContinuousClock.now - loadStarted
        for name in ["bramble-recipe.png", "oxford-timetable.png"] {
            let url = try Integration.library(name)
            // Once to warm the kernels, once timed.
            _ = try await captioner.caption(.fileURL(url))
            let started = ContinuousClock.now
            let caption = try await captioner.caption(.fileURL(url))
            let elapsed = ContinuousClock.now - started
            Integration.measure(
                "caption.library",
                [
                    "file": name,
                    "model": Integration.captionModel.identity,
                    "prompt": await captioner.configuration.prompt,
                    "load_ms": String(format: "%.0f", Integration.milliseconds(loaded)),
                    "caption_ms": String(format: "%.0f", Integration.milliseconds(elapsed)),
                    "caption": caption,
                    "peak_mb": String(format: "%.0f", Integration.peakFootprintMegabytes()),
                ])
            XCTAssertFalse(caption.isEmpty, name)
        }
        await captioner.release()
    }
}

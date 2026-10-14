import AVFoundation
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import SemanticSearch
import SemanticSearchApple
import SemanticSearchMLX
import XCTest

/// The multimodal reference profile: Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B.
///
/// What this profile buys, and what these tests check, is one shared space for
/// text and pixels — image-to-text and text-to-image retrieval in the same
/// index, without a second model and without fusing across two spaces on ranks.
final class MultimodalIntegrationTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(
            Integration.runsMultimodalTests,
            "set SEMANTIC_SEARCH_RUN_MULTIMODAL_TESTS=1 to run the 2B multimodal tests")
    }

    // MARK: - Fixtures

    /// Render a word into a bitmap, so a visual document has known content.
    private func textImage(_ text: String) throws -> CGImage {
        let size = CGSize(width: 800, height: 240)
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: Int(size.width), height: Int(size.height),
                bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(origin: .zero, size: size))

        let font = CTFontCreateWithName("Helvetica" as CFString, 84, nil)
        let attributed = NSAttributedString(
            string: text,
            attributes: [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                    CGColor(red: 0, green: 0, blue: 0, alpha: 1),
            ])
        context.textPosition = CGPoint(x: 40, y: 90)
        CTLineDraw(CTLineCreateWithAttributedString(attributed), context)
        return try XCTUnwrap(context.makeImage())
    }

    private func imageData(_ text: String) throws -> Data {
        try ImageDecoding.encode(try textImage(text))
    }

    private func makeProvider() -> MLXEmbeddingProvider {
        ModelProfile.Multimodal.embeddingProvider(model: Integration.multimodalEmbeddingModel)
    }

    private func makeEngine(storeURL: URL?) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "multimodal",
                        provider: makeProvider(),
                        kinds: Set(RepresentationKind.allCases),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                reranker: ModelProfile.Multimodal.rerankingProvider(
                    model: Integration.multimodalRerankerModel),
                // One model resident at a time: each is ~1.9 GB at 5 bits.
                runtime: RuntimePolicy(sequentialModelResidency: true)))
        try await engine.prepare()
        return engine
    }

    // MARK: - Budgets

    /// Text beyond the provider's budget is cut and the cut is reported —
    /// before, the multimodal backend embedded any length and always said
    /// `wasTruncated: false`.
    func testLongTextIsTruncatedAndReported() async throws {
        let provider = MLXEmbeddingProvider(
            configuration: MLXEmbeddingProvider.Configuration(
                model: Integration.multimodalEmbeddingModel,
                defaultQueryInstruction: MLXEmbeddingProvider.qwen3VLDefaultInstruction,
                dimensions: ModelProfile.Multimodal.dimensions,
                maximumBatchSize: 1,
                maximumInputTokens: 32))
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let long = String(repeating: "The harbour lights came on one by one. ", count: 40)
        let outputs = try await provider.embed([
            EmbeddingInput(text: long), EmbeddingInput(text: "A short caption."),
        ])
        XCTAssertEqual(outputs.count, 2)
        XCTAssertTrue(outputs[0].wasTruncated)
        XCTAssertFalse(outputs[1].wasTruncated)
        XCTAssertEqual(outputs[0].vector.count, ModelProfile.Multimodal.dimensions)
    }

    /// Parent context goes first, then the candidate's own text; both are
    /// reported.
    func testRerankerFitsLongPairsIntoItsBudget() async throws {
        let reranker = MLXRerankingProvider(
            configuration: MLXRerankingProvider.Configuration(
                model: Integration.multimodalRerankerModel,
                maximumPairTokens: 160, batchSize: 1, maximumImagesPerPair: 2))
        try await reranker.prepare()
        defer { Task { await reranker.release() } }

        let long = String(repeating: "Tide tables for the northern harbour. ", count: 60)
        let scores = try await reranker.score(
            query: RerankQuery(text: "harbour tides", instruction: "Find tide tables."),
            candidates: [
                RerankCandidate(text: long, parentContext: long),
                RerankCandidate(text: "Tide tables.", parentContext: long),
                RerankCandidate(text: "Tide tables."),
            ])
        XCTAssertEqual(scores.count, 3)
        XCTAssertTrue(scores[0].wasTruncated)
        XCTAssertTrue(scores[0].droppedParentContext)
        XCTAssertFalse(scores[1].wasTruncated)
        XCTAssertTrue(scores[1].droppedParentContext)
        XCTAssertFalse(scores[2].wasTruncated)
        XCTAssertFalse(scores[2].droppedParentContext)
        XCTAssertTrue(scores.allSatisfy { $0.score.isFinite })
    }

    // MARK: - Orientation

    /// `image` as a phone stores a portrait: the pixels a quarter turn
    /// counter-clockwise, the JPEG tagged 6 — turn clockwise to show — so it
    /// shows as `image`.
    private func phoneJPEG(_ image: CGImage, tagged: Bool) throws -> Data {
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: image.height, height: image.width, bitsPerComponent: 8,
                bytesPerRow: 0, space: try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB)),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.translateBy(x: CGFloat(image.height), y: 0)
        context.rotate(by: .pi / 2)
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return try jpeg(try XCTUnwrap(context.makeImage()), orientation: tagged ? 6 : nil)
    }

    private func jpeg(_ image: CGImage, orientation: Int?) throws -> Data {
        let data = NSMutableData()
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil))
        var properties: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 1.0]
        if let orientation { properties[kCGImagePropertyOrientation] = orientation }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return data as Data
    }

    /// The runtime decodes an image URL with `CIImage(contentsOf:)`, which
    /// ignores the EXIF orientation: a portrait was embedded as the sideways
    /// file below. Read upright, it embeds as the picture it shows.
    func testPhonePortraitEmbedsUpright() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let picture = try textImage("LIGHTHOUSE")
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tagged = directory.appendingPathComponent("portrait.jpg")
        let sideways = directory.appendingPathComponent("sideways.jpg")
        try phoneJPEG(picture, tagged: true).write(to: tagged)
        try phoneJPEG(picture, tagged: false).write(to: sideways)

        func embed(_ payload: ImagePayload) async throws -> [Float] {
            try await provider.embed(EmbeddingInput(images: [payload])).vector
        }
        let upright = try await embed(
            .encoded(try jpeg(picture, orientation: nil), contentType: "public.jpeg"))
        let fromTagged = try await embed(.fileURL(tagged))
        let fromSideways = try await embed(.fileURL(sideways))
        let fromTaggedBytes = try await embed(
            .encoded(try Data(contentsOf: tagged), contentType: "public.jpeg"))

        let taggedCosine = VectorMath.dot(upright, fromTagged)
        let sidewaysCosine = VectorMath.dot(upright, fromSideways)
        print(
            String(
                format: "[orientation] embed upright·tagged %.4f upright·sideways %.4f",
                taggedCosine, sidewaysCosine))
        XCTAssertEqual(VectorMath.dot(fromTagged, fromTaggedBytes), 1, accuracy: 1e-4)
        // 1.0000 and 0.3433 with the 5-bit pair.
        XCTAssertGreaterThan(taggedCosine, 0.99)
        XCTAssertLessThan(sidewaysCosine, taggedCosine - 0.1)
    }

    /// The reranker reads the same file the embedder does, upright, on its
    /// fast path (the default) as on mlx-swift-lm's.
    func testPhonePortraitIsRerankedUpright() async throws {
        let picture = try textImage("LIGHTHOUSE")
        let upright = ImagePayload.encoded(
            try jpeg(picture, orientation: nil), contentType: "public.jpeg")
        let tagged = ImagePayload.encoded(
            try phoneJPEG(picture, tagged: true), contentType: "public.jpeg")
        let sideways = ImagePayload.encoded(
            try phoneJPEG(picture, tagged: false), contentType: "public.jpeg")
        let query = RerankQuery(
            text: "a sign that says LIGHTHOUSE",
            instruction: "Retrieve images or text relevant to the user's query.")

        for scoring in [MLXRerankingProvider.Scoring.fast, .stock] {
            let reranker = MLXRerankingProvider(
                configuration: MLXRerankingProvider.Configuration(
                    model: Integration.multimodalRerankerModel, scoring: scoring))
            try await reranker.prepare()
            let scores = try await reranker.score(
                query: query,
                candidates: [upright, tagged, sideways].map {
                    RerankCandidate(text: "", kind: .image, images: [$0])
                }
            ).map(\.score)
            await reranker.release()
            print(
                "[orientation] rerank \(scoring) upright, tagged, sideways "
                    + scores.map { String(format: "%.4f", $0) }.joined(separator: " "))
            // 0.540 / 0.541 / 0.250 with the 5-bit pair, either path.
            XCTAssertEqual(scores[1], scores[0], accuracy: 0.02, "\(scoring)")
            XCTAssertLessThan(scores[2], scores[0] - 0.1, "\(scoring)")
        }
    }

    // MARK: - Capabilities

    func testProviderAdvertisesASharedSpace() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let capabilities = await provider.capabilities
        XCTAssertTrue(capabilities.acceptsText)
        XCTAssertTrue(capabilities.acceptsImages)
        XCTAssertTrue(
            capabilities.sharedTextVisualSpace,
            "a separate visual space cannot answer an image-to-text query")
    }

    /// Text and image vectors have to be the same width and the same length, or
    /// they are not in one space whatever the provider claims.
    func testTextAndImageVectorsShareTheSpace() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let text = try await provider.embed(EmbeddingInput(text: "a photograph of a bridge"))
        let image = try await provider.embed(
            EmbeddingInput(images: [.encoded(try imageData("BRIDGE"), contentType: "public.png")]))

        XCTAssertEqual(text.vector.count, image.vector.count)
        XCTAssertEqual(VectorMath.l2Norm(text.vector), 1, accuracy: 1e-3)
        XCTAssertEqual(VectorMath.l2Norm(image.vector), 1, accuracy: 1e-3)
    }

    /// Capabilities do not depend on whether the weights are loaded. The engine
    /// reads them before making a model resident, and under sequential
    /// residency the reranker is always unloaded at that point — it used to
    /// report text-only then, so the VL reranker never saw an image.
    func testCapabilitiesDoNotChangeWithResidency() async throws {
        let reranker = ModelProfile.Multimodal.rerankingProvider(
            model: Integration.multimodalRerankerModel)
        let before = await reranker.capabilities
        XCTAssertTrue(before.acceptsImages)
        XCTAssertGreaterThan(before.maximumImagesPerPair, 0)

        try await reranker.prepare()
        await reranker.release()
        let after = await reranker.capabilities
        XCTAssertTrue(after.acceptsImages)
        XCTAssertEqual(after.maximumImagesPerPair, before.maximumImagesPerPair)

        let embedder = makeProvider()
        let unloaded = await embedder.capabilities
        XCTAssertTrue(unloaded.acceptsImages)
        XCTAssertTrue(unloaded.sharedTextVisualSpace)
    }

    /// The full pipeline with an image query, one model resident at a time: the
    /// reranker has been released by the query's embedding step when the engine
    /// decides what to send it, and must still be sent the image.
    func testImageQueryIsRerankedUnderSequentialResidency() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"))
        try await engine.upsert([
            SearchItem(
                id: "text-bridge",
                representations: [.text("A suspension bridge spanning a wide river.")]),
            SearchItem(
                id: "text-penguin",
                representations: [.text("Emperor penguins huddle together through the winter.")]),
        ])

        let results = try await engine.search(
            SearchQuery(images: [.encoded(try imageData("BRIDGE"), contentType: "public.png")]))
        XCTAssertNil(results.diagnostics.rerankSkippedReason)
        XCTAssertEqual(results.diagnostics.rerankedCandidates, 2)
        XCTAssertTrue(results.hits.allSatisfy { $0.evidence.rerankScore != nil })
        XCTAssertEqual(results.hits.first?.itemID, "text-bridge")
        try await engine.close()
    }

    // MARK: - Cross-modal retrieval

    func testTextQueryRetrievesAnImage() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"))

        try await engine.upsert([
            SearchItem(
                id: "image-bridge",
                representations: [
                    ContentRepresentation(
                        id: "image", kind: .image,
                        content: .imageData(
                            try imageData("BRIDGE"), contentType: "public.png"))
                ]),
            SearchItem(
                id: "image-penguin",
                representations: [
                    ContentRepresentation(
                        id: "image", kind: .image,
                        content: .imageData(
                            try imageData("PENGUIN"), contentType: "public.png"))
                ]),
        ])

        let results = try await engine.search(
            "a picture containing the word bridge",
            options: SearchOptions(mode: .denseOnly, limit: 2))
        XCTAssertEqual(results.hits.first?.itemID, "image-bridge")
        XCTAssertEqual(results.hits.first?.kind, .image)
        try await engine.close()
    }

    func testImageQueryRetrievesText() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"))

        try await engine.upsert([
            SearchItem(
                id: "text-bridge",
                representations: [.text("A suspension bridge spanning a wide river.")]),
            SearchItem(
                id: "text-penguin",
                representations: [.text("Emperor penguins huddle together through the winter.")]),
        ])

        // Image in, text out — the retrieval direction the profile exists for.
        let results = try await engine.search(
            SearchQuery(images: [.encoded(try imageData("BRIDGE"), contentType: "public.png")]),
            options: SearchOptions(mode: .denseOnly, limit: 2))

        XCTAssertFalse(results.hits.isEmpty)
        XCTAssertEqual(results.hits.first?.itemID, "text-bridge")
        try await engine.close()
    }

    func testMixedQueryUsesBothModalities() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"))
        try await engine.upsert([
            SearchItem(id: "a", representations: [.text("a suspension bridge at sunset")]),
            SearchItem(id: "b", representations: [.text("a colony of emperor penguins")]),
        ])

        let results = try await engine.search(
            SearchQuery(
                text: "at sunset",
                images: [.encoded(try imageData("BRIDGE"), contentType: "public.png")]),
            options: SearchOptions(mode: .denseOnly, limit: 2))
        XCTAssertEqual(results.hits.first?.itemID, "a")
        try await engine.close()
    }

    // MARK: - Video and OCR evidence

    /// Visual evidence and OCR evidence are separately addressable, and both
    /// survive to the result.
    func testVisualAndOCREvidenceAreSeparate() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: directory.appendingPathComponent("index.sqlite"),
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "multimodal", provider: makeProvider(),
                        kinds: Set(RepresentationKind.allCases), indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                processor: VisionOCRProcessor(),
                runtime: RuntimePolicy(sequentialModelResidency: true)))
        try await engine.prepare()

        try await engine.ingest([
            RawContent(
                id: "page",
                payload: .imageData(try imageData("BATHYSPHERE"), contentType: "public.png"))
        ])

        let storedPage = try await engine.item("page")
        let page = try XCTUnwrap(storedPage)
        XCTAssertTrue(page.representations.contains { $0.kind == .image })
        XCTAssertTrue(page.representations.contains { $0.kind == .ocrText })

        // The OCR text is retrievable lexically, with no model involved.
        let lexical = try await engine.search(
            "bathysphere", options: SearchOptions(mode: .lexicalOnly))
        XCTAssertEqual(lexical.hits.first?.kind, .ocrText)

        // And the image itself is retrievable through the dense channel.
        let dense = try await engine.search(
            "a page with a single large word on it",
            options: SearchOptions(mode: .denseOnly, limit: 5))
        XCTAssertTrue(dense.hits.contains { $0.kind == .image })

        try await engine.close()
    }

    func testVideoClipIsEmbedded() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // A short clip built from a rendered frame, so the file is real video
        // without needing a fixture on disk.
        let movie = directory.appendingPathComponent("clip.mov")
        try await makeMovie(at: movie)

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"))
        // The whole clip as well as its frames: whole-video embedding is opt-in,
        // and this is the test that exercises it.
        let items = try await VideoFrameExtractor(
            configuration: .init(interval: 0.5, maximumFrames: 4, embedsWholeVideo: true)
        ).items(from: movie, id: "video")

        let report = try await engine.upsert(items)
        XCTAssertEqual(report.vectorsWritten, items.count, "every frame and the clip itself")
        XCTAssertTrue(report.skipped.isEmpty, "\(report.skipped)")

        // Frame children carry their own time range, so a visual hit resolves
        // to a moment rather than to the file.
        let results = try await engine.search(
            "a frame from a video", options: SearchOptions(mode: .denseOnly, limit: 5))
        XCTAssertTrue(results.hits.contains { $0.locator.timeRange != nil })

        try await engine.close()
    }

    private func makeMovie(at url: URL) async throws {
        let image = try textImage("FRAME")
        let data = try ImageDecoding.encode(image)
        // Written through the same helper the pure suite uses, kept local so
        // the integration target has no dependency on the test target.
        try await MovieWriter.write(image: try ImageDecoding.decode(
            .encoded(data, contentType: "public.png")), to: url, seconds: 2)
    }
}

/// Minimal H.264 writer for the video fixtures.
enum MovieWriter {
    static func write(image: CGImage, to url: URL, seconds: Double, fps: Int32 = 8) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: image.width,
                AVVideoHeightKey: image.height,
            ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: image.width,
                kCVPixelBufferHeightKey as String: image.height,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for frame in 0..<Int(Double(fps) * seconds) {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(5))
            }
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(
                nil, image.width, image.height, kCVPixelFormatType_32ARGB, nil, &buffer)
            guard let buffer else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer),
                let context = CGContext(
                    data: base, width: image.width, height: image.height, bitsPerComponent: 8,
                    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue)
            {
                context.draw(
                    image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(
                buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: fps))
        }

        input.markAsFinished()
        await writer.finishWriting()
    }
}

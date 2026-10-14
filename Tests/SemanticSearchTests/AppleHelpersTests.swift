import AVFoundation
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Testing

@testable import SemanticSearch
@testable import SemanticSearchApple

/// Vision OCR, image decoding, frame extraction and audio extraction.
///
/// Everything here runs against images and media this file generates, so the
/// suite needs no fixtures on disk and no model downloads.
@Suite("Apple helpers")
struct AppleHelpersTests {

    // MARK: - Fixtures

    /// Render text into a bitmap Vision can read.
    static func textImage(
        _ text: String, size: CGSize = CGSize(width: 900, height: 260), fontSize: CGFloat = 72
    ) throws -> CGImage {
        let context = try #require(
            CGContext(
                data: nil, width: Int(size.width), height: Int(size.height),
                bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(origin: .zero, size: size))

        // Core Text keys rather than AppKit/UIKit ones, so the fixture builds
        // on both platforms without importing a UI framework.
        let font = CTFontCreateWithName("Helvetica" as CFString, fontSize, nil)
        let attributed = NSAttributedString(
            string: text,
            attributes: [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                    CGColor(red: 0, green: 0, blue: 0, alpha: 1),
            ])
        let line = CTLineCreateWithAttributedString(attributed)
        context.textPosition = CGPoint(x: 40, y: size.height / 2 - fontSize / 3)
        CTLineDraw(line, context)

        return try #require(context.makeImage())
    }

    static func solidImage(
        width: Int, height: Int, red: CGFloat = 1, green: CGFloat = 0, blue: CGFloat = 0
    ) throws -> CGImage {
        let context = try #require(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: red, green: green, blue: blue, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return try #require(context.makeImage())
    }

    /// A short silent-video-plus-tone file, written with AVAssetWriter.
    static func makeMovie(
        at url: URL, seconds: Double = 2, fps: Int32 = 10, metadata: [AVMetadataItem] = [],
        level: Int32? = nil
    ) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        writer.metadata = metadata
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: 320,
                AVVideoHeightKey: 240,
            ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: 320,
                kCVPixelBufferHeightKey as String: 240,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let frameCount = Int(Double(fps) * seconds)
        for frame in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(5))
            }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferCreate(
                nil, 320, 240, kCVPixelFormatType_32ARGB, nil, &pixelBuffer)
            guard let pixelBuffer else { continue }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                // A ramp, so consecutive frames differ; or one `level`.
                memset(
                    base, level ?? Int32(frame * 8 % 255), CVPixelBufferGetDataSize(pixelBuffer))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            adaptor.append(
                pixelBuffer,
                withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: fps))
        }

        input.markAsFinished()
        await writer.finishWriting()
    }

    /// A WAV file holding a sine tone.
    static func makeAudio(at url: URL, seconds: Double = 1, sampleRate: Double = 44_100) throws {
        let format = try #require(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1,
                interleaved: false))
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let frames = AVAudioFrameCount(seconds * sampleRate)
        let buffer = try #require(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        buffer.frameLength = frames
        if let channel = buffer.floatChannelData {
            for index in 0..<Int(frames) {
                channel[0][index] = Float(
                    sin(2 * Double.pi * 440 * Double(index) / sampleRate)) * 0.4
            }
        }
        try file.write(from: buffer)
    }

    // MARK: - Image decoding

    @Test("An encoded image decodes and re-encodes")
    func imageRoundTrip() throws {
        let original = try Self.solidImage(width: 64, height: 32)
        let data = try ImageDecoding.encode(original)
        let decoded = try ImageDecoding.decode(.encoded(data, contentType: "public.png"))
        #expect(decoded.width == 64)
        #expect(decoded.height == 32)
    }

    @Test("Decoding downscales to a bound")
    func decodingRespectsMaximumDimension() throws {
        // The bound has to be applied inside ImageIO — decoding a 48-megapixel
        // photo full-size and resampling afterwards is 200 MB of transient
        // memory instead of four.
        let large = try Self.solidImage(width: 2_000, height: 1_000)
        let data = try ImageDecoding.encode(large)
        let decoded = try ImageDecoding.decode(
            .encoded(data, contentType: "public.png"), maximumDimension: 256)
        #expect(max(decoded.width, decoded.height) <= 256)
    }

    @Test("Dimensions are read without a full decode")
    func dimensionsWithoutDecoding() throws {
        let data = try ImageDecoding.encode(try Self.solidImage(width: 123, height: 45))
        let size = try #require(ImageDecoding.dimensions(of: .encoded(data, contentType: "public.png")))
        #expect(size.width == 123)
        #expect(size.height == 45)
    }

    @Test("A missing file is reported as unavailable media")
    func missingImageFile() {
        let missing = URL(fileURLWithPath: "/tmp/no-such-image-\(UUID().uuidString).png")
        #expect(throws: SemanticSearchError.self) {
            _ = try ImageDecoding.decode(.fileURL(missing))
        }
    }

    @Test("Bytes that are not an image are rejected")
    func undecodableBytes() {
        #expect(throws: SemanticSearchError.self) {
            _ = try ImageDecoding.decode(
                .encoded(Data("not an image".utf8), contentType: "public.png"))
        }
    }

    // MARK: - Upright decoding

    /// 64×32 as stored: red, green / blue, yellow. Every orientation sends
    /// each quadrant somewhere else, mirrors included.
    static let quadrantColors: [[Int]] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

    static var sRGB: CGColorSpace { CGColorSpace(name: CGColorSpace.sRGB)! }

    /// A JPEG of the quadrants with EXIF `orientation`, as a phone writes a
    /// portrait: the pixels stay as the sensor read them, the tag says how to
    /// turn them.
    static func orientedJPEG(orientation: Int) throws -> Data {
        let context = try #require(
            CGContext(
                data: nil, width: 64, height: 32, bitsPerComponent: 8, bytesPerRow: 0,
                space: Self.sRGB,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        // CoreGraphics' origin is bottom-left: stored row 0 is the top.
        for (index, color) in quadrantColors.enumerated() {
            context.setFillColor(
                CGColor(
                    srgbRed: CGFloat(color[0]) / 255, green: CGFloat(color[1]) / 255,
                    blue: CGFloat(color[2]) / 255, alpha: 1))
            context.fill(
                CGRect(x: index % 2 * 32, y: index < 2 ? 16 : 0, width: 32, height: 16))
        }
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil))
        CGImageDestinationAddImage(
            destination, try #require(context.makeImage()),
            [
                kCGImagePropertyOrientation: orientation,
                kCGImageDestinationLossyCompressionQuality: 1.0,
            ] as CFDictionary)
        try #require(CGImageDestinationFinalize(destination))
        return data as Data
    }

    /// The stored pixel shown at `x`, `y` of the upright image, as Pillow's
    /// `exif_transpose` maps EXIF orientations.
    static func storedPixel(x: Int, y: Int, width: Int, height: Int, orientation: Int)
        -> (x: Int, y: Int)
    {
        switch orientation {
        case 2: return (width - 1 - x, y)
        case 3: return (width - 1 - x, height - 1 - y)
        case 4: return (x, height - 1 - y)
        case 5: return (y, x)
        case 6: return (y, height - 1 - x)
        case 7: return (width - 1 - y, height - 1 - x)
        case 8: return (width - 1 - y, x)
        default: return (x, y)
        }
    }

    /// RGBA8 of `image` drawn into sRGB at its own size: an sRGB image's own
    /// values; anything else colour-matched to sRGB.
    static func srgbPixels(_ image: CGImage) throws -> (bytes: [UInt8], width: Int) {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        try bytes.withUnsafeMutableBytes { buffer in
            let context = try #require(
                CGContext(
                    data: buffer.baseAddress, width: image.width, height: image.height,
                    bitsPerComponent: 8, bytesPerRow: image.width * 4,
                    space: Self.sRGB,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            context.draw(
                image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return (bytes, image.width)
    }

    static func rgb(_ pixels: (bytes: [UInt8], width: Int), x: Int, y: Int) -> [Int] {
        let offset = (y * pixels.width + x) * 4
        return pixels.bytes[offset..<offset + 3].map(Int.init)
    }

    /// Each upright quadrant's centre against the stored quadrant it shows.
    static func expectUpright(_ image: CGImage, orientation: Int) throws {
        let turned = (5...8).contains(orientation)
        #expect(image.width == (turned ? 32 : 64))
        #expect(image.height == (turned ? 64 : 32))
        let pixels = try srgbPixels(image)
        for y in [image.height / 4, image.height * 3 / 4] {
            for x in [image.width / 4, image.width * 3 / 4] {
                let stored = storedPixel(
                    x: x, y: y, width: 64, height: 32, orientation: orientation)
                let expected = quadrantColors[(stored.y < 16 ? 0 : 2) + (stored.x < 32 ? 0 : 1)]
                let actual = rgb(pixels, x: x, y: y)
                #expect(
                    zip(actual, expected).allSatisfy { abs($0 - $1) <= 24 },
                    "orientation \(orientation) at (\(x), \(y)): \(actual), want \(expected)")
            }
        }
    }

    @Test("Every EXIF orientation is turned upright for the vision models", arguments: 1...8)
    func decodeUprightAppliesOrientation(orientation: Int) throws {
        let data = try Self.orientedJPEG(orientation: orientation)
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        let properties = try #require(
            CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        #expect(properties[kCGImagePropertyOrientation] as? Int == orientation)

        try Self.expectUpright(
            try ImageDecoding.decodeUpright(.encoded(data, contentType: "public.jpeg")),
            orientation: orientation)
    }

    @Test("An oriented file on disk is turned upright too")
    func decodeUprightFromFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("portrait-\(UUID().uuidString).jpg")
        try Self.orientedJPEG(orientation: 6).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.expectUpright(try ImageDecoding.decodeUpright(.fileURL(url)), orientation: 6)
    }

    @Test("A full-size decode is upright, as a bounded one always was", arguments: [3, 6, 8])
    func fullSizeDecodeAppliesOrientation(orientation: Int) throws {
        let data = try Self.orientedJPEG(orientation: orientation)
        try Self.expectUpright(
            try ImageDecoding.decode(.encoded(data, contentType: "public.jpeg")),
            orientation: orientation)
    }

    @Test("The vision models see a Display P3 photo's stored values, not sRGB's")
    func decodeUprightKeepsStoredValues() throws {
        // Pillow ignores the embedded profile; Core Image would convert.
        let p3 = try #require(CGColorSpace(name: CGColorSpace.displayP3))
        let context = try #require(
            CGContext(
                data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                space: p3, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let color = CGColor(colorSpace: p3, components: [200 / 255, 100 / 255, 50 / 255, 1])
        context.setFillColor(try #require(color))
        context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        let data = try ImageDecoding.encode(try #require(context.makeImage()))

        let stored = try ImageDecoding.decode(.encoded(data, contentType: "public.png"))
        #expect(stored.colorSpace?.name == CGColorSpace.displayP3)
        let converted = Self.rgb(try Self.srgbPixels(stored), x: 4, y: 4)
        #expect(zip(converted, [200, 100, 50]).contains { abs($0 - $1) > 5 })

        let upright = try ImageDecoding.decodeUpright(.encoded(data, contentType: "public.png"))
        let actual = Self.rgb(try Self.srgbPixels(upright), x: 4, y: 4)
        #expect(zip(actual, [200, 100, 50]).allSatisfy { abs($0 - $1) <= 1 }, "\(actual)")
    }

    @Test("Grey reaches all three channels as stored")
    func decodeUprightKeepsGreyValues() throws {
        // Dark, where Gray Gamma 2.2 and sRGB's curve are furthest apart: a
        // colour-matched 20 comes out near 12.
        let context = try #require(
            CGContext(
                data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue))
        context.setFillColor(gray: 20 / 255, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        let data = try ImageDecoding.encode(try #require(context.makeImage()))

        let upright = try ImageDecoding.decodeUpright(.encoded(data, contentType: "public.png"))
        let actual = Self.rgb(try Self.srgbPixels(upright), x: 4, y: 4)
        #expect(actual.allSatisfy { abs($0 - 20) <= 1 }, "\(actual)")
    }

    @Test("Transparency is composited on white, as qwen_vl_utils' to_rgb does")
    func decodeUprightCompositesOnWhite() throws {
        let context = try #require(
            CGContext(
                data: nil, width: 16, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                space: Self.sRGB,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.clear(CGRect(x: 0, y: 0, width: 16, height: 8))
        context.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 128 / 255))
        context.fill(CGRect(x: 8, y: 0, width: 8, height: 8))
        let data = try ImageDecoding.encode(try #require(context.makeImage()))

        let pixels = try Self.srgbPixels(
            try ImageDecoding.decodeUpright(.encoded(data, contentType: "public.png")))
        let clear = Self.rgb(pixels, x: 4, y: 4)
        let half = Self.rgb(pixels, x: 12, y: 4)
        #expect(clear == [255, 255, 255], "\(clear)")
        // Pillow: 255·a + 255·(1 − a) = 255 red, 255·(1 − 128/255) = 127 else.
        #expect(zip(half, [255, 127, 127]).allSatisfy { abs($0 - $1) <= 2 }, "\(half)")
    }

    @Test("A missing file is unavailable media for the vision models too")
    func decodeUprightMissingFile() {
        let missing = URL(fileURLWithPath: "/tmp/no-such-image-\(UUID().uuidString).jpg")
        #expect(throws: SemanticSearchError.self) {
            _ = try ImageDecoding.decodeUpright(.fileURL(missing))
        }
    }

    // MARK: - Vision OCR

    @Test("OCR reads rendered text and reports a bounding box")
    func ocrReadsText() async throws {
        let image = try Self.textImage("HYDROTHERMAL")
        let lines = try await VisionOCRProcessor().recognize(image)
        #expect(!lines.isEmpty)
        #expect(lines.contains { $0.text.uppercased().contains("HYDROTHERMAL") })

        let box = try #require(lines.first?.boundingBox)
        // Normalised, origin top-left. Vision reports bottom-left, and the flip
        // happens once so nothing downstream has to remember which it holds.
        #expect(box.x >= 0 && box.x <= 1)
        #expect(box.y >= 0 && box.y <= 1)
        #expect(box.width > 0 && box.height > 0)
    }

    @Test("On macOS 26 a wrapped paragraph comes back as one block, not a line each")
    func documentStructureKeepsParagraphs() async throws {
        guard VisionOCRProcessor.documentRequestAvailable else { return }
        let image = try Self.page([
            ("Harbour Report", 64),
            ("The storm reached the northern harbour shortly after midnight and", 34),
            ("the ferries stayed in port until the wind dropped in the morning.", 34),
        ])
        let structured = try await VisionOCRProcessor().recognize(image)
        let lines = try VisionOCRProcessor().recognizeLines(image)
        #expect(lines.count >= 3, "\(lines.map(\.text))")
        #expect(structured.count < lines.count, "\(structured.map(\.text))")
        let paragraph = try #require(structured.first { $0.text.contains("ferries") })
        #expect(paragraph.text.contains("midnight"), "one block holds both lines")

        let flat = try await VisionOCRProcessor(
            configuration: .init(prefersDocumentStructure: false)
        ).recognize(image)
        #expect(flat.count == lines.count)
    }

    /// Lines of text at given point sizes, top to bottom, on a white page.
    static func page(_ lines: [(String, CGFloat)]) throws -> CGImage {
        let width = 1_400
        let height = 600
        let context = try #require(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        var y = CGFloat(height) - 100
        for (text, size) in lines {
            let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
            let attributed = NSAttributedString(
                string: text,
                attributes: [
                    NSAttributedString.Key(kCTFontAttributeName as String): font,
                    NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                        CGColor(red: 0, green: 0, blue: 0, alpha: 1),
                ])
            context.textPosition = CGPoint(x: 60, y: y)
            CTLineDraw(CTLineCreateWithAttributedString(attributed), context)
            y -= size * 1.5 + (size > 40 ? 30 : 0)
        }
        return try #require(context.makeImage())
    }

    @Test("The processor emits OCR text and keeps the image")
    func ocrProcessorKeepsTheImage() async throws {
        // OCR is an additional view of the content, never a replacement for it.
        let image = try Self.textImage("PENGUINS")
        let data = try ImageDecoding.encode(image)
        let items = try await VisionOCRProcessor().process(
            RawContent(id: "page", payload: .imageData(data, contentType: "public.png")))

        let page = try #require(items.first)
        #expect(page.representations.contains { $0.kind == .image })
        let ocr = try #require(page.representations.first { $0.kind == .ocrText })
        #expect(ocr.content.text?.uppercased().contains("PENGUINS") == true)
        #expect(ocr.provenance.producer == "vision.ocr")
        #expect(ocr.locator?.boundingBoxes.isEmpty == false)
    }

    @Test("A blank page yields no OCR representation and still keeps the image")
    func blankPage() async throws {
        let data = try ImageDecoding.encode(try Self.solidImage(width: 200, height: 200))
        let items = try await VisionOCRProcessor().process(
            RawContent(id: "blank", payload: .imageData(data, contentType: "public.png")))
        let page = try #require(items.first)
        #expect(page.representations.contains { $0.kind == .image })
        #expect(!page.representations.contains { $0.kind == .ocrText })
    }

    @Test("The processor's fingerprint changes with its configuration")
    func ocrFingerprint() {
        // The fingerprint is what makes an upsert redo derived work; a
        // configuration change that does not move it leaves stale OCR in place.
        let base = VisionOCRProcessor()
        let different = VisionOCRProcessor(
            configuration: .init(accurate: false, minimumConfidence: 0.8))
        #expect(base.configurationFingerprint != different.configurationFingerprint)
        #expect(base.configurationFingerprint == VisionOCRProcessor().configurationFingerprint)
    }

    @Test("OCR output is indexed and retrievable")
    func ocrEndToEnd() async throws {
        let image = try Self.textImage("BATHYSPHERE")
        let data = try ImageDecoding.encode(image)

        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(acceptsVisual: true),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                processor: VisionOCRProcessor()))
        try await engine.prepare()
        try await engine.ingest([
            RawContent(id: "page", payload: .imageData(data, contentType: "public.png"))
        ])

        let results = try await engine.search(
            "bathysphere", options: SearchOptions(mode: .lexicalOnly))
        #expect(results.hits.first?.kind == .ocrText)
        #expect(results.hits.first?.provenance.producer == "vision.ocr")
    }

    // MARK: - Video frames

    @Test("Frames are extracted with timestamps and bounded in number")
    func frameExtraction() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("clip.mov")
        try await Self.makeMovie(at: movie, seconds: 3)

        let extractor = VideoFrameExtractor(
            configuration: .init(interval: 0.5, maximumFrames: 4, maximumDimension: 128))
        let frames = try await extractor.frames(from: movie)

        // The cap widens the interval rather than truncating: a cap applied by
        // stopping early indexes the first minute of a film and none of the rest.
        #expect(frames.count <= 4)
        #expect(!frames.isEmpty)
        #expect(frames.map(\.time).sorted() == frames.map(\.time))
        #expect(frames.last!.time > frames.first!.time)
        #expect(frames.allSatisfy { max($0.image.width, $0.image.height) <= 128 })
    }

    @Test("A video without sound is indexed by its frames, not failed")
    func silentVideo() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("time-lapse.mov")
        try await Self.makeMovie(at: movie, seconds: 2)

        #expect(try await !AudioExtractor.hasAudioTrack(movie))
        let silent = try #require(try await Transcript.ofSilentFile(movie))
        #expect(silent.segments.isEmpty)
        #expect(abs(silent.duration - 2) < 0.2)
        #expect(try await AppleSpeechTranscriber().transcribe(url: movie).segments.isEmpty)
        #expect(try await AppleSoundClassifier().classify(movie).isEmpty)

        // Sounds beside frames, as the recipes combine them: the frames stay.
        let items = try await CombinedContentProcessor([
            await SoundEventProcessor(classifier: AppleSoundClassifier()),
            KeyframeProcessor(),
        ]).process(RawContent(id: "video", payload: .fileURL(movie)), onProgress: nil)
        #expect(items.first?.id == "video")
        #expect(items.contains { $0.representations.contains { $0.kind == .image } })
        #expect(!items.contains { $0.representations.contains { $0.kind == .soundEvent } })
    }

    @Test("A video dark from end to end keeps a frame instead of failing")
    func darkVideo() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("night.mov")
        try await Self.makeMovie(at: movie, seconds: 3, level: 0)

        let frames = try await KeyframeExtractor().frames(from: movie)
        // Identical black seconds are one picture.
        #expect(frames.count == 1)
    }

    @Test("A transparent frame is not a black one")
    func transparentFrameIsNotBlack() throws {
        func image(_ fill: CGColor?) throws -> CGImage {
            let context = try #require(
                CGContext(
                    data: nil, width: 64, height: 36, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            if let fill {
                context.setFillColor(fill)
                context.fill(CGRect(x: 0, y: 0, width: 64, height: 36))
            }
            // A ball.
            context.setFillColor(CGColor(red: 0.9, green: 0.2, blue: 0.1, alpha: 1))
            context.fillEllipse(in: CGRect(x: 24, y: 10, width: 16, height: 16))
            return try #require(context.makeImage())
        }
        let blackLuma = KeyframeExtractor.Configuration.default.blackLuma
        // Clear background, as HEVC with alpha decodes.
        #expect(KeyframeExtractor.meanLuma(of: try image(nil)) >= blackLuma)
        // A black frame with the same ball is still black.
        #expect(
            KeyframeExtractor.meanLuma(of: try image(CGColor(red: 0, green: 0, blue: 0, alpha: 1)))
                < blackLuma)
    }

    @Test("Frames become items with time ranges under one parent")
    func frameItems() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("clip.mov")
        try await Self.makeMovie(at: movie, seconds: 2)

        let items = try await VideoFrameExtractor(
            configuration: .init(interval: 0.5, maximumFrames: 4)
        ).items(from: movie, id: "video")

        let parent = try #require(items.first)
        #expect(parent.id == "video")
        #expect(parent.representations.contains { $0.kind == .video })

        let children = items.dropFirst()
        #expect(!children.isEmpty)
        #expect(children.allSatisfy { $0.parentID == "video" })
        #expect(children.allSatisfy { $0.locator.timeRange != nil })
        #expect(children.allSatisfy { $0.locator.focusTime != nil })
        #expect(children.allSatisfy { $0.representations.first?.kind == .image })
    }

    @Test("A missing video is reported as unavailable media")
    func missingVideo() async {
        let missing = URL(fileURLWithPath: "/tmp/no-such-video-\(UUID().uuidString).mov")
        await #expect(throws: SemanticSearchError.self) {
            _ = try await VideoFrameExtractor().frames(from: missing)
        }
    }

    // MARK: - Audio

    @Test("Audio extracts as 16 kHz mono float samples")
    func audioExtraction() async throws {
        let directory = try TemporaryDirectory()
        let audio = directory.url.appendingPathComponent("tone.wav")
        try Self.makeAudio(at: audio, seconds: 1, sampleRate: 44_100)

        let samples = try await AudioExtractor().samples(from: audio)
        // Resampled by AVFoundation inside the reader, which beats resampling
        // afterwards.
        #expect(abs(Double(samples.count) - AudioExtractor.targetSampleRate) < 2_000)
        #expect(samples.contains { $0 != 0 })
    }

    @Test("A time range extracts only that span")
    func audioTimeRange() async throws {
        let directory = try TemporaryDirectory()
        let audio = directory.url.appendingPathComponent("tone.wav")
        try Self.makeAudio(at: audio, seconds: 2)

        let whole = try await AudioExtractor().samples(from: audio)
        let half = try await AudioExtractor().samples(
            from: audio, timeRange: SourceLocator.TimeRange(start: 0, end: 1))
        #expect(half.count < whole.count)
        #expect(abs(Double(half.count) - AudioExtractor.targetSampleRate) < 2_000)
    }

    @Test("Audio duration is reported")
    func audioDuration() async throws {
        let directory = try TemporaryDirectory()
        let audio = directory.url.appendingPathComponent("tone.wav")
        try Self.makeAudio(at: audio, seconds: 1.5)
        let duration = try await AudioExtractor().duration(of: audio)
        #expect(abs(duration - 1.5) < 0.1)
    }

    @Test("A file with no audio track is reported")
    func noAudioTrack() async throws {
        let directory = try TemporaryDirectory()
        let movie = directory.url.appendingPathComponent("silent.mov")
        try await Self.makeMovie(at: movie, seconds: 1)

        await #expect(throws: SemanticSearchError.self) {
            _ = try await AudioExtractor().samples(from: movie)
        }
    }

    @Test("Missing audio is reported as unavailable media")
    func missingAudio() async {
        let missing = URL(fileURLWithPath: "/tmp/no-such-audio-\(UUID().uuidString).wav")
        await #expect(throws: SemanticSearchError.self) {
            _ = try await AudioExtractor().samples(from: missing)
        }
    }
}

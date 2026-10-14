// Copied from FluidInference/FluidUse (Apache-2.0) and modified for SwiftSearch;
// see NOTICE in this package for the source revision and the list of changes.
// The image resize is ported from Pillow's libImaging/Resample.c (MIT-CMU); its
// copyright and permission notice are in NOTICE too.

@preconcurrency import CoreML
import CoreGraphics
import Foundation
import ImageIO

/// EmbeddingGemma 2 image embeddings: the image is resized and cut into 16 px patches the way the Hugging Face
/// `Gemma4ImageProcessor` does, encoded by `EmbeddingGemma2Vision` (ViT + 3x3 pooling, on the GPU, where it beats the
/// Neural Engine at every size), and the pooled tokens are embedded by the text model, so images share the text space.
public final class EmbeddingGemma2Vision: Sendable {
    /// Soft tokens per image. More tokens keep more detail and cost more time (zero-shot Oxford Pets on 370 photos:
    /// 87.0% / 88.4% / 89.5%; M5 Pro GPU: about 15 / 34 / 64+ ms per image). `.detailed` is the default, as it is
    /// the reference's (`max_soft_tokens: 280` in google/embeddinggemma-2's processor_config.json): an image embedded
    /// at another budget is a different vector of the same picture.
    public enum Budget: Int, CaseIterable, Sendable {
        case fast = 70
        case balanced = 140
        case detailed = 280

        var patches: Int { rawValue * 9 }
    }

    static let patchSize = 16
    static let pooling = 3
    static let hidden = 768
    static let tableRows = 1024
    public static let maxInFlight = 4

    public let text: EmbeddingGemma2Manager
    private let models: [Budget: MLModel]
    private let positionTable: Data

    init(text: EmbeddingGemma2Manager, models: [Budget: MLModel], positionTable: Data) {
        self.text = text
        self.models = models
        self.positionTable = positionTable
    }

    /// Loads the vision model and its position tables from `directory` (`EmbeddingGemma2ModelStore.ensureVision(in:)`)
    /// next to an already loaded text model. Only the functions of `budgets` are loaded: each is its own compiled
    /// program, and one a caller never uses costs load time and memory for nothing.
    public static func load(
        from directory: URL, text: EmbeddingGemma2Manager, budgets: [Budget] = Budget.allCases,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> EmbeddingGemma2Vision {
        guard #available(macOS 15, iOS 18, *) else {
            throw EmbeddingGemma2Error.unsupported("multifunction Core ML models need macOS 15 / iOS 18")
        }
        guard !budgets.isEmpty else { throw EmbeddingGemma2Error.unsupported("no image budget to load") }
        let compiled = directory.appendingPathComponent("EmbeddingGemma2Vision.mlmodelc")
        let package = directory.appendingPathComponent("EmbeddingGemma2Vision.mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try await MLModel.compileModel(at: package)
        } else {
            throw EmbeddingGemma2Error.invalidAsset("Missing EmbeddingGemma2Vision in \(directory.path)")
        }
        guard
            let table = try? Data(
                contentsOf: directory.appendingPathComponent("position_embeddings.f16"), options: .alwaysMapped),
            table.count == 2 * tableRows * hidden * 2
        else { throw EmbeddingGemma2Error.invalidAsset("Missing or malformed position_embeddings.f16") }
        var models: [Budget: MLModel] = [:]
        for budget in Budget.allCases where budgets.contains(budget) {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = "vision_\(budget.rawValue)"
            models[budget] = try await MLModel.load(contentsOf: url, configuration: configuration)
        }
        return EmbeddingGemma2Vision(text: text, models: models, positionTable: table)
    }

    /// Decodes an image file (anything ImageIO reads).
    public static func image(contentsOf url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw EmbeddingGemma2Error.invalidAsset("Cannot decode \(url.lastPathComponent)") }
        return image
    }

    /// Resized size in pixels: the largest aspect-preserving size whose sides are multiples of 48 px and which has at
    /// most `budget.patches` 16 px patches (HF `get_aspect_ratio_preserving_size`).
    static func targetSize(width: Int, height: Int, budget: Budget) -> (width: Int, height: Int) {
        let side = patchSize * pooling
        let factor = (Double(budget.patches * patchSize * patchSize) / Double(width * height)).squareRoot()
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        let maxSide = (budget.patches / (pooling * pooling)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min((width / height) * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min((height / width) * side, maxSide)
        }
        return (targetWidth, targetHeight)
    }

    /// Embeddings of many images, in order; up to `maxInFlight` run at once.
    public func embed(
        images: [CGImage], budget: Budget = .detailed, maxInFlight: Int = EmbeddingGemma2Vision.maxInFlight
    ) async throws -> [[Float]] {
        try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            var result = [[Float]](repeating: [], count: images.count)
            var next = 0
            func addImage() {
                guard next < images.count else { return }
                let index = next
                let image = images[index]
                next += 1
                group.addTask { (index, try await self.embed(image: image, budget: budget)) }
            }
            for _ in 0..<min(max(maxInFlight, 1), images.count) { addImage() }
            for try await (index, vector) in group {
                result[index] = vector
                addImage()
            }
            return result
        }
    }

    /// L2-normalized embedding of one image.
    public func embed(image: CGImage, budget: Budget = .detailed) async throws -> [Float] {
        try await embed(textTokens: [], images: [image], budget: budget)
    }

    /// Soft tokens an image of `width` x `height` becomes at `budget`, known before the vision model runs: what a
    /// caller needs to fit text next to it in 512 tokens.
    public static func tokenCount(width: Int, height: Int, budget: Budget) -> Int {
        let size = targetSize(width: width, height: height, budget: budget)
        return (size.width / patchSize) * (size.height / patchSize) / (pooling * pooling)
    }

    /// L2-normalized embedding of one sequence: `<bos>`, `textTokens` (a text's ids without `<bos>`/`<eos>`), each
    /// image as `<|image> tokens <image|>`, `<eos>`. Text before images, as the reference lays out a text with its
    /// images. It has to fit 512 tokens (`tokenCount`); nothing is cut here.
    public func embed(textTokens: [Int32], images: [CGImage], budget: Budget = .detailed) async throws -> [Float] {
        var media: [EmbeddingGemma2Manager.Media] = []
        for image in images {
            let encoded = try await tokens(of: image, budget: budget)
            media.append(
                EmbeddingGemma2Manager.Media(
                    tokens: encoded.tokens, count: encoded.count, start: EmbeddingGemma2Manager.imageStartId,
                    end: EmbeddingGemma2Manager.imageEndId))
        }
        return try await text.embed(textTokens: textTokens, media: media)
    }

    /// The vision model's output for one image and how many of its rows are the image's soft tokens.
    private func tokens(of image: CGImage, budget: Budget) async throws -> (tokens: MLMultiArray, count: Int) {
        guard let model = models[budget] else {
            throw EmbeddingGemma2Error.predictionFailed("the \(budget.rawValue)-token vision function is not loaded")
        }
        let size = Self.targetSize(width: image.width, height: image.height, budget: budget)
        let pixels = Self.resized(
            try Self.rgba(image), width: image.width, height: image.height, to: size.width, size.height)
        let columns = size.width / Self.patchSize
        let rows = size.height / Self.patchSize
        let count = columns * rows
        let total = budget.patches
        let tokens = budget.rawValue
        guard count <= total, columns <= Self.tableRows, rows <= Self.tableRows else {
            throw EmbeddingGemma2Error.predictionFailed("image needs \(count) patches; the budget has \(total)")
        }
        let patches = try MLMultiArray(shape: [1, NSNumber(value: total), 768], dataType: .float16)
        let positions = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .float16)
        let positionEmbeddings = try MLMultiArray(shape: [1, NSNumber(value: total), 768], dataType: .float16)
        let valid = try MLMultiArray(shape: [1, NSNumber(value: total)], dataType: .float16)
        let pool = try MLMultiArray(shape: [NSNumber(value: tokens), NSNumber(value: total)], dataType: .float16)
        let patchPointer = patches.dataPointer.assumingMemoryBound(to: Float16.self)
        let positionPointer = positions.dataPointer.assumingMemoryBound(to: Float16.self)
        let embeddingPointer = positionEmbeddings.dataPointer.assumingMemoryBound(to: Float16.self)
        let validPointer = valid.dataPointer.assumingMemoryBound(to: Float16.self)
        let poolPointer = pool.dataPointer.assumingMemoryBound(to: Float16.self)
        patchPointer.update(repeating: 0, count: total * 768)
        embeddingPointer.update(repeating: 0, count: total * 768)
        positionPointer.update(repeating: -1, count: total * 2)
        validPointer.update(repeating: 0, count: total)
        poolPointer.update(repeating: 0, count: tokens * total)
        positionTable.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            for row in 0..<rows {
                for column in 0..<columns {
                    let patch = row * columns + column
                    // Patch layout: [16 rows][16 columns][RGB], values in [0, 1] (the model maps them to [-1, 1]).
                    let base = patch * 768
                    for y in 0..<Self.patchSize {
                        for x in 0..<Self.patchSize {
                            let pixel = ((row * Self.patchSize + y) * size.width + column * Self.patchSize + x) * 4
                            let offset = base + (y * Self.patchSize + x) * 3
                            for channel in 0..<3 {
                                patchPointer[offset + channel] = Float16(Float(pixels[pixel + channel]) / 255)
                            }
                        }
                    }
                    positionPointer[patch * 2] = Float16(column)
                    positionPointer[patch * 2 + 1] = Float16(row)
                    validPointer[patch] = 1
                    let xRow = column * Self.hidden
                    let yRow = (Self.tableRows + row) * Self.hidden
                    for channel in 0..<Self.hidden {
                        embeddingPointer[base + channel] = Float16(
                            Float(table[xRow + channel]) + Float(table[yRow + channel]))
                    }
                    // 3x3 average pooling over the patch grid, groups numbered row-major.
                    let group = (column / Self.pooling) + (columns / Self.pooling) * (row / Self.pooling)
                    poolPointer[group * total + patch] = Float16(1.0 / 9.0)
                }
            }
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray: patches), "positions": MLFeatureValue(multiArray: positions),
            "position_embeddings": MLFeatureValue(multiArray: positionEmbeddings),
            "valid": MLFeatureValue(multiArray: valid), "pool": MLFeatureValue(multiArray: pool),
        ])
        let output = try await model.prediction(from: input)
        guard let imageTokens = output.featureValue(for: "image_tokens")?.multiArrayValue else {
            throw EmbeddingGemma2Error.predictionFailed("missing image_tokens output")
        }
        return (imageTokens, count / (Self.pooling * Self.pooling))
    }

    /// RGBX8 pixels of `image` at its own size, so nothing is resampled here, drawn into an sRGB context: an image in
    /// another colour space is colour matched, and alpha is composited over black.
    static func rgba(_ image: CGImage) throws -> [UInt8] {
        let width = image.width
        let height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw EmbeddingGemma2Error.predictionFailed("cannot draw the image") }
        return pixels
    }

    // From here to `cubic`: Pillow's resampler, ported from libImaging/Resample.c (Pillow 12.3.0) under the
    // MIT-CMU License, whose permission notice is in NOTICE.
    //   Copyright © 1997-2011 by Secret Labs AB
    //   Copyright © 1995-2011 by Fredrik Lundh and contributors
    //   Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

    /// Pillow's fixed-point precision for 8-bit images (`PRECISION_BITS` in libImaging/Resample.c).
    private static let precisionBits = 32 - 8 - 2

    /// RGBX8 `pixels` of `width` x `height` resized to `targetWidth` x `targetHeight` as Pillow's
    /// `Image.resize(size, BICUBIC)` does it: the antialiased bicubic filter (a = -0.5) widened by the scale, 22-bit
    /// fixed-point weights, the horizontal pass first and rounded to 8 bits before the vertical one. The same size is
    /// returned untouched, as the processor skips it.
    ///
    /// That is the resize of transformers' PIL backend (`Gemma4ImageProcessorPil`). With torchvision installed,
    /// transformers picks `Gemma4ImageProcessor` instead, whose `tvF.resize(antialias=True)` is PyTorch's kernel: a
    /// level or two off Pillow on under 1% of values. Byte parity is with a reference made on the PIL backend.
    static func resized(
        _ pixels: [UInt8], width: Int, height: Int, to targetWidth: Int, _ targetHeight: Int
    ) -> [UInt8] {
        var image = pixels
        var imageHeight = height
        var vertical = coefficients(input: height, output: targetHeight)
        if targetWidth != width {
            // Only the source rows the vertical pass reads, as Pillow crops them before the horizontal pass.
            let first = vertical.first[0]
            let last = vertical.first[targetHeight - 1] + vertical.count[targetHeight - 1]
            let horizontal = coefficients(input: width, output: targetWidth)
            // Rows of `width` pixels in, rows of `targetWidth` out.
            image = pass(
                image, lines: last - first, outputs: targetWidth, coefficients: horizontal,
                inputOffset: first * width * 4, inputLine: width * 4, inputTap: 4,
                outputLine: targetWidth * 4, outputStep: 4)
            imageHeight = last - first
            vertical.first = vertical.first.map { $0 - first }
        }
        if targetHeight != height {
            // Columns, `targetWidth` pixels apart in both.
            image = pass(
                image, lines: targetWidth, outputs: targetHeight, coefficients: vertical,
                inputOffset: 0, inputLine: 4, inputTap: targetWidth * 4,
                outputLine: 4, outputStep: targetWidth * 4)
            imageHeight = targetHeight
        }
        precondition(image.count == targetWidth * imageHeight * 4)
        return image
    }

    /// One separable pass over `lines` rows or columns of RGBX8: each of the `outputs` values along a line is the
    /// weighted sum of the taps `coefficients` names, from Pillow's rounding offset, clipped to 8 bits. Byte offsets:
    /// tap t of line l is read at `inputOffset + l * inputLine + t * inputTap`, output p of line l is written at
    /// `l * outputLine + p * outputStep`.
    private static func pass(
        _ input: [UInt8], lines: Int, outputs: Int, coefficients: Coefficients,
        inputOffset: Int, inputLine: Int, inputTap: Int, outputLine: Int, outputStep: Int
    ) -> [UInt8] {
        let half = 1 << (precisionBits - 1)
        var result = [UInt8](repeating: 0, count: lines * outputs * 4)
        input.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { result in
                coefficients.weights.withUnsafeBufferPointer { weights in
                    for line in 0..<lines {
                        let lineStart = inputOffset + line * inputLine
                        for position in 0..<outputs {
                            let start = lineStart + coefficients.first[position] * inputTap
                            let row = position * coefficients.taps
                            var red = half
                            var green = half
                            var blue = half
                            for tap in 0..<coefficients.count[position] {
                                let weight = Int(weights[row + tap])
                                let index = start + tap * inputTap
                                red += Int(input[index]) * weight
                                green += Int(input[index + 1]) * weight
                                blue += Int(input[index + 2]) * weight
                            }
                            let target = line * outputLine + position * outputStep
                            result[target] = clip8(red)
                            result[target + 1] = clip8(green)
                            result[target + 2] = clip8(blue)
                        }
                    }
                }
            }
        }
        return result
    }

    private static func clip8(_ value: Int) -> UInt8 {
        if value >= (1 << precisionBits) << 8 { return 255 }
        if value <= 0 { return 0 }
        return UInt8(value >> precisionBits)
    }

    /// Per output position: the first input tap, the number of taps, and `taps` fixed-point weights.
    struct Coefficients {
        var first: [Int]
        var count: [Int]
        var weights: [Int32]
        var taps: Int
    }

    /// Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc` for its bicubic filter, operation for operation:
    /// the weights are rounded to fixed point, so a different order of the same arithmetic can move a pixel.
    static func coefficients(input: Int, output: Int) -> Coefficients {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        let taps = Int(support.rounded(.up)) * 2 + 1
        let inverse = 1 / filterScale
        var result = Coefficients(
            first: [Int](repeating: 0, count: output), count: [Int](repeating: 0, count: output),
            weights: [Int32](repeating: 0, count: output * taps), taps: taps)
        var values = [Double](repeating: 0, count: taps)
        for position in 0..<output {
            let center = (Double(position) + 0.5) * scale
            // C's (int) truncates toward zero, as Int(_: Double) does.
            let first = max(Int(center - support + 0.5), 0)
            let count = min(Int(center + support + 0.5), input) - first
            var total = 0.0
            for tap in 0..<count {
                values[tap] = cubic((Double(tap + first) - center + 0.5) * inverse)
                total += values[tap]
            }
            for tap in 0..<count {
                let weight = total != 0 ? values[tap] / total : values[tap]
                let scaled = weight * Double(1 << precisionBits)
                result.weights[position * taps + tap] = Int32(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            result.first[position] = first
            result.count[position] = count
        }
        return result
    }

    /// Pillow's `bicubic_filter`.
    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
}

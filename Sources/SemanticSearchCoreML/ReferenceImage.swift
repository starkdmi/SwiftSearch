import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import SemanticSearch

/// Images decoded the way EmbeddingGemma 2's reference reads them.
///
/// transformers' `load_image` opens a file with Pillow, turns it upright with
/// `ImageOps.exif_transpose` and calls `convert("RGB")`. Pillow ignores an
/// embedded colour profile, so the stored values are the pixels; and
/// `convert("RGB")` drops the alpha channel, it does not composite it, so a
/// transparent pixel keeps the colour it was stored with. ImageIO and
/// CoreGraphics do neither by default: drawing colour-matches a Display P3
/// photo to sRGB and premultiplies or flattens alpha, and a decoded `CGImage`
/// is not turned by its orientation at all. Each of those moves every pixel
/// the model sees, so this does what Pillow does instead and hands the vision
/// runtime an upright, opaque image whose sRGB values are the file's own.
///
/// The image is decoded at full size. ImageIO's thumbnail path would bound the
/// memory, but it resamples, and the reference resizes once, from the full
/// image; a second filter in front of it is a different picture. The copies
/// are what is bounded instead: one of a 48 MP photo is 195 MB, and a copy per
/// step (packing rows, dropping alpha, turning, wrapping) would hold three at
/// once. Each pixel goes from vImage's decode straight to its upright place
/// in the bytes the returned image keeps, so decoding holds two at most.
///
/// 16-bit images are where this parts from Pillow, on purpose. Pillow opens a
/// 16-bit gray PNG as `I;16`, and `convert("RGB")` clips every value above 255,
/// so the reference sees an almost white picture; of 16-bit colour it keeps
/// the high byte. vImage scales 16 bits to 8 instead: the picture itself for
/// gray, and colour that can be a level off the high byte. Such files do not
/// belong in a parity comparison.
enum ReferenceImage {

    /// `payload` decoded, upright, its stored RGB values tagged sRGB, alpha
    /// dropped.
    static func decode(_ payload: ImagePayload) throws -> CGImage {
        let source = try imageSource(payload)
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, options) else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "could not decode image")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        let upright = try uprightPixels(image, orientation: orientation)
        return try sRGBImage(upright.bytes, width: upright.width, height: upright.height)
    }

    private static func imageSource(_ payload: ImagePayload) throws -> CGImageSource {
        switch payload {
        case .fileURL(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
            }
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
                throw SemanticSearchError.unsupportedContent(
                    itemID: ItemID(url.lastPathComponent), detail: "not a decodable image")
            }
            return source
        case .encoded(let data, let contentType):
            guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
                throw SemanticSearchError.unsupportedContent(
                    itemID: "image", detail: "not a decodable \(contentType) image")
            }
            return source
        }
    }

    /// RGBX8 of `image`'s stored values, upright for EXIF `orientation`, the
    /// X byte 255: one pass from vImage's decode, which is freed on return,
    /// before the image is built over these bytes.
    private static func uprightPixels(_ image: CGImage, orientation: Int) throws
        -> (bytes: Data, width: Int, height: Int)
    {
        let stored = try storedPixels(image)
        defer { stored.buffer.free() }
        let width = image.width
        let height = image.height
        let swapsSides = (5...8).contains(orientation)
        let outputWidth = swapsSides ? height : width
        let outputHeight = swapsSides ? width : height
        let step = stored.bytesPerPixel
        let rowBytes = stored.buffer.rowBytes
        let input = UnsafeRawBufferPointer(
            UnsafeMutableRawBufferPointer(
                start: stored.buffer.data, count: rowBytes * Int(stored.buffer.height)))
        var bytes = Data(count: width * height * 4)
        bytes.withUnsafeMutableBytes { (output: UnsafeMutableRawBufferPointer) in
            for y in 0..<outputHeight {
                for x in 0..<outputWidth {
                    let source = storedPixel(
                        x: x, y: y, width: width, height: height, orientation: orientation)
                    let from = source.y * rowBytes + source.x * step
                    let to = (y * outputWidth + x) * 4
                    if stored.isGray {
                        output[to] = input[from]
                        output[to + 1] = input[from]
                        output[to + 2] = input[from]
                    } else {
                        output[to] = input[from]
                        output[to + 1] = input[from + 1]
                        output[to + 2] = input[from + 2]
                    }
                    output[to + 3] = 255
                }
            }
        }
        return (bytes, outputWidth, outputHeight)
    }

    /// A decode vImage made: its buffer, the bytes of one pixel, and whether
    /// the first of them is gray, for all three channels.
    private struct StoredPixels {
        var buffer: vImage_Buffer
        var bytesPerPixel: Int
        var isGray: Bool
    }

    /// The values the file stores, as RGBA, gray and alpha, or gray.
    ///
    /// vImage converts the pixel layout and leaves the colour alone when the
    /// destination is in the image's own colour space, which is how a P3 or
    /// Adobe RGB file keeps its numbers. Its destination is RGBA with
    /// non-premultiplied alpha, so a straight-alpha PNG comes through as stored
    /// and the alpha is then ignored, as `convert("RGB")` ignores it. Gray is
    /// read as gray and copied into all three channels, which is Pillow's
    /// `L`→`RGB` for 8-bit gray (16-bit: see above). Anything else (CMYK, Lab,
    /// a profile vImage will not write to) is converted to sRGB: Pillow has no
    /// colour management to compare with there. The caller frees the buffer.
    private static func storedPixels(_ image: CGImage) throws -> StoredPixels {
        if let space = image.colorSpace, space.model == .monochrome {
            // Gray and alpha if vImage writes it, else gray alone (alpha flattened over black).
            let layouts: [(bitsPerPixel: Int, alpha: CGImageAlphaInfo)] = [(16, .last), (8, .none)]
            for (bitsPerPixel, alpha) in layouts {
                guard
                    let format = vImage_CGImageFormat(
                        bitsPerComponent: 8, bitsPerPixel: bitsPerPixel, colorSpace: space,
                        bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue)),
                    let buffer = try? vImage_Buffer(cgImage: image, format: format)
                else { continue }
                return StoredPixels(
                    buffer: buffer, bytesPerPixel: bitsPerPixel / 8, isGray: true)
            }
        }

        var spaces: [CGColorSpace] = []
        if let own = image.colorSpace, own.model == .rgb { spaces.append(own) }
        if let sRGB = CGColorSpace(name: CGColorSpace.sRGB) { spaces.append(sRGB) }
        for space in spaces {
            guard
                let format = vImage_CGImageFormat(
                    bitsPerComponent: 8, bitsPerPixel: 32, colorSpace: space,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)),
                let buffer = try? vImage_Buffer(cgImage: image, format: format)
            else { continue }
            return StoredPixels(buffer: buffer, bytesPerPixel: 4, isGray: false)
        }
        throw SemanticSearchError.unsupportedContent(
            itemID: "image", detail: "pixel format vImage cannot read")
    }

    /// The stored pixel shown at `x`, `y` of the upright image, for EXIF
    /// `orientation` as Pillow's `exif_transpose` maps it: 2 mirror, 3 rotate
    /// 180°, 4 flip, 5 transpose, 6 rotate 90° clockwise, 7 transverse, 8
    /// rotate 90° counter-clockwise, anything else as stored. `width` and
    /// `height` are the stored image's. Checked against Pillow's
    /// `Image.transpose` for every value.
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

    /// An opaque 8-bit sRGB image over `bytes` (RGBX8): drawn into the vision
    /// runtime's sRGB context at its own size, it is copied, not converted.
    /// The provider keeps `bytes` themselves, where `Data(_:)` of an array
    /// would be one more full-size copy.
    private static func sRGBImage(_ bytes: Data, width: Int, height: Int) throws -> CGImage {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let provider = CGDataProvider(data: bytes as CFData),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "could not rebuild the decoded image")
        }
        return image
    }
}

import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import SemanticSearch
import UniformTypeIdentifiers

/// Decoding and downscaling for the image payloads providers are handed.
///
/// Every provider that accepts images needs the same three things — a `CGImage`
/// from either bytes or a file, a bound on how large it may be, and a way to
/// encode one back to bytes for storage — so they live here rather than being
/// reimplemented once per provider.
public enum ImageDecoding {

    /// Decode a payload, optionally downscaling so neither side exceeds
    /// `maximumDimension`.
    ///
    /// The downscale happens inside ImageIO via `kCGImageSourceThumbnailMax…`,
    /// which decodes at the reduced size rather than decoding full-size and
    /// resampling. On a 48-megapixel photo that is the difference between 200 MB
    /// of transient memory and 4 MB.
    ///
    /// The image is upright either way. The EXIF orientation is applied only on
    /// ImageIO's thumbnail path, so a full-size decode of a turned image takes
    /// that path too, unbounded: `CGImageSourceCreateImageAtIndex` returns an
    /// iPhone portrait as stored, on its side.
    public static func decode(_ payload: ImagePayload, maximumDimension: Int? = nil) throws
        -> CGImage
    {
        let source = try imageSource(payload)

        var options: [CFString: Any] = [
            kCGImageSourceShouldCache: false,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        let bounded = maximumDimension.map { $0 > 0 } ?? false
        if bounded || orientation(of: source) != .up {
            options[kCGImageSourceCreateThumbnailFromImageAlways] = true
            if bounded { options[kCGImageSourceThumbnailMaxPixelSize] = maximumDimension }
            guard
                let image = CGImageSourceCreateThumbnailAtIndex(
                    source, 0, options as CFDictionary)
            else {
                throw SemanticSearchError.unsupportedContent(
                    itemID: "image", detail: "could not decode image at the requested size")
            }
            return image
        }

        guard let image = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary)
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "could not decode image")
        }
        return image
    }

    /// A payload as Qwen3-VL's reference reads a file, but upright.
    ///
    /// `qwen_vl_utils.fetch_image` opens it with Pillow and `to_rgb`s it: the
    /// stored values are the pixels — an embedded profile is ignored, where
    /// `CIImage(contentsOf:)` converts a Display P3 photo to sRGB — and RGBA is
    /// composited on white, where Core Image's premultiplied alpha leaves a
    /// transparent background black. Neither applies the EXIF orientation, so
    /// an iPhone portrait, stored landscape with orientation 6, was embedded on
    /// its side; here it is turned upright, as transformers' `load_image` does.
    /// Pillow drops rather than composites the alpha of grey and palette
    /// images; here every alpha is composited.
    ///
    /// Full size, so the model's processor resamples once, from the whole
    /// picture, as the reference does. Rendered with colour management off:
    /// the values pass through unconverted — grey into all three channels, as
    /// `convert("RGB")` puts it — and are tagged sRGB, which the processor reads
    /// as they are.
    public static func decodeUpright(_ payload: ImagePayload) throws -> CGImage {
        // Core Image autoreleases what it renders, and the providers call this
        // from async code that no pool drains: a long reranking run held
        // gigabytes of decoded bitmaps by its end.
        try autoreleasepool {
            let source = try imageSource(payload)
            guard
                let decoded = CGImageSourceCreateImageAtIndex(
                    source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
            else {
                throw SemanticSearchError.unsupportedContent(
                    itemID: "image", detail: "could not decode image")
            }
            let image = CIImage(cgImage: decoded).oriented(orientation(of: source))
            let extent = image.extent
            guard !extent.isEmpty, !extent.isInfinite,
                let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
                let rendered = unmanagedContext.createCGImage(
                    image.composited(over: CIImage(color: .white).cropped(to: extent)),
                    from: extent, format: .RGBA8, colorSpace: sRGB)
            else {
                throw SemanticSearchError.unsupportedContent(
                    itemID: "image", detail: "could not decode image")
            }
            return rendered
        }
    }

    private static let unmanagedContext = CIContext(options: [
        .workingColorSpace: NSNull(),
        .outputColorSpace: NSNull(),
        .cacheIntermediates: false,
    ])

    private static func imageSource(_ payload: ImagePayload) throws -> CGImageSource {
        switch payload {
        case .fileURL(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
            }
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
                throw SemanticSearchError.unsupportedContent(
                    itemID: ItemID(url.lastPathComponent),
                    detail: "not a decodable image")
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

    private static func orientation(of source: CGImageSource) -> CGImagePropertyOrientation {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        return (properties?[kCGImagePropertyOrientation] as? NSNumber)
            .flatMap { CGImagePropertyOrientation(rawValue: $0.uint32Value) } ?? .up
    }

    /// Encode a `CGImage` for storage in a representation.
    ///
    /// PNG by default: an index stores images that were often produced by
    /// rendering or frame extraction, and re-encoding those as JPEG adds
    /// artefacts to the thing an OCR pass will read next.
    public static func encode(
        _ image: CGImage, as type: UTType = .png, quality: Double = 0.9
    ) throws -> Data {
        let data = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                data as CFMutableData, type.identifier as CFString, 1, nil)
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "no encoder for \(type.identifier)")
        }
        CGImageDestinationAddImage(
            destination, image,
            [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw SemanticSearchError.unsupportedContent(
                itemID: "image", detail: "could not encode image as \(type.identifier)")
        }
        return data as Data
    }

    /// Pixel dimensions without decoding the image.
    public static func dimensions(of payload: ImagePayload) -> CGSize? {
        let source: CGImageSource?
        switch payload {
        case .fileURL(let url): source = CGImageSourceCreateWithURL(url as CFURL, nil)
        case .encoded(let data, _): source = CGImageSourceCreateWithData(data as CFData, nil)
        }
        guard let source,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return CGSize(width: width, height: height)
    }
}

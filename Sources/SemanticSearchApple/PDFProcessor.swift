import CoreGraphics
import Foundation
import PDFKit
import SemanticSearch

/// PDFs, page by page: the text layer where a page has one, Vision's reading
/// of a rendering where it does not.
///
/// A born-digital page already holds its exact text, and reading it back costs
/// milliseconds; a scan holds only pixels. A page whose layer has fewer than 48
/// characters, or more than 8 % unmapped glyphs — subset fonts with no
/// ToUnicode map — is rendered at 200 dpi and read by ``VisionOCRProcessor`` as
/// a document, which keeps the rendering as the page's picture. Everything else
/// is chunked as text.
///
/// The PDF is the parent; each page is a child `<id>#page-<n>`, 1-based as a
/// reader counts, with ``SourceLocator/pageIndex`` 0-based; a long page's
/// passages are children of the page.
public struct PDFProcessor: ContentProcessor {

    public struct Configuration: Sendable, Hashable {
        public var text: TextChunkingProcessor.Configuration
        public var ocr: VisionOCRProcessor.Configuration
        /// Resolution scans are rendered at for recognition.
        public var dpi: Double
        /// Fewer characters than this and the page is read as a scan. Above
        /// the length of a folio, so a picture with a page number is still
        /// read from its pixels.
        public var minimumCharacters: Int
        /// Share of characters that may be unmapped glyphs before the layer
        /// is judged broken.
        public var maximumUnmappedRatio: Double

        public init(
            text: TextChunkingProcessor.Configuration = .default,
            ocr: VisionOCRProcessor.Configuration = .default,
            dpi: Double = 200,
            minimumCharacters: Int = 48,
            maximumUnmappedRatio: Double = 0.08
        ) {
            self.text = text
            self.ocr = ocr
            self.dpi = dpi
            self.minimumCharacters = minimumCharacters
            self.maximumUnmappedRatio = maximumUnmappedRatio
        }

        public static let `default` = Configuration()
    }

    public let configuration: Configuration

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    public var identifier: String { "apple.pdf" }
    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsText: true, acceptsImages: true, contentTypes: ["com.adobe.pdf"],
            producedKinds: [.originalText, .ocrText, .image])
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("apple.pdf.v1")
            hasher.append(TextChunkingProcessor(configuration: configuration.text)
                .configurationFingerprint)
            hasher.append(VisionOCRProcessor(configuration: configuration.ocr)
                .configurationFingerprint)
            hasher.append(configuration.dpi)
            hasher.append(configuration.minimumCharacters)
            hasher.append(configuration.maximumUnmappedRatio)
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        guard case .fileURL(let url) = content.payload else { return false }
        return url.pathExtension.lowercased() == "pdf"
    }

    /// How a page was read.
    public enum Route: String, Sendable {
        case textLayer
        case recognized
    }

    /// The route a page's text layer calls for.
    public func route(forLayer text: String) -> Route {
        let characters = text.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0)
        }
        guard characters.count >= configuration.minimumCharacters else { return .recognized }
        let unmapped = characters.filter {
            $0.value == 0xFFFD || (0xE000...0xF8FF).contains($0.value)
        }.count
        return Double(unmapped) / Double(characters.count) > configuration.maximumUnmappedRatio
            ? .recognized : .textLayer
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .fileURL(let url) = content.payload else {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "PDFProcessor needs a PDF file URL")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: content.id)
        }
        guard let document = PDFDocument(url: url) else {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "not a readable PDF")
        }
        if document.isLocked {
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "the PDF is encrypted")
        }

        var metadata = content.metadata
        if let title = document.documentAttributes?[PDFDocumentAttribute.titleAttribute]
            as? String, !title.isEmpty, metadata["title"] == nil
        {
            metadata["title"] = .string(title)
        }
        var items = [
            SearchItem(
                id: content.id, parentID: content.parentID, metadata: metadata,
                locator: content.locator, representations: [])
        ]

        let text = TextChunkingProcessor(configuration: configuration.text)
        let ocr = VisionOCRProcessor(configuration: configuration.ocr)
        for index in 0..<document.pageCount {
            try Task.checkCancellation()
            guard let page = document.page(at: index) else { continue }
            onProgress?(
                ProcessingProgress(
                    itemID: content.id, stage: "page \(index + 1)",
                    fraction: Double(index) / Double(max(1, document.pageCount))))

            var locator = content.locator
            locator.pageIndex = index
            let pageID = ItemID("\(content.id.rawValue)#page-\(index + 1)")
            let layer = page.string ?? ""
            var pageItems: [SearchItem]
            switch route(forLayer: layer) {
            case .textLayer:
                pageItems = try await text.process(
                    RawContent(
                        id: pageID, parentID: content.id, payload: .text(layer),
                        metadata: metadata, locator: locator,
                        languageHint: content.languageHint),
                    onProgress: nil)
            case .recognized:
                let png = try ImageDecoding.encode(render(page, id: pageID))
                pageItems = try await ocr.process(
                    RawContent(
                        id: pageID, parentID: content.id,
                        payload: .imageData(png, contentType: "public.png"),
                        metadata: metadata, locator: locator,
                        languageHint: content.languageHint),
                    onProgress: nil)
            }
            // An empty page is still a page of the document.
            if pageItems.isEmpty {
                pageItems = [
                    SearchItem(
                        id: pageID, parentID: content.id, metadata: metadata,
                        locator: locator, representations: [])
                ]
            }
            items += pageItems
        }
        onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
        return items
    }

    /// The page as printed, on white, at ``Configuration/dpi``.
    func render(_ page: PDFPage, id: ItemID) throws -> CGImage {
        let bounds = page.bounds(for: .cropBox)
        let scale = configuration.dpi / 72
        let width = max(1, Int((bounds.width * scale).rounded()))
        let height = max(1, Int((bounds.height * scale).rounded()))
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw SemanticSearchError.unsupportedContent(
                itemID: id, detail: "could not render a \(width)×\(height) page")
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -bounds.minX, y: -bounds.minY)
        page.draw(with: .cropBox, to: context)
        guard let image = context.makeImage() else {
            throw SemanticSearchError.unsupportedContent(
                itemID: id, detail: "could not render the page")
        }
        return image
    }
}

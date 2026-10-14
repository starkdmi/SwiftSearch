import CoreGraphics
import Foundation
import NaturalLanguage
import SemanticSearch
import Vision

/// One line or block Vision recognised.
public struct RecognizedText: Sendable, Hashable {
    public var text: String
    /// Normalised, origin top-left. Vision reports bottom-left; the conversion
    /// happens once, here, so nothing downstream has to remember which
    /// convention it is holding.
    public var boundingBox: SourceLocator.BoundingBox
    public var confidence: Float
    /// BCP-47 tag Vision was asked for or detected, when it reports one.
    public var language: String?

    public init(
        text: String, boundingBox: SourceLocator.BoundingBox, confidence: Float,
        language: String? = nil
    ) {
        self.text = text
        self.boundingBox = boundingBox
        self.confidence = confidence
        self.language = language
    }
}

/// Apple Vision text recognition.
///
/// The cheap, always-available recogniser: no weights to download, no GPU
/// memory held, a page in tens of milliseconds. Emits an `ocrText`
/// representation carrying the recognised text, the language, and a bounding
/// box per line — all three preserved rather than flattened to a string.
///
/// The original image is preserved as its own representation. OCR is an
/// additional view of the content, never a replacement for it.
public struct VisionOCRProcessor: ContentProcessor {

    public struct Configuration: Sendable, Hashable {
        /// Languages passed to the recogniser. Empty means automatic detection.
        public var recognitionLanguages: [String]
        /// Trade speed for accuracy. `false` selects the fast path, roughly 5×
        /// quicker and noticeably worse on small type.
        public var accurate: Bool
        /// Apply the recogniser's language model. Helps prose, hurts serial
        /// numbers, part numbers and code.
        public var usesLanguageCorrection: Bool
        /// Drop lines the recogniser is less sure of than this.
        public var minimumConfidence: Float
        /// Downscale before recognition so neither side exceeds this. Vision
        /// does not benefit from more than a few thousand pixels and the memory
        /// cost is quadratic.
        public var maximumDimension: Int
        /// Keep the source image as its own representation alongside the text.
        public var retainsImageRepresentation: Bool
        /// Chunk recognised text when it exceeds the chunker's budget.
        public var chunker: ChunkerConfiguration?
        /// On macOS 26 / iOS 26, read the page as a document — title,
        /// paragraphs, table rows, list items, in reading order — through
        /// `RecognizeDocumentsRequest`. Line OCR hands back a table as a column
        /// of unrelated cells and a paragraph as a dozen fragments; a passage
        /// is what gets searched. Earlier systems use line recognition
        /// regardless.
        public var prefersDocumentStructure: Bool

        public init(
            recognitionLanguages: [String] = [],
            accurate: Bool = true,
            usesLanguageCorrection: Bool = true,
            minimumConfidence: Float = 0.3,
            maximumDimension: Int = 3_000,
            retainsImageRepresentation: Bool = true,
            chunker: ChunkerConfiguration? = nil,
            prefersDocumentStructure: Bool = true
        ) {
            self.recognitionLanguages = recognitionLanguages
            self.accurate = accurate
            self.usesLanguageCorrection = usesLanguageCorrection
            self.minimumConfidence = minimumConfidence
            self.maximumDimension = maximumDimension
            self.retainsImageRepresentation = retainsImageRepresentation
            self.chunker = chunker
            self.prefersDocumentStructure = prefersDocumentStructure
        }

        public static let `default` = Configuration()
    }

    public let configuration: Configuration

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    public var identifier: String { "vision.ocr" }

    /// The recognisers' revisions, pinned as the keyframes' feature prints are.
    /// A revision is Vision's promise of one algorithm; the OS version it used
    /// to be is not, and since a chain's fingerprint covers every processor in
    /// it, each minor macOS update processed every file of a library again —
    /// videos and recordings included.
    public var version: String {
        "lines.r\(Self.lineRevision)+documents.r1"
    }

    static let lineRevision = VNRecognizeTextRequestRevision3

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsImages: true,
            contentTypes: ["public.image"],
            producedKinds: configuration.retainsImageRepresentation
                ? [.ocrText, .image] : [.ocrText])
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("vision.ocr.v3")
            hasher.append(configuration.prefersDocumentStructure && Self.documentRequestAvailable)
            hasher.append(version)
            hasher.append(configuration.recognitionLanguages.sorted())
            hasher.append(configuration.accurate)
            hasher.append(configuration.usesLanguageCorrection)
            hasher.append(Double(configuration.minimumConfidence))
            hasher.append(configuration.maximumDimension)
            hasher.append(configuration.retainsImageRepresentation)
            hasher.append(configuration.chunker?.fingerprint ?? "")
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        switch content.payload {
        case .imageData: return true
        case .fileURL(let url): return Self.imageExtensions.contains(
                url.pathExtension.lowercased())
        case .text: return false
        }
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        try Task.checkCancellation()
        onProgress?(ProcessingProgress(itemID: content.id, stage: "decode", fraction: 0))

        let payload: ImagePayload
        switch content.payload {
        case .imageData(let data, let contentType):
            payload = .encoded(data, contentType: contentType)
        case .fileURL(let url):
            payload = .fileURL(url)
        case .text:
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "VisionOCRProcessor does not accept text")
        }

        let image = try ImageDecoding.decode(
            payload, maximumDimension: configuration.maximumDimension)

        try Task.checkCancellation()
        onProgress?(ProcessingProgress(itemID: content.id, stage: "recognize", fraction: 0.2))

        let lines = try await recognize(image)
        try Task.checkCancellation()
        onProgress?(ProcessingProgress(itemID: content.id, stage: "assemble", fraction: 0.9))

        let provenance = Provenance(
            producer: identifier, version: version,
            configurationFingerprint: configurationFingerprint,
            confidence: lines.isEmpty
                ? nil : lines.map(\.confidence).reduce(0, +) / Float(lines.count))

        var representations: [ContentRepresentation] = []

        if configuration.retainsImageRepresentation {
            // The source, unchanged. A multimodal space embeds this; OCR text
            // is an additional channel, not a substitute for the pixels.
            switch content.payload {
            case .imageData(let data, let contentType):
                representations.append(
                    ContentRepresentation(
                        id: "image", kind: .image,
                        content: .imageData(data, contentType: contentType),
                        locator: content.locator))
            case .fileURL(let url):
                representations.append(
                    ContentRepresentation(
                        id: "image", kind: .image, content: .mediaReference(url),
                        locator: content.locator))
            case .text:
                break
            }
        }

        let recognizedText = lines.map(\.text).joined(separator: "\n")
        guard !recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
            return [
                SearchItem(
                    id: content.id, parentID: content.parentID, metadata: content.metadata,
                    locator: content.locator, representations: representations)
            ]
        }

        let language = pageLanguage(of: recognizedText, lines: lines, hint: content.languageHint)

        if let chunkerConfiguration = configuration.chunker,
            TokenEstimator.estimate(recognizedText) > chunkerConfiguration.maximumTokens
        {
            // A long page becomes children of the page rather than a single
            // over-length representation. The page keeps its image, so a visual
            // match still lands on the page and a text match lands on the
            // passage within it.
            let chunker = TextChunker(configuration: chunkerConfiguration)
            let chunks = chunker.chunk(recognizedText, language: language)
            var items: [SearchItem] = [
                SearchItem(
                    id: content.id, parentID: content.parentID, metadata: content.metadata,
                    locator: content.locator, representations: representations)
            ]
            for chunk in chunks {
                items.append(
                    SearchItem(
                        id: ItemID("\(content.id.rawValue)#chunk-\(chunk.ordinal)"),
                        parentID: content.id,
                        metadata: content.metadata,
                        locator: content.locator.narrowed(toCharacters: chunk.range),
                        representations: [
                            ContentRepresentation(
                                id: "ocr", kind: .ocrText, content: .text(chunk.text),
                                language: language, provenance: provenance,
                                locator: locator(
                                    for: chunk, lines: lines, base: content.locator))
                        ],
                        ordinal: chunk.ordinal))
            }
            onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
            return items
        }

        var ocrLocator = content.locator
        ocrLocator.boundingBoxes = lines.map(\.boundingBox)
        representations.append(
            ContentRepresentation(
                id: "ocr", kind: .ocrText, content: .text(recognizedText),
                language: language, provenance: provenance, locator: ocrLocator))

        onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
        return [
            SearchItem(
                id: content.id, parentID: content.parentID, metadata: content.metadata,
                locator: content.locator, representations: representations)
        ]
    }

    // MARK: - Recognition

    /// True when this OS has `RecognizeDocumentsRequest`.
    static var documentRequestAvailable: Bool {
        if #available(macOS 26, iOS 26, visionOS 26, *) { return true }
        return false
    }

    /// Recognise text: document structure where the OS has it and it was
    /// asked for, lines otherwise. Each element is a block (structured) or a
    /// line, in reading order, with its box.
    public func recognize(_ image: CGImage) async throws -> [RecognizedText] {
        if configuration.prefersDocumentStructure,
            #available(macOS 26, iOS 26, visionOS 26, *)
        {
            let blocks = try await recognizeDocument(image)
            // A page with no document structure Vision can find — a photo with
            // a sign in it — still has lines.
            if !blocks.isEmpty { return blocks }
        }
        return try recognizeLines(image)
    }

    /// Title, paragraphs, tables and lists, in reading order.
    ///
    /// Cell and list-item text is also reported as paragraphs, so paragraphs
    /// inside a table or list region are dropped rather than emitted twice. A
    /// table becomes one block, a row per line with cells separated by " | "; a
    /// list becomes one block, an item per line.
    @available(macOS 26, iOS 26, visionOS 26, *)
    func recognizeDocument(_ image: CGImage) async throws -> [RecognizedText] {
        var request = RecognizeDocumentsRequest(.revision1)
        request.textRecognitionOptions.useLanguageCorrection =
            configuration.usesLanguageCorrection
        if configuration.recognitionLanguages.isEmpty {
            request.textRecognitionOptions.automaticallyDetectLanguage = true
        } else {
            request.textRecognitionOptions.recognitionLanguages =
                configuration.recognitionLanguages.map { Locale.Language(identifier: $0) }
        }
        let observations = try await request.perform(on: image)
        guard let observation = observations.first else { return [] }
        let document = observation.document
        let language =
            configuration.recognitionLanguages.count == 1
            ? configuration.recognitionLanguages[0] : nil
        let confidence = observation.confidence

        var blocks: [RecognizedText] = []
        var structured: [SourceLocator.BoundingBox] = []
        func add(_ text: String, _ box: SourceLocator.BoundingBox) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            blocks.append(
                RecognizedText(
                    text: trimmed, boundingBox: box, confidence: confidence, language: language))
        }

        if let title = document.title {
            let box = Self.flip(title.boundingRegion.normalizedPath.boundingBox)
            structured.append(box)
            add(title.transcript, box)
        }
        for table in document.tables {
            let box = Self.flip(table.boundingRegion.normalizedPath.boundingBox)
            structured.append(box)
            add(
                table.rows.map { row in
                    row.map {
                        $0.content.text.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                    }.joined(separator: " | ")
                }.joined(separator: "\n"), box)
        }
        for list in document.lists {
            let box = Self.flip(list.boundingRegion.normalizedPath.boundingBox)
            structured.append(box)
            add(
                list.items.map {
                    "- " + $0.itemString.trimmingCharacters(in: .whitespacesAndNewlines)
                }.joined(separator: "\n"), box)
        }
        for paragraph in document.paragraphs {
            let box = Self.flip(paragraph.boundingRegion.normalizedPath.boundingBox)
            guard !structured.contains(where: { Self.contains($0, box) }) else { continue }
            add(paragraph.transcript, box)
        }

        // Top to bottom, then left to right — Vision reports tables, lists and
        // paragraphs in separate arrays with no shared order.
        return blocks.sorted { lhs, rhs in
            let lhsMid = lhs.boundingBox.minY + lhs.boundingBox.height / 2
            let rhsMid = rhs.boundingBox.minY + rhs.boundingBox.height / 2
            if abs(lhsMid - rhsMid) > 0.01 { return lhsMid < rhsMid }
            return lhs.boundingBox.minX < rhs.boundingBox.minX
        }
    }

    /// True when `inner` lies almost entirely inside `outer`.
    static func contains(
        _ outer: SourceLocator.BoundingBox, _ inner: SourceLocator.BoundingBox
    ) -> Bool {
        let width = min(outer.maxX, inner.maxX) - max(outer.minX, inner.minX)
        let height = min(outer.maxY, inner.maxY) - max(outer.minY, inner.minY)
        let area = inner.width * inner.height
        guard width > 0, height > 0, area > 0 else { return false }
        return width * height / area > 0.8
    }

    /// Line recognition: every system, and the fallback.
    public func recognizeLines(_ image: CGImage) throws -> [RecognizedText] {
        let request = VNRecognizeTextRequest()
        request.revision = Self.lineRevision
        request.recognitionLevel = configuration.accurate ? .accurate : .fast
        request.usesLanguageCorrection = configuration.usesLanguageCorrection
        if !configuration.recognitionLanguages.isEmpty {
            request.recognitionLanguages = configuration.recognitionLanguages
        } else {
            request.automaticallyDetectsLanguage = true
        }

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        return (request.results ?? []).compactMap { observation in
            guard observation.confidence >= configuration.minimumConfidence,
                let candidate = observation.topCandidates(1).first
            else { return nil }
            return RecognizedText(
                text: candidate.string,
                boundingBox: Self.flip(observation.boundingBox),
                confidence: observation.confidence,
                // Only when it is certain. Vision does not say which of several
                // requested languages a line was read as, and labelling every
                // line with the first one would be a guess dressed as a fact.
                language: configuration.recognitionLanguages.count == 1
                    ? configuration.recognitionLanguages[0] : nil)
        }
    }

    /// The language of the recognised page.
    ///
    /// Vision does not report one, so: the configured language when exactly one
    /// was requested; otherwise the language `NLLanguageRecognizer` is
    /// confident of, constrained to the requested set when there is one; and
    /// the caller's hint when neither applies. It matters downstream — the
    /// lexical analyzer segments Chinese and Japanese by it.
    func pageLanguage(of text: String, lines: [RecognizedText], hint: String?) -> String? {
        if configuration.recognitionLanguages.count == 1 {
            return configuration.recognitionLanguages[0]
        }
        // Vision's tags are regional (`ru-RU`, `ja-JP`); NaturalLanguage's are
        // not (`ru`, `ja`), except for the two Chinese scripts. A detected
        // language is reported as the tag the caller configured for it.
        var configuredFor: [NLLanguage: String] = [:]
        for tag in configuration.recognitionLanguages {
            configuredFor[Self.naturalLanguage(for: tag)] = tag
        }

        let recognizer = NLLanguageRecognizer()
        if !configuredFor.isEmpty { recognizer.languageConstraints = Array(configuredFor.keys) }
        recognizer.processString(text)
        if let (language, probability) = recognizer.languageHypotheses(withMaximum: 1).first,
            probability >= 0.6, language != .undetermined
        {
            return configuredFor[language] ?? language.rawValue
        }
        return hint
    }

    static func naturalLanguage(for tag: String) -> NLLanguage {
        let lowered = tag.lowercased()
        if lowered.hasPrefix("zh") {
            return lowered.contains("hant") || lowered.hasSuffix("-tw") || lowered.hasSuffix("-hk")
                ? .traditionalChinese : .simplifiedChinese
        }
        return NLLanguage(rawValue: TextNormalization.baseLanguageCode(tag))
    }

    /// Bounding boxes for the lines a chunk covers.
    ///
    /// By offset: the chunked text is the lines joined with newlines, the
    /// chunk's range is a UTF-16 range into it, and a line belongs to every chunk its own
    /// range overlaps. Matching by content instead puts a short line — "OK",
    /// a page number — into every chunk that happens to contain those
    /// characters, and loses a line the chunker split across two chunks.
    func locator(
        for chunk: TextChunk, lines: [RecognizedText], base: SourceLocator
    ) -> SourceLocator {
        var locator = base.narrowed(toCharacters: chunk.range)
        let chunkStart = chunk.range.location
        let chunkEnd = chunk.range.location + chunk.range.length

        var boxes: [SourceLocator.BoundingBox] = []
        var lineStart = 0
        for line in lines {
            let length = (line.text as NSString).length
            if length > 0, lineStart < chunkEnd, lineStart + length > chunkStart {
                boxes.append(line.boundingBox)
            }
            lineStart += length + 1  // the joining newline
        }
        if !boxes.isEmpty { locator.boundingBoxes = boxes }
        return locator
    }

    /// Vision normalises with the origin at the bottom left; ``SourceLocator``
    /// uses the top left, matching every drawing API a caller will hand the box
    /// to next.
    static func flip(_ rect: CGRect) -> SourceLocator.BoundingBox {
        SourceLocator.BoundingBox(
            x: Double(rect.minX),
            y: Double(1 - rect.maxY),
            width: Double(rect.width),
            height: Double(rect.height))
    }

    static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "heif", "tiff", "tif", "gif", "bmp", "webp",
    ]
}

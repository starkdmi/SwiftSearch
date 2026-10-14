import Foundation

/// Turns a block of text into a parent item and one child per chunk.
///
/// The default processor for text: everything above the chunk budget is split
/// on paragraph and sentence boundaries, and everything below it is indexed as
/// one representation. The parent keeps the whole text so `searchInside` has
/// something to scope to and the reranker has context to draw on.
public struct TextChunkingProcessor: ContentProcessor {

    public struct Configuration: Sendable, Hashable {
        public var chunker: ChunkerConfiguration
        /// Keep the full text on the parent. Costs a copy of the source in the
        /// database and buys parent context for reranking without a re-read of
        /// the original file.
        public var retainsFullText: Bool
        /// Kind assigned to the chunks.
        public var kind: RepresentationKind
        /// Which inputs are cut at their Markdown headings before chunking.
        public var markdown: MarkdownHandling

        public enum MarkdownHandling: String, Sendable, Hashable {
            /// `.md` and `.markdown` files.
            case byExtension
            /// Every input, text payloads included.
            case always
            /// None: Markdown is chunked as plain text.
            case never
        }

        public init(
            chunker: ChunkerConfiguration = .default,
            retainsFullText: Bool = true,
            kind: RepresentationKind = .originalText,
            markdown: MarkdownHandling = .byExtension
        ) {
            self.chunker = chunker
            self.retainsFullText = retainsFullText
            self.kind = kind
            self.markdown = markdown
        }

        public static let `default` = Configuration()
    }

    public let configuration: Configuration

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    /// Build one constrained to a provider's advertised input limit.
    ///
    /// A 512-token chunk handed to a 256-token provider is a 256-token chunk
    /// with its tail silently discarded.
    public init(chunker: ChunkerConfiguration = .default, providerTokenLimit: Int) {
        self.configuration = Configuration(
            chunker: chunker.constrained(toProviderLimit: providerTokenLimit))
    }

    public var identifier: String { "text.chunker" }
    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        ProcessorCapabilities(
            acceptsText: true,
            contentTypes: ["public.plain-text", "public.utf8-plain-text"],
            producedKinds: [configuration.kind])
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("text.chunker.v1")
            hasher.append(configuration.chunker.fingerprint)
            hasher.append(configuration.retainsFullText)
            hasher.append(configuration.kind.rawValue)
            if configuration.markdown != .never {
                hasher.append("markdown-sections.v1")
                hasher.append(configuration.markdown.rawValue)
            }
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        switch content.payload {
        case .text: return true
        case .fileURL(let url):
            return ["txt", "md", "markdown", "text", "csv", "json", "log"].contains(
                url.pathExtension.lowercased())
        case .imageData: return false
        }
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        try Task.checkCancellation()

        let text: String
        var isMarkdown = configuration.markdown == .always
        switch content.payload {
        case .text(let value):
            text = value
        case .fileURL(let url):
            if configuration.markdown == .byExtension {
                isMarkdown = ["md", "markdown"].contains(url.pathExtension.lowercased())
            }
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SemanticSearchError.mediaUnavailable(url: url, itemID: content.id)
            }
            text = try String(contentsOf: url, encoding: .utf8)
        case .imageData:
            throw SemanticSearchError.unsupportedContent(
                itemID: content.id, detail: "TextChunkingProcessor does not accept images")
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let provenance = Provenance(
            producer: identifier, version: version,
            configurationFingerprint: configurationFingerprint)
        let language = content.languageHint

        onProgress?(ProcessingProgress(itemID: content.id, stage: "chunk", fraction: 0.1))

        if TokenEstimator.estimate(text) <= configuration.chunker.maximumTokens {
            // Short enough to index whole. Splitting it anyway would produce a
            // parent and exactly one child, which doubles the row count and
            // gives fusion two units where the corpus has one.
            onProgress?(ProcessingProgress(itemID: content.id, stage: "done", fraction: 1))
            return [
                SearchItem(
                    id: content.id, parentID: content.parentID, metadata: content.metadata,
                    locator: content.locator,
                    representations: [
                        ContentRepresentation(
                            id: "text", kind: configuration.kind, content: .text(text),
                            language: language, provenance: provenance,
                            locator: content.locator)
                    ])
            ]
        }

        let chunks =
            isMarkdown
            ? markdownChunks(text, language: language)
            : TextChunker(configuration: configuration.chunker).chunk(text, language: language)

        var parentRepresentations: [ContentRepresentation] = []
        if configuration.retainsFullText {
            parentRepresentations.append(
                ContentRepresentation(
                    id: "text", kind: .summary, content: .text(text), language: language,
                    provenance: provenance, locator: content.locator, weight: 0))
        }

        var items: [SearchItem] = [
            SearchItem(
                id: content.id, parentID: content.parentID, metadata: content.metadata,
                locator: content.locator, representations: parentRepresentations)
        ]

        for chunk in chunks {
            try Task.checkCancellation()
            let locator = content.locator.narrowed(toCharacters: chunk.range)
            items.append(
                SearchItem(
                    id: ItemID("\(content.id.rawValue)#chunk-\(chunk.ordinal)"),
                    parentID: content.id,
                    metadata: content.metadata,
                    locator: locator,
                    representations: [
                        ContentRepresentation(
                            id: "text", kind: configuration.kind, content: .text(chunk.text),
                            language: language, provenance: provenance, locator: locator)
                    ],
                    ordinal: chunk.ordinal))
            onProgress?(
                ProcessingProgress(
                    itemID: content.id, stage: "chunk",
                    fraction: 0.1 + 0.9 * Double(chunk.ordinal + 1) / Double(chunks.count)))
        }

        return items
    }

    /// Each section chunked on its own, so no chunk spans two headings, and
    /// each chunk prefixed with the headings above it: "Install › macOS",
    /// then the text. A chunk that opens with its own heading line gets only
    /// the headings above that one. Ranges stay on the source, without the
    /// prefix.
    func markdownChunks(_ text: String, language: String?) -> [TextChunk] {
        let source = text as NSString
        var chunks: [TextChunk] = []
        for section in MarkdownSections.split(text) where section.hasBody {
            let body = source.substring(
                with: NSRange(location: section.range.location, length: section.range.length))
            // The prefix counts against the budget, so a chunk and its
            // headings still fit the provider.
            let reserved = TokenEstimator.estimate(
                section.headings.joined(separator: " › ") + "\n\n")
            let chunker = TextChunker(
                configuration: section.headings.isEmpty
                    ? configuration.chunker
                    : configuration.chunker.constrained(
                        toProviderLimit: max(1, configuration.chunker.maximumTokens - reserved)))
            for var chunk in chunker.chunk(body, language: language) {
                let opensWithHeading = chunk.range.location == 0
                let path = opensWithHeading ? section.headings.dropLast() : section.headings[...]
                if !path.isEmpty {
                    let prefix = path.joined(separator: " › ") + "\n\n"
                    chunk.text = prefix + chunk.text
                    chunk.overlapPrefixLength += (prefix as NSString).length
                    chunk.estimatedTokens += TokenEstimator.estimate(prefix)
                }
                chunk.range = SourceLocator.CharacterRange(
                    location: chunk.range.location + section.range.location,
                    length: chunk.range.length)
                chunk.ordinal = chunks.count
                chunks.append(chunk)
            }
        }
        return chunks
    }
}

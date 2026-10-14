import Foundation

/// Raw input handed to a processor before it is anything the engine can index.
public struct RawContent: Sendable {
    public enum Payload: Sendable {
        case text(String)
        case imageData(Data, contentType: String)
        case fileURL(URL)
    }

    public var id: ItemID
    public var parentID: ItemID?
    public var payload: Payload
    public var metadata: Metadata
    public var locator: SourceLocator
    /// BCP-47 hint. A processor may override it if it detects otherwise, and
    /// should say so in the representation's provenance when it does.
    public var languageHint: String?

    public init(
        id: ItemID,
        parentID: ItemID? = nil,
        payload: Payload,
        metadata: Metadata = [:],
        locator: SourceLocator = .unspecified,
        languageHint: String? = nil
    ) {
        self.id = id
        self.parentID = parentID
        self.payload = payload
        self.metadata = metadata
        self.locator = locator
        self.languageHint = languageHint
    }
}

/// What a processor can turn into representations.
public struct ProcessorCapabilities: Sendable, Hashable, Codable {
    public var acceptsText: Bool
    public var acceptsImages: Bool
    public var acceptsAudio: Bool
    public var acceptsVideo: Bool
    /// Content types the processor claims, as UTIs or MIME types. Empty means
    /// it decides per input.
    public var contentTypes: [String]
    public var producedKinds: Set<RepresentationKind>

    public init(
        acceptsText: Bool = false,
        acceptsImages: Bool = false,
        acceptsAudio: Bool = false,
        acceptsVideo: Bool = false,
        contentTypes: [String] = [],
        producedKinds: Set<RepresentationKind> = []
    ) {
        self.acceptsText = acceptsText
        self.acceptsImages = acceptsImages
        self.acceptsAudio = acceptsAudio
        self.acceptsVideo = acceptsVideo
        self.contentTypes = contentTypes
        self.producedKinds = producedKinds
    }
}

/// Progress from a long-running processor. Reported per input, not per corpus,
/// so a caller can drive a determinate bar over its own work list.
public struct ProcessingProgress: Sendable {
    public var itemID: ItemID
    public var stage: String
    /// 0...1 within the current stage, or `nil` when the stage has no measure.
    public var fraction: Double?

    public init(itemID: ItemID, stage: String, fraction: Double? = nil) {
        self.itemID = itemID
        self.stage = stage
        self.fraction = fraction
    }
}

/// Turns raw input into searchable items.
///
/// Replace this to substitute a different OCR engine, a different ASR model, a
/// different chunking policy, or any enrichment step invented later. The engine
/// calls it and then indexes whatever comes back; it has no opinion about how
/// the representations were produced.
public protocol ContentProcessor: Sendable {
    /// Stable name, recorded as ``Provenance/producer`` on what this emits.
    var identifier: String { get }
    var version: String { get }
    var capabilities: ProcessorCapabilities { get }

    /// Fingerprint of this processor's configuration. Changing it is what makes
    /// the engine redo derived work on the next upsert; leaving it alone is
    /// what makes the engine skip it.
    var configurationFingerprint: String { get }

    func canProcess(_ content: RawContent) -> Bool

    /// Process one input.
    ///
    /// Returns one or more items: a processor may split its input (a long page
    /// into chunks, a recording into utterances) and the pieces are returned as
    /// children of the input's id.
    ///
    /// Must honour `Task.isCancelled`.
    func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem]

    /// What this processor made from a file, for the same bytes at another
    /// path — a file that was moved or renamed. `items` are already under
    /// `content`'s ids.
    ///
    /// The default, ``SearchItem/relocated(_:from:to:)``, points every
    /// locator source and media reference at the new URL and applies
    /// `content`'s metadata; a processor whose output depends on the path in
    /// any other way — the name as words — says so here, and one that wraps
    /// others passes the call on to them. Nothing is transcribed, captioned
    /// or embedded again.
    func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem]
}

extension ContentProcessor {
    public func process(_ content: RawContent) async throws -> [SearchItem] {
        try await process(content, onProgress: nil)
    }

    public func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem] {
        SearchItem.relocated(items, from: old, to: content)
    }
}

extension SearchItem {
    /// `items` with every locator source and media reference that named
    /// `old` naming `content`'s file, and `content`'s metadata applied.
    public static func relocated(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) -> [SearchItem] {
        guard case .fileURL(let new) = content.payload else { return items }
        let before = old.standardizedFileURL.absoluteString
        // Compared standardized: the id `old` comes from is, and a processor
        // writes its sources from the URL it was given, which need not be.
        func isOld(_ url: URL) -> Bool { url.standardizedFileURL.absoluteString == before }
        // A source may point into the file — `a.pdf#page=3` — and
        // standardizing drops the fragment, so it is carried over by hand.
        func moved(_ url: URL) -> URL {
            guard let fragment = url.fragment(percentEncoded: true),
                var components = URLComponents(url: new, resolvingAgainstBaseURL: false)
            else { return new }
            components.percentEncodedFragment = fragment
            return components.url ?? new
        }
        func moved(_ source: String) -> String {
            guard let url = URL(string: source), isOld(url) else { return source }
            return moved(url).absoluteString
        }
        return items.map { item in
            var item = item
            item.locator.source = item.locator.source.map(moved)
            for (key, value) in content.metadata { item.metadata[key] = value }
            item.representations = item.representations.map { representation in
                var representation = representation
                if let source = representation.locator?.source {
                    representation.locator?.source = moved(source)
                }
                if case .mediaReference(let url) = representation.content, isOld(url) {
                    representation.content = .mediaReference(moved(url))
                }
                return representation
            }
            return item
        }
    }
}

/// Runs the first processor that claims each input.
///
/// Order matters: the first match wins, so put specific processors before
/// general ones.
public struct ContentProcessorChain: ContentProcessor {
    public let processors: [any ContentProcessor]

    public init(_ processors: [any ContentProcessor]) {
        self.processors = processors
    }

    public var identifier: String { "chain" }

    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        var merged = ProcessorCapabilities()
        for processor in processors {
            let other = processor.capabilities
            merged.acceptsText = merged.acceptsText || other.acceptsText
            merged.acceptsImages = merged.acceptsImages || other.acceptsImages
            merged.acceptsAudio = merged.acceptsAudio || other.acceptsAudio
            merged.acceptsVideo = merged.acceptsVideo || other.acceptsVideo
            merged.contentTypes.append(contentsOf: other.contentTypes)
            merged.producedKinds.formUnion(other.producedKinds)
        }
        return merged
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("chain.v1")
            for processor in processors {
                hasher.append(processor.identifier)
                hasher.append(processor.version)
                hasher.append(processor.configurationFingerprint)
            }
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        processors.contains { $0.canProcess(content) }
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        for processor in processors where processor.canProcess(content) {
            return try await processor.process(content, onProgress: onProgress)
        }
        throw SemanticSearchError.unsupportedContent(
            itemID: content.id,
            detail: "no processor in the chain accepts this input")
    }

    /// The processor that made them moves them, as it made them.
    public func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem] {
        for processor in processors where processor.canProcess(content) {
            return try await processor.relocate(items, from: old, to: content)
        }
        return SearchItem.relocated(items, from: old, to: content)
    }
}

/// Runs every processor that accepts an input and merges what they produce.
///
/// For inputs with more than one kind of evidence: a video's shot frames and
/// its transcript are two independent sets of children under one parent.
/// ``ContentProcessorChain`` would run only the first processor that claims the
/// file; this runs all of them, merges their parent items into one, and keeps
/// every child. Children must not collide by id — processors that emit them
/// name them after their own kind (`#utterance-3`, `#shot-3`).
public struct CombinedContentProcessor: ContentProcessor {
    public let processors: [any ContentProcessor]

    public init(_ processors: [any ContentProcessor]) {
        self.processors = processors
    }

    public var identifier: String { "combined" }

    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        ContentProcessorChain(processors).capabilities
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("combined.v1")
            for processor in processors {
                hasher.append(processor.identifier)
                hasher.append(processor.version)
                hasher.append(processor.configurationFingerprint)
            }
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        processors.contains { $0.canProcess(content) }
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        var parent: SearchItem?
        var children: [SearchItem] = []
        var seen: Set<ItemID> = []

        for processor in processors where processor.canProcess(content) {
            try Task.checkCancellation()
            for item in try await processor.process(content, onProgress: onProgress) {
                if item.id == content.id {
                    guard var merged = parent else {
                        parent = item
                        continue
                    }
                    // One parent: the first processor's metadata and locator,
                    // every processor's representations. A representation id
                    // two processors both use keeps the first.
                    let existing = Set(merged.representations.map(\.id))
                    merged.representations += item.representations.filter {
                        !existing.contains($0.id)
                    }
                    parent = merged
                } else if seen.insert(item.id).inserted {
                    children.append(item)
                }
            }
        }

        guard let parent else {
            if children.isEmpty {
                throw SemanticSearchError.unsupportedContent(
                    itemID: content.id, detail: "no processor accepts this input")
            }
            return children
        }
        return [parent] + children
    }

    /// Every processor that took part moves the items in turn: each one's
    /// own relocation touches what it made, the path rewrite is the same
    /// every time.
    public func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem] {
        var items = SearchItem.relocated(items, from: old, to: content)
        for processor in processors where processor.canProcess(content) {
            items = try await processor.relocate(items, from: old, to: content)
        }
        return items
    }
}

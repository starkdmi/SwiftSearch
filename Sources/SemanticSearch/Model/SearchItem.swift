import Foundation

/// What a representation *is*, as opposed to what it says.
///
/// The distinction is load-bearing. An OCR representation and a caller-supplied
/// summary of the same page are both text, but a search that returns the
/// summary should say so — the words the user matched were written by a model,
/// not by the author. Kind also drives which channels a representation
/// participates in, and it is the unit of channel-family weighting in fusion.
public enum RepresentationKind: String, Sendable, Hashable, Codable, CaseIterable {
    /// Text as the author wrote it.
    case originalText
    /// Text a recogniser read off an image or page.
    case ocrText
    /// Text a recogniser heard in audio.
    case transcript
    /// A still image.
    case image
    /// A video clip.
    case video
    /// A recording with no picture. Searchable today through its transcript;
    /// embedding the sound itself needs a provider that accepts audio, which
    /// none of the bundled ones does.
    case audio
    /// A caption or description the caller generated for non-text content.
    case caption
    /// A summary or abstract of this item or its children.
    case summary
    /// A translation of another representation of the same item.
    case translation
    /// A sound a classifier heard — "glass clink", "applause" — over the
    /// stretch it heard it: a recording's non-speech evidence, as words.
    case soundEvent
    /// What a file says about itself rather than what it contains: its name,
    /// the title, artist, album, author and keywords in its tags, the place
    /// its coordinates name. Its own family, so a file's details never take
    /// the vote of the text the author wrote.
    case metadata

    /// Whether this kind carries text that lexical search should index.
    public var isLexicallySearchable: Bool {
        switch self {
        case .originalText, .ocrText, .transcript, .caption, .summary, .translation,
            .soundEvent, .metadata:
            true
        case .image, .video, .audio: false
        }
    }

    /// Whether this kind requires a provider with visual input support.
    public var isVisual: Bool {
        switch self {
        case .image, .video: true
        default: false
        }
    }

    /// Fusion channel family. Representations in the same family compete for a
    /// single vote per unit, so adding a second OCR pass or a translation
    /// cannot multiply an item's weight in the fused ranking.
    public var channelFamily: ChannelFamily {
        switch self {
        case .originalText, .ocrText, .translation: .text
        case .transcript: .transcript
        case .image, .video: .visual
        case .audio: .audio
        case .caption: .caption
        case .summary: .summary
        case .soundEvent: .sound
        case .metadata: .metadata
        }
    }
}

/// Grouping used to normalise fusion weights. See ``RepresentationKind/channelFamily``.
public enum ChannelFamily: String, Sendable, Hashable, Codable, CaseIterable {
    case text
    case transcript
    case visual
    case audio
    case caption
    case summary
    case sound
    case metadata
    case lexical
}

/// Payload of a representation: what actually gets embedded or indexed.
public enum RepresentationContent: Sendable, Hashable, Codable {
    case text(String)
    /// Bytes of an encoded still image, with its UTI or MIME type.
    case imageData(Data, contentType: String)
    /// A reference to media on disk or at a URL, loaded lazily at embed time.
    ///
    /// The store never copies media. A 4 GB video indexed by reference costs
    /// the index its locator and nothing else; the cost of reading it is paid
    /// once, during embedding, and again only if the caller asks to reindex.
    case mediaReference(URL)

    public var text: String? {
        if case .text(let value) = self { return value }
        return nil
    }

    public var isInline: Bool {
        if case .mediaReference = self { return false }
        return true
    }
}

/// How a representation came to exist.
///
/// Carried onto every result so a caller can tell a matched sentence the author
/// wrote from a matched sentence a model produced, and can invalidate exactly
/// the representations a changed processor affects.
public struct Provenance: Sendable, Hashable, Codable {
    /// Identifier of whatever produced this representation: `"caller"`,
    /// `"vision.recognizeDocuments"`, `"parakeet-tdt-0.6b-v3"`.
    public var producer: String
    /// Version of that producer, when it has one.
    public var version: String?
    /// Fingerprint of the producer's configuration. Changing it invalidates
    /// derived representations on the next upsert without touching the
    /// original content.
    public var configurationFingerprint: String?
    public var createdAt: Date
    /// Producer's own confidence, where it reports one. Not comparable across
    /// producers, and never used for ranking.
    public var confidence: Float?

    public init(
        producer: String,
        version: String? = nil,
        configurationFingerprint: String? = nil,
        createdAt: Date = Date(),
        confidence: Float? = nil
    ) {
        self.producer = producer
        self.version = version
        self.configurationFingerprint = configurationFingerprint
        self.createdAt = createdAt
        self.confidence = confidence
    }

    public static let caller = Provenance(producer: "caller")
}

/// One view of an item's content.
///
/// An item has as many of these as the caller cares to give it. A video frame
/// might carry the frame itself, the OCR of the text burned into it, and a
/// caption — three representations, independently retrievable, all pointing at
/// the same moment.
public struct ContentRepresentation: Sendable, Hashable, Codable, Identifiable {
    public var id: RepresentationID
    public var kind: RepresentationKind
    public var content: RepresentationContent
    /// BCP-47 language tag of the content, when known. Used to keep
    /// multilingual retrieval in the original language rather than collapsing
    /// everything into one pivot language.
    public var language: String?
    public var provenance: Provenance
    /// Where this representation sits in its source. Falls back to the item's
    /// locator when unspecified.
    public var locator: SourceLocator?
    /// The representation this one was derived from, if any — a translation
    /// points at its source text, a caption at the frame it describes.
    public var derivedFrom: RepresentationID?
    /// Per-representation weight applied to its fusion contribution. Defaults
    /// to 1; values below it demote a noisy channel without removing it.
    ///
    /// Zero or less means *context only*: the representation is stored and can
    /// be shown as parent context or read back by the caller, but it is neither
    /// embedded nor lexically indexed. That is how a parent can hold a copy of
    /// its children's text for reranking context without the same words being
    /// retrievable twice — once as the passage and once as the whole document.
    public var weight: Float

    public init(
        id: RepresentationID,
        kind: RepresentationKind,
        content: RepresentationContent,
        language: String? = nil,
        provenance: Provenance = .caller,
        locator: SourceLocator? = nil,
        derivedFrom: RepresentationID? = nil,
        weight: Float = 1
    ) {
        self.id = id
        self.kind = kind
        self.content = content
        self.language = language
        self.provenance = provenance
        self.locator = locator
        self.derivedFrom = derivedFrom
        self.weight = weight
    }

    /// Convenience for the common case: a block of original text.
    public static func text(
        _ value: String,
        id: RepresentationID = "text",
        kind: RepresentationKind = .originalText,
        language: String? = nil,
        locator: SourceLocator? = nil,
        provenance: Provenance = .caller
    ) -> ContentRepresentation {
        ContentRepresentation(
            id: id, kind: kind, content: .text(value), language: language,
            provenance: provenance, locator: locator)
    }
}

/// A searchable unit.
///
/// The engine indexes items, ranks items, and returns items. Everything below
/// an item — chunks, OCR passes, transcripts — is a representation of it;
/// everything above it is its parent chain.
public struct SearchItem: Sendable, Hashable, Codable, Identifiable {
    public var id: ItemID
    /// Enclosing unit: the document a page belongs to, the video a clip belongs
    /// to. Parents supply bounded context for reranking and are what
    /// `searchInside` scopes to.
    public var parentID: ItemID?
    public var metadata: Metadata
    public var locator: SourceLocator
    public var representations: [ContentRepresentation]
    /// Ordering hint within the parent, ascending. Used only to make ties
    /// deterministic and to render sibling context in document order.
    public var ordinal: Int?

    public init(
        id: ItemID,
        parentID: ItemID? = nil,
        metadata: Metadata = [:],
        locator: SourceLocator = .unspecified,
        representations: [ContentRepresentation] = [],
        ordinal: Int? = nil
    ) {
        self.id = id
        self.parentID = parentID
        self.metadata = metadata
        self.locator = locator
        self.representations = representations
        self.ordinal = ordinal
    }

    public subscript(representation id: RepresentationID) -> ContentRepresentation? {
        representations.first { $0.id == id }
    }

    /// Effective locator for a representation: its own if it has one, the
    /// item's otherwise.
    public func locator(for representation: ContentRepresentation) -> SourceLocator {
        representation.locator ?? locator
    }

    /// Content fingerprint used to skip unchanged work on upsert.
    ///
    /// Covers everything an embedding or a posting list depends on and nothing
    /// else, so re-upserting an item whose metadata changed re-writes the row
    /// without re-running OCR or re-embedding anything.
    public var contentFingerprint: String {
        var hasher = StableHash()
        hasher.append("item.v1")
        hasher.append(id.rawValue)
        hasher.append(parentID?.rawValue ?? "")
        for representation in representations.sorted(by: { $0.id.rawValue < $1.id.rawValue }) {
            hasher.append(representation.id.rawValue)
            hasher.append(representation.kind.rawValue)
            hasher.append(representation.language ?? "")
            hasher.append(representation.weight.bitPattern)
            switch representation.content {
            case .text(let value):
                hasher.append("t")
                hasher.append(value)
            case .imageData(let data, let contentType):
                hasher.append("i")
                hasher.append(contentType)
                hasher.append(data)
            case .mediaReference(let url):
                hasher.append("m")
                hasher.append(url.absoluteString)
            }
            hasher.append(representation.provenance.producer)
            hasher.append(representation.provenance.version ?? "")
            hasher.append(representation.provenance.configurationFingerprint ?? "")
        }
        return hasher.digest
    }
}

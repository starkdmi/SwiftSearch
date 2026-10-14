import Foundation

/// Caller-owned identity for a searchable unit.
///
/// The engine never mints these. An upsert with an ID the store already holds
/// replaces that item, which is the whole mechanism behind incremental
/// indexing: the caller decides what "the same thing" means, because only the
/// caller knows whether a re-parsed page is the same page.
public struct ItemID: Sendable, Hashable, Codable, RawRepresentable,
    ExpressibleByStringLiteral, CustomStringConvertible
{
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public var description: String { rawValue }

    public init(from decoder: any Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// Identity of one representation within its item.
///
/// Unique per item, not globally: `("page-3", "ocr")` and `("page-4", "ocr")`
/// are different representations with the same local ID. Keeping it local means
/// a caller can name representations by what they are rather than having to
/// mint globally unique strings for them.
public struct RepresentationID: Sendable, Hashable, Codable, RawRepresentable,
    ExpressibleByStringLiteral, CustomStringConvertible
{
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public var description: String { rawValue }

    public init(from decoder: any Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// Globally unique address of one representation.
public struct RepresentationKey: Sendable, Hashable, Codable, CustomStringConvertible {
    public let item: ItemID
    public let representation: RepresentationID

    public init(item: ItemID, representation: RepresentationID) {
        self.item = item
        self.representation = representation
    }

    public var description: String { "\(item.rawValue)#\(representation.rawValue)" }
}

/// Name of an embedding space: one model, one preprocessing configuration, one
/// set of vectors that may legitimately be compared with each other.
public struct EmbeddingSpaceID: Sendable, Hashable, Codable, RawRepresentable,
    ExpressibleByStringLiteral, CustomStringConvertible
{
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public var description: String { rawValue }

    public init(from decoder: any Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

import Foundation

/// A typed, filterable metadata value.
///
/// Deliberately small. Metadata exists to be filtered on, and every case here
/// has an unambiguous SQL representation and an unambiguous equality; a nested
/// `Any` would have neither.
public enum MetadataValue: Sendable, Hashable, Codable {
    case string(String)
    case integer(Int64)
    case double(Double)
    case boolean(Bool)
    case date(Date)
    case stringList([String])

    private enum CodingKeys: String, CodingKey { case type, value }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "string": self = .string(try container.decode(String.self, forKey: .value))
        case "integer": self = .integer(try container.decode(Int64.self, forKey: .value))
        case "double": self = .double(try container.decode(Double.self, forKey: .value))
        case "boolean": self = .boolean(try container.decode(Bool.self, forKey: .value))
        case "date": self = .date(try container.decode(Date.self, forKey: .value))
        case "stringList": self = .stringList(try container.decode([String].self, forKey: .value))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container, debugDescription: "unknown metadata type \(type)")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .string(let value):
            try container.encode("string", forKey: .type)
            try container.encode(value, forKey: .value)
        case .integer(let value):
            try container.encode("integer", forKey: .type)
            try container.encode(value, forKey: .value)
        case .double(let value):
            try container.encode("double", forKey: .type)
            try container.encode(value, forKey: .value)
        case .boolean(let value):
            try container.encode("boolean", forKey: .type)
            try container.encode(value, forKey: .value)
        case .date(let value):
            try container.encode("date", forKey: .type)
            try container.encode(value, forKey: .value)
        case .stringList(let value):
            try container.encode("stringList", forKey: .type)
            try container.encode(value, forKey: .value)
        }
    }

    /// Ordering used by range filters. `nil` for values that have no total
    /// order against each other — comparing a string to a date is a caller
    /// mistake, and silently coercing would hide it.
    func comparable(to other: MetadataValue) -> (Double, Double)? {
        switch (self, other) {
        case (.integer(let a), .integer(let b)): return (Double(a), Double(b))
        case (.integer(let a), .double(let b)): return (Double(a), b)
        case (.double(let a), .integer(let b)): return (a, Double(b))
        case (.double(let a), .double(let b)): return (a, b)
        case (.date(let a), .date(let b)):
            return (a.timeIntervalSinceReferenceDate, b.timeIntervalSinceReferenceDate)
        default: return nil
        }
    }

    /// Membership test used by `.anyOf`, and by `.equals` against a list value.
    func contains(_ other: MetadataValue) -> Bool {
        guard case .stringList(let values) = self, case .string(let needle) = other else {
            return self == other
        }
        return values.contains(needle)
    }
}

public typealias Metadata = [String: MetadataValue]

/// A predicate over item metadata and structure.
///
/// Applied identically to every retrieval channel. That matters more than it
/// sounds: filtering lexical results but not dense ones (or filtering after
/// fusion) silently changes which channel gets to contribute candidates, and
/// the ranking shifts for reasons that have nothing to do with relevance.
public indirect enum SearchFilter: Sendable, Hashable, Codable {
    case all
    case none
    case equals(key: String, value: MetadataValue)
    /// Items whose value differs, *including* items without the key: the
    /// complement of ``equals(key:value:)``.
    case notEquals(key: String, value: MetadataValue)
    case anyOf(key: String, values: [MetadataValue])
    case exists(key: String)
    case lessThan(key: String, value: MetadataValue)
    case greaterThan(key: String, value: MetadataValue)
    case between(key: String, lower: MetadataValue, upper: MetadataValue)
    /// The item itself, or any descendant of it.
    case withinParent(ItemID)
    case hasItemID(Set<ItemID>)
    case hasKind(Set<RepresentationKind>)
    case and([SearchFilter])
    case or([SearchFilter])
    /// Everything the inner filter does not match, including items that lack
    /// the keys it tests.
    case not(SearchFilter)

    /// Evaluate against one candidate.
    ///
    /// `ancestors` is the full chain from the item's direct parent to the root,
    /// so `withinParent` matches at any depth without the evaluator needing to
    /// walk the store.
    public func matches(
        itemID: ItemID,
        kind: RepresentationKind,
        metadata: Metadata,
        ancestors: [ItemID]
    ) -> Bool {
        switch self {
        case .all: return true
        case .none: return false
        case .equals(let key, let value):
            guard let actual = metadata[key] else { return false }
            return actual.contains(value)
        case .notEquals(let key, let value):
            guard let actual = metadata[key] else { return true }
            return !actual.contains(value)
        case .anyOf(let key, let values):
            guard let actual = metadata[key] else { return false }
            return values.contains { actual.contains($0) }
        case .exists(let key):
            return metadata[key] != nil
        case .lessThan(let key, let value):
            guard let actual = metadata[key], let (lhs, rhs) = actual.comparable(to: value) else {
                return false
            }
            return lhs < rhs
        case .greaterThan(let key, let value):
            guard let actual = metadata[key], let (lhs, rhs) = actual.comparable(to: value) else {
                return false
            }
            return lhs > rhs
        case .between(let key, let lower, let upper):
            guard let actual = metadata[key],
                let (value, low) = actual.comparable(to: lower),
                let (_, high) = actual.comparable(to: upper)
            else { return false }
            return value >= low && value <= high
        case .withinParent(let parent):
            return itemID == parent || ancestors.contains(parent)
        case .hasItemID(let ids):
            return ids.contains(itemID)
        case .hasKind(let kinds):
            return kinds.contains(kind)
        case .and(let filters):
            return filters.allSatisfy {
                $0.matches(itemID: itemID, kind: kind, metadata: metadata, ancestors: ancestors)
            }
        case .or(let filters):
            return filters.contains {
                $0.matches(itemID: itemID, kind: kind, metadata: metadata, ancestors: ancestors)
            }
        case .not(let filter):
            return !filter.matches(
                itemID: itemID, kind: kind, metadata: metadata, ancestors: ancestors)
        }
    }

    /// True when the filter can be answered without looking at any item.
    var isUnconstrained: Bool {
        switch self {
        case .all: return true
        case .and(let filters): return filters.allSatisfy(\.isUnconstrained)
        default: return false
        }
    }
}

extension MetadataValue: CustomStringConvertible {
    /// Human-readable form, for diagnostics and for display.
    ///
    /// Dates are ISO-8601 so a printed value matches what the filter compiler
    /// puts in SQL; every other case is its payload, unadorned.
    public var description: String {
        switch self {
        case .string(let value): return value
        case .integer(let value): return String(value)
        case .double(let value): return String(value)
        case .boolean(let value): return value ? "true" : "false"
        case .date(let value): return MetadataDateFormat.string(from: value)
        case .stringList(let values): return values.joined(separator: ", ")
        }
    }

    /// The payload when this is a string, and nothing otherwise.
    ///
    /// Separate from ``description`` on purpose: this is for code that means
    /// "a string metadata value", which an integer is not.
    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }
}

import Foundation

/// Every failure the engine reports.
///
/// Degraded search must be requested rather than substituted, and this enum
/// is how that rule is enforced: a missing model, a stale media file or a
/// failed processor surfaces here instead of quietly dropping a retrieval
/// channel.
public enum SemanticSearchError: Error, Sendable {
    // MARK: Configuration

    case configuration(String)
    case invalidQuery(String)

    // MARK: Model compatibility

    /// The query provider's space does not match the space the vectors were
    /// indexed in. `detail` names the field that differs.
    case incompatibleEmbeddingSpace(
        space: EmbeddingSpaceID, expected: String, actual: String, detail: String)
    /// Dimension reduction requested from a provider that does not support it.
    case dimensionReductionUnsupported(model: String, requested: Int, native: Int)
    case unknownEmbeddingSpace(EmbeddingSpaceID)
    case providerNotPrepared(String)
    case providerReturnedNoOutput(String)
    case modelUnavailable(String, underlying: (any Error)?)

    // MARK: Content

    case unsupportedContent(itemID: ItemID, detail: String)
    case unsupportedModality(itemID: ItemID, kind: RepresentationKind, provider: String)
    case processorFailed(itemID: ItemID, processor: String, underlying: any Error)
    /// Referenced media is gone or unreadable. The index still holds the item;
    /// the caller decides whether to re-ingest or remove it.
    case mediaUnavailable(url: URL, itemID: ItemID?)
    case itemNotFound(ItemID)

    // MARK: Storage

    case storage(String, code: Int32?)
    case indexCorrupted(String)
    case schemaVersionUnsupported(found: Int, supported: Int)

    // MARK: Lifecycle

    case cancelled
}

extension SemanticSearchError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .configuration(let detail):
            return "configuration error: \(detail)"
        case .invalidQuery(let detail):
            return "invalid query: \(detail)"
        case .incompatibleEmbeddingSpace(let space, let expected, let actual, let detail):
            return """
                embedding space '\(space)' is incompatible with the query provider \
                (\(detail): indexed with \(expected), queried with \(actual))
                """
        case .dimensionReductionUnsupported(let model, let requested, let native):
            return """
                '\(model)' does not support dimension reduction; \
                requested \(requested) of \(native)
                """
        case .unknownEmbeddingSpace(let space):
            return "no embedding space named '\(space)' is configured"
        case .providerNotPrepared(let model):
            return "provider '\(model)' has not been prepared; call prepare() first"
        case .providerReturnedNoOutput(let model):
            return "provider '\(model)' returned no output for a non-empty batch"
        case .modelUnavailable(let model, let underlying):
            let suffix = underlying.map { ": \($0)" } ?? ""
            return "model '\(model)' is unavailable\(suffix)"
        case .unsupportedContent(let itemID, let detail):
            return "cannot process content for '\(itemID)': \(detail)"
        case .unsupportedModality(let itemID, let kind, let provider):
            return "'\(provider)' does not accept \(kind.rawValue) content (item '\(itemID)')"
        case .processorFailed(let itemID, let processor, let underlying):
            return "processor '\(processor)' failed on '\(itemID)': \(underlying)"
        case .mediaUnavailable(let url, let itemID):
            let suffix = itemID.map { " referenced by '\($0)'" } ?? ""
            return "media at \(url.path) is unavailable\(suffix)"
        case .itemNotFound(let itemID):
            return "no item with id '\(itemID)'"
        case .storage(let detail, let code):
            let suffix = code.map { " (sqlite \($0))" } ?? ""
            return "storage error: \(detail)\(suffix)"
        case .indexCorrupted(let detail):
            return "index is corrupted: \(detail)"
        case .schemaVersionUnsupported(let found, let supported):
            return "index schema version \(found) is newer than this build supports (\(supported))"
        case .cancelled:
            return "cancelled"
        }
    }
}

extension SemanticSearchError: LocalizedError {
    public var errorDescription: String? { description }
}

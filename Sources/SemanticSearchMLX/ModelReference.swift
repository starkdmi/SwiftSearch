import Foundation

/// Where a model's weights come from.
///
/// Deliberately does not reach into any project's directory layout. A caller
/// that keeps weights somewhere unusual passes ``directory(_:)``; everything
/// else is a Hub id that the MLX runtime resolves against its own cache, which
/// is what makes a prepared installation work offline.
public enum ModelReference: Sendable, Hashable {
    /// A Hugging Face repository id, e.g. `"mlx-community/Qwen3-Embedding-0.6B-8bit"`,
    /// optionally pinned to a commit: `"<org>/<name>@<revision>"`.
    ///
    /// Pin it. A repository id names whatever `main` holds today, and an index
    /// knows its vectors' model by this reference: weights uploaded again
    /// under the same id would load, and their vectors would mix with the old
    /// ones with nothing to tell them apart. A revision names one set of
    /// files, and a different revision opens a different space.
    case hub(String)
    /// A directory holding `config.json` and the weights.
    case directory(URL)

    /// Build from a string, treating anything path-shaped that exists on disk
    /// as a directory and everything else as a Hub id.
    public init(_ raw: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let expanded = (trimmed as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            let url = URL(fileURLWithPath: expanded, isDirectory: true)
            // A Hub cache repo directory has the weights one level down, under
            // `snapshots/<revision>`. Pointing at the repo root is the natural
            // thing to try and would otherwise fail with "no config.json".
            if Self.containsConfig(url) {
                self = .directory(url)
            } else if let snapshot = Self.newestSnapshot(in: url) {
                self = .directory(snapshot)
            } else {
                self = .directory(url)
            }
            return
        }
        self = .hub(trimmed)
    }

    /// Stable identity for the embedding space descriptor, with the revision
    /// when there is one.
    ///
    /// A directory reference resolves to its path, which is not portable
    /// between machines — deliberately. Two indexes built from different local
    /// checkouts are not known to be compatible, and claiming otherwise is
    /// exactly the mistake the descriptor exists to prevent. Pass a Hub id, or
    /// override ``EmbeddingSpaceDescriptor/modelIdentity`` yourself, when the
    /// weights are known to be the same.
    public var identity: String {
        switch self {
        case .hub(let id): return id
        case .directory(let url): return url.standardizedFileURL.path
        }
    }

    public var isDirectory: Bool {
        if case .directory = self { return true }
        return false
    }

    public var directoryURL: URL? {
        if case .directory(let url) = self { return url }
        return nil
    }

    /// The repository, without the revision.
    public var hubID: String? {
        guard case .hub(let id) = self else { return nil }
        return id.split(separator: "@", maxSplits: 1).first.map(String.init) ?? id
    }

    /// The pinned revision, when the Hub id names one.
    public var revision: String? {
        guard case .hub(let id) = self else { return nil }
        let parts = id.split(separator: "@", maxSplits: 1)
        return parts.count == 2 && !parts[1].isEmpty ? String(parts[1]) : nil
    }

    private static func containsConfig(_ url: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: url.appendingPathComponent("config.json").path)
    }

    private static func newestSnapshot(in repositoryDirectory: URL) -> URL? {
        let snapshots = repositoryDirectory.appendingPathComponent(
            "snapshots", isDirectory: true)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(atPath: snapshots.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            let entries = try? FileManager.default.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles])
        else { return nil }

        return
            entries
            .filter(containsConfig)
            .max { lhs, rhs in
                let lhsDate =
                    (try? lhs.resourceValues(forKeys: [.contentModificationDateKey])
                        .contentModificationDate) ?? .distantPast
                let rhsDate =
                    (try? rhs.resourceValues(forKeys: [.contentModificationDateKey])
                        .contentModificationDate) ?? .distantPast
                return lhsDate < rhsDate
            }
    }
}

extension ModelReference: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self.init(value)
    }
}

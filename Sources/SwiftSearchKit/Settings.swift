import Foundation

/// Which models index and search the library.
public enum Profile: String, Codable, Sendable, CaseIterable {
    /// No models: words only — text, PDFs, OCR of pictures, sounds, file
    /// details. Nothing to download.
    case lexical
    /// Qwen3 0.6B embedder and reranker; pictures and video frames captioned
    /// by SmolVLM2-500M, speech by Parakeet. About 2 GB of models.
    case fast
    /// Qwen3-VL 2B embedder and reranker over pixels and text in one space;
    /// speech by Parakeet. About 5 GB; a picture can be the query.
    case multimodal
}

/// What the library is and how it answers, stored as `settings.json` in its
/// home directory.
public struct Settings: Codable, Sendable, Equatable {
    /// Folders to index, as absolute paths.
    public var folders: [String]
    public var profile: Profile
    /// Name places from photo and video coordinates with MapKit — online, so
    /// off unless asked for.
    public var namesPlaces: Bool
    /// Keep coordinates in the index at all.
    public var recordsLocation: Bool
    /// Characters of matched text an answer quotes; 0 for none — paths,
    /// kinds and times only.
    public var snippetCharacters: Int
    /// Where models are downloaded; `nil` for the loaders' own defaults.
    public var modelDirectory: String?
    /// Take a video's keyframes from OmniShotCut's shots, never from a
    /// transition. Off, candidates are one a second, and a shot shorter than
    /// that can go unseen.
    public var shots: Bool
    /// A local OmniShotCut Core ML conversion to use instead of downloading
    /// the published one (103 MB, into the model directory).
    public var shotModel: String?

    public init(
        folders: [String] = [],
        profile: Profile = .fast,
        namesPlaces: Bool = false,
        recordsLocation: Bool = true,
        snippetCharacters: Int = 300,
        modelDirectory: String? = nil,
        shots: Bool = true,
        shotModel: String? = nil
    ) {
        self.folders = folders
        self.profile = profile
        self.namesPlaces = namesPlaces
        self.recordsLocation = recordsLocation
        self.snippetCharacters = snippetCharacters
        self.modelDirectory = modelDirectory
        self.shots = shots
        self.shotModel = shotModel
    }

    public init(from decoder: any Decoder) throws {
        // Every key optional, so a settings file written by an older version
        // still opens.
        let defaults = Settings()
        let container = try decoder.container(keyedBy: CodingKeys.self)
        folders = try container.decodeIfPresent([String].self, forKey: .folders) ?? defaults.folders
        profile = try container.decodeIfPresent(Profile.self, forKey: .profile) ?? defaults.profile
        namesPlaces =
            try container.decodeIfPresent(Bool.self, forKey: .namesPlaces) ?? defaults.namesPlaces
        recordsLocation =
            try container.decodeIfPresent(Bool.self, forKey: .recordsLocation)
            ?? defaults.recordsLocation
        snippetCharacters =
            try container.decodeIfPresent(Int.self, forKey: .snippetCharacters)
            ?? defaults.snippetCharacters
        modelDirectory = try container.decodeIfPresent(String.self, forKey: .modelDirectory)
        shots = try container.decodeIfPresent(Bool.self, forKey: .shots) ?? defaults.shots
        shotModel = try container.decodeIfPresent(String.self, forKey: .shotModel)
    }
}

/// The directory a library lives in: settings, one index per profile, the
/// lock.
public struct Home: Sendable {
    public let url: URL

    /// `SWIFTSEARCH_HOME`, or `~/Library/Application Support/SwiftSearch`.
    public static func `default`() -> Home {
        if let path = ProcessInfo.processInfo.environment["SWIFTSEARCH_HOME"], !path.isEmpty {
            return Home(URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
        }
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return Home(support.appendingPathComponent("SwiftSearch", isDirectory: true))
    }

    public init(_ url: URL) {
        self.url = url.standardizedFileURL
    }

    public var settingsURL: URL { url.appendingPathComponent("settings.json") }
    public var lockURL: URL { url.appendingPathComponent("lock") }

    /// One index per profile: vectors from different models are never
    /// comparable, and switching back should not mean indexing again.
    public func indexURL(for profile: Profile) -> URL {
        url.appendingPathComponent("index-\(profile.rawValue).sqlite")
    }

    public func loadSettings() throws -> Settings {
        guard FileManager.default.fileExists(atPath: settingsURL.path) else { return Settings() }
        return try JSONDecoder().decode(Settings.self, from: Data(contentsOf: settingsURL))
    }

    /// Creates the home, or closes an existing one, to everyone but its
    /// owner. The index holds what was extracted from the files — text, OCR,
    /// transcripts — and SQLite creates its files at the umask's 0644, so a
    /// home outside the user's own Library (`--home /tmp/…`) would be
    /// readable by every local user.
    public func createPrivately() throws {
        let manager = FileManager.default
        try manager.createDirectory(
            at: url, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
    }

    public func save(_ settings: Settings) throws {
        try createPrivately()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(settings).write(to: settingsURL, options: .atomic)
    }
}

import ArgumentParser
import Foundation
import SwiftSearchKit

@main
struct SwiftSearch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "swiftsearch",
        abstract: "Search your files by what is in them — for you, or for an AI agent over MCP.",
        subcommands: [
            Add.self, Remove.self, Sync.self, Search.self, List.self, Info.self, Status.self,
            Config.self, Serve.self,
        ])
}

struct HomeOption: ParsableArguments {
    @Option(help: "Library directory (default: $SWIFTSEARCH_HOME or ~/Library/Application Support/SwiftSearch).")
    var home: String?

    var value: Home {
        home.map { Home(URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)) }
            ?? .default()
    }

    func open() async throws -> Library { try await Library.open(home: value) }
}

struct FilterOptions: ParsableArguments {
    @Option(help: "image, video, audio, pdf, text, or a UTI.")
    var type: String?
    @Option(help: "Only files under this folder.")
    var folder: String?
    @Option(help: "Only files dated after this ISO date (2024-06-01).")
    var after: String?
    @Option(help: "Only files dated before this ISO date.")
    var before: String?
    @Option(help: "Which date: created (taken or written) or modified.")
    var dateField: String = "created"

    func query() throws -> FileQuery {
        FileQuery(
            type: type, folder: folder, after: try Self.date(after), before: try Self.date(before),
            dateField: FileQuery.DateField(rawValue: dateField) ?? .created)
    }

    static func date(_ string: String?) throws -> Date? {
        guard let string else { return nil }
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: string) { return date }
        formatter.formatOptions = [.withFullDate]
        guard let date = formatter.date(from: string) else {
            throw ValidationError("\(string) is not an ISO date like 2024-06-01")
        }
        return date
    }
}

/// Progress on standard error, one line rewritten in place.
func syncWithProgress(_ library: Library) async throws -> SyncSummary {
    let reporter = Task {
        var last = ""
        while !Task.isCancelled {
            try? await Task.sleep(for: .milliseconds(500))
            guard let sync = await library.syncProgress(), let current = sync.current else {
                continue
            }
            let line = "[\(sync.completed + 1)/\(sync.total)] \((current as NSString).lastPathComponent)"
            if line != last {
                FileHandle.standardError.write(Data("\r\u{1B}[K\(line)".utf8))
                last = line
            }
        }
    }
    defer {
        reporter.cancel()
        FileHandle.standardError.write(Data("\r\u{1B}[K".utf8))
    }
    return try await library.sync()
}

func printSummary(_ sync: SyncSummary) {
    var parts = [
        "\(sync.added) added", "\(sync.updated) updated", "\(sync.unchanged) unchanged",
        "\(sync.removed) removed",
    ]
    if sync.moved > 0 { parts.insert("\(sync.moved) moved", at: 3) }
    if sync.placesNamed > 0 { parts.append("\(sync.placesNamed) places named") }
    if sync.unsupported > 0 { parts.append("\(sync.unsupported) not indexable") }
    print(parts.joined(separator: ", "))
    for failure in sync.failed { print("failed: \(failure)") }
    for path in sync.unreadable { print("unreadable, kept: \(path)") }
}

struct Add: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Add a folder to the library and index it.")
    @OptionGroup var home: HomeOption
    @Argument(help: "Folder to index.") var folder: String
    @Flag(help: "Only add it; index on the next sync.") var noSync = false

    func run() async throws {
        let library = try await home.open()
        let added = try await library.addFolder(folder)
        print("added \(added)")
        if !noSync { printSummary(try await syncWithProgress(library)) }
        try await library.close()
    }
}

struct Remove: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Remove a folder from the library, and what was indexed from it.")
    @OptionGroup var home: HomeOption
    @Argument(help: "Folder to stop indexing.") var folder: String

    func run() async throws {
        let library = try await home.open()
        let removed = try await library.removeFolder(folder)
        print("removed \(removed) files from the index")
        try await library.close()
    }
}

struct Sync: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Index new and changed files, drop deleted ones.")
    @OptionGroup var home: HomeOption

    func run() async throws {
        let library = try await home.open()
        printSummary(try await syncWithProgress(library))
        try await library.close()
    }
}

struct Search: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Search the library.")
    @OptionGroup var home: HomeOption
    @OptionGroup var filters: FilterOptions
    @Argument(help: "What to look for.") var query: [String] = []
    @Option(help: "A picture to search with.") var image: String?
    @Option(help: "Search inside this one file.") var within: String?
    @Option(help: "Files to show, or places with --within.") var limit = 10
    @Flag(help: "Print JSON, as the MCP tool returns it.") var json = false

    func run() async throws {
        let text = query.joined(separator: " ")
        guard !text.isEmpty || image != nil else { throw ValidationError("give a query or --image") }
        let library = try await home.open()
        let answer = try await library.search(
            text, image: image, imageFromAnywhere: true, in: try filters.query(), within: within,
            limit: limit)
        try await library.close()
        if json { return print(try Output.json(answer, pretty: true)) }
        for note in answer.notes { print("note: \(note)") }
        if answer.files.isEmpty { print("nothing found") }
        for (rank, file) in answer.files.enumerated() {
            print(
                "\(rank + 1). \(file.path)  (\(file.type ?? "file"), \(String(format: "%.2f", file.score)))")
            for match in file.matches {
                var place = match.kind
                if let page = match.page { place += " p.\(page)" }
                if let start = match.start {
                    place += " \(clock(start))" + (match.end.map { "–\(clock($0))" } ?? "")
                }
                print("     \(place)" + (match.text.map { "  \"\($0)\"" } ?? ""))
            }
        }
    }
}

func clock(_ seconds: Double) -> String {
    let total = Int(seconds)
    return total >= 3600
        ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
        : String(format: "%d:%02d", total / 60, total % 60)
}

struct List: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "List indexed files.")
    @OptionGroup var home: HomeOption
    @OptionGroup var filters: FilterOptions
    @Option(help: "path, newest or oldest.") var order = "path"
    @Option var limit = 50
    @Option var offset = 0
    @Flag(help: "Print JSON.") var json = false

    func run() async throws {
        let library = try await home.open()
        let listing = try await library.list(
            try filters.query(), order: ListOrder(rawValue: order) ?? .path, limit: limit,
            offset: offset)
        try await library.close()
        if json { return print(try Output.json(listing, pretty: true)) }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        for file in listing.files {
            let date = (file.created ?? file.modified).map { formatter.string(from: $0) } ?? ""
            print([date, file.type ?? "", file.path].joined(separator: "\t"))
        }
        print("\(listing.offset + listing.files.count) of \(listing.total)")
    }
}

struct Info: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "What the index knows about a file.")
    @OptionGroup var home: HomeOption
    @Argument var path: String

    func run() async throws {
        let library = try await home.open()
        let info = try await library.info(path)
        try await library.close()
        print(try Output.json(info, pretty: true))
    }
}

struct Status: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Folders, counts, index size.")
    @OptionGroup var home: HomeOption

    func run() async throws {
        let library = try await home.open()
        let status = try await library.status()
        try await library.close()
        print(try Output.json(status, pretty: true))
    }
}

struct Config: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show or change the library's settings.")
    @OptionGroup var home: HomeOption
    @Option(help: "lexical (no models), fast (0.6B + captions) or multimodal (2B VL).")
    var profile: Profile?
    @Option(help: "Name places from coordinates with MapKit (online): on or off.")
    var placeNames: Switch?
    @Option(help: "Keep photo and video coordinates in the index: on or off.")
    var location: Switch?
    @Option(help: "Characters of matched text in answers; 0 for none.")
    var snippets: Int?
    @Option(help: "Where to download models, as <directory>/models/<org>/<name>.")
    var modelDirectory: String?
    @Option(
        help: """
            Take video keyframes from OmniShotCut's shots (on, the default; 103 MB \
            downloaded), or sample a candidate a second (off).
            """)
    var shots: Switch?
    @Option(
        help: """
            A local OmniShotCut Core ML conversion to use instead of the published \
            one; none to download it.
            """)
    var shotModel: String?

    enum Switch: String, ExpressibleByArgument { case on, off }

    func run() async throws {
        let home = home.value
        let changes = profile != nil || placeNames != nil || location != nil || snippets != nil
            || modelDirectory != nil || shots != nil || shotModel != nil
        // Not while another process has the library open: it would keep
        // running on the old settings. Taken before the settings are read,
        // or two `config` runs would each change what they read and the
        // second would write the first's change away.
        let lock = changes ? try IndexLock(home.lockURL) : nil
        defer { lock?.release() }
        var settings = try home.loadSettings()
        if changes {
            if let profile { settings.profile = profile }
            if let placeNames { settings.namesPlaces = placeNames == .on }
            if let location { settings.recordsLocation = location == .on }
            if let snippets { settings.snippetCharacters = max(0, snippets) }
            if let modelDirectory {
                settings.modelDirectory = (modelDirectory as NSString).expandingTildeInPath
            }
            if let shots { settings.shots = shots == .on }
            if let shotModel {
                settings.shotModel =
                    shotModel == "none" ? nil : (shotModel as NSString).expandingTildeInPath
            }
            try home.save(settings)
        }
        print(try Output.json(settings, pretty: true))
        if changes { print("run `swiftsearch sync` to index under the new settings") }
    }
}

extension Profile: ExpressibleByArgument {}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mcp", abstract: "Serve the library to an agent over MCP (stdio).")
    @OptionGroup var home: HomeOption
    @Flag(help: "Do not index new and changed files on start.") var noSync = false
    @Option(help: "Unload models after this many idle seconds; 0 keeps them.")
    var releaseAfter = 300

    func run() async throws {
        let library = try await home.open()
        try await MCPServer.run(
            library: library,
            options: .init(
                syncsOnStart: !noSync,
                releasesModelsAfter: releaseAfter > 0 ? .seconds(releaseAfter) : nil))
    }
}

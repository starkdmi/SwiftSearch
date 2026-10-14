import Foundation
import MCP

/// The MCP tools: the library as an agent sees it.
///
/// Read-only but for `sync_library`, and bounded by the library's folders:
/// `read_file` hands over an indexed file inside them and nothing else. The
/// agent finds files by what is in them and asks for the few it needs,
/// instead of opening a folder of private photos to look.
public enum Tools {

    public static let all: [Tool] = [
        Tool(
            name: "search",
            description: """
                Search the user's indexed files — documents, PDFs, photos, recordings, \
                videos — by meaning, in any language: what is written, said, shown or \
                heard in them, and their titles, tags and places. Returns files, best \
                first, each with up to three places it matched: a page, or start/end \
                seconds into a recording or video, with the matched text, and `time`, \
                the exact second of a frame or sentence — pass it to read_file to see \
                that frame (`end` is already the next shot). \
                Give `path` to search inside that one file instead — the moment in a long \
                video, the page in a PDF — and get up to `limit` places in it. \
                Open what you need with read_file.
                """,
            inputSchema: schema(
                [
                    "query": ["type": "string", "description": "What to look for, in words."],
                    "path": [
                        "type": "string",
                        "description": "Search inside this one indexed file only.",
                    ],
                    "image": [
                        "type": "string",
                        "description": .string(
                            "Path of an indexed picture in the library to search with — "
                                + "\"more like this one\" — alone or with a query."),
                    ],
                    "limit": [
                        "type": "integer",
                        "description":
                            "Files to return, default 10 — or, with `path`, places in the file.",
                    ],
                ].merging(filterProperties) { a, _ in a }),
            annotations: .init(readOnlyHint: true, openWorldHint: false)),
        Tool(
            name: "list_files",
            description: """
                List indexed files without a query, by type, folder and date, ordered by \
                path or by date — "photos from June", "every PDF in Invoices", "the \
                newest videos". Returns paths with type, size, dates, title, duration \
                and place, and the total for paging.
                """,
            inputSchema: schema(
                [
                    "order": [
                        "type": "string", "enum": .array(ListOrder.allCases.map { .string($0.rawValue) }),
                        "description": "path (default), newest or oldest.",
                    ],
                    "limit": ["type": "integer", "description": "Default 50, at most 500."],
                    "offset": ["type": "integer", "description": "Skip this many, for the next page."],
                ].merging(filterProperties) { a, _ in a }),
            annotations: .init(readOnlyHint: true, openWorldHint: false)),
        Tool(
            name: "file_info",
            description: """
                What the index knows about one file, without opening it: metadata (type, \
                size, dates, title, artist, album, author, keywords, place, camera, \
                duration, pixel size) and what was extracted from it, by kind (text, \
                speech, frames, frame descriptions, sounds).
                """,
            inputSchema: schema(
                ["path": ["type": "string", "description": "Absolute path of an indexed file."]],
                required: ["path"]),
            annotations: .init(readOnlyHint: true, openWorldHint: false)),
        Tool(
            name: "read_file",
            description: """
                Open one indexed file. A picture comes back as the image. A video with \
                `time` comes back as the actual frame at that second: to know what is on \
                screen, look at the frame rather than relying on a frame description. A \
                video without `time`, or a recording, comes back as its transcript with \
                timestamps and the sounds heard, optionally between `start` and `end` \
                seconds. A PDF comes back as text, one `page` or all, scanned pages as \
                their OCR. A text file comes back as its contents, from `offset`.
                """,
            inputSchema: schema(
                [
                    "path": ["type": "string", "description": "Absolute path of an indexed file."],
                    "time": [
                        "type": "number",
                        "description": "Video: return the frame at this second, as an image.",
                    ],
                    "start": ["type": "number", "description": "Transcript from this second."],
                    "end": ["type": "number", "description": "Transcript up to this second."],
                    "page": ["type": "integer", "description": "PDF page, 1-based."],
                    "offset": [
                        "type": "integer",
                        "description": "Text, PDF or transcript: characters to skip, to continue.",
                    ],
                    "max_characters": [
                        "type": "integer",
                        "description": "Text: at most this many, default 20000, at most 100000.",
                    ],
                ],
                required: ["path"]),
            annotations: .init(readOnlyHint: true, openWorldHint: false)),
        Tool(
            name: "library_status",
            description: """
                The library's folders, how many files of each type are indexed, and \
                whether indexing is running — search results may be incomplete while it is.
                """,
            inputSchema: schema([:]),
            annotations: .init(readOnlyHint: true, openWorldHint: false)),
        Tool(
            name: "sync_library",
            description: """
                Index new and changed files in the library's folders and drop deleted \
                ones, in the background; library_status shows progress. The server does \
                this when it starts; call it after files change.
                """,
            inputSchema: schema([:]),
            annotations: .init(
                readOnlyHint: false, destructiveHint: false, idempotentHint: true,
                openWorldHint: false)),
    ]

    /// The most text one `read_file` answer carries; more is paged with
    /// `offset`, `page` or `start`.
    static let maximumCharacters = 100_000

    static let filterProperties: [String: Value] = [
        "type": [
            "type": "string", "description": "image, video, audio, pdf, text, or a UTI.",
        ],
        "folder": ["type": "string", "description": "Only files under this absolute folder."],
        "after": ["type": "string", "description": "ISO date, e.g. 2024-06-01."],
        "before": ["type": "string", "description": "ISO date."],
        "date_field": [
            "type": "string", "enum": ["created", "modified"],
            "description": .string(
                "created (default): when the photo or video was taken or the document "
                    + "written; modified: the file's modification date."),
        ],
    ]

    static func schema(_ properties: [String: Value], required: [String] = []) -> Value {
        var object: [String: Value] = ["type": "object", "properties": .object(properties)]
        if !required.isEmpty { object["required"] = .array(required.map { .string($0) }) }
        return .object(object)
    }

    // MARK: - Calls

    public static func call(_ name: String, _ arguments: [String: Value], library: Library) async
        -> CallTool.Result
    {
        do {
            switch name {
            case "search":
                let query = arguments["query"]?.stringValue ?? ""
                let image = arguments["image"]?.stringValue
                guard !query.isEmpty || image != nil else {
                    return failure("search needs a query or an image")
                }
                let answer = try await library.search(
                    query, image: image, in: try fileQuery(arguments),
                    within: arguments["path"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 },
                    limit: max(1, min(50, arguments["limit"]?.intValue ?? 10)))
                return try json(answer)

            case "list_files":
                let order = arguments["order"]?.stringValue.flatMap(ListOrder.init) ?? .path
                let listing = try await library.list(
                    try fileQuery(arguments), order: order,
                    limit: max(1, min(500, arguments["limit"]?.intValue ?? 50)),
                    offset: max(0, arguments["offset"]?.intValue ?? 0))
                return try json(listing)

            case "file_info":
                return try json(try await library.info(try path(arguments)))

            case "read_file":
                let content = try await library.read(
                    try path(arguments), page: arguments["page"]?.intValue,
                    time: number(arguments["time"]), start: number(arguments["start"]),
                    end: number(arguments["end"]), offset: arguments["offset"]?.intValue ?? 0,
                    maximumCharacters: max(
                        1, min(maximumCharacters, arguments["max_characters"]?.intValue ?? 20_000)))
                switch content {
                case .text(let text, let truncated):
                    return .init(
                        content: [
                            .text(
                                text: truncated
                                    ? text + "\n[truncated — ask again with offset, page or start]"
                                    : text, annotations: nil, _meta: nil)
                        ])
                case .image(let data, let mimeType):
                    return .init(
                        content: [
                            .image(
                                data: data.base64EncodedString(), mimeType: mimeType,
                                annotations: nil, _meta: nil)
                        ])
                }

            case "library_status":
                return try json(try await library.status())

            case "sync_library":
                await library.startSync()
                return try json(try await library.status())

            default:
                return failure("unknown tool \(name)")
            }
        } catch {
            return failure(String(describing: error))
        }
    }

    static func fileQuery(_ arguments: [String: Value]) throws -> FileQuery {
        FileQuery(
            type: arguments["type"]?.stringValue, folder: arguments["folder"]?.stringValue,
            after: try date(arguments["after"]), before: try date(arguments["before"]),
            dateField: arguments["date_field"]?.stringValue.flatMap(FileQuery.DateField.init)
                ?? .created)
    }

    static func path(_ arguments: [String: Value]) throws -> String {
        guard let path = arguments["path"]?.stringValue, !path.isEmpty else {
            throw ArgumentError("path is required")
        }
        return path
    }

    static func number(_ value: Value?) -> Double? {
        value?.doubleValue ?? value?.intValue.map(Double.init)
    }

    /// `2024-06-01`, or a full ISO 8601 date and time.
    static func date(_ value: Value?) throws -> Date? {
        guard let string = value?.stringValue, !string.isEmpty else { return nil }
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: string) { return date }
        formatter.formatOptions = [.withFullDate]
        if let date = formatter.date(from: string) { return date }
        throw ArgumentError("\(string) is not an ISO date like 2024-06-01")
    }

    struct ArgumentError: Error, CustomStringConvertible {
        var description: String
        init(_ description: String) { self.description = description }
    }

    static func json(_ value: some Encodable) throws -> CallTool.Result {
        .init(content: [.text(text: try Output.json(value), annotations: nil, _meta: nil)])
    }

    static func failure(_ message: String) -> CallTool.Result {
        .init(content: [.text(text: message, annotations: nil, _meta: nil)], isError: true)
    }
}

/// JSON as both front ends print it: compact, ISO dates, stable key order.
public enum Output {
    public static func json(_ value: some Encodable, pretty: Bool = false) throws -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting =
            pretty
            ? [.sortedKeys, .withoutEscapingSlashes, .prettyPrinted]
            : [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

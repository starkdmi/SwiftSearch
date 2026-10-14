import AVFoundation
import Foundation
import ImageIO
import PDFKit
import SemanticSearch
import UniformTypeIdentifiers

/// Names the place at a coordinate — "Eiffel Tower, Paris, France".
///
/// Replaceable because the one Apple ships asks a server: an app that must not
/// send coordinates anywhere passes none, and keeps the coordinates alone.
public protocol PlaceNamer: Sendable {
    var identifier: String { get }
    var configurationFingerprint: String { get }
    /// `nil` where there is no place to name, such as the open sea.
    func placeName(latitude: Double, longitude: Double) async throws -> String?
}

/// Adds what a file says about itself to what another processor reads in it.
///
/// Wraps any processor, as ``CaptioningProcessor`` does. Two things are added:
///
/// - **Filterable facts** on every item the file produces — the file, its
///   chunks, frames, utterances — because a filter tests an item's own
///   metadata: "photos taken in 2024" has to match the photo, "videos over
///   ten minutes" each moment in them. Keys, all optional:
///   `fileName`, `contentType` (the type and every type it conforms to, so
///   `.equals("contentType", "public.image")` finds every picture),
///   `fileSize`, `fileCreated`, `fileModified`, `contentCreated` (when the
///   photo or video was taken, the PDF written — Spotlight's content creation
///   date), `latitude`, `longitude`, `place`, `camera`, `pixelWidth`,
///   `pixelHeight`, `duration`,
///   `title`, `artist`, `album`, `author`, `description`, `keywords`. A key
///   the caller or the inner processor already set is left alone.
/// - **Words** on the file itself, a `.metadata` representation: the file
///   name, the folders a sync found it in below the synced folder, title,
///   artist, album, author, description, keywords and place, one per line —
///   so "Abbey Road", "invoice march" or "Fethiye 2022" finds a file whose
///   content never says it. Dates and the camera stay out of it: a photo taken with
///   an iPhone is not a photo *of* an iPhone, and "2024" would match every
///   picture of the year. Those are for filters.
///
/// Read from the file, not guessed: EXIF, TIFF, IPTC, GPS and PNG text through
/// ImageIO; AVFoundation's common metadata (ID3, iTunes, QuickTime) for audio
/// and video; the PDF's document attributes; a Markdown file's YAML front
/// matter. Place names come from `placeNamer`, when there is one.
public struct FileMetadataProcessor: ContentProcessor {

    public struct Configuration: Sendable, Hashable {
        /// Index the title, artist, album, author, description, keywords and
        /// place as words.
        public var indexesWords: Bool
        /// Index the file's name as words too. Off where names are meant to
        /// say nothing — a test that hides what a file is behind `IMG_0412`.
        public var indexesFileName: Bool
        /// Keep the coordinates a photo or video was taken at, and name the
        /// place. Location in an index is sensitive; off, neither is stored.
        public var recordsLocation: Bool
        /// Index the folders a sync found the file in, below the synced
        /// folder (`folderPath`), as words: "Trips / Fethiye 2022" is how
        /// people file what a photo's own tags never say.
        public var indexesFolderNames: Bool

        public init(
            indexesWords: Bool = true, indexesFileName: Bool = true, recordsLocation: Bool = true,
            indexesFolderNames: Bool = true
        ) {
            self.indexesWords = indexesWords
            self.indexesFileName = indexesFileName
            self.recordsLocation = recordsLocation
            self.indexesFolderNames = indexesFolderNames
        }
    }

    public let processor: any ContentProcessor
    public let placeNamer: (any PlaceNamer)?
    public let configuration: Configuration

    public init(
        wrapping processor: any ContentProcessor,
        placeNamer: (any PlaceNamer)? = nil,
        configuration: Configuration = Configuration()
    ) {
        self.processor = processor
        self.placeNamer = placeNamer
        self.configuration = configuration
    }

    public var identifier: String { "file-metadata" }

    public var version: String { "1" }

    public var capabilities: ProcessorCapabilities {
        var capabilities = processor.capabilities
        capabilities.producedKinds.insert(.metadata)
        return capabilities
    }

    public var configurationFingerprint: String {
        StableHash.of { hasher in
            // v2: Android videos' location and creation date.
            hasher.append("file-metadata.v2")
            hasher.append(processor.identifier)
            hasher.append(processor.version)
            hasher.append(processor.configurationFingerprint)
            hasher.append(configuration.indexesWords ? "words" : "-")
            hasher.append(configuration.indexesFileName ? "name" : "-")
            hasher.append(configuration.recordsLocation ? "location" : "-")
            // Only when off: the words follow `folderPath`, which a sync
            // compares for every unchanged file and redoes alone, so turning
            // this on — the default — needs no second processing of anything.
            if !configuration.indexesFolderNames { hasher.append("no-folders") }
            if configuration.recordsLocation, let placeNamer {
                hasher.append(placeNamer.identifier)
                hasher.append(placeNamer.configurationFingerprint)
            }
        }
    }

    public func canProcess(_ content: RawContent) -> Bool {
        processor.canProcess(content)
    }

    public func process(
        _ content: RawContent,
        onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        var items = try await processor.process(content, onProgress: onProgress)
        guard case .fileURL(let url) = content.payload, url.isFileURL else { return items }
        try Task.checkCancellation()
        onProgress?(ProcessingProgress(itemID: content.id, stage: "metadata"))

        var facts = await FileFacts.read(url)
        if configuration.recordsLocation, let (latitude, longitude) = facts.coordinate {
            facts.metadata["latitude"] = .double(latitude)
            facts.metadata["longitude"] = .double(longitude)
            if let placeNamer {
                do {
                    if let place = try await placeNamer.placeName(
                        latitude: latitude, longitude: longitude)
                    {
                        facts.metadata["place"] = .string(place)
                        facts.words.append(place)
                    }
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    // Offline, or refused: the file is still worth indexing
                    // with its coordinates, which filters can use. Marked, so
                    // `namePendingPlaces` asks again: an unchanged file is
                    // never processed again, and one refusal would otherwise
                    // leave it without a place for good.
                    facts.metadata[Self.placePendingKey] = .boolean(true)
                }
            }
        }

        for index in items.indices {
            for (key, value) in facts.metadata where items[index].metadata[key] == nil {
                items[index].metadata[key] = value
            }
        }

        let folders = folderWords(content.metadata).map { [$0] } ?? []
        let representation = words(name: facts.nameWords, folders: folders, facts.words)

        if let parent = items.firstIndex(where: { $0.id == content.id }) {
            if let representation { items[parent].representations.append(representation) }
        } else {
            // A processor that returned only children, or nothing, still
            // leaves the file there to be found by its details and filtered.
            var metadata = content.metadata
            for (key, value) in facts.metadata where metadata[key] == nil {
                metadata[key] = value
            }
            items.insert(
                SearchItem(
                    id: content.id, parentID: content.parentID, metadata: metadata,
                    locator: content.locator, representations: representation.map { [$0] } ?? []),
                at: 0)
        }
        return items
    }

    /// Set on a file's items while its place could not be looked up.
    public static let placePendingKey = "placePending"

    /// The file's words, one per line, each once: its name, its folders,
    /// then `lines`.
    func words(
        name: String?, folders: [String] = [], _ lines: [String]
    ) -> ContentRepresentation? {
        var all: [String] = []
        if configuration.indexesWords {
            if configuration.indexesFileName, let name { all.append(name) }
            if configuration.indexesFolderNames { all += folders }
            all += lines
        }
        var seen: Set<String> = []
        all = all.filter { seen.insert($0.lowercased()).inserted }
        guard !all.isEmpty else { return nil }
        return ContentRepresentation(
            id: "metadata", kind: .metadata, content: .text(all.joined(separator: "\n")),
            provenance: Provenance(
                producer: identifier, version: version,
                configurationFingerprint: configurationFingerprint))
    }

    /// The folders a sync found the file in, below the folder it synced —
    /// `Trips/Fethiye 2022` as "Trips / Fethiye 2022", separators as spaces
    /// as in a file name. `nil` at the top, or for content not from a sync.
    func folderWords(_ metadata: Metadata) -> String? {
        guard case .string(let path)? = metadata[SemanticSearchEngine.folderPathKey] else {
            return nil
        }
        let names = path.split(separator: "/").map { name in
            name.split(whereSeparator: { "_-".contains($0) || $0.isWhitespace })
                .joined(separator: " ")
        }.filter { !$0.isEmpty }
        return names.isEmpty ? nil : names.joined(separator: " / ")
    }

    /// The keys that depend on where a file is rather than what it holds.
    static let pathKeys = ["fileName", "contentType", "fileCreated", "fileModified"]

    /// A moved file: the same bytes, so the same tags, coordinates and place
    /// — no second lookup. What depends on the path is read again: the name,
    /// the type its extension gives it, the dates the file system keeps, and
    /// the name and folders among the words. Also how an unchanged file
    /// whose `folderPath` changed gets its words redone.
    public func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem] {
        var items = try await processor.relocate(items, from: old, to: content)
        guard case .fileURL(let url) = content.payload else { return items }
        let facts = await FileFacts.read(url)
        for index in items.indices {
            for key in Self.pathKeys { items[index].metadata[key] = facts.metadata[key] }
        }
        guard let root = items.firstIndex(where: { $0.id == content.id }) else { return items }
        var lines = facts.words
        if case .string(let place)? = items[root].metadata["place"] { lines.append(place) }
        items[root].representations.removeAll { $0.id == "metadata" }
        let folders = folderWords(content.metadata).map { [$0] } ?? []
        if let representation = words(name: facts.nameWords, folders: folders, lines) {
            items[root].representations.append(representation)
        }
        return items
    }

    /// Ask again for the places a failed lookup left out.
    ///
    /// Only the place is filled in — on every item of the file, and among its
    /// words — so nothing is read, transcribed or embedded again but that one
    /// line of words. A lookup that fails again stays pending for the next
    /// call. Returns how many files were resolved.
    public func namePendingPlaces(in engine: SemanticSearchEngine) async throws -> Int {
        guard configuration.recordsLocation, let placeNamer else { return 0 }
        let pending = try await engine.listItems(
            matching: .equals(key: Self.placePendingKey, value: .boolean(true)), limit: 100_000)
        var resolved = 0
        for file in pending.items {
            try Task.checkCancellation()
            guard case .double(let latitude)? = file.metadata["latitude"],
                case .double(let longitude)? = file.metadata["longitude"]
            else { continue }
            let place: String?
            do {
                place = try await placeNamer.placeName(latitude: latitude, longitude: longitude)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                continue
            }

            var items = try await engine.listItems(
                matching: .withinParent(file.id), topLevelOnly: false, limit: 1_000_000,
                includesRepresentations: true
            ).items
            for index in items.indices {
                items[index].metadata[Self.placePendingKey] = nil
                if let place { items[index].metadata["place"] = .string(place) }
            }
            if let place, let root = items.firstIndex(where: { $0.id == file.id }) {
                var lines: [String] = []
                if let current = items[root].representations.first(where: { $0.id == "metadata" }),
                    case .text(let text) = current.content
                {
                    lines = text.components(separatedBy: "\n")
                }
                items[root].representations.removeAll { $0.id == "metadata" }
                // The name is already the first line when there is one; the
                // words are passed whole and the place added last, as
                // `process` orders them.
                if let representation = words(name: nil, lines + [place]) {
                    items[root].representations.append(representation)
                }
            }
            try await engine.upsert(items)
            resolved += 1
        }
        return resolved
    }
}

/// What one file says about itself.
struct FileFacts {
    var metadata: Metadata = [:]
    /// Title, artist, album, author, description, keywords — in that order.
    var words: [String] = []
    /// The file name without its extension, separators as spaces.
    var nameWords: String?
    var coordinate: (Double, Double)?

    static func read(_ url: URL) async -> FileFacts {
        var facts = FileFacts()
        let values = try? url.resourceValues(forKeys: [
            .nameKey, .contentTypeKey, .fileSizeKey, .creationDateKey,
            .contentModificationDateKey,
        ])
        let name = values?.name ?? url.lastPathComponent
        facts.metadata["fileName"] = .string(name)
        facts.nameWords = Self.nameWords(name)
        let type = values?.contentType ?? UTType(filenameExtension: url.pathExtension)
        if let type {
            let supertypes = type.supertypes.map(\.identifier).sorted()
            facts.metadata["contentType"] = .stringList([type.identifier] + supertypes)
        }
        if let size = values?.fileSize { facts.metadata["fileSize"] = .integer(Int64(size)) }
        if let date = values?.creationDate { facts.metadata["fileCreated"] = .date(date) }
        if let date = values?.contentModificationDate {
            facts.metadata["fileModified"] = .date(date)
        }

        var tags = Tags()
        if let type {
            if type.conforms(to: .image) {
                tags = Self.imageTags(url)
            } else if type.conforms(to: .audiovisualContent) {
                tags = await Self.assetTags(url)
            } else if type.conforms(to: .pdf) {
                tags = Self.pdfTags(url)
            } else if ["md", "markdown"].contains(url.pathExtension.lowercased()) {
                tags = Self.frontMatterTags(url)
            }
        }
        facts.apply(tags)
        return facts
    }

    /// `IMG_0412.HEIC` → `IMG 0412`; `invoice-march_2024.pdf` →
    /// `invoice march 2024`. The analyzer splits on spaces, not on `_`.
    static func nameWords(_ name: String) -> String? {
        let stem = (name as NSString).deletingPathExtension
        let words = stem.split(whereSeparator: { "_-.".contains($0) || $0.isWhitespace })
            .joined(separator: " ")
        return words.isEmpty ? nil : words
    }

    // MARK: - Tags

    struct Tags {
        var title: String?
        var artists: [String] = []
        var album: String?
        var authors: [String] = []
        var description: String?
        var keywords: [String] = []
        var contentCreated: Date?
        var coordinate: (Double, Double)?
        var camera: String?
        var pixelSize: (Int, Int)?
        var duration: Double?
    }

    private mutating func apply(_ tags: Tags) {
        func clean(_ value: String?) -> String? {
            let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters))
            return trimmed?.isEmpty == false ? trimmed : nil
        }
        func unique(_ values: [String]) -> [String] {
            var seen: Set<String> = []
            return values.compactMap(clean).filter { seen.insert($0).inserted }
        }
        let artists = unique(tags.artists)
        let authors = unique(tags.authors)
        let keywords = unique(tags.keywords)
        if let title = clean(tags.title) {
            metadata["title"] = .string(title)
            words.append(title)
        }
        if !artists.isEmpty {
            metadata["artist"] = .string(artists.joined(separator: ", "))
            words.append(artists.joined(separator: ", "))
        }
        if let album = clean(tags.album) {
            metadata["album"] = .string(album)
            words.append(album)
        }
        if !authors.isEmpty {
            metadata["author"] = .string(authors.joined(separator: ", "))
            words.append(authors.joined(separator: ", "))
        }
        if let description = clean(tags.description) {
            metadata["description"] = .string(description)
            words.append(description)
        }
        if !keywords.isEmpty {
            metadata["keywords"] = .stringList(keywords)
            words.append(keywords.joined(separator: ", "))
        }
        if let date = tags.contentCreated { metadata["contentCreated"] = .date(date) }
        if let camera = clean(tags.camera) { metadata["camera"] = .string(camera) }
        if let (width, height) = tags.pixelSize, width > 0, height > 0 {
            metadata["pixelWidth"] = .integer(Int64(width))
            metadata["pixelHeight"] = .integer(Int64(height))
        }
        if let duration = tags.duration, duration.isFinite, duration > 0 {
            metadata["duration"] = .double(duration)
        }
        if let (latitude, longitude) = tags.coordinate,
            (-90...90).contains(latitude), (-180...180).contains(longitude),
            !(latitude == 0 && longitude == 0)
        {
            // 0, 0 is what a camera without a fix writes, not a place.
            coordinate = (latitude, longitude)
        }
    }

    /// "Canon" + "Canon EOS R5" is one camera, not "Canon Canon EOS R5".
    static func camera(make: String?, model: String?) -> String? {
        let make = make?.trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters))
        let model = model?.trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters))
        switch (make?.isEmpty == false ? make : nil, model?.isEmpty == false ? model : nil) {
        case (let make?, let model?):
            return model.lowercased().hasPrefix(make.lowercased()) ? model : "\(make) \(model)"
        case (let make?, nil): return make
        case (nil, let model?): return model
        case (nil, nil): return nil
        }
    }

    // MARK: Images

    static func imageTags(_ url: URL) -> Tags {
        var tags = Tags()
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any]
        else { return tags }

        if let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        {
            // Orientations 5–8 are rotated a quarter turn: what is shown is
            // the stored image on its side.
            let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
            tags.pixelSize = (5...8).contains(orientation) ? (height, width) : (width, height)
        }

        let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
        if let original = exif[kCGImagePropertyExifDateTimeOriginal] as? String {
            tags.contentCreated = exifDate(
                original, offset: exif[kCGImagePropertyExifOffsetTimeOriginal] as? String)
        }

        let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]
        tags.camera = camera(
            make: tiff[kCGImagePropertyTIFFMake] as? String,
            model: tiff[kCGImagePropertyTIFFModel] as? String)
        tags.description = tiff[kCGImagePropertyTIFFImageDescription] as? String
        if let artist = tiff[kCGImagePropertyTIFFArtist] as? String { tags.authors.append(artist) }

        let iptc = properties[kCGImagePropertyIPTCDictionary] as? [CFString: Any] ?? [:]
        tags.title = iptc[kCGImagePropertyIPTCObjectName] as? String
        if let caption = iptc[kCGImagePropertyIPTCCaptionAbstract] as? String {
            tags.description = caption
        }
        tags.keywords += strings(iptc[kCGImagePropertyIPTCKeywords])
        tags.authors += strings(iptc[kCGImagePropertyIPTCByline])

        let png = properties[kCGImagePropertyPNGDictionary] as? [CFString: Any] ?? [:]
        if tags.title == nil { tags.title = png[kCGImagePropertyPNGTitle] as? String }
        if tags.description == nil {
            tags.description = png[kCGImagePropertyPNGDescription] as? String
        }
        if let author = png[kCGImagePropertyPNGAuthor] as? String { tags.authors.append(author) }

        let gps = properties[kCGImagePropertyGPSDictionary] as? [CFString: Any] ?? [:]
        if let latitude = gps[kCGImagePropertyGPSLatitude] as? Double,
            let longitude = gps[kCGImagePropertyGPSLongitude] as? Double
        {
            let south = (gps[kCGImagePropertyGPSLatitudeRef] as? String)?.uppercased() == "S"
            let west = (gps[kCGImagePropertyGPSLongitudeRef] as? String)?.uppercased() == "W"
            tags.coordinate = (south ? -latitude : latitude, west ? -longitude : longitude)
        }
        return tags
    }

    /// EXIF writes local time, `2024:06:14 18:23:11`, and since 2.31 its
    /// offset beside it. Without one the zone is unknown; the Mac's own is
    /// the guess Photos makes too.
    static func exifDate(_ value: String, offset: String?) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy:MM:dd HH:mm:ss"
        formatter.timeZone = offset.flatMap(timeZone(offset:)) ?? .current
        return formatter.date(from: value.trimmingCharacters(in: .whitespaces))
    }

    /// `+03:00` or `-0530`.
    static func timeZone(offset: String) -> TimeZone? {
        let trimmed = offset.trimmingCharacters(in: .whitespaces)
        guard let sign = trimmed.first, sign == "+" || sign == "-" else { return nil }
        let digits = trimmed.dropFirst().filter(\.isNumber)
        guard digits.count == 4, let hours = Int(digits.prefix(2)), let minutes = Int(digits.suffix(2))
        else { return nil }
        let seconds = (hours * 3600 + minutes * 60) * (sign == "-" ? -1 : 1)
        return TimeZone(secondsFromGMT: seconds)
    }

    private static func strings(_ value: Any?) -> [String] {
        if let list = value as? [String] { return list }
        if let single = value as? String { return [single] }
        return []
    }

    // MARK: Audio and video

    static func assetTags(_ url: URL) async -> Tags {
        var tags = Tags()
        let asset = AVURLAsset(url: url)
        if let duration = try? await asset.load(.duration), duration.isNumeric {
            tags.duration = duration.seconds
        }
        if let track = try? await asset.loadTracks(withMediaType: .video).first,
            let (size, transform) = try? await track.load(.naturalSize, .preferredTransform)
        {
            let shown = size.applying(transform)
            tags.pixelSize = (Int(abs(shown.width).rounded()), Int(abs(shown.height).rounded()))
        }

        var make: String?
        var model: String?
        for item in (try? await asset.load(.commonMetadata)) ?? [] {
            guard let key = item.commonKey else { continue }
            if key == .commonKeyCreationDate {
                if let date = try? await item.load(.dateValue) {
                    tags.contentCreated = date
                } else if let string = try? await item.load(.stringValue) {
                    tags.contentCreated = isoDate(string)
                }
                continue
            }
            guard let value = try? await item.load(.stringValue) else { continue }
            switch key {
            case .commonKeyTitle: tags.title = tags.title ?? value
            case .commonKeyArtist: tags.artists.append(value)
            case .commonKeyAlbumName: tags.album = tags.album ?? value
            case .commonKeyAuthor, .commonKeyCreator: tags.authors.append(value)
            case .commonKeyDescription: tags.description = tags.description ?? value
            case .commonKeySubject: tags.description = tags.description ?? value
            case .commonKeyLocation: tags.coordinate = tags.coordinate ?? iso6709(value)
            case .commonKeyMake: make = value
            case .commonKeyModel: model = value
            default: break
            }
        }
        tags.camera = camera(make: make, model: model)

        // An Android camera writes neither into QuickTime's keys: the
        // capture time is the movie header's, and the place is the MP4
        // `©xyz` atom — raw bytes AVFoundation does not decode.
        if tags.contentCreated == nil, let item = try? await asset.load(.creationDate),
            let date = try? await item.load(.dateValue),
            // An unset header is zero: 1904 in QuickTime's epoch, or 1970.
            date > Date(timeIntervalSince1970: 0)
        {
            tags.contentCreated = date
        }
        if tags.coordinate == nil,
            let items = try? await asset.loadMetadata(for: AVMetadataFormat(rawValue: "org.mp4ra"))
        {
            for item in items where item.identifier?.rawValue == "uiso/%A9xyz" {
                if let coordinate = try? await xyz(item) { tags.coordinate = coordinate }
            }
        }
        return tags
    }

    /// The `©xyz` atom's value: a 16-bit length, a 16-bit language code,
    /// then ISO 6709 text.
    static func xyz(_ item: AVMetadataItem) async throws -> (Double, Double)? {
        if let string = try await item.load(.stringValue) { return iso6709(string) }
        guard let data = try await item.load(.dataValue), data.count > 4 else { return nil }
        return iso6709(String(decoding: data.dropFirst(4), as: UTF8.self))
    }

    /// QuickTime's creation date: `2024-06-14T18:23:11+0100`.
    static func isoDate(_ value: String) -> Date? {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        for options: ISO8601DateFormatter.Options in [
            [.withInternetDateTime], [.withFullDate, .withFullTime, .withTimeZone],
            [.withFullDate],
        ] {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = options
            if let date = formatter.date(from: trimmed) { return date }
        }
        return nil
    }

    /// ISO 6709 in decimal degrees, as iPhones and QuickTime write it:
    /// `+38.7223-009.1393+045.000/`. Android writes the longitude's degrees
    /// in two digits where the standard has three (`+36.6672+29.1057/`); both
    /// are read. The degrees-and-minutes forms the standard also allows are
    /// not.
    static func iso6709(_ value: String) -> (Double, Double)? {
        guard
            let match = value.firstMatch(
                of: /^([+-]\d{2}(?:\.\d+)?)([+-]\d{2,3}(?:\.\d+)?)/),
            let latitude = Double(match.1), let longitude = Double(match.2)
        else { return nil }
        return (latitude, longitude)
    }

    // MARK: PDF

    static func pdfTags(_ url: URL) -> Tags {
        var tags = Tags()
        guard let attributes = PDFDocument(url: url)?.documentAttributes else { return tags }
        tags.title = attributes[PDFDocumentAttribute.titleAttribute] as? String
        if let author = attributes[PDFDocumentAttribute.authorAttribute] as? String {
            tags.authors.append(author)
        }
        tags.description = attributes[PDFDocumentAttribute.subjectAttribute] as? String
        // PDFKit gives keywords as an array, or as the string the file holds.
        if let keywords = attributes[PDFDocumentAttribute.keywordsAttribute] as? [String] {
            tags.keywords = keywords.flatMap(splitList)
        } else if let keywords = attributes[PDFDocumentAttribute.keywordsAttribute] as? String {
            tags.keywords = splitList(keywords)
        }
        tags.contentCreated = attributes[PDFDocumentAttribute.creationDateAttribute] as? Date
        return tags
    }

    static func splitList(_ value: String) -> [String] {
        value.split(whereSeparator: { ",;".contains($0) }).map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }
    }

    // MARK: Markdown

    static func frontMatterTags(_ url: URL) -> Tags {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return Tags() }
        return frontMatterTags(text)
    }

    /// The flat YAML front matter static site generators read: `title:`,
    /// `author:`, `description:`, and `tags:` / `keywords:` as `[a, b]`,
    /// `a, b` or a `- a` list. Anything nested is skipped, not parsed.
    /// A quoted value is one value, commas and all — `"Doe, Jane"`,
    /// `["Paris, France", travel]` — as in YAML; so is a plain `author:`.
    /// Only a plain `tags:` / `keywords:` scalar is split on commas.
    static func frontMatterTags(_ text: String) -> Tags {
        var tags = Tags()
        var lines = text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
            .map(String.init)[...]
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else { return tags }
        lines = lines.dropFirst()

        func unquoted(_ value: String) -> String {
            let value = value.trimmingCharacters(in: .whitespaces)
            return isQuoted(value) ? flowItems(value).first ?? "" : value
        }
        var listKey: String?
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed == "---" || trimmed == "..." { break }
            if let key = listKey, trimmed.hasPrefix("- ") {
                let item = unquoted(String(trimmed.dropFirst(2)))
                if key == "author" { tags.authors.append(item) } else { tags.keywords.append(item) }
                continue
            }
            listKey = nil
            guard !line.hasPrefix(" "), !line.hasPrefix("\t"),
                let colon = line.firstIndex(of: ":")
            else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            switch key {
            case "title": tags.title = unquoted(value)
            case "description", "summary": tags.description = unquoted(value)
            case "author", "authors", "tags", "keywords", "categories":
                let normalized = key == "authors" ? "author" : key
                if value.isEmpty {
                    listKey = normalized
                    continue
                }
                let items: [String]
                if value.hasPrefix("["), value.hasSuffix("]") {
                    items = flowItems(String(value.dropFirst().dropLast()))
                } else if isQuoted(value) || normalized == "author" {
                    items = [unquoted(value)]
                } else {
                    items = splitList(value)
                }
                if normalized == "author" { tags.authors += items } else { tags.keywords += items }
            default: break
            }
        }
        return tags
    }

    private static func isQuoted(_ value: String) -> Bool {
        guard value.count >= 2, let first = value.first, first == "\"" || first == "'" else {
            return false
        }
        return value.last == first
    }

    /// The items of a YAML flow sequence's inside, `"Paris, France", travel`,
    /// split on the commas outside quotes. Quotes are removed; `\"` in double
    /// quotes and `''` in single quotes stand for the quote itself.
    static func flowItems(_ list: String) -> [String] {
        var items: [String] = []
        var current = ""
        var quote: Character?
        var wasQuoted = false
        var characters = list.makeIterator()
        func finish() {
            let item = wasQuoted ? current : current.trimmingCharacters(in: .whitespaces)
            if !item.isEmpty || wasQuoted { items.append(item) }
            current = ""
            wasQuoted = false
        }
        var pending: Character? = nil
        while let character = pending ?? characters.next() {
            pending = nil
            switch quote {
            case "\""?:
                if character == "\\", let next = characters.next() {
                    current.append(next)
                } else if character == "\"" {
                    quote = nil
                } else {
                    current.append(character)
                }
            case "'"?:
                if character == "'" {
                    let next = characters.next()
                    if next == "'" {
                        current.append("'")
                    } else {
                        quote = nil
                        pending = next
                    }
                } else {
                    current.append(character)
                }
            default:
                if (character == "\"" || character == "'")
                    && current.trimmingCharacters(in: .whitespaces).isEmpty
                {
                    current = ""
                    quote = character
                    wasQuoted = true
                } else if character == "," {
                    finish()
                } else if !(wasQuoted && character.isWhitespace) {
                    current.append(character)
                }
            }
        }
        finish()
        return items
    }
}

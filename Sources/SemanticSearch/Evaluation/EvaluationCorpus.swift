import Foundation

/// A corpus and its labelled queries, as one file.
///
/// Portable on purpose. A baseline taken before changing an algorithm is only
/// useful if the corpus, the queries and the model and configuration identity
/// travel together and can be re-run against any build.
public struct EvaluationCorpus: Sendable, Codable {

    /// One indexable document in the corpus.
    public struct Document: Sendable, Codable {
        public var id: String
        public var parentID: String?
        public var text: String
        public var language: String?
        public var kind: RepresentationKind
        public var metadata: [String: String]
        /// Relative path to an image or media file, resolved against the
        /// corpus file's own directory.
        public var mediaPath: String?

        public init(
            id: String,
            parentID: String? = nil,
            text: String = "",
            language: String? = nil,
            kind: RepresentationKind = .originalText,
            metadata: [String: String] = [:],
            mediaPath: String? = nil
        ) {
            self.id = id
            self.parentID = parentID
            self.text = text
            self.language = language
            self.kind = kind
            self.metadata = metadata
            self.mediaPath = mediaPath
        }
    }

    public var name: String
    public var documents: [Document]
    public var queries: [EvaluationQuery]
    /// Drop a hit whose id is the query's own before the ranking is scored.
    ///
    /// BEIR's `ignore_identical_ids`, which MTEB sets for ArguAna: 1,298 of
    /// its 1,406 test queries are also documents in its corpus, word for word.
    /// A model that finds that copy first scores ~0.66 nDCG@10 at best,
    /// where published numbers reach 0.71. Part of the corpus rather than of
    /// a run, so the set cannot be scored without it. Absent from a file
    /// means off.
    public var ignoresIdenticalIDs: Bool?

    public init(
        name: String,
        documents: [Document],
        queries: [EvaluationQuery],
        ignoresIdenticalIDs: Bool? = nil
    ) {
        self.name = name
        self.documents = documents
        self.queries = queries
        self.ignoresIdenticalIDs = ignoresIdenticalIDs
    }

    public static func load(from url: URL) throws -> EvaluationCorpus {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(EvaluationCorpus.self, from: data)
    }

    public func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    /// Materialise the corpus as items.
    ///
    /// `baseURL` resolves `mediaPath` entries, so a fixture can carry a handful
    /// of images beside its JSON and stay relocatable.
    public func items(baseURL: URL? = nil) -> [SearchItem] {
        documents.map { document in
            var metadata: Metadata = [:]
            for (key, value) in document.metadata { metadata[key] = .string(value) }

            var representations: [ContentRepresentation] = []
            if !document.text.isEmpty {
                representations.append(
                    ContentRepresentation(
                        id: "text", kind: document.kind, content: .text(document.text),
                        language: document.language))
            }
            if let mediaPath = document.mediaPath, let baseURL {
                let url = baseURL.appendingPathComponent(mediaPath)
                representations.append(
                    ContentRepresentation(
                        id: "media",
                        kind: Self.isVideo(mediaPath) ? .video : .image,
                        content: .mediaReference(url),
                        locator: SourceLocator(source: url.absoluteString)))
            }

            return SearchItem(
                id: ItemID(document.id),
                parentID: document.parentID.map { ItemID($0) },
                metadata: metadata,
                representations: representations)
        }
    }

    private static func isVideo(_ path: String) -> Bool {
        ["mp4", "mov", "m4v", "mkv", "avi"].contains(
            (path as NSString).pathExtension.lowercased())
    }

    /// Filter for a query's slice constraint, if it has one.
    public func filter(for query: EvaluationQuery) -> SearchFilter {
        guard let key = query.filterKey, let value = query.filterValue else { return .all }
        return .equals(key: key, value: .string(value))
    }
}

/// The configuration a baseline was measured under.
///
/// Recorded alongside the numbers because a baseline compared against a
/// different configuration is not a comparison. Two runs are comparable only
/// over identical corpora and candidate pools, and this is what makes a
/// mismatch visible rather than silent.
public struct EvaluationIdentity: Sendable, Codable, Equatable {

    /// One embedding space as the engine searched it.
    ///
    /// What changes every vector under one model name: a run at 128
    /// dimensions, one that read 2,048 tokens, one under another instruction
    /// and the MLX and Core ML conversions of one model all report the same
    /// ``EvaluationIdentity/embeddingModels``.
    public struct Space: Sendable, Codable, Equatable {
        public var id: String
        /// ``EmbeddingSpaceDescriptor/fingerprint`` of the vectors the space
        /// holds.
        public var fingerprint: String
        /// Width the space is searched at: the one it is stored at — which a
        /// reopened index keeps, whatever the configuration asks — or the
        /// narrower one the search options ask for.
        public var dimensions: Int
        /// The provider's ``EmbeddingCapabilities/maximumInputTokens``. No
        /// fingerprint records it, and it decides how much of a long
        /// document a vector reads.
        public var maximumInputTokens: Int
        /// Query instruction per channel family the space embeds queries
        /// for, keyed by the family's raw value, as the engine resolves it.
        /// Such a family without one is embedded under the provider's
        /// default; the instruction of a family the space does not route is
        /// never read, and is left out.
        public var queryInstructions: [String: String]

        public init(
            id: String,
            fingerprint: String,
            dimensions: Int,
            maximumInputTokens: Int,
            queryInstructions: [String: String] = [:]
        ) {
            self.id = id
            self.fingerprint = fingerprint
            self.dimensions = dimensions
            self.maximumInputTokens = maximumInputTokens
            self.queryInstructions = queryInstructions
        }
    }

    public var corpusName: String
    public var documentCount: Int
    public var queryCount: Int
    public var embeddingModels: [String]
    public var rerankerModel: String?
    public var analyzerVersion: String
    public var bm25: BM25Parameters
    public var fusionK: Float
    public var candidatesPerChannel: Int
    public var rerankPoolSize: Int
    public var limit: Int
    public var mode: String
    /// How the first space was searched: `exact` by brute force — an `hnsw`
    /// space below its graph threshold included — `hnsw` through its graph.
    /// Read off the engine rather than the configuration, because a reopened
    /// index keeps the kind it was created with.
    public var indexKind: String
    /// Every space, sorted by id. `nil` in identities recorded before it was.
    public var spaces: [Space]?
    /// Whether a hit with the query's own id was dropped before scoring
    /// (``EvaluationCorpus/ignoresIdenticalIDs``). `nil` in identities
    /// recorded before it was; recorded runs always say which.
    public var ignoresIdenticalIDs: Bool?

    public init(
        corpusName: String,
        documentCount: Int,
        queryCount: Int,
        embeddingModels: [String],
        rerankerModel: String?,
        analyzerVersion: String,
        bm25: BM25Parameters,
        fusionK: Float,
        candidatesPerChannel: Int,
        rerankPoolSize: Int,
        limit: Int,
        mode: String,
        indexKind: String,
        spaces: [Space]? = nil,
        ignoresIdenticalIDs: Bool? = nil
    ) {
        self.corpusName = corpusName
        self.documentCount = documentCount
        self.queryCount = queryCount
        self.embeddingModels = embeddingModels
        self.rerankerModel = rerankerModel
        self.analyzerVersion = analyzerVersion
        self.bm25 = bm25
        self.fusionK = fusionK
        self.candidatesPerChannel = candidatesPerChannel
        self.rerankPoolSize = rerankPoolSize
        self.limit = limit
        self.mode = mode
        self.indexKind = indexKind
        self.spaces = spaces
        self.ignoresIdenticalIDs = ignoresIdenticalIDs
    }

    /// Fields that must match for two runs to be comparable.
    ///
    /// The index kind is excluded on purpose: exact versus HNSW is precisely
    /// the comparison the ANN recall gate exists to make. A field one side
    /// recorded before it existed is `nil`, and compares as it did then.
    public func isComparable(with other: EvaluationIdentity) -> Bool {
        corpusName == other.corpusName
            && documentCount == other.documentCount
            && queryCount == other.queryCount
            && embeddingModels == other.embeddingModels
            && rerankerModel == other.rerankerModel
            && analyzerVersion == other.analyzerVersion
            && bm25 == other.bm25
            && fusionK == other.fusionK
            && candidatesPerChannel == other.candidatesPerChannel
            && rerankPoolSize == other.rerankPoolSize
            && limit == other.limit
            && mode == other.mode
            && Self.agree(spaces, other.spaces)
            && Self.agree(ignoresIdenticalIDs, other.ignoresIdenticalIDs)
    }

    /// Differences from `baseline` that make a gate against it meaningless
    /// rather than merely different, one `baseline → this` line each: per
    /// space the width, the token cap and the query instructions, and the
    /// identical-id rule.
    ///
    /// The rest of ``isComparable(with:)`` is a change a gate run is made to
    /// judge. These move every number under the same model and corpus names,
    /// are set by flags rather than by the code under test, and are easy to
    /// carry over by mistake — an instruction table that missed a set, a
    /// width left off — so a pass or a failure across them would be credited
    /// to the code. A field one side recorded before it existed is `nil` and
    /// is not compared.
    public func measurementDifferences(from baseline: EvaluationIdentity) -> [String] {
        var differences: [String] = []
        if let current = ignoresIdenticalIDs, let base = baseline.ignoresIdenticalIDs,
            current != base
        {
            func rule(_ ignores: Bool) -> String { ignores ? "dropped" : "scored" }
            differences.append("identical ids: \(rule(base)) → \(rule(current))")
        }
        guard let spaces, let baseSpaces = baseline.spaces else { return differences }

        // A space only one side has is a change of models, which
        // `isComparable` reports.
        let baseByID = Dictionary(
            baseSpaces.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        for space in spaces {
            guard let base = baseByID[space.id] else { continue }
            let name = "space '\(space.id)'"
            if space.dimensions != base.dimensions {
                differences.append(
                    "\(name) dimensions: \(base.dimensions) → \(space.dimensions)")
            }
            if space.maximumInputTokens != base.maximumInputTokens {
                differences.append(
                    "\(name) max input tokens: \(base.maximumInputTokens) → "
                        + "\(space.maximumInputTokens)")
            }
            differences += Self.instructionDifferences(
                name, from: base.queryInstructions, to: space.queryInstructions)
        }
        return differences
    }

    /// One line per distinct change, naming the families it applies to: a
    /// run that sets one instruction for every family would otherwise print
    /// the same change once for each of them.
    private static func instructionDifferences(
        _ name: String, from base: [String: String], to current: [String: String]
    ) -> [String] {
        struct Change: Hashable {
            var base: String?
            var current: String?
        }
        func describe(_ instruction: String?) -> String {
            instruction.map { "\"\($0)\"" } ?? "provider default"
        }

        var families: [Change: [String]] = [:]
        for family in Set(base.keys).union(current.keys).sorted() {
            let change = Change(base: base[family], current: current[family])
            if change.base != change.current { families[change, default: []].append(family) }
        }
        return families.sorted { $0.value[0] < $1.value[0] }.map { entry in
            "\(name) query instruction (\(entry.value.joined(separator: ", "))): "
                + "\(describe(entry.key.base)) → \(describe(entry.key.current))"
        }
    }

    /// Equal, or not recorded on one side.
    private static func agree<Value: Equatable>(_ lhs: Value?, _ rhs: Value?) -> Bool {
        guard let lhs, let rhs else { return true }
        return lhs == rhs
    }
}

/// A recorded baseline: what was measured, and under what.
public struct EvaluationBaseline: Sendable, Codable {
    public var identity: EvaluationIdentity
    public var report: EvaluationReport
    public var recordedAt: Date

    public init(
        identity: EvaluationIdentity, report: EvaluationReport, recordedAt: Date = Date()
    ) {
        self.identity = identity
        self.report = report
        self.recordedAt = recordedAt
    }

    public static func load(from url: URL) throws -> EvaluationBaseline {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(EvaluationBaseline.self, from: try Data(contentsOf: url))
    }

    public func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(self).write(to: url, options: .atomic)
    }
}

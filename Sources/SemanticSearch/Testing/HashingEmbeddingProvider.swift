import Foundation

/// A deterministic embedding provider with no model behind it.
///
/// Hashes character trigrams into a fixed-width vector. That gives it two
/// properties the tests need and nothing else: identical input always produces
/// an identical vector, and lexically similar strings land near each other.
///
/// It is **not** a semantic model and must never be used to judge retrieval
/// quality. What it is for is everything about the pipeline that is not the
/// model: fusion arithmetic, filtering, persistence, cancellation, dimension
/// handling, compatibility rejection and scale — all of which can be tested
/// exhaustively in milliseconds, and none of which needs a gigabyte of weights.
public actor HashingEmbeddingProvider: EmbeddingProvider {

    private let dimensions: Int
    private let identity: String
    private let matryoshka: [Int]
    private let acceptsVisual: Bool
    private var prepared = false

    public init(
        dimensions: Int = 64,
        identity: String = "hashing-trigram",
        matryoshkaDimensions: [Int] = [],
        acceptsVisual: Bool = false
    ) {
        self.dimensions = max(4, dimensions)
        self.identity = identity
        self.matryoshka = matryoshkaDimensions.filter { $0 > 0 && $0 <= dimensions }.sorted()
        self.acceptsVisual = acceptsVisual
    }

    public var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: identity,
            tokenizerVersion: "trigram.v1",
            preprocessingVersion: "lowercase-fold.v1",
            pooling: "sum",
            normalized: true,
            dimensions: dimensions)
    }

    public var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(
            acceptsText: true,
            acceptsImages: acceptsVisual,
            acceptsVideo: acceptsVisual,
            sharedTextVisualSpace: acceptsVisual,
            supportsDimensionReduction: !matryoshka.isEmpty,
            supportedDimensions: matryoshka,
            maximumBatchSize: 32,
            maximumInputTokens: 512)
    }

    public var isPrepared: Bool { prepared }

    public func prepare() async throws { prepared = true }

    public func release() async { prepared = false }

    public func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard prepared else {
            throw SemanticSearchError.providerNotPrepared(identity)
        }
        return try inputs.map { input in
            try Task.checkCancellation()

            var source = input.text
            // Media has no text to hash. Its path stands in, which is stable
            // and distinguishes one image from another — enough for a test that
            // cares whether the image channel produced a candidate at all.
            for url in input.videos { source += " " + url.lastPathComponent }
            for image in input.images {
                switch image {
                case .fileURL(let url): source += " " + url.lastPathComponent
                case .encoded(let data, _): source += " img\(data.count)"
                }
            }

            guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return EmbeddingOutput(vector: [])
            }

            var vector = Self.hashVector(source, dimensions: dimensions)
            if let requested = input.requestedDimensions, matryoshka.contains(requested),
                requested < vector.count
            {
                vector = VectorMath.normalized(Array(vector.prefix(requested)))
            }
            return EmbeddingOutput(
                vector: vector, consumedTokens: source.count / 4, wasTruncated: false)
        }
    }

    /// Character trigrams, hashed into buckets with a signed contribution.
    ///
    /// The sign comes from a second hash so that unrelated trigrams landing in
    /// one bucket cancel on average instead of piling up, which is what keeps
    /// unrelated strings from all looking similar at small dimensions.
    static func hashVector(_ text: String, dimensions: Int) -> [Float] {
        let normalized = " " + text.lowercased().folding(
            options: [.diacriticInsensitive], locale: nil) + " "
        let characters = Array(normalized.unicodeScalars)
        guard characters.count >= 3 else {
            return VectorMath.normalized(
                bucketed(characters.map { UInt64($0.value) }, dimensions: dimensions))
        }

        var keys: [UInt64] = []
        keys.reserveCapacity(characters.count - 2)
        for index in 0..<(characters.count - 2) {
            var hasher = StableHash()
            hasher.append(UInt64(characters[index].value))
            hasher.append(UInt64(characters[index + 1].value))
            hasher.append(UInt64(characters[index + 2].value))
            keys.append(UInt64(hasher.digest.hashValueForBucketing))
        }
        return VectorMath.normalized(bucketed(keys, dimensions: dimensions))
    }

    private static func bucketed(_ keys: [UInt64], dimensions: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimensions)
        for key in keys {
            let bucket = Int(key % UInt64(dimensions))
            let sign: Float = (key / UInt64(dimensions)) % 2 == 0 ? 1 : -1
            vector[bucket] += sign
        }
        return vector
    }
}

extension String {
    /// Parse the hex digest back to an integer for bucketing.
    fileprivate var hashValueForBucketing: UInt64 {
        UInt64(self, radix: 16) ?? 0
    }
}

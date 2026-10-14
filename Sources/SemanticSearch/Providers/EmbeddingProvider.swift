import Foundation

/// Which modalities a provider accepts, and how its vectors behave.
public struct EmbeddingCapabilities: Sendable, Hashable, Codable {
    public var acceptsText: Bool
    public var acceptsImages: Bool
    public var acceptsVideo: Bool
    /// Native audio input, as opposed to audio made searchable through a
    /// transcript. No bundled provider sets it.
    public var acceptsAudio: Bool
    /// True when text and visual inputs land in one comparable space, which is
    /// what makes image-to-text and text-to-image retrieval possible. A
    /// provider that embeds images into a *separate* space must report `false`
    /// here even though it accepts both.
    public var sharedTextVisualSpace: Bool
    /// True when a prefix of the vector is itself a valid, normalisable
    /// embedding (Matryoshka). Dimension reduction is rejected unless this is
    /// set — truncating a non-Matryoshka vector produces something that still
    /// has a cosine similarity and no meaning.
    public var supportsDimensionReduction: Bool
    /// Dimensions the provider will return on request, ascending. Empty when
    /// only the native dimension is available.
    public var supportedDimensions: [Int]
    /// Largest batch the provider will accept in one call.
    public var maximumBatchSize: Int
    /// Token budget per input. Inputs above it are truncated, and the
    /// truncation is reported in diagnostics rather than swallowed.
    public var maximumInputTokens: Int

    public init(
        acceptsText: Bool = true,
        acceptsImages: Bool = false,
        acceptsVideo: Bool = false,
        sharedTextVisualSpace: Bool = false,
        supportsDimensionReduction: Bool = false,
        supportedDimensions: [Int] = [],
        maximumBatchSize: Int = 1,
        maximumInputTokens: Int = 512,
        acceptsAudio: Bool = false
    ) {
        self.acceptsText = acceptsText
        self.acceptsImages = acceptsImages
        self.acceptsVideo = acceptsVideo
        self.acceptsAudio = acceptsAudio
        self.sharedTextVisualSpace = sharedTextVisualSpace
        self.supportsDimensionReduction = supportsDimensionReduction
        self.supportedDimensions = supportedDimensions
        self.maximumBatchSize = maximumBatchSize
        self.maximumInputTokens = maximumInputTokens
    }

    /// Whether this provider can embed a representation of `kind`.
    ///
    /// Per modality: a provider that takes images is not thereby able to take
    /// a video, and one that takes text cannot take a recording.
    public func accepts(_ kind: RepresentationKind) -> Bool {
        switch kind {
        case .image: acceptsImages
        case .video: acceptsVideo
        case .audio: acceptsAudio
        case .originalText, .ocrText, .transcript, .caption, .summary, .translation, .soundEvent,
            .metadata:
            acceptsText
        }
    }
}

/// Everything that has to match for two vectors to be comparable.
///
/// Stored alongside every embedding space and checked on every query. Equal
/// dimensions are not enough and never were: two 1024-dimensional models
/// produce vectors whose cosine similarity is a number with no meaning.
public struct EmbeddingSpaceDescriptor: Sendable, Hashable, Codable {
    /// Stable identity of the weights — a Hub repo id, a local checkout hash.
    public var modelIdentity: String
    /// Quantisation or conversion variant, when the same weights ship several.
    public var weightsVariant: String?
    public var tokenizerVersion: String
    /// Version of the preprocessing the provider applies: prompt templates,
    /// image resizing, frame sampling.
    public var preprocessingVersion: String
    /// Instruction prefixed to documents at index time, verbatim. Part of
    /// identity because changing it changes every vector in the space.
    public var documentInstruction: String?
    /// Pooling strategy, e.g. `"last-token"`, `"mean"`, `"cls"`.
    public var pooling: String
    /// True when the provider L2-normalises its output.
    public var normalized: Bool
    public var dimensions: Int

    public init(
        modelIdentity: String,
        weightsVariant: String? = nil,
        tokenizerVersion: String,
        preprocessingVersion: String,
        documentInstruction: String? = nil,
        pooling: String,
        normalized: Bool,
        dimensions: Int
    ) {
        self.modelIdentity = modelIdentity
        self.weightsVariant = weightsVariant
        self.tokenizerVersion = tokenizerVersion
        self.preprocessingVersion = preprocessingVersion
        self.documentInstruction = documentInstruction
        self.pooling = pooling
        self.normalized = normalized
        self.dimensions = dimensions
    }

    /// Identity fingerprint. Two descriptors with the same fingerprint describe
    /// vectors that may be compared; anything else may not.
    public var fingerprint: String {
        StableHash.of { hasher in
            hasher.append("embedding-space.v1")
            hasher.append(modelIdentity)
            hasher.append(weightsVariant ?? "")
            hasher.append(tokenizerVersion)
            hasher.append(preprocessingVersion)
            hasher.append(documentInstruction ?? "")
            hasher.append(pooling)
            hasher.append(normalized)
            hasher.append(dimensions)
        }
    }

    /// Whether a query embedded under `self` may be compared with vectors
    /// indexed under `other`.
    ///
    /// `reducedDimensions` allows the one legitimate mismatch: a Matryoshka
    /// provider queried at a prefix length the index was also built at.
    public func isCompatible(with other: EmbeddingSpaceDescriptor, reducedDimensions: Int? = nil)
        -> Bool
    {
        guard modelIdentity == other.modelIdentity,
            weightsVariant == other.weightsVariant,
            tokenizerVersion == other.tokenizerVersion,
            preprocessingVersion == other.preprocessingVersion,
            documentInstruction == other.documentInstruction,
            pooling == other.pooling,
            normalized == other.normalized
        else { return false }
        if let reducedDimensions {
            return reducedDimensions <= dimensions && reducedDimensions <= other.dimensions
        }
        return dimensions == other.dimensions
    }
}

/// What the provider is being asked to embed.
///
/// Documents and queries are separated because most retrieval models are
/// asymmetric: the query side gets an instruction prefix, the document side
/// does not, and swapping them quietly costs several points of nDCG.
public enum EmbeddingRole: Sendable, Hashable, Codable {
    case document
    /// A query, with the instruction to prefix it with. `nil` uses the
    /// provider's default.
    case query(instruction: String?)
}

/// One unit of work for an embedding provider.
public struct EmbeddingInput: Sendable {
    public var text: String
    public var images: [ImagePayload]
    public var videos: [URL]
    public var role: EmbeddingRole
    /// Prefix length to return, for Matryoshka providers. Rejected by providers
    /// that do not advertise ``EmbeddingCapabilities/supportsDimensionReduction``.
    public var requestedDimensions: Int?

    public init(
        text: String = "",
        images: [ImagePayload] = [],
        videos: [URL] = [],
        role: EmbeddingRole = .document,
        requestedDimensions: Int? = nil
    ) {
        self.text = text
        self.images = images
        self.videos = videos
        self.role = role
        self.requestedDimensions = requestedDimensions
    }

    public var isEmpty: Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && images.isEmpty
            && videos.isEmpty
    }
}

/// An image handed to a provider, either already decoded or still on disk.
///
/// Media stays a reference for as long as possible. An index build over ten
/// thousand frames that decoded every one up front would need the whole corpus
/// resident; this way each is read as its batch reaches the provider.
public enum ImagePayload: Sendable, Hashable {
    case fileURL(URL)
    case encoded(Data, contentType: String)
}

/// Result of embedding one input.
public struct EmbeddingOutput: Sendable {
    public var vector: [Float]
    /// Tokens the provider actually consumed, after truncation.
    public var consumedTokens: Int?
    /// True when the input did not fit the provider's budget. Surfaced in
    /// diagnostics: a silently truncated document is a silently wrong vector.
    public var wasTruncated: Bool

    public init(vector: [Float], consumedTokens: Int? = nil, wasTruncated: Bool = false) {
        self.vector = vector
        self.consumedTokens = consumedTokens
        self.wasTruncated = wasTruncated
    }
}

/// Produces vectors. Implement this to replace the bundled models.
///
/// Everything model-specific — prompt wire format, pooling, tokenisation,
/// normalisation — lives behind this protocol. The engine knows a provider's
/// descriptor and capabilities and nothing else about it.
public protocol EmbeddingProvider: Actor {
    var descriptor: EmbeddingSpaceDescriptor { get }
    /// Must not change with ``isPrepared``. The engine routes representations
    /// and queries on capabilities before it makes the model resident, and an
    /// idle or sequential-residency release happens between most calls.
    var capabilities: EmbeddingCapabilities { get }

    /// Load weights. Separate from construction so an engine can be built,
    /// configured and handed around before any model is resident.
    func prepare() async throws

    /// Release weights and any cached buffers.
    func release() async

    var isPrepared: Bool { get }

    /// Embed a batch.
    ///
    /// Must honour `Task.isCancelled` between inputs and return exactly as many
    /// outputs as it was given inputs, in order. An input the provider cannot
    /// handle yields an empty vector rather than a thrown error, so one bad
    /// frame does not abandon a batch of a hundred.
    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput]
}

extension EmbeddingProvider {
    public func embed(_ input: EmbeddingInput) async throws -> EmbeddingOutput {
        let outputs = try await embed([input])
        guard let first = outputs.first else {
            throw SemanticSearchError.providerReturnedNoOutput(descriptor.modelIdentity)
        }
        return first
    }
}

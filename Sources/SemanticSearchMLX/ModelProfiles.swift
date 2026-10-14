import Foundation
import SemanticSearch

/// The reference configurations.
///
/// The Qwen3 profiles name 5-bit conversions, the text pair and the VL pair
/// from the same converter. Other quantisations are explicit configurations —
/// pass a different ``ModelReference`` — and are subject to the same evaluation
/// gates before they are used for anything.
public enum ModelProfile {

    /// Text-only: Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B.
    ///
    /// 1024-dimensional, ~820 MB on disk at 5 bits for the pair. The right
    /// default for documents, transcripts, OCR text and captioned frames, and
    /// unable to embed a pixel.
    ///
    /// 5-bit. Measured against the 8-bit conversions
    /// (`mlx-community/Qwen3-Embedding-0.6B-8bit`,
    /// `mku64/Qwen3-Reranker-0.6B-mlx-8Bit`): identical nDCG@10 and MRR on the
    /// multilingual corpus, and a warm query 22 % faster. Override with
    /// `SEMANTIC_SEARCH_TEXT_EMBEDDING_MODEL` / `…_RERANKER_MODEL` or by passing
    /// a `ModelReference`.
    ///
    /// Every default here is pinned to a commit (``ModelReference/hub(_:)``).
    /// Moving a pin names other weights: it opens a new space, and its old
    /// vectors are not searched alongside the new ones.
    public enum Text {
        public static let embeddingModel: ModelReference =
            "starkdmi/Qwen3-Embedding-0.6B-mlx-5bit@9f2ea09a790aad476b73a1e1934f2068a6b3acd9"
        public static let rerankerModel: ModelReference =
            "starkdmi/Qwen3-Reranker-0.6B-mlx-5bit@ae152c2f65103697b46755fdcad619fe884b4298"
        public static let dimensions = 1_024
        /// Longest input the provider will embed without truncating. Size the
        /// chunker from this rather than from a guess.
        public static let maximumInputTokens = 512

        public static func embeddingProvider(
            model: ModelReference = embeddingModel,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> MLXEmbeddingProvider {
            MLXEmbeddingProvider(
                configuration: MLXEmbeddingProvider.Configuration(
                    model: model,
                    dimensions: dimensions,
                    // Qwen3-Embedding is trained with Matryoshka objectives at
                    // these widths. Any other prefix is rejected rather than
                    // silently truncated.
                    matryoshkaDimensions: [128, 256, 512, 768, 1_024],
                    maximumBatchSize: 8,
                    maximumInputTokens: maximumInputTokens,
                    multimodal: false,
                    runtime: runtime))
        }

        public static func rerankingProvider(
            model: ModelReference = rerankerModel,
            scoring: MLXRerankingProvider.Scoring = .fast,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> MLXRerankingProvider {
            MLXRerankingProvider(
                configuration: MLXRerankingProvider.Configuration(
                    model: model,
                    maximumPairTokens: 1_024,
                    batchSize: 4,
                    maximumImagesPerPair: 0,
                    multimodal: false,
                    scoring: scoring,
                    runtime: runtime))
        }
    }

    /// Multimodal: Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B.
    ///
    /// Text and visual representations land in one space, which is what makes
    /// image-to-text and text-to-image retrieval work. Audio becomes searchable
    /// through transcription; there is no acoustic retrieval here and a
    /// provider that offers it would be a new space, not a setting on this one.
    ///
    /// ~1.9 GB on disk per model at 5 bits (2.5 GB at 8). On a phone, pair it
    /// with ``RuntimePolicy/constrainedDevice`` so the two models are never
    /// resident together.
    ///
    /// 5-bit. Measured against the 8-bit conversions
    /// (`mlx-community/Qwen3-VL-Embedding-2B-8bit`,
    /// `mlx-community/Qwen3-VL-Reranker-2B-8bit`) and Qwen's bf16 reference
    /// on labelled frames: the embedder retrieves as well or better, and the
    /// reranker alone makes the same top-1 calls as the 8-bit one. Its raw
    /// scores sit lower than bf16's across the board — the ordering is kept,
    /// so compare them within a query, not against a fixed threshold.
    public enum Multimodal {
        public static let embeddingModel: ModelReference =
            "starkdmi/Qwen3-VL-Embedding-2B-mlx-5bit@6b63622e1024d34cf43292552ddf22de546a01ad"
        public static let rerankerModel: ModelReference =
            "starkdmi/Qwen3-VL-Reranker-2B-mlx-5bit@2abf82837f13308a03ee2e06ad8b8b7f4eb79980"
        public static let dimensions = 2_048
        /// Longest text, in tokens, the provider will embed without
        /// truncating; longer text is cut and reported as truncated. Visual
        /// tokens are not counted here — the image processor's resolution
        /// limit bounds them — so this is the text share of the sequence.
        public static let maximumInputTokens = 1_024

        public static func embeddingProvider(
            model: ModelReference = embeddingModel,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> MLXEmbeddingProvider {
            MLXEmbeddingProvider(
                configuration: MLXEmbeddingProvider.Configuration(
                    model: model,
                    defaultQueryInstruction: MLXEmbeddingProvider.qwen3VLDefaultInstruction,
                    dimensions: dimensions,
                    // Not advertised for the VL checkpoints: the conversions
                    // this was validated against do not document Matryoshka
                    // widths, and guessing one produces vectors that are
                    // comparable-looking and wrong.
                    matryoshkaDimensions: [],
                    maximumBatchSize: 1,
                    maximumInputTokens: maximumInputTokens,
                    multimodal: true,
                    runtime: runtime))
        }

        public static func rerankingProvider(
            model: ModelReference = rerankerModel,
            scoring: MLXRerankingProvider.Scoring = .fast,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> MLXRerankingProvider {
            MLXRerankingProvider(
                configuration: MLXRerankingProvider.Configuration(
                    model: model,
                    maximumPairTokens: 2_048,
                    batchSize: 1,
                    // Two query pictures and the frame being judged: a query
                    // can pair two complementary images.
                    maximumImagesPerPair: 3,
                    multimodal: true,
                    scoring: scoring,
                    runtime: runtime))
        }
    }

    /// Frames in words, for the text profile: SmolVLM2-500M captions each
    /// frame, and the caption is indexed and reranked like any other text.
    ///
    /// Pictures searched with the 0.6B text pair plus a 500M captioner, no 2B
    /// model resident at query time. Wrap the frame source with
    /// ``SemanticSearch/CaptioningProcessor``:
    ///
    /// ```swift
    /// let captioner = ModelProfile.Captioning.captioner()
    /// var configuration = ModelProfile.textConfiguration(storeURL: url)
    /// configuration.processor = CombinedContentProcessor([
    ///     TranscriptionProcessor(),
    ///     await CaptioningProcessor(wrapping: KeyframeProcessor(), captioner: captioner),
    ///     await SoundEventProcessor(classifier: AppleSoundClassifier()),  // sounds as words
    /// ])
    /// configuration.queryCaptioner = captioner  // search with pictures too
    /// ```
    ///
    /// Under the multimodal profile the same captions give the frame a second,
    /// textual route in, and let the reranker judge it without pixels
    /// (``SemanticSearch/SearchOptions/reranksImages``).
    public enum Captioning {
        public static let model: ModelReference = MLXImageCaptioner.defaultModel

        public static func captioner(
            model: ModelReference = model,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> MLXImageCaptioner {
            MLXImageCaptioner(
                configuration: MLXImageCaptioner.Configuration(model: model, runtime: runtime))
        }
    }

    /// EmbeddingGemma 2: text, images and video in one 768-dimensional space.
    ///
    /// bf16, as Google's checkpoint ships: the mlx-community conversion is
    /// bit-identical to it, and the model card forbids float16 — its
    /// activations overflow it and the vectors come back NaN. Resident:
    /// ~0.88 GB with vision (the profile's default), ~0.54 GB text-only
    /// (`vision: false`), which embeds text into the same space, ~1.5 GB
    /// with the audio tower too. Trained with Matryoshka at 512, 256 and
    /// 128; the card finds 256 near-lossless and 128 costly for media.
    ///
    /// The rerankers are the Qwen3 pair: a reranker reads candidates, not
    /// vectors, so the multimodal one judges frames by their pixels and the
    /// text one by their captions, as under the other profiles.
    public enum EmbeddingGemma {
        public static let embeddingModel: ModelReference =
            "mlx-community/embeddinggemma-2-bf16@1a4ffddb7905d3f63486748deabe091a01fb6201"
        public static let dimensions = 768
        public static let matryoshkaDimensions = [128, 256, 512, 768]
        /// Longest text, in tokens counting the prompt, `<bos>` and `<eos>`,
        /// the provider embeds without truncating.
        public static let maximumInputTokens = 2_048
        /// Chunk size the EG2 profiles index at, in `TokenEstimator` tokens,
        /// so that the MLX and Core ML providers both read a chunk whole: the
        /// Core ML conversion stops at 512 Gemma tokens, 9 of them the
        /// document prompt, `<bos>` and `<eos>`.
        ///
        /// The chunker cannot ask Gemma's tokenizer, and the estimate is not
        /// it. On the multilingual corpus fixture a document runs 0.56–0.77
        /// Gemma tokens an estimated one in Japanese and Chinese, 0.85–1.17 in
        /// English, 1.23–1.41 in Russian and Turkish and up to 1.696 in Arabic
        /// (39 for 23); 296 is (512 − 9) / 1.696, rounded down. English prose
        /// varies more chunk to chunk than per language: BEIR's ArguAna packed
        /// into chunks passed 512 in 17% of them at 480, in none at 296
        /// (longest 449). Punctuation counts nothing in the estimate and a
        /// number one word, where Gemma has a token for every digit and most
        /// marks: a table of figures runs ~3 tokens an estimated one and still
        /// passes 512, as do citation lists and contents pages (6 of 3,020
        /// SciDocs chunks at 296). Core ML keeps the head of such a chunk and
        /// reports it truncated (`IndexingReport.truncatedRepresentations`);
        /// MLX reads it whole.
        /// Measured with Google's tokenizer.json.
        public static let chunkTokenLimit = 296

        public static func embeddingProvider(
            model: ModelReference = embeddingModel,
            vision: Bool = true,
            audio: Bool = false,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> EmbeddingGemma2Provider {
            EmbeddingGemma2Provider(
                configuration: EmbeddingGemma2Provider.Configuration(
                    model: model,
                    vision: vision,
                    audio: audio,
                    maximumInputTokens: maximumInputTokens,
                    runtime: runtime))
        }

        /// An engine over one EmbeddingGemma 2 space, reranked by the Qwen3
        /// reranker that matches it: Qwen3-VL-Reranker-2B with `vision`,
        /// Qwen3-Reranker-0.6B without.
        ///
        /// With `vision` every kind the provider accepts is routed to the
        /// space; `embedsCaptions: false` leaves captions to BM25 and the
        /// reranker, as `multimodalConfiguration` explains. Without it, the
        /// text kinds `textConfiguration` routes. EmbeddingGemma 2 with vision
        /// (~0.88 GB) and the 5-bit 2B reranker (~1.9 GB) are ~2.8 GB
        /// together; on a small device pass `RuntimePolicy.constrainedDevice`.
        public static func configuration(
            storeURL: URL?,
            space: EmbeddingSpaceID = "embeddinggemma2",
            indexKind: VectorIndexKind = .hnsw,
            vision: Bool = true,
            embedsCaptions: Bool = true,
            runtime: RuntimePolicy = .default,
            mlxRuntime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> EngineConfiguration {
            configuration(
                storeURL: storeURL,
                space: space,
                indexKind: indexKind,
                vision: vision,
                embedsCaptions: embedsCaptions,
                reranker: vision
                    ? Multimodal.rerankingProvider(runtime: mlxRuntime)
                    : ModelProfile.Text.rerankingProvider(runtime: mlxRuntime),
                runtime: runtime,
                mlxRuntime: mlxRuntime)
        }

        /// The same engine with a reranker of the caller's choosing; `nil`
        /// reranks nothing.
        ///
        /// An overload rather than a default argument: the default reranker
        /// depends on `vision`, which a default argument cannot see.
        public static func configuration(
            storeURL: URL?,
            space: EmbeddingSpaceID = "embeddinggemma2",
            indexKind: VectorIndexKind = .hnsw,
            vision: Bool = true,
            embedsCaptions: Bool = true,
            reranker: (any RerankingProvider)?,
            runtime: RuntimePolicy = .default,
            mlxRuntime: MLXRuntime.Configuration = .fromEnvironment()
        ) -> EngineConfiguration {
            var kinds = Set(RepresentationKind.allCases)
            if !vision {
                kinds = [
                    .originalText, .ocrText, .transcript, .caption, .summary, .translation,
                    .soundEvent, .metadata,
                ]
            }
            if !embedsCaptions { kinds.remove(.caption) }
            return EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: space,
                        provider: embeddingProvider(vision: vision, runtime: mlxRuntime),
                        kinds: kinds,
                        indexKind: indexKind)
                ],
                reranker: reranker,
                processor: TextChunkingProcessor(
                    chunker: .default, providerTokenLimit: chunkTokenLimit),
                runtime: runtime)
        }
    }

    // MARK: - Engine configurations

    /// A text-only engine: one space over every text-bearing representation.
    ///
    /// Pass the captioner that captions the frames as `queryCaptioner` to
    /// search with pictures too: a query image is captioned the same way and
    /// searched as text (``EngineConfiguration/queryCaptioner``).
    public static func textConfiguration(
        storeURL: URL?,
        space: EmbeddingSpaceID = "text",
        indexKind: VectorIndexKind = .hnsw,
        queryCaptioner: (any ImageCaptioner)? = nil,
        runtime: RuntimePolicy = .default,
        mlxRuntime: MLXRuntime.Configuration = .fromEnvironment()
    ) -> EngineConfiguration {
        EngineConfiguration(
            storeURL: storeURL,
            spaces: [
                EmbeddingSpaceConfiguration(
                    id: space,
                    provider: Text.embeddingProvider(runtime: mlxRuntime),
                    kinds: [
                        .originalText, .ocrText, .transcript, .caption, .summary, .translation,
                        .soundEvent, .metadata,
                    ],
                    indexKind: indexKind)
            ],
            reranker: Text.rerankingProvider(runtime: mlxRuntime),
            processor: TextChunkingProcessor(
                chunker: .default, providerTokenLimit: Text.maximumInputTokens),
            runtime: runtime,
            queryCaptioner: queryCaptioner)
    }

    /// A multimodal engine: one space holding text and pixels together.
    ///
    /// One space rather than two on purpose. Two spaces would need their
    /// candidates fused across models, and RRF over ranks is the only honest
    /// way to do that; a shared space compares them directly, which is both
    /// better and the reason to pay for a 2B model.
    ///
    /// `embedsCaptions: false` is a layout for captioned frames: the pixels
    /// are embedded, the caption is indexed for BM25 and read by the reranker,
    /// but not embedded. A frame then has two routes in, as a transcript line
    /// does; embedding the caption as well gives it a third, and exact
    /// transcript queries start losing to frames.
    public static func multimodalConfiguration(
        storeURL: URL?,
        space: EmbeddingSpaceID = "multimodal",
        indexKind: VectorIndexKind = .hnsw,
        embedsCaptions: Bool = true,
        runtime: RuntimePolicy = .default,
        mlxRuntime: MLXRuntime.Configuration = .fromEnvironment()
    ) -> EngineConfiguration {
        EngineConfiguration(
            storeURL: storeURL,
            spaces: [
                EmbeddingSpaceConfiguration(
                    id: space,
                    provider: Multimodal.embeddingProvider(runtime: mlxRuntime),
                    kinds: embedsCaptions
                        ? Set(RepresentationKind.allCases)
                        : Set(RepresentationKind.allCases).subtracting([.caption]),
                    indexKind: indexKind)
            ],
            reranker: Multimodal.rerankingProvider(runtime: mlxRuntime),
            processor: TextChunkingProcessor(
                chunker: .default, providerTokenLimit: Multimodal.maximumInputTokens),
            runtime: runtime)
    }
}

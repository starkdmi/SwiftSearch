import CoreImage
import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import SemanticSearch
import SemanticSearchApple

@preconcurrency import MLX

/// MLX-backed reranker for the Qwen3-Reranker families.
///
/// Qwen3-Reranker is a causal LM asked a yes/no question; the score is
/// `sigmoid(logit("yes") − logit("no"))` read at the last real token. Two
/// details in that sentence are load-bearing:
///
/// - **The last *real* token.** With right padding, the final position of a
///   batched row is a pad token whose logits mean nothing. Reading it makes a
///   candidate's score depend on which other candidates shared its batch.
/// - **`sigmoid` of the *difference*.** Not a softmax over the whole
///   vocabulary, and not a probability of anything. The value is monotone in
///   relevance and otherwise uncalibrated, which is why
///   ``RerankingCapabilities/producesCalibratedScores`` is `false` and why the
///   engine never thresholds on it.
public actor MLXRerankingProvider: RerankingProvider {

    private enum Backend {
        case text(ModelContainer)
        case vlm(ModelContainer)
        case fastText(FastTextRerankScorer, any Tokenizer)
        case fastVision(FastVisionRerankScorer)
    }

    /// How pairs are run through the model.
    public enum Scoring: Sendable {
        /// A query's shared prefix once, then each candidate against it, on a
        /// model restructured for scoring (`FastQwen3Reranker.swift`). Qwen3
        /// and Qwen3-VL checkpoints with tied embeddings; anything else loads
        /// the stock way.
        case fast
        /// mlx-swift-lm's model, each pair from scratch. Kept to compare
        /// against.
        case stock
    }

    public struct Configuration: Sendable {
        public var model: ModelReference
        /// Tokens per (query, document) pair, prompt scaffolding included.
        public var maximumPairTokens: Int
        public var batchSize: Int
        /// Images per pair, query and document together. Zero disables visual
        /// reranking even on a multimodal checkpoint.
        public var maximumImagesPerPair: Int
        public var modelIdentity: String?
        public var weightsVariant: String?
        /// Whether the checkpoint is a vision-language model. Decides the
        /// capabilities reported before the weights are loaded; `nil` guesses
        /// from the model id. Set it when the weights live in a directory whose
        /// name does not say "VL".
        public var multimodal: Bool?
        public var scoring: Scoring
        /// Bytes of vision-tower output kept per image across queries, with
        /// ``Scoring/fast``; 0 keeps none. A keyframe of 1024 px is ~9 MB. The
        /// cache outlives ``release()`` — under sequential residency the
        /// reranker is released between queries, and the features depend only
        /// on the image and the weights.
        public var visionCacheBytes: Int
        public var runtime: MLXRuntime.Configuration

        public init(
            model: ModelReference,
            maximumPairTokens: Int = 1_024,
            batchSize: Int = 4,
            maximumImagesPerPair: Int = 2,
            modelIdentity: String? = nil,
            weightsVariant: String? = nil,
            multimodal: Bool? = nil,
            scoring: Scoring = .fast,
            visionCacheBytes: Int = 512 * 1_024 * 1_024,
            runtime: MLXRuntime.Configuration = .fromEnvironment()
        ) {
            self.model = model
            self.maximumPairTokens = max(64, maximumPairTokens)
            self.batchSize = max(1, batchSize)
            self.maximumImagesPerPair = max(0, maximumImagesPerPair)
            self.modelIdentity = modelIdentity
            self.weightsVariant = weightsVariant
            self.multimodal = multimodal
            self.scoring = scoring
            self.visionCacheBytes = max(0, visionCacheBytes)
            self.runtime = runtime
        }
    }

    /// Bumped when the prompt scaffolding changes. Part of the reranker's
    /// identity so a changed prompt is visible in diagnostics; unlike an
    /// embedding change it does not require rebuilding anything.
    /// v6: images upright, as the embedder sees them (`ImageDecoding.decodeUpright`).
    private static let promptVersion = "qwen3-reranker.v6"

    private let configuration: Configuration
    private var backend: Backend?
    /// What the weights turned out to be, remembered across `release()`.
    private var loadedVisual: Bool?
    private let visionCache: VisionFeatureCache

    public init(configuration: Configuration) {
        self.configuration = configuration
        self.visionCache = VisionFeatureCache(capacity: configuration.visionCacheBytes)
    }

    public init(model: ModelReference) {
        self.init(configuration: Configuration(model: model))
    }

    // MARK: - Contract

    public var identity: RerankerIdentity {
        RerankerIdentity(
            modelIdentity: configuration.modelIdentity ?? configuration.model.identity,
            weightsVariant: configuration.weightsVariant,
            promptVersion: Self.promptVersion)
    }

    public var capabilities: RerankingCapabilities {
        // The same answer loaded or not. The engine reads capabilities before
        // it makes the reranker resident, and under sequential residency the
        // reranker is released by every query's embedding step — reporting
        // text-only while unloaded meant a VL reranker never saw an image.
        let isVisual =
            loadedVisual ?? configuration.multimodal ?? Self.looksMultimodal(configuration.model)
        return RerankingCapabilities(
            acceptsImages: isVisual && configuration.maximumImagesPerPair > 0,
            acceptsVideo: false,
            maximumPairTokens: configuration.maximumPairTokens,
            maximumImagesPerPair: isVisual ? configuration.maximumImagesPerPair : 0,
            maximumBatchSize: configuration.batchSize,
            // Never. The scores are monotone in relevance and not calibrated
            // against anything; asserting otherwise invites a threshold that
            // means nothing.
            producesCalibratedScores: false)
    }

    public var isPrepared: Bool { backend != nil }

    public func prepare() async throws {
        guard backend == nil else { return }
        MLXRuntime.apply(configuration.runtime)

        let modelConfiguration = configuration.model.runtimeConfiguration

        if configuration.scoring == .fast {
            do {
                if try await prepareFast(modelConfiguration) { return }
            } catch {
                throw SemanticSearchError.modelUnavailable(
                    configuration.model.identity, underlying: error)
            }
        }

        let downloader = MLXRuntime.downloader(configuration.runtime, for: .languageModel)
        do {
            backend = .text(
                try await LLMModelFactory.shared.loadContainer(
                    from: downloader, using: MLXRuntime.tokenizerLoader,
                    configuration: modelConfiguration))
            loadedVisual = false
        } catch let error as ModelFactoryError {
            guard case .unsupportedModelType = error else {
                throw SemanticSearchError.modelUnavailable(
                    configuration.model.identity, underlying: error)
            }
            do {
                backend = .vlm(
                    try await VLMModelFactory.shared.loadContainer(
                        from: downloader, using: MLXRuntime.tokenizerLoader,
                        configuration: modelConfiguration))
                loadedVisual = true
            } catch {
                throw SemanticSearchError.modelUnavailable(
                    configuration.model.identity, underlying: error)
            }
        } catch {
            throw SemanticSearchError.modelUnavailable(
                configuration.model.identity, underlying: error)
        }
    }

    /// Loads the checkpoint onto the scoring models. `false` when it is not one
    /// they take — another architecture, or an untied output head.
    private func prepareFast(_ modelConfiguration: ModelConfiguration) async throws -> Bool {
        let directory = try await resolve(
            configuration: modelConfiguration,
            from: MLXRuntime.downloader(configuration.runtime, for: .languageModel),
            useLatest: false, progressHandler: { _ in }
        ).modelDirectory
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        struct ModelType: Decodable { let model_type: String }
        let modelType = try JSONDecoder().decode(ModelType.self, from: configData).model_type
        guard modelType == "qwen3" || modelType == "qwen3_vl" else { return false }

        let tokenizer = try await MLXRuntime.tokenizerLoader.load(from: directory)
        guard
            let yes = Self.tokenID(["yes", " yes"], convert: tokenizer.convertTokenToId),
            let no = Self.tokenID(["no", " no"], convert: tokenizer.convertTokenToId)
        else {
            throw SemanticSearchError.modelUnavailable(
                "reranker tokenizer has no yes/no token ids", underlying: nil)
        }
        let quantization = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
            .perLayerQuantization

        if modelType == "qwen3" {
            let text = try JSONDecoder().decode(
                Qwen3VLConfiguration.TextConfiguration.self, from: configData)
            let language = FastLanguage(text)
            guard
                try loadScoringWeights(
                    directory: directory, quantization: quantization, into: language)
            else { return false }
            language.prepareScoreHead(yes: yes, no: no)
            backend = .fastText(
                FastTextRerankScorer(language: language, ropeBase: Float(text.ropeTheta)),
                tokenizer)
            loadedVisual = false
            return true
        }

        let vl = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: configData)
        guard let imagePad = tokenizer.convertTokenToId("<|image_pad|>") else { return false }
        let processorConfiguration = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: Data(
                contentsOf: directory.appendingPathComponent("preprocessor_config.json")))
        let model = FastQwen3VL(vl)
        guard try loadScoringWeights(directory: directory, quantization: quantization, into: model)
        else { return false }
        model.prepare(yesToken: yes, noToken: no)
        backend = .fastVision(
            FastVisionRerankScorer(
                model: model, tokenizer: tokenizer,
                processor: Qwen3VLProcessor(processorConfiguration, tokenizer: tokenizer),
                configuration: vl, imagePad: imagePad))
        loadedVisual = true
        return true
    }

    /// Heuristic for the capabilities reported before the weights have ever
    /// loaded. ``Configuration/multimodal`` overrides it.
    private static func looksMultimodal(_ model: ModelReference) -> Bool {
        let identity = model.identity.lowercased()
        return identity.contains("-vl") || identity.contains("_vl") || identity.contains("vlm")
    }

    public func release() async {
        backend = nil
        MLXRuntime.releaseCache()
    }

    public func score(query: RerankQuery, candidates: [RerankCandidate]) async throws
        -> [RerankScore]
    {
        guard let backend else {
            throw SemanticSearchError.providerNotPrepared(configuration.model.identity)
        }
        guard !candidates.isEmpty else { return [] }

        switch backend {
        case .text(let container):
            return try await scoreText(query: query, candidates: candidates, container: container)
        case .vlm(let container):
            return try await scoreMultimodal(
                query: query, candidates: candidates, container: container)
        case .fastText(let scorer, let tokenizer):
            let rows = Self.textRows(
                query: query, candidates: candidates, tokenizer: tokenizer,
                budget: configuration.maximumPairTokens)
            try Task.checkCancellation()
            let scores = try scorer.score(rows.map(\.tokens))
            return zip(rows, scores).map { row, score in
                RerankScore(
                    score: Self.sigmoid(score), wasTruncated: row.truncated,
                    droppedParentContext: row.droppedContext, droppedImages: row.droppedImages)
            }
        case .fastVision(let scorer):
            return try scoreFast(query: query, candidates: candidates, scorer: scorer)
        }
    }

    // MARK: - Text backend

    struct TextRow {
        var tokens: [Int]
        var truncated: Bool
        var droppedContext: Bool
        var droppedImages: Int
    }

    /// Each candidate's prompt tokens: scaffolding, pair, assistant opening.
    static func textRows(
        query: RerankQuery, candidates: [RerankCandidate], tokenizer: any Tokenizer, budget: Int
    ) -> [TextRow] {
        let prefixTokens = tokenizer.encode(text: systemPrefix, addSpecialTokens: false)
        let suffixTokens = tokenizer.encode(text: assistantSuffix, addSpecialTokens: false)
        let pairBudget = max(1, budget - prefixTokens.count - suffixTokens.count)

        return candidates.map { candidate in
            // Parent context is dropped before the evidence is cut.
            // Truncating the other way round removes the text that caused the
            // match and leaves the surrounding document, which is exactly
            // backwards.
            var pairTokens = tokenizer.encode(
                text: pair(query: query, candidate: candidate, includeContext: true),
                addSpecialTokens: false)
            var lostContext = false
            if pairTokens.count > pairBudget, candidate.parentContext != nil {
                pairTokens = tokenizer.encode(
                    text: pair(query: query, candidate: candidate, includeContext: false),
                    addSpecialTokens: false)
                lostContext = true
            }
            let wasTruncated = pairTokens.count > pairBudget
            if wasTruncated { pairTokens = Array(pairTokens.prefix(pairBudget)) }
            return TextRow(
                tokens: prefixTokens + pairTokens + suffixTokens, truncated: wasTruncated,
                droppedContext: lostContext, droppedImages: candidate.images.count)
        }
    }

    private func scoreText(
        query: RerankQuery, candidates: [RerankCandidate], container: ModelContainer
    ) async throws -> [RerankScore] {
        var results = Array(repeating: RerankScore(score: 0), count: candidates.count)
        let step = configuration.batchSize
        let budget = configuration.maximumPairTokens

        for start in stride(from: 0, to: candidates.count, by: step) {
            try Task.checkCancellation()
            let end = min(start + step, candidates.count)
            let chunk = Array(candidates[start..<end])

            let scores = try await container.perform { context in
                let tokenizer = context.tokenizer
                guard
                    let yesToken = Self.tokenID(
                        ["yes", " yes"], convert: tokenizer.convertTokenToId),
                    let noToken = Self.tokenID(
                        ["no", " no"], convert: tokenizer.convertTokenToId)
                else {
                    throw SemanticSearchError.modelUnavailable(
                        "reranker tokenizer has no yes/no token ids", underlying: nil)
                }

                let rows = Self.textRows(
                    query: query, candidates: chunk, tokenizer: tokenizer, budget: budget)
                let lengths = rows.map(\.tokens.count)
                let padToken = tokenizer.eosTokenId ?? 0

                let maxLength = max(1, lengths.max() ?? 1)
                let padded: [MLXArray] = rows.map { row in
                    MLXArray(
                        (row.tokens + Array(repeating: padToken, count: maxLength - row.tokens.count))
                            .map(Int32.init))
                }
                let masks: [MLXArray] = lengths.map { length in
                    MLXArray(
                        Array(repeating: Int32(1), count: length)
                            + Array(repeating: Int32(0), count: maxLength - length))
                }

                let output = context.model(
                    .init(tokens: stacked(padded), mask: stacked(masks) .== Int32(1)),
                    cache: nil, state: nil)

                let delta = output.logits[0..., 0..., yesToken] - output.logits[0..., 0..., noToken]
                eval(delta)

                let shape = delta.shape
                guard shape.count == 2, shape[0] == chunk.count else {
                    throw SemanticSearchError.modelUnavailable(
                        "unexpected reranker logits shape \(shape)", underlying: nil)
                }
                let columns = shape[1]
                let flat = delta.asArray(Float.self)
                guard flat.count == shape[0] * columns else {
                    throw SemanticSearchError.modelUnavailable(
                        "unexpected reranker logit count", underlying: nil)
                }

                return rows.enumerated().map { index, row -> RerankScore in
                    // The last *real* token of this row, not of the padded
                    // batch.
                    let position = min(max(0, lengths[index] - 1), max(0, columns - 1))
                    return RerankScore(
                        score: Self.sigmoid(flat[index * columns + position]),
                        wasTruncated: row.truncated,
                        droppedParentContext: row.droppedContext,
                        droppedImages: row.droppedImages)
                }
            }

            for (offset, score) in scores.enumerated() { results[start + offset] = score }
        }

        return results
    }

    // MARK: - Vision-language backend

    /// Query images first, then the candidate's, within the pair's visual
    /// budget — but one place is kept for the candidate. A query with more
    /// pictures than the budget would otherwise crowd out the frame being
    /// judged, and the frame would be scored on its text alone. Kept per side:
    /// the prompt places each image under its own heading. `nil` is an image
    /// that could not be loaded; it is dropped.
    static func selectImages<Image>(
        query: [Image?], document: [Image?], budget: Int
    ) -> (query: [Image], document: [Image], dropped: Int) {
        let queryBudget = max(0, budget - min(1, document.count))
        var queryImages: [Image] = []
        var documentImages: [Image] = []
        var dropped = 0
        for (image, isQuery) in query.map({ ($0, true) }) + document.map({ ($0, false) }) {
            guard queryImages.count + documentImages.count < budget,
                !isQuery || queryImages.count < queryBudget,
                let image
            else {
                dropped += 1
                continue
            }
            if isQuery { queryImages.append(image) } else { documentImages.append(image) }
        }
        return (queryImages, documentImages, dropped)
    }

    private func scoreFast(
        query: RerankQuery, candidates: [RerankCandidate], scorer: FastVisionRerankScorer
    ) throws -> [RerankScore] {
        func load(_ payload: ImagePayload) -> ScoringImage? {
            switch payload {
            case .fileURL(let url):
                return (try? Data(contentsOf: url)).flatMap(ScoringImage.init(data:))
            case .encoded(let data, _):
                return ScoringImage(data: data)
            }
        }
        let queryImages = query.images.map(load)

        var prompts: [VisionRerankPrompt] = []
        var fits: [(truncated: Bool, droppedContext: Bool, dropped: Int)] = []
        for candidate in candidates {
            try Task.checkCancellation()
            let images = Self.selectImages(
                query: queryImages, document: candidate.images.map(load),
                budget: configuration.maximumImagesPerPair)
            let fitted = Self.fittedPair(
                query: query, candidate: candidate, budget: configuration.maximumPairTokens,
                tokenizer: scorer.tokenizer)
            let (prefix, suffix) = Self.renderedPrompt(
                query: query, queryImageCount: images.query.count,
                document: Self.documentText(
                    fitted.candidate, includeContext: fitted.includesContext),
                documentImageCount: images.document.count)
            prompts.append(
                VisionRerankPrompt(
                    prefix: prefix, queryImages: images.query, suffix: suffix,
                    documentImages: images.document))
            fits.append((fitted.truncated, fitted.droppedContext, images.dropped))
        }

        let scores = try scorer.score(prompts, cache: visionCache)
        return zip(scores, fits).map { score, fit in
            RerankScore(
                score: Self.sigmoid(score), wasTruncated: fit.truncated,
                droppedParentContext: fit.droppedContext, droppedImages: fit.dropped)
        }
    }

    private func scoreMultimodal(
        query: RerankQuery, candidates: [RerankCandidate], container: ModelContainer
    ) async throws -> [RerankScore] {
        var results: [RerankScore] = []
        results.reserveCapacity(candidates.count)

        // One at a time. A VLM pair carries its own vision tokens, so batching
        // would pad every row out to the widest image in the batch and spend
        // most of the compute on padding.
        for candidate in candidates {
            try Task.checkCancellation()

            // Not `.url`, which the runtime decodes on its side (MLXEmbeddingProvider).
            func load(_ payload: ImagePayload) -> UserInput.Image? {
                (try? ImageDecoding.decodeUpright(payload)).map {
                    .ciImage(CIImage(cgImage: $0))
                }
            }
            let images = Self.selectImages(
                query: query.images.map(load), document: candidate.images.map(load),
                budget: configuration.maximumImagesPerPair)

            let budget = configuration.maximumPairTokens
            let fitted = await container.perform { context in
                Self.fittedPair(
                    query: query, candidate: candidate, budget: budget,
                    tokenizer: context.tokenizer)
            }
            let messages = Self.multimodalMessages(
                query: query, queryImageCount: images.query.count,
                document: Self.documentText(
                    fitted.candidate, includeContext: fitted.includesContext),
                documentImageCount: images.document.count)

            let input = UserInput(
                messages: messages,
                images: images.query + images.document,
                additionalContext: [
                    // The answer is read at the first assistant position, as
                    // the reference implementation does. Read at the end of
                    // the user turn, where the model is predicting
                    // `<|im_start|>`, the order still comes out right, but
                    // image candidates are separated far less.
                    "add_generation_prompt": true,
                ])

            let box = UncheckedBox(input)
            let prepared = try await container.prepare(input: box.take())
            let score = try await container.perform(nonSendable: prepared) {
                context, prepared in
                let tokenizer = context.tokenizer
                guard
                    let yesToken = Self.tokenID(
                        ["yes", " yes"], convert: tokenizer.convertTokenToId),
                    let noToken = Self.tokenID(
                        ["no", " no"], convert: tokenizer.convertTokenToId)
                else {
                    throw SemanticSearchError.modelUnavailable(
                        "reranker tokenizer has no yes/no token ids", underlying: nil)
                }

                let cache = try context.model.newCache(parameters: nil)
                let output: LMOutput
                // One forward, so the logits cover every position the mask
                // names; a windowed prefill returns only the last window's.
                let oneForward = PrefillParameters(stepSize: .max)
                switch try context.model.prepare(
                    prepared, cache: cache, state: nil, prefill: oneForward)
                {
                case .logits(let logits):
                    output = logits
                case .tokens(let textInput):
                    output = context.model(
                        textInput, cache: cache.isEmpty ? nil : cache, state: nil)
                }

                return try Self.yesMinusNo(
                    logits: output.logits, mask: prepared.text.mask,
                    yesToken: yesToken, noToken: noToken)
            }

            results.append(
                RerankScore(
                    score: score, wasTruncated: fitted.truncated,
                    droppedParentContext: fitted.droppedContext, droppedImages: images.dropped))
        }

        return results
    }

    /// Qwen3-VL-Reranker's user turn, as its reference implementation builds
    /// it.
    ///
    /// Each side is its heading, then its images, then its text — `NULL` when
    /// it has neither — so the model can tell the query's picture from the
    /// document's. Putting every image at the head of the turn, before the
    /// instruction, leaves it no way to know which is which.
    static func userContent(
        query: RerankQuery, queryImageCount: Int, document: String, documentImageCount: Int
    ) -> [[String: String]] {
        func side(_ heading: String, text: String, images: Int) -> [[String: String]] {
            var parts: [[String: String]] = [["type": "text", "text": heading]]
            parts += Array(repeating: ["type": "image"], count: images)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                parts.append(["type": "text", "text": trimmed])
            } else if images == 0 {
                parts.append(["type": "text", "text": "NULL"])
            }
            return parts
        }

        return [["type": "text", "text": "<Instruct>: \(query.instruction)"]]
            + side("<Query>:", text: query.text, images: queryImageCount)
            + side(documentHeading, text: document, images: documentImageCount)
    }

    private static let documentHeading = "\n<Document>:"

    static func multimodalMessages(
        query: RerankQuery, queryImageCount: Int, document: String, documentImageCount: Int
    ) -> [[String: any Sendable]] {
        [
            ["role": "system", "content": [["type": "text", "text": systemInstruction]]],
            [
                "role": "user",
                "content": userContent(
                    query: query, queryImageCount: queryImageCount, document: document,
                    documentImageCount: documentImageCount),
            ],
        ]
    }

    /// The prompt ``multimodalMessages`` renders to through Qwen3-VL's chat
    /// template with `add_generation_prompt`, split after the document
    /// heading: everything before the split is the same for every candidate.
    static func renderedPrompt(
        query: RerankQuery, queryImageCount: Int, document: String, documentImageCount: Int
    ) -> (prefix: String, suffix: String) {
        var prefix = systemPrefix
        var suffix = ""
        var inDocument = false
        for part in userContent(
            query: query, queryImageCount: queryImageCount, document: document,
            documentImageCount: documentImageCount)
        {
            let rendered =
                part["type"] == "image"
                ? "<|vision_start|><|image_pad|><|vision_end|>" : (part["text"] ?? "")
            if inDocument { suffix += rendered } else { prefix += rendered }
            if !inDocument, part["text"] == documentHeading { inDocument = true }
        }
        return (prefix, suffix + "<|im_end|>\n<|im_start|>assistant\n")
    }

    /// The candidate cut to the text part of `budget`.
    ///
    /// Same policy as the text backend: parent context goes first, and only
    /// then is the candidate's own text shortened — the other order keeps the
    /// surrounding document and drops the passage that matched. Vision tokens
    /// are budgeted separately, by ``Configuration/maximumImagesPerPair``.
    static func fittedPair(
        query: RerankQuery, candidate: RerankCandidate, budget: Int, tokenizer: any Tokenizer
    ) -> (
        candidate: RerankCandidate, includesContext: Bool, truncated: Bool, droppedContext: Bool
    ) {
        func count(_ text: String) -> Int {
            tokenizer.encode(text: text, addSpecialTokens: false).count
        }
        let scaffolding = count(systemPrefix) + count(assistantSuffix)
        let available = max(1, budget - scaffolding)

        let full = pair(query: query, candidate: candidate, includeContext: true)
        if count(full) <= available { return (candidate, true, false, false) }

        let droppedContext = candidate.parentContext != nil
        let bare = pair(query: query, candidate: candidate, includeContext: false)
        if count(bare) <= available { return (candidate, false, false, droppedContext) }

        var shortened = candidate
        shortened.text = ""
        let frame = count(pair(query: query, candidate: shortened, includeContext: false))
        let textTokens = tokenizer.encode(text: candidate.text, addSpecialTokens: false)
        let kept = max(0, min(textTokens.count, available - frame))
        shortened.text = tokenizer.decode(tokenIds: Array(textTokens.prefix(kept)))
        return (shortened, false, true, droppedContext)
    }

    private static func yesMinusNo(
        logits: MLXArray, mask: MLXArray?, yesToken: Int, noToken: Int
    ) throws -> Float {
        eval(logits)
        let shape = logits.shape
        switch shape.count {
        case 2:
            guard shape[0] > 0 else {
                throw SemanticSearchError.modelUnavailable(
                    "unexpected reranker logits shape \(shape)", underlying: nil)
            }
            return sigmoid(
                logits[0, yesToken].item(Float.self) - logits[0, noToken].item(Float.self))
        case 3:
            guard shape[0] > 0, shape[1] > 0 else {
                throw SemanticSearchError.modelUnavailable(
                    "unexpected reranker logits shape \(shape)", underlying: nil)
            }
            let position = MLXEmbeddingProvider.lastValidIndex(
                mask: mask, sequenceLength: shape[1])
            return sigmoid(
                logits[0, position, yesToken].item(Float.self)
                    - logits[0, position, noToken].item(Float.self))
        default:
            throw SemanticSearchError.modelUnavailable(
                "unexpected reranker logits shape \(shape)", underlying: nil)
        }
    }

    // MARK: - Prompting

    private static let systemInstruction =
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        + "provided. Note that the answer can only be \"yes\" or \"no\"."

    private static let systemPrefix =
        "<|im_start|>system\n" + systemInstruction + "<|im_end|>\n<|im_start|>user\n"

    private static let assistantSuffix =
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    /// The document side of a pair: title, kind, bounded parent context and
    /// the candidate's own text.
    static func documentText(_ candidate: RerankCandidate, includeContext: Bool) -> String {
        var document = ""
        if let title = candidate.title, !title.isEmpty { document += "Title: \(title)\n" }
        switch candidate.kind {
        case .soundEvent:
            // A sound event's text is a class name — "glass clink" — which
            // alone reads as a phrase about glass; said to be a sound heard in
            // the recording, it is judged as one.
            document += "Source: a sound heard in the recording\n"
        case .metadata:
            // A file's name, folders, tags and place say what the file is,
            // not what is in it.
            document += "Source: the file's name, folders, tags and place\n"
        default:
            document += "Source: \(candidate.kind.rawValue)\n"
        }
        if includeContext, let context = candidate.parentContext, !context.isEmpty {
            document += "Context: \(context)\n"
        }
        document += "Text: \(candidate.text)"
        return document
    }

    static func pair(query: RerankQuery, candidate: RerankCandidate, includeContext: Bool)
        -> String
    {
        let document = documentText(candidate, includeContext: includeContext)
        let queryText =
            query.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "[image query]" : query.text
        return "<Instruct>: \(query.instruction)\n<Query>: \(queryText)\n<Document>: \(document)"
    }

    private static func tokenID(_ candidates: [String], convert: (String) -> Int?) -> Int? {
        for token in candidates {
            if let id = convert(token) { return id }
        }
        return nil
    }

    /// Numerically stable on both signs; the naive form overflows for large
    /// negative logits.
    static func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            return Float(1 / (1 + Foundation.exp(Double(-value))))
        }
        let exponential = Foundation.exp(Double(value))
        return Float(exponential / (1 + exponential))
    }
}

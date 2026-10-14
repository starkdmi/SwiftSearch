import Foundation
import SemanticSearch
import SemanticSearchMLX
import XCTest

/// EmbeddingGemma 2 on MLX: the provider's contract, its tokens against the
/// reference tokenizer, and one space for text and pixels.
///
/// The expected token ids are what Hugging Face `tokenizers` produces from the
/// checkpoint's own `tokenizer.json` with special tokens added, so these tests
/// also hold swift-transformers to the reference on Gemma's 262k vocabulary —
/// a tokenizer that splits differently gives every vector a different input.
///
/// Each test loads its own provider and releases it before returning: the
/// text tower alone is ~0.54 GB at bf16, with vision ~0.88 GB, every tower
/// ~1.5 GB, and two copies resident at once is what a 16 GB machine cannot
/// afford.
final class EmbeddingGemmaIntegrationTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(
            Integration.runsEmbeddingGemmaTests,
            "set SEMANTIC_SEARCH_RUN_EMBEDDINGGEMMA_TESTS=1 to run the EmbeddingGemma 2 tests")
    }

    private func makeProvider(
        vision: Bool = true,
        audio: Bool = false,
        maximumInputTokens: Int = ModelProfile.EmbeddingGemma.maximumInputTokens
    ) -> EmbeddingGemma2Provider {
        EmbeddingGemma2Provider(
            configuration: EmbeddingGemma2Provider.Configuration(
                model: Integration.embeddingGemmaModel,
                vision: vision,
                audio: audio,
                maximumInputTokens: maximumInputTokens))
    }

    /// Prepare, run, release — released before the next test loads its copy,
    /// not in a detached task that may run after it.
    private func withPrepared<T: Sendable>(
        _ provider: EmbeddingGemma2Provider,
        _ body: (EmbeddingGemma2Provider) async throws -> T
    ) async throws -> T {
        try await provider.prepare()
        do {
            let result = try await body(provider)
            await provider.release()
            return result
        } catch {
            await provider.release()
            throw error
        }
    }

    private func library(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(
                forResource: (name as NSString).deletingPathExtension,
                withExtension: (name as NSString).pathExtension,
                subdirectory: "Fixtures/library")
        else { throw XCTSkip("fixture library/\(name) is not in the test bundle") }
        return url
    }

    // MARK: - Vectors

    func testVectorsAreUnitLengthAt768() async throws {
        try await withPrepared(makeProvider(vision: false)) { provider in
            let outputs = try await provider.embed([
                EmbeddingInput(
                    text: "What causes the northern lights?", role: .query(instruction: nil)),
                EmbeddingInput(
                    text: "The northern lights are caused by charged particles from the sun."),
                EmbeddingInput(text: "Emperor penguins huddle together through the winter."),
            ])
            XCTAssertEqual(outputs.count, 3)
            for output in outputs {
                XCTAssertEqual(output.vector.count, ModelProfile.EmbeddingGemma.dimensions)
                XCTAssertEqual(VectorMath.l2Norm(output.vector), 1, accuracy: 1e-3)
                XCTAssertFalse(output.wasTruncated)
            }
            XCTAssertGreaterThan(
                VectorMath.dot(outputs[0].vector, outputs[1].vector),
                VectorMath.dot(outputs[0].vector, outputs[2].vector),
                "the answering passage is not nearer the question than an unrelated one")
        }
    }

    /// The query prompt has to differ from the document prompt, or the
    /// asymmetric retrieval the model was trained for is not happening.
    func testQueryAndDocumentPromptsDiffer() async throws {
        try await withPrepared(makeProvider(vision: false)) { provider in
            let text = "arctic tern migration"
            let document = try await provider.embed(
                EmbeddingInput(text: text, role: .document)
            ).vector
            let query = try await provider.embed(
                EmbeddingInput(text: text, role: .query(instruction: nil))
            ).vector
            XCTAssertLessThan(
                VectorMath.dot(document, query), 0.9999,
                "the prompts are not reaching the model")
        }
    }

    /// A trained width is the full vector's prefix, renormalised; any other
    /// width is refused rather than cut.
    func testMatryoshkaWidthsAreRenormalisedPrefixes() async throws {
        try await withPrepared(makeProvider(vision: false)) { provider in
            let text = "Matryoshka representation learning nests smaller embeddings"
            let full = try await provider.embed(EmbeddingInput(text: text)).vector
            for width in [128, 256, 512] {
                let reduced = try await provider.embed(
                    EmbeddingInput(text: text, requestedDimensions: width)
                ).vector
                XCTAssertEqual(reduced.count, width)
                XCTAssertEqual(VectorMath.l2Norm(reduced), 1, accuracy: 1e-3)
                let prefix = VectorMath.normalizedPrefix(full, dimensions: width)
                XCTAssertGreaterThan(
                    VectorMath.dot(reduced, prefix), 0.9999,
                    "\(width) is not the renormalised prefix")
            }
            do {
                _ = try await provider.embed(
                    EmbeddingInput(text: text, requestedDimensions: 300))
                XCTFail("a width the model was not trained at was accepted")
            } catch let error as SemanticSearchError {
                guard case .configuration = error else {
                    return XCTFail("expected a configuration error, got \(error)")
                }
            }
        }
    }

    /// A vector must not depend on what else shared its batch: padding is
    /// masked out of attention and out of the mean.
    func testBatchedAndSingleEmbeddingsAgree() async throws {
        let texts = [
            "short",
            "a considerably longer passage about the migration of arctic terns across two "
                + "hemispheres, with enough words to pad every other row in the batch",
            "medium length text about hydrothermal vents",
        ]
        try await withPrepared(makeProvider(vision: false)) { provider in
            var singles: [[Float]] = []
            for text in texts {
                singles.append(try await provider.embed(EmbeddingInput(text: text)).vector)
            }
            let batched = try await provider.embed(texts.map { EmbeddingInput(text: $0) })
            XCTAssertEqual(batched.count, texts.count)
            for (index, output) in batched.enumerated() {
                XCTAssertGreaterThanOrEqual(
                    VectorMath.dot(singles[index], output.vector), 0.9999,
                    "batched and single embeddings diverged for \(texts[index].prefix(24))…")
            }
        }
    }

    // MARK: - Identity

    /// The descriptor is fixed before anything loads and does not move on
    /// `prepare` or `release`, and a text-only load describes the same space
    /// as the full one. Capabilities do not move with residency either.
    func testDescriptorAndCapabilitiesDoNotDependOnLoadStateOrTowers() async throws {
        let full = makeProvider(vision: true)
        let textOnly = makeProvider(vision: false)

        let unloaded = await full.descriptor
        let capabilitiesUnloaded = await full.capabilities
        try await full.prepare()
        let loaded = await full.descriptor
        let capabilitiesLoaded = await full.capabilities
        await full.release()
        let released = await full.descriptor
        let capabilitiesReleased = await full.capabilities

        XCTAssertEqual(unloaded.fingerprint, loaded.fingerprint)
        XCTAssertEqual(unloaded.fingerprint, released.fingerprint)
        XCTAssertEqual(capabilitiesUnloaded, capabilitiesLoaded)
        XCTAssertEqual(capabilitiesUnloaded, capabilitiesReleased)

        let textDescriptor = await textOnly.descriptor
        XCTAssertEqual(
            textDescriptor.fingerprint, unloaded.fingerprint,
            "a text-only load embeds text with the same weights; its space is the full one's")

        XCTAssertEqual(unloaded.pooling, "mean")
        XCTAssertTrue(unloaded.normalized)
        XCTAssertEqual(unloaded.dimensions, ModelProfile.EmbeddingGemma.dimensions)
        XCTAssertEqual(unloaded.documentInstruction, EmbeddingGemma2Provider.documentPrefix)

        XCTAssertTrue(capabilitiesUnloaded.acceptsImages)
        XCTAssertTrue(capabilitiesUnloaded.acceptsVideo)
        XCTAssertTrue(capabilitiesUnloaded.sharedTextVisualSpace)
        XCTAssertFalse(capabilitiesUnloaded.acceptsAudio)
        XCTAssertTrue(capabilitiesUnloaded.supportsDimensionReduction)
        XCTAssertEqual(
            capabilitiesUnloaded.supportedDimensions,
            ModelProfile.EmbeddingGemma.matryoshkaDimensions)

        let textCapabilities = await textOnly.capabilities
        XCTAssertTrue(textCapabilities.acceptsText)
        XCTAssertFalse(textCapabilities.acceptsImages)
        XCTAssertFalse(textCapabilities.acceptsVideo)
        XCTAssertFalse(textCapabilities.sharedTextVisualSpace)
    }

    // MARK: - Tokens

    /// `<bos>`, the prompt and the text tokenised as one string, `<eos>` — the
    /// ids Hugging Face `tokenizers` gives for the same strings.
    func testTokenIDsMatchTheReferenceTokenizer() async throws {
        try await withPrepared(makeProvider(vision: false)) { provider in
            // "task: search result | query: arctic tern migration"
            let query = try await provider.tokenIDs(
                EmbeddingInput(text: "arctic tern migration", role: .query(instruction: nil)))
            XCTAssertEqual(
                query,
                [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 108197, 49588, 16064, 1])

            // A document that begins "title:" is text, not a prompt: it still
            // gets the document prefix ("title: none | text: title: Notes").
            let document = try await provider.tokenIDs(
                EmbeddingInput(text: "title: Notes", role: .document))
            XCTAssertEqual(
                document, [2, 3250, 236787, 7293, 1109, 1816, 236787, 3822, 236787, 24118, 1])

            // A task name and the whole prompt it stands for are the same input
            // ("task: question answering | query: x").
            let named = try await provider.tokenIDs(
                EmbeddingInput(text: "x", role: .query(instruction: "question answering")))
            let verbatim = try await provider.tokenIDs(
                EmbeddingInput(
                    text: "x", role: .query(instruction: "task: question answering | query:")))
            XCTAssertEqual(named, [2, 8071, 236787, 2934, 38020, 1109, 7609, 236787, 1123, 1])
            XCTAssertEqual(verbatim, named)

            // `<|image|>` (258880) typed in text is padding, as the reference
            // embeds it.
            let placeholder = try await provider.tokenIDs(
                EmbeddingInput(text: "hello <|image|> world", role: .document))
            XCTAssertEqual(
                placeholder,
                [2, 3250, 236787, 7293, 1109, 1816, 236787, 29104, 236743, 0, 1902, 1])
        }
    }

    /// Long text keeps its head and `<eos>`, consumes exactly the budget, and
    /// says it was cut.
    func testLongTextIsTruncatedKeepingEndOfSequence() async throws {
        let provider = makeProvider(vision: false, maximumInputTokens: 64)
        try await withPrepared(provider) { provider in
            let long = EmbeddingInput(
                text: String(repeating: "A sentence about oceanography. ", count: 40))
            let tokens = try await provider.tokenIDs(long)
            XCTAssertEqual(tokens.count, 64)
            XCTAssertEqual(tokens.first, 2)
            XCTAssertEqual(tokens.last, 1, "truncation dropped <eos>")

            let output = try await provider.embed(long)
            XCTAssertTrue(output.wasTruncated)
            XCTAssertEqual(output.consumedTokens, 64)
            XCTAssertEqual(output.vector.count, ModelProfile.EmbeddingGemma.dimensions)

            let short = try await provider.embed(
                EmbeddingInput(text: "A sentence about oceanography."))
            XCTAssertFalse(short.wasTruncated)
        }
    }

    /// The Qwen provider would load this checkpoint and embed it with the
    /// wrong prompts, pooling and normalisation. It refuses and names the
    /// provider that is right.
    func testMLXEmbeddingProviderRefusesTheCheckpoint() async throws {
        let provider = MLXEmbeddingProvider(
            configuration: MLXEmbeddingProvider.Configuration(
                model: Integration.embeddingGemmaModel,
                dimensions: ModelProfile.EmbeddingGemma.dimensions))
        do {
            try await provider.prepare()
            await provider.release()
            XCTFail("an EmbeddingGemma 2 checkpoint loaded as a Qwen3-style embedder")
        } catch let error as SemanticSearchError {
            guard case .modelUnavailable = error else {
                return XCTFail("expected modelUnavailable, got \(error)")
            }
            XCTAssertTrue("\(error)".contains("EmbeddingGemma2Provider"), "\(error)")
        }
    }

    // MARK: - Media

    /// A picture lands nearer the words that describe it than unrelated
    /// words, whether it arrives as a file or as bytes.
    func testImageIsNearestToItsOwnDescription() async throws {
        let url = try library("bramble-recipe.png")
        let bytes = try Data(contentsOf: url)
        try await withPrepared(makeProvider(vision: true)) { provider in
            let image = try await provider.embed(EmbeddingInput(images: [.fileURL(url)]))
            XCTAssertEqual(image.vector.count, ModelProfile.EmbeddingGemma.dimensions)
            XCTAssertEqual(VectorMath.l2Norm(image.vector), 1, accuracy: 1e-3)

            let encoded = try await provider.embed(
                EmbeddingInput(images: [.encoded(bytes, contentType: "public.png")]))
            XCTAssertGreaterThanOrEqual(
                VectorMath.dot(image.vector, encoded.vector), 0.9999,
                "the same picture as a file and as bytes")

            let matching = try await provider.embed(
                EmbeddingInput(
                    text: "The Bramble: a gin cocktail recipe with lemon juice and "
                        + "blackberry liqueur",
                    role: .query(instruction: nil))
            ).vector
            let unrelated = try await provider.embed(
                EmbeddingInput(
                    text: "Emperor penguins huddling together through the antarctic winter",
                    role: .query(instruction: nil))
            ).vector
            XCTAssertGreaterThan(
                VectorMath.dot(image.vector, matching), VectorMath.dot(image.vector, unrelated))

            // Text and a picture make one vector, longer than the picture's
            // soft tokens alone.
            let both = try await provider.embed(
                EmbeddingInput(text: "Bar handbook, page 14", images: [.fileURL(url)]))
            XCTAssertEqual(both.vector.count, ModelProfile.EmbeddingGemma.dimensions)
            XCTAssertEqual(VectorMath.l2Norm(both.vector), 1, accuracy: 1e-3)
            XCTAssertGreaterThan(both.consumedTokens ?? 0, image.consumedTokens ?? 0)
        }
    }

    /// Dropping the vision tower leaves the text path untouched: the same
    /// text gives the same vector, and a picture is refused with an empty
    /// vector rather than embedded as something else.
    func testTextOnlyLoadEmbedsTextLikeTheFullLoad() async throws {
        let inputs = [
            EmbeddingInput(
                text: "What causes the northern lights?", role: .query(instruction: nil)),
            EmbeddingInput(
                text: "The northern lights are caused by charged particles from the sun."),
        ]
        let picture = try library("bramble-recipe.png")

        let full = try await withPrepared(makeProvider(vision: true)) { provider -> [[Float]] in
            try await provider.embed(inputs).map(\.vector)
        }
        let textOnly = try await withPrepared(makeProvider(vision: false)) {
            provider -> [[Float]] in
            let refused = try await provider.embed(EmbeddingInput(images: [.fileURL(picture)]))
            XCTAssertTrue(refused.vector.isEmpty, "a text-only provider embedded a picture")
            return try await provider.embed(inputs).map(\.vector)
        }

        XCTAssertEqual(full.count, inputs.count)
        XCTAssertEqual(textOnly.count, inputs.count)
        for (lhs, rhs) in zip(full, textOnly) {
            XCTAssertGreaterThanOrEqual(VectorMath.dot(lhs, rhs), 0.9999)
        }
    }

    /// Audio is a side channel: refused without the audio tower, embedded
    /// unprompted with it.
    func testAudioIsEmbeddedOnlyWhenConfigured() async throws {
        let recording = try Integration.synthesisedAudio(seconds: 5)
        defer { try? FileManager.default.removeItem(at: recording.deletingLastPathComponent()) }

        try await withPrepared(makeProvider(vision: false)) { provider in
            let capabilities = await provider.capabilities
            XCTAssertFalse(capabilities.acceptsAudio)
            do {
                _ = try await provider.embedAudio(recording)
                XCTFail("a provider without the audio tower embedded a recording")
            } catch let error as SemanticSearchError {
                guard case .unsupportedModality = error else {
                    return XCTFail("expected unsupportedModality, got \(error)")
                }
            }
        }

        try await withPrepared(makeProvider(vision: false, audio: true)) { provider in
            let output = try await provider.embedAudio(recording)
            XCTAssertEqual(output.vector.count, ModelProfile.EmbeddingGemma.dimensions)
            XCTAssertEqual(VectorMath.l2Norm(output.vector), 1, accuracy: 1e-3)
            XCTAssertFalse(
                output.wasTruncated, "a 5-second recording fits the 30-second window")
        }
    }
}

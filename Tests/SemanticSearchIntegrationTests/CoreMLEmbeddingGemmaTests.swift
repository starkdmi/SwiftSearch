import Foundation
import SemanticSearch
import SemanticSearchCoreML
import SemanticSearchMLX
import XCTest

/// EmbeddingGemma 2 on Core ML: FluidInference's fp16 conversion through the
/// FluidUse runtime vendored in `Vendor/FluidUseEmbeddingGemma2`.
///
/// Gated by `SEMANTIC_SEARCH_RUN_COREML_EMBEDDINGGEMMA_TESTS`.
/// `SEMANTIC_SEARCH_COREML_EMBEDDINGGEMMA_DIRECTORY` points at a local copy of
/// the repository; without it the pinned revision is downloaded (text and
/// vision ~0.9 GB, audio another 0.6 GB). The first load compiles every
/// function for the provider's compute units (the CPU by default; minutes on
/// the Neural Engine).
///
/// The expected token ids are Hugging Face's: tokenizers 0.23.3 on
/// google/embeddinggemma-2's tokenizer.json, the file the Core ML repository
/// ships (sha256 4d777ef5…).
final class CoreMLEmbeddingGemmaTests: XCTestCase {

    private static let unavailable = "the Core ML conversion needs macOS 15 or iOS 18"

    override func setUpWithError() throws {
        try XCTSkipUnless(
            Integration.runsCoreMLEmbeddingGemmaTests,
            "set SEMANTIC_SEARCH_RUN_COREML_EMBEDDINGGEMMA_TESTS=1 to run the Core ML "
                + "EmbeddingGemma 2 tests")
    }

    @available(macOS 15, iOS 18, *)
    private func makeProvider(vision: Bool = false, audio: Bool = false)
        -> CoreMLEmbeddingGemma2Provider
    {
        CoreMLEmbeddingGemma2Provider(
            configuration: .init(
                modelDirectory: Integration.coreMLEmbeddingGemmaDirectory,
                vision: vision, audio: audio))
    }

    /// Prepare, run, release — released before the next test loads its copy,
    /// not in a task that may run after it: text and vision are ~0.9 GB and
    /// audio another 0.6, and two resident copies are what a 16 GB machine
    /// cannot afford.
    @available(macOS 15, iOS 18, *)
    private func withPrepared<T: Sendable>(
        _ provider: CoreMLEmbeddingGemma2Provider,
        _ body: (CoreMLEmbeddingGemma2Provider) async throws -> T
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

    private static func fixture(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(
                forResource: (name as NSString).deletingPathExtension,
                withExtension: (name as NSString).pathExtension,
                subdirectory: "Fixtures/library")
        else { throw XCTSkip("fixture library/\(name) is not in the test bundle") }
        return url
    }

    // MARK: - Text

    /// 768-d unit vectors, and the document and query prompts are part of the
    /// input: the same words under the two roles are two vectors.
    func testVectorsAreUnitLengthAndRolesDiffer() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        try await withPrepared(makeProvider()) { provider in
            let text = "How do lighthouses warn ships about rocks at night?"
            let outputs = try await provider.embed([
                EmbeddingInput(text: text, role: .document),
                EmbeddingInput(text: text, role: .query(instruction: nil)),
            ])
            XCTAssertEqual(outputs.count, 2)
            for output in outputs {
                XCTAssertEqual(output.vector.count, 768)
                XCTAssertEqual(VectorMath.l2Norm(output.vector), 1, accuracy: 1e-4)
                XCTAssertFalse(output.wasTruncated)
            }
            let similarity = VectorMath.dot(outputs[0].vector, outputs[1].vector)
            XCTAssertLessThan(
                similarity, 0.9999, "the document and query prompts made one vector")

            // Packing is off by default, so a text's vector does not depend on
            // what it was batched with.
            let alone = try await provider.embed(EmbeddingInput(text: text, role: .document))
            XCTAssertGreaterThan(VectorMath.dot(alone.vector, outputs[0].vector), 0.99999)
        }
    }

    /// The ids fed are the reference tokenizer's for the prompt and the text
    /// as one string, between `<bos>` (2) and `<eos>` (1).
    func testTokenIDsAreTheReferenceTokenizers() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        try await withPrepared(makeProvider()) { provider in
            let document = try await provider.tokenIDs(
                EmbeddingInput(text: "hello", role: .document))
            XCTAssertEqual(document, [2, 3250, 236787, 7293, 1109, 1816, 236787, 29104, 1])
            let query = try await provider.tokenIDs(
                EmbeddingInput(text: "hello", role: .query(instruction: nil)))
            XCTAssertEqual(query, [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 29104, 1])

            // The prompts themselves: "title: none | text:" and
            // "task: search result | query:", their last space joined to the word.
            XCTAssertEqual(
                Array(document.dropFirst().prefix(6)), [3250, 236787, 7293, 1109, 1816, 236787])
            XCTAssertEqual(
                Array(query.dropFirst().prefix(7)),
                [8071, 236787, 3927, 1354, 1109, 7609, 236787])
            for ids in [document, query] {
                XCTAssertEqual(ids.first, 2)
                XCTAssertEqual(ids.last, 1)
            }

            // Special tokens typed in text become their ids, as Hugging Face splits
            // them out; <|image|> (258880 there) is fed as <pad>, the row the
            // reference model reads for it.
            let special = try await provider.tokenIDs(
                EmbeddingInput(text: "a<|image|>b <eos>c<bos>", role: .document))
            XCTAssertEqual(
                special,
                [
                    2, 3250, 236787, 7293, 1109, 1816, 236787, 496, 0, 236763, 236743, 1,
                    236755, 2, 1,
                ])

            // A task by name and the same task written as a whole prompt are one
            // prompt.
            let named = try await provider.tokenIDs(
                EmbeddingInput(text: "hello", role: .query(instruction: "question answering")))
            let written = try await provider.tokenIDs(
                EmbeddingInput(
                    text: "hello",
                    role: .query(instruction: "task: question answering | query: ")))
            XCTAssertEqual(named, written)
            XCTAssertNotEqual(named, query)
        }
    }

    /// 609 tokens with the prompt: the conversion's 512 are embedded — `<bos>`,
    /// the first 510, `<eos>` — and the cut is reported.
    func testLongTextIsCutToTheConversionsLimitAndReported() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        try await withPrepared(makeProvider()) { provider in
            let text = String(
                repeating: "The quick brown fox jumps over the lazy dog. ", count: 60)
            let long = try await provider.embed(EmbeddingInput(text: text, role: .document))
            XCTAssertTrue(long.wasTruncated)
            XCTAssertEqual(long.consumedTokens, 512)
            XCTAssertEqual(long.vector.count, 768)
            let ids = try await provider.tokenIDs(EmbeddingInput(text: text, role: .document))
            XCTAssertEqual(ids.count, 512)
            XCTAssertEqual(ids.first, 2)
            XCTAssertEqual(ids.last, 1)

            let short = try await provider.embed(
                EmbeddingInput(text: "The quick brown fox", role: .document))
            XCTAssertFalse(short.wasTruncated)
            XCTAssertEqual(short.consumedTokens, 12)
        }
    }

    /// A Matryoshka width is the full vector's prefix, renormalised; a width
    /// the model was not trained at is refused.
    func testMatryoshkaPrefixIsRenormalised() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        try await withPrepared(makeProvider()) { provider in
            let text = "Kelp forests grow fastest in cold, nutrient-rich water."
            let full = try await provider.embed(EmbeddingInput(text: text, role: .document))
            let reduced = try await provider.embed(
                EmbeddingInput(text: text, role: .document, requestedDimensions: 256))
            XCTAssertEqual(reduced.vector.count, 256)
            XCTAssertEqual(VectorMath.l2Norm(reduced.vector), 1, accuracy: 1e-4)
            let expected = VectorMath.normalized(Array(full.vector.prefix(256)))
            for (value, reference) in zip(reduced.vector, expected) {
                XCTAssertEqual(value, reference, accuracy: 1e-4)
            }

            do {
                _ = try await provider.embed(
                    EmbeddingInput(text: text, role: .document, requestedDimensions: 300))
                XCTFail("a width the model was not trained at was accepted")
            } catch SemanticSearchError.dimensionReductionUnsupported {
            }
        }
    }

    // MARK: - Identity

    /// The same model as the MLX provider, other numerics: the spaces must not
    /// be comparable. Read before anything loads, as the engine reads them.
    func testFingerprintIsNotTheMLXProviders() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        let coreML = makeProvider(vision: true)
        let mlx = ModelProfile.EmbeddingGemma.embeddingProvider()
        let descriptor = await coreML.descriptor
        let mlxDescriptor = await mlx.descriptor
        XCTAssertNotEqual(descriptor.fingerprint, mlxDescriptor.fingerprint)
        XCTAssertFalse(descriptor.isCompatible(with: mlxDescriptor))
        XCTAssertEqual(descriptor.modelIdentity, "google/embeddinggemma-2")
        XCTAssertEqual(
            descriptor.weightsVariant, "FluidInference/embeddinggemma-2-coreml@3f978ddf:fp16")
        XCTAssertEqual(descriptor.dimensions, 768)
        XCTAssertEqual(descriptor.pooling, "mean")
        XCTAssertEqual(
            descriptor.documentInstruction, CoreMLEmbeddingGemma2Provider.documentPrefix)

        // Loading the vision encoder or not leaves text in one space; another
        // image budget is another space.
        let textOnly = await makeProvider(vision: false).descriptor
        XCTAssertEqual(textOnly.fingerprint, descriptor.fingerprint)
        let smaller = await CoreMLEmbeddingGemma2Provider(
            configuration: .init(imageTokenBudget: 140)
        ).descriptor
        XCTAssertNotEqual(smaller.fingerprint, descriptor.fingerprint)

        let capabilities = await coreML.capabilities
        XCTAssertTrue(capabilities.acceptsImages)
        XCTAssertTrue(capabilities.sharedTextVisualSpace)
        XCTAssertFalse(capabilities.acceptsVideo)
        XCTAssertFalse(capabilities.acceptsAudio)
        XCTAssertEqual(capabilities.maximumInputTokens, 512)
        XCTAssertEqual(capabilities.supportedDimensions, [128, 256, 512, 768])
    }

    // MARK: - Media

    /// Pictures land in the text space: each page is nearer the caption that
    /// describes it than the one that describes the other. An image goes
    /// unprompted and fills the sequence the reference builds for it.
    func testImageIsNearerItsCaption() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        let recipe = try Self.fixture("bramble-recipe.png")
        let timetable = try Self.fixture("oxford-timetable.png")
        try await withPrepared(makeProvider(vision: true)) { provider in
            let outputs = try await provider.embed([
                EmbeddingInput(images: [.fileURL(recipe)], role: .document),
                EmbeddingInput(images: [.fileURL(timetable)], role: .document),
                EmbeddingInput(
                    text: "a cocktail recipe with gin, lemon juice and blackberry liqueur",
                    role: .query(instruction: nil)),
                EmbeddingInput(
                    text: "a railway timetable of trains between Oxford and London",
                    role: .query(instruction: nil)),
            ])
            for output in outputs {
                XCTAssertEqual(output.vector.count, 768)
                XCTAssertEqual(VectorMath.l2Norm(output.vector), 1, accuracy: 1e-4)
            }
            let recipeCocktail = VectorMath.dot(outputs[0].vector, outputs[2].vector)
            let recipeTrains = VectorMath.dot(outputs[0].vector, outputs[3].vector)
            let timetableTrains = VectorMath.dot(outputs[1].vector, outputs[3].vector)
            let timetableCocktail = VectorMath.dot(outputs[1].vector, outputs[2].vector)
            print(
                "[coreml-eg2] recipe: \(recipeCocktail) vs \(recipeTrains); "
                    + "timetable: \(timetableTrains) vs \(timetableCocktail)")
            XCTAssertGreaterThan(recipeCocktail, recipeTrains)
            XCTAssertGreaterThan(timetableTrains, timetableCocktail)

            // 850 x 1100 at 280 tokens is resized to 672 x 912: 42 x 57 patches,
            // 266 soft tokens, as transformers' get_aspect_ratio_preserving_size.
            XCTAssertEqual(outputs[0].consumedTokens, 270)
            let ids = try await provider.tokenIDs(
                EmbeddingInput(images: [.fileURL(recipe)], role: .document))
            XCTAssertEqual(ids.count, 270)
            XCTAssertEqual(Array(ids.prefix(2)), [2, 255_999])
            XCTAssertEqual(Array(ids.suffix(2)), [258_882, 1])
            XCTAssertEqual(ids.filter { $0 == 258_880 }.count, 266)

            // A picture with a question is one sequence: the prompted text, then
            // the image.
            let question = EmbeddingInput(
                text: "What is this cocktail called?", images: [.fileURL(recipe)],
                role: .query(instruction: nil))
            let mixed = try await provider.embed(question)
            XCTAssertEqual(mixed.vector.count, 768)
            XCTAssertEqual(mixed.consumedTokens, 283)
            let mixedIDs = try await provider.tokenIDs(question)
            let questionIDs = [
                2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 2900, 563, 672, 32238, 2760,
                236881,
            ]
            XCTAssertEqual(Array(mixedIDs.prefix(15)), questionIDs + [255_999])
        }
    }

    /// One 10 s window of a recording, through the audio encoder and the text
    /// model, is a unit vector in the same space.
    func testAudioWindowIsAUnitVector() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip(Self.unavailable) }
        let recording = try Integration.synthesisedAudio(seconds: 5)
        defer { try? FileManager.default.removeItem(at: recording.deletingLastPathComponent()) }

        // Not configured with audio: refused before anything loads.
        do {
            _ = try await makeProvider().embedAudio(recording)
            XCTFail("a provider without audio embedded a recording")
        } catch SemanticSearchError.configuration {
        }

        try await withPrepared(makeProvider(audio: true)) { provider in
            let output = try await provider.embedAudio(recording)
            XCTAssertEqual(output.vector.count, 768)
            XCTAssertEqual(VectorMath.l2Norm(output.vector), 1, accuracy: 1e-4)
            // 5 s: one window, nothing cut.
            XCTAssertFalse(output.wasTruncated)
            let tokens = try XCTUnwrap(output.consumedTokens)
            XCTAssertGreaterThan(tokens, 4)
            XCTAssertLessThanOrEqual(tokens, 254)
        }
    }
}

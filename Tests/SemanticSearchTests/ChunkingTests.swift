import Foundation
import Testing

@testable import SemanticSearch

@Suite("Chunking")
struct ChunkingTests {

    @Test("Defaults are 512 target and 64 overlap")
    func planDefaults() {
        #expect(ChunkerConfiguration.default.targetTokens == 512)
        #expect(ChunkerConfiguration.default.overlapTokens == 64)
    }

    @Test("Overlap is capped at half the target")
    func overlapIsBounded() {
        // Above half, consecutive chunks are mostly duplicates of each other:
        // the index grows and the same passage gets several votes in retrieval.
        let configuration = ChunkerConfiguration(targetTokens: 100, overlapTokens: 90)
        #expect(configuration.overlapTokens == 50)
    }

    @Test("Short text is one chunk")
    func shortTextIsNotSplit() {
        let chunks = TextChunker().chunk("A short sentence about foxes.")
        #expect(chunks.count == 1)
        #expect(chunks[0].text.contains("foxes"))
    }

    @Test("Empty text yields no chunks")
    func emptyText() {
        #expect(TextChunker().chunk("").isEmpty)
        #expect(TextChunker().chunk("   \n\n  ").isEmpty)
    }

    @Test("Character ranges point back into the source")
    func rangesAreAccurate() {
        let source = """
            First paragraph about alpha.

            Second paragraph about beta.

            Third paragraph about gamma.
            """
        let chunker = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 6, maximumTokens: 8, overlapTokens: 0, minimumTokens: 1))
        let chunks = chunker.chunk(source)
        #expect(chunks.count >= 2)

        for chunk in chunks {
            // The range has to select the text the chunk actually holds —
            // otherwise a hit highlights the wrong span of the document.
            let selected = (source as NSString).substring(
                with: NSRange(location: chunk.range.location, length: chunk.range.length))
            let body = chunk.text.dropFirst(chunk.overlapPrefixLength)
            #expect(selected.contains(body.trimmingCharacters(in: .whitespacesAndNewlines)))
        }
    }

    @Test("Ranges are ascending and do not overlap")
    func rangesAreOrdered() {
        let paragraphs = (1...20).map { "Paragraph number \($0) about a topic of interest." }
        let source = paragraphs.joined(separator: "\n\n")
        let chunker = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 20, maximumTokens: 24, overlapTokens: 4, minimumTokens: 1))
        let chunks = chunker.chunk(source)
        #expect(chunks.count > 1)

        // Overlap is excluded from a chunk's range, so concatenating the ranges
        // walks the source exactly once.
        for pair in zip(chunks, chunks.dropFirst()) {
            #expect(pair.0.range.location < pair.1.range.location)
            #expect(pair.0.range.upperBound <= pair.1.range.location)
        }
    }

    @Test("Overlap carries the previous chunk's tail forward")
    func overlapIsCarried() {
        let source = (1...30).map { "Sentence \($0) discusses a distinct subject entirely." }
            .joined(separator: " ")
        let chunker = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 30, maximumTokens: 40, overlapTokens: 10, minimumTokens: 1))
        let chunks = chunker.chunk(source)
        #expect(chunks.count > 1)
        #expect(chunks.dropFirst().allSatisfy { $0.overlapPrefixLength > 0 })
    }

    @Test("No chunk exceeds the ceiling by much")
    func chunksRespectTheCeiling() {
        let source = (1...200).map { "Word\($0)" }.joined(separator: " ")
        let configuration = ChunkerConfiguration(
            targetTokens: 40, maximumTokens: 50, overlapTokens: 5, minimumTokens: 1)
        let chunks = TextChunker(configuration: configuration).chunk(source)
        for chunk in chunks {
            // The ceiling applies to the chunk's own content; the carried
            // overlap sits on top of it by design.
            #expect(chunk.estimatedTokens <= configuration.maximumTokens + configuration.overlapTokens)
        }
    }

    @Test("One oversized paragraph splits on sentences")
    func oversizedParagraphSplitsOnSentences() {
        let sentences = (1...40).map { "This is sentence number \($0) in a very long paragraph." }
        let chunks = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 30, maximumTokens: 40, overlapTokens: 0, minimumTokens: 1)
        ).chunk(sentences.joined(separator: " "))

        #expect(chunks.count > 1)
        // Sentence-boundary splitting means every chunk ends on a full stop.
        for chunk in chunks.dropLast() {
            #expect(chunk.text.hasSuffix("."))
        }
    }

    @Test("An unbroken run splits on characters")
    func unbrokenRunHardSplits() {
        // A base64 blob, or a CJK paragraph with no punctuation. There is no
        // word boundary to back off to and cutting mid-run is correct.
        let blob = String(repeating: "A", count: 20_000)
        let chunks = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 100, maximumTokens: 100, overlapTokens: 0, minimumTokens: 1)
        ).chunk(blob)
        #expect(chunks.count > 1)
        #expect(chunks.map(\.text.count).reduce(0, +) == blob.count)
    }

    @Test("A dense-script paragraph is chunked, not dropped")
    func chineseParagraph() {
        let source = String(repeating: "这是一段关于机器学习的中文文本内容", count: 80)
        let chunks = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 80, maximumTokens: 100, overlapTokens: 8, minimumTokens: 1)
        ).chunk(source, language: "zh")
        #expect(chunks.count > 1)
        #expect(chunks.allSatisfy { !$0.text.isEmpty })
    }

    @Test("Japanese sentences split on 。")
    func japaneseSentences() {
        let source = (1...30).map { "これは\($0)番目の文章です。" }.joined()
        let chunks = TextChunker(
            configuration: ChunkerConfiguration(
                targetTokens: 40, maximumTokens: 50, overlapTokens: 0, minimumTokens: 1)
        ).chunk(source, language: "ja")
        #expect(chunks.count > 1)
    }

    @Test("A provider's token limit clamps the budget")
    func providerLimitClampsBudget() {
        // A 512-token chunk handed to a 256-token provider is a 256-token chunk
        // with its tail silently discarded.
        let clamped = ChunkerConfiguration.default.constrained(toProviderLimit: 256)
        #expect(clamped.maximumTokens == 256)
        #expect(clamped.targetTokens <= 256)
    }

    @Test("A limit above the ceiling changes nothing")
    func generousProviderLimitIsIgnored() {
        let unchanged = ChunkerConfiguration.default.constrained(toProviderLimit: 8_192)
        #expect(unchanged == ChunkerConfiguration.default)
    }

    // MARK: - Token estimation

    @Test("Dense scripts are charged one token per character")
    func denseScriptEstimation() {
        #expect(TokenEstimator.estimate("東京都") == 3)
        #expect(TokenEstimator.estimate("机器学习") == 4)
    }

    @Test("Alphabetic text is charged per word")
    func alphabeticEstimation() {
        #expect(TokenEstimator.estimate("the quick brown fox") == 4)
    }

    @Test("Empty text costs nothing")
    func emptyEstimation() {
        #expect(TokenEstimator.estimate("") == 0)
    }

    // MARK: - Processor

    @Test("The processor emits a parent and one child per chunk")
    func processorProducesHierarchy() async throws {
        let source = (1...60).map { "Paragraph \($0) covers a separate topic in detail." }
            .joined(separator: "\n\n")
        let processor = TextChunkingProcessor(
            configuration: .init(
                chunker: ChunkerConfiguration(
                    targetTokens: 30, maximumTokens: 40, overlapTokens: 4, minimumTokens: 1)))

        let items = try await processor.process(
            RawContent(id: "doc", payload: .text(source)))

        #expect(items.count > 2)
        #expect(items[0].id == "doc")
        #expect(items.dropFirst().allSatisfy { $0.parentID == "doc" })
        #expect(items.dropFirst().allSatisfy { $0.ordinal != nil })
    }

    @Test("The parent's copy of the text is context-only")
    func parentTextIsNotSearchable() async throws {
        // Indexing it as well would give every document a second vote in
        // retrieval and put the whole text in competition with the passage
        // inside it that actually matches.
        let source = (1...60).map { "Paragraph \($0) covers a separate topic in detail." }
            .joined(separator: "\n\n")
        let items = try await TextChunkingProcessor(
            configuration: .init(
                chunker: ChunkerConfiguration(targetTokens: 30, maximumTokens: 40))
        ).process(RawContent(id: "doc", payload: .text(source)))

        let parentText = items[0].representations.first { $0.kind == .summary }
        #expect(parentText != nil)
        #expect(parentText!.weight <= 0)
    }

    @Test("Short input stays a single item with no children")
    func processorKeepsShortInputWhole() async throws {
        let items = try await TextChunkingProcessor().process(
            RawContent(id: "doc", payload: .text("A brief note.")))
        #expect(items.count == 1)
        #expect(items[0].representations.count == 1)
        #expect(items[0].representations[0].kind == .originalText)
    }
}

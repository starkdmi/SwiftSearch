import Foundation
import Testing

@testable import SemanticSearch

@Suite("Markdown sections")
struct MarkdownTests {

    @Test("ATX and setext headings nest into paths; code and front matter do not")
    func sections() {
        let text = """
            ---
            title: Guide
            tags: a
            ---
            Intro line.

            # Setup ##

            Before installing.

            ## Install
            ```sh
            # not a heading
            brew install thing
            ```

            Usage
            -----
            Run it.

            Reference
            =========
            ### Flags
            """
        let sections = MarkdownSections.split(text)
        #expect(sections.map(\.headings) == [
            [], ["Setup"], ["Setup", "Install"], ["Setup", "Usage"], ["Reference"],
            ["Reference", "Flags"],
        ])
        #expect(sections.map(\.hasBody) == [true, true, true, true, false, false])
        // Ranges tile the source.
        let ns = text as NSString
        #expect(sections.first?.range.location == 0)
        #expect(sections.last.map { $0.range.upperBound } == ns.length)
        for (a, b) in zip(sections, sections.dropFirst()) {
            #expect(a.range.upperBound == b.range.location)
        }
        #expect(
            ns.substring(with: NSRange(location: sections[2].range.location, length: 10))
                == "## Install")
    }

    @Test("Only real headings: seven hashes, no space, indented code, thematic breaks")
    func notHeadings() {
        #expect(MarkdownSections.atxHeading("####### seven") == nil)
        #expect(MarkdownSections.atxHeading("#hashtag") == nil)
        #expect(MarkdownSections.atxHeading("    # indented code") == nil)
        #expect(MarkdownSections.atxHeading("   # three spaces")?.title == "three spaces")
        #expect(MarkdownSections.atxHeading("# C# notes")?.title == "C# notes")
        let text = "Paragraph\n\n---\n\nMore\n* item\n---\n"
        #expect(MarkdownSections.split(text).map(\.headings) == [[]])
    }

    static func longGuide() -> String {
        let filler = (1...40).map { "Sentence \($0) about the general subject of this tool." }
            .joined(separator: " ")
        return """
            # Tool

            \(filler)

            ## Install

            ### macOS

            \(filler)

            Use the installer package and drag the app into Applications.

            ### Linux

            \(filler)
            """
    }

    @Test("A long document is chunked per section, with its headings in front")
    func chunks() async throws {
        let text = Self.longGuide()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("guide-\(UUID().uuidString).md")
        try text.write(to: url, atomically: true, encoding: .utf8)
        let items = try await TextChunkingProcessor().process(
            RawContent(id: "guide", payload: .fileURL(url)))
        let chunks = items.dropFirst()
        #expect(chunks.count >= 3)
        let ns = text as NSString
        for chunk in chunks {
            let body = try #require(chunk.representations.first?.content.text)
            let range = try #require(chunk.locator.characterRange)
            let source = ns.substring(with: NSRange(location: range.location, length: range.length))
            // No chunk crosses a heading of the same or a higher level.
            #expect(!source.dropFirst().contains("\n### "))
            // The chunker joins paragraphs with single spaces.
            func collapsed(_ text: String) -> String {
                text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            }
            #expect(collapsed(body).hasSuffix(collapsed(source)))
        }
        let mac = try #require(chunks.first { $0.representations[0].content.text!.contains("installer") })
        #expect(mac.representations[0].content.text!.hasPrefix("Tool › Install"))

        // The same file as .txt is chunked as plain text.
        let plain = url.deletingPathExtension().appendingPathExtension("txt")
        try text.write(to: plain, atomically: true, encoding: .utf8)
        let plainItems = try await TextChunkingProcessor().process(
            RawContent(id: "guide", payload: .fileURL(plain)))
        #expect(!plainItems.dropFirst().contains { $0.representations[0].content.text!.hasPrefix("Tool ›") })
    }

    @Test("A word only the heading has finds the text under it")
    func headingFindsSection() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text", provider: HashingEmbeddingProvider(dimensions: 64),
                        kinds: [.originalText], indexKind: .exact)
                ],
                reranker: nil, processor: TextChunkingProcessor()))
        try await engine.prepare()
        _ = try await engine.ingest([
            RawContent(
                id: "guide",
                payload: .fileURL({
                    let url = FileManager.default.temporaryDirectory
                        .appendingPathComponent("guide-\(UUID().uuidString).md")
                    try? Self.longGuide().write(to: url, atomically: true, encoding: .utf8)
                    return url
                }()))
        ])
        let results = try await engine.search(
            "macOS installer", options: SearchOptions(mode: .lexicalOnly))
        let top = try #require(results.hits.first)
        #expect(top.text?.contains("Use the installer package") == true)
        try await engine.close()
    }

    @Test("Short documents stay whole")
    func shortStaysWhole() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("short-\(UUID().uuidString).md")
        try "# Title\n\nOne paragraph.\n\n## Part\n\nAnother.".write(
            to: url, atomically: true, encoding: .utf8)
        let items = try await TextChunkingProcessor().process(
            RawContent(id: "short", payload: .fileURL(url)))
        #expect(items.count == 1)
    }
}

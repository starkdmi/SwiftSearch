import Foundation
import Testing

@testable import SemanticSearch

/// Lexical retrieval across the fixture's six languages.
///
/// These run against the hashing embedder, so they test tokenisation,
/// normalisation and the lexical channel — not semantic quality, which needs a
/// real model and lives in the integration suite.
@Suite("Multilingual retrieval")
struct MultilingualTests {

    private func multilingualEngine() async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("en", "The migration of arctic terns spans the entire globe.", language: "en"),
            textItem("ru", "Миграция полярных крачек охватывает весь земной шар.", language: "ru"),
            textItem("tr", "Kuzey sumrularının göçü tüm dünyayı kapsar.", language: "tr"),
            textItem("ar", "تمتد هجرة الخرشنة القطبية عبر الكرة الأرضية بأكملها.", language: "ar"),
            textItem("zh", "北极燕鸥的迁徙横跨整个地球。", language: "zh"),
            textItem("ja", "キョクアジサシの渡りは地球全体に及びます。", language: "ja"),
        ])
        return engine
    }

    @Test("English retrieves the English document")
    func english() async throws {
        let engine = try await multilingualEngine()
        let results = try await engine.search(
            SearchQuery(text: "arctic terns migration", languageHint: "en"))
        #expect(results.hits.first?.itemID == "en")
    }

    @Test("Russian retrieves the Russian document, inflection included")
    func russian() async throws {
        let engine = try await multilingualEngine()
        // "миграция" in the document, "миграции" in the query: they have to
        // share a lemma or the lexical channel finds nothing.
        let results = try await engine.search(
            SearchQuery(text: "миграции полярных крачек", languageHint: "ru"))
        #expect(results.hits.first?.itemID == "ru")
    }

    @Test("Turkish retrieves the Turkish document across the dotted i")
    func turkish() async throws {
        let engine = try await multilingualEngine()
        let results = try await engine.search(
            SearchQuery(text: "KUZEY SUMRULARININ GÖÇÜ", languageHint: "tr"))
        #expect(results.hits.first?.itemID == "tr")
    }

    @Test("Arabic retrieves the Arabic document with vowel marks in the query")
    func arabic() async throws {
        let engine = try await multilingualEngine()
        // The corpus is unvocalised and the query is vocalised, which is the
        // normal asymmetry and the reason harakat are stripped.
        let results = try await engine.search(
            SearchQuery(text: "هِجرة الخرشنة القطبية", languageHint: "ar"))
        #expect(results.hits.first?.itemID == "ar")
    }

    @Test("A single Chinese character retrieves the Chinese document")
    func chineseSingleCharacter() async throws {
        let engine = try await multilingualEngine()
        // The extracted analyzer dropped every one-character non-digit token,
        // which made this query match nothing at all.
        let results = try await engine.search(SearchQuery(text: "鸥", languageHint: "zh"))
        #expect(results.hits.contains { $0.itemID == "zh" })
    }

    @Test("Chinese phrases retrieve the Chinese document")
    func chinese() async throws {
        let engine = try await multilingualEngine()
        let results = try await engine.search(SearchQuery(text: "北极燕鸥迁徙", languageHint: "zh"))
        #expect(results.hits.first?.itemID == "zh")
    }

    @Test("Japanese retrieves the Japanese document")
    func japanese() async throws {
        let engine = try await multilingualEngine()
        let results = try await engine.search(
            SearchQuery(text: "キョクアジサシ 渡り", languageHint: "ja"))
        #expect(results.hits.first?.itemID == "ja")
    }

    @Test("The query language is detected when no hint is given")
    func languageDetection() async throws {
        let engine = try await multilingualEngine()
        let results = try await engine.search("Миграция полярных крачек")
        #expect(results.diagnostics.detectedQueryLanguage == "ru")
    }

    @Test("Documents keep their original language; nothing is pivoted")
    func originalLanguageIsPreserved() async throws {
        // By default multilingual retrieval runs on original-language content
        // and original-language queries. No translation happens unless a query
        // transform is installed.
        let engine = try await multilingualEngine()
        let item = try #require(try await engine.item("ja"))
        #expect(item.representations[0].content.text?.contains("キョクアジサシ") == true)
        #expect(item.representations[0].language == "ja")
    }

    @Test("Query expansion is off by default")
    func expansionIsOffByDefault() async throws {
        let engine = try await multilingualEngine()
        let configuration = await engine.configuration
        #expect(configuration.queryTransform.maximumAdditionalVariants == 0)
    }

    @Test("A query transform retains the original query and is bounded")
    func queryTransformIsBounded() async throws {
        struct RunawayTransform: QueryTransform {
            let identifier = "runaway"
            let maximumAdditionalVariants = 2
            func transform(_ query: SearchQuery) async throws -> [QueryVariant] {
                // Returns far more than it promised. The engine has to hold it
                // to its own advertised bound, or an unbounded expansion gives
                // its family unbounded say in the fused ranking.
                (0..<50).map {
                    QueryVariant(query: SearchQuery("variant \($0)"), variant: "v\($0)")
                }
            }
        }

        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(), indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                queryTransform: RunawayTransform()))
        try await engine.prepare()
        try await engine.upsert([textItem("a", "the original subject of the query")])

        let results = try await engine.search("original subject")
        let variants = Set(results.diagnostics.channels.compactMap(\.channel.variant))
        #expect(variants.count <= 2)
        // The original is always kept.
        #expect(results.diagnostics.channels.contains { $0.channel.variant == nil })
    }
}

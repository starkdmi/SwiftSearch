import Foundation
import Testing

@testable import SemanticSearch

/// Tokenisation and normalisation, across the fixture's six languages.
@Suite("Lexical analysis")
struct LexicalAnalyzerTests {

    let analyzer = LinguisticLexicalAnalyzer()

    // MARK: - Lemma and surface handling

    @Test("Documents post both the lemma and the surface form")
    func documentsPostLemmaAndSurface() {
        let document = analyzer.analyzeDocument("jumped quickly", language: "en")
        #expect(document.terms.contains("jump"))
        #expect(document.terms.contains("jumped"))
        #expect(document.terms.contains("quickly"))
    }

    @Test("Document length counts primary tokens, not posted terms")
    func documentLengthExcludesSurfaceVariants() {
        // The surface variant is an extra posting for a token that was already
        // counted. Charging it to the length would make an inflected document
        // look longer than it is, and BM25 penalises length.
        let document = analyzer.analyzeDocument("jumped quickly", language: "en")
        #expect(document.tokenCount == 2)
        #expect(document.terms.count > document.tokenCount)
    }

    @Test("Queries emit primary terms only")
    func queriesAreLemmaOnly() {
        let terms = analyzer.analyzeQuery("jumped quickly", language: "en")
        #expect(terms.contains("jump"))
        #expect(!terms.contains("jumped"))
    }

    @Test("Closed-class words are dropped from queries but kept in documents")
    func stopwordsAreQuerySideOnly() {
        let terms = analyzer.analyzeQuery("the cat on the mat", language: "en")
        #expect(!terms.contains("the"))
        #expect(terms.contains("cat"))

        let document = analyzer.analyzeDocument("the cat on the mat", language: "en")
        #expect(document.tokenCount >= 5)
    }

    // MARK: - Single-character tokens

    @Test("Single CJK characters survive; single Latin letters do not")
    func singleCharacterHandling() {
        // 水 is a word. A lone "a" matches nearly every English document and
        // costs a posting list the length of the corpus.
        #expect(LinguisticLexicalAnalyzer.isUsableTerm("水"))
        #expect(LinguisticLexicalAnalyzer.isUsableTerm("日"))
        #expect(LinguisticLexicalAnalyzer.isUsableTerm("가"))
        #expect(LinguisticLexicalAnalyzer.isUsableTerm("7"))
        #expect(!LinguisticLexicalAnalyzer.isUsableTerm("a"))
        #expect(!LinguisticLexicalAnalyzer.isUsableTerm("в"))
    }

    @Test("A one-character Chinese query produces a term")
    func chineseSingleCharacterQuery() {
        let terms = analyzer.analyzeQuery("水", language: "zh")
        #expect(terms == ["水"])
    }

    @Test("Chinese text segments into terms")
    func chineseSegmentation() {
        let document = analyzer.analyzeDocument("这是一个关于机器学习的文档", language: "zh")
        #expect(!document.terms.isEmpty)
        #expect(document.tokenCount > 1)
    }

    @Test("A CJK word is also posted character by character")
    func denseScriptCharactersArePosted() {
        // 北极燕鸥 segments as one token, so a one-character query — an
        // ordinary thing to search for in Chinese — would otherwise match
        // nothing.
        let document = analyzer.analyzeDocument("北极燕鸥的迁徙", language: "zh")
        #expect(document.terms.contains("鸥"))
        #expect(document.terms.contains("北"))
    }

    @Test("Latin words are not posted letter by letter")
    func latinIsNotSplitIntoCharacters() {
        // Splitting "printing" into letters would post eight terms that match
        // every English document in the corpus.
        let document = analyzer.analyzeDocument("printing press", language: "en")
        #expect(!document.terms.contains("p"))
        #expect(!document.terms.contains("r"))
    }

    @Test("Character postings do not inflate document length")
    func characterPostingsAreNotCounted() {
        let document = analyzer.analyzeDocument("北极燕鸥", language: "zh")
        #expect(document.tokenCount < document.terms.count)
    }

    @Test("Japanese text segments into terms")
    func japaneseSegmentation() {
        let document = analyzer.analyzeDocument("東京は日本の首都です", language: "ja")
        #expect(!document.terms.isEmpty)
        #expect(document.terms.contains { $0.contains("東京") || $0 == "東京" })
    }

    // MARK: - Script normalisation

    @Test("Russian ё folds to е but й stays distinct from и")
    func cyrillicNormalization() {
        #expect(TextNormalization.normalizeToken("ёлка") == TextNormalization.normalizeToken("елка"))
        #expect(TextNormalization.normalizeToken("мой") != TextNormalization.normalizeToken("мои"))
    }

    @Test("Russian inflections share a lemma")
    func russianLemmatization() {
        let document = analyzer.analyzeDocument("собаки бегут", language: "ru")
        let query = analyzer.analyzeQuery("собака", language: "ru")
        #expect(!query.isEmpty)
        #expect(document.terms.contains(query[0]))
    }

    @Test("Turkish dotted and dotless i normalise together")
    func turkishDotlessI() {
        // `lowercased()` without a locale maps I→i and leaves ı alone, so IŞIK
        // and ışık would otherwise be different terms.
        #expect(TextNormalization.normalizeToken("IŞIK") == TextNormalization.normalizeToken("ışık"))
        #expect(TextNormalization.normalizeToken("İstanbul") == TextNormalization.normalizeToken("istanbul"))
    }

    @Test("Arabic harakat, tatweel and alef variants normalise away")
    func arabicNormalization() {
        // A query written with vowel marks has to reach a corpus written
        // without them, which is most corpora.
        #expect(TextNormalization.normalizeToken("كِتَاب") == TextNormalization.normalizeToken("كتاب"))
        #expect(TextNormalization.normalizeToken("أحمد") == TextNormalization.normalizeToken("احمد"))
        #expect(TextNormalization.normalizeToken("مـــدرسة") == TextNormalization.normalizeToken("مدرسة"))
    }

    @Test("Arabic text produces terms")
    func arabicTokenization() {
        let document = analyzer.analyzeDocument("البحث الدلالي عن النصوص", language: "ar")
        #expect(document.tokenCount >= 3)
    }

    @Test("Latin diacritics fold")
    func latinFolding() {
        #expect(TextNormalization.normalizeToken("café") == TextNormalization.normalizeToken("cafe"))
        #expect(TextNormalization.normalizeToken("naïve") == TextNormalization.normalizeToken("naive"))
    }

    // MARK: - Numbers

    @Test("Numbers keep their decimal point and lose their grouping")
    func numericNormalization() {
        #expect(TextNormalization.normalizeToken("3.14") == "3.14")
        #expect(TextNormalization.normalizeToken("3,14") == "3.14")
        #expect(TextNormalization.normalizeToken("1,234") == "1234")
        #expect(TextNormalization.normalizeToken("1.234") == "1234")
        #expect(TextNormalization.normalizeToken("+1234.50") == "1234.5")
        #expect(TextNormalization.normalizeToken("007") == "7")
    }

    @Test("A decimal does not collide with the integer of the same digits")
    func decimalsDoNotCollideWithIntegers() {
        // Stripping every non-alphanumeric character turned 3.14 into 314.
        #expect(TextNormalization.normalizeToken("3.14") != TextNormalization.normalizeToken("314"))
    }

    @Test("Numbers survive tokenisation")
    func numbersAreIndexed() {
        let document = analyzer.analyzeDocument("version 2.5 released in 2026", language: "en")
        #expect(document.terms.contains("2.5"))
        #expect(document.terms.contains("2026"))
    }

    // MARK: - Versioning

    @Test("The analyzer reports a version")
    func analyzerIsVersioned() {
        // The version is baked into stored posting lists; an unversioned
        // analyzer can change its rules and leave an index disagreeing with the
        // queries running against it.
        #expect(!analyzer.analyzerVersion.isEmpty)
    }

    @Test("Empty and whitespace-only input yields no terms")
    func emptyInput() {
        #expect(analyzer.analyzeQuery("", language: nil).isEmpty)
        #expect(analyzer.analyzeQuery("   \n\t ", language: nil).isEmpty)
        #expect(analyzer.analyzeDocument("", language: nil).terms.isEmpty)
    }

    @Test("Punctuation-only input falls back without crashing")
    func punctuationOnly() {
        let document = analyzer.analyzeDocument("!!! ??? ...", language: "en")
        #expect(document.terms.isEmpty || document.tokenCount >= 0)
    }
}

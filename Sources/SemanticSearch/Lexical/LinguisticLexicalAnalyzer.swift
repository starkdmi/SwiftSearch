import Foundation
import NaturalLanguage

/// The default analyzer, built on `NSLinguisticTagger`.
///
/// Two multilingual defects are avoided on purpose. Dropping every
/// one-character token that is not a digit is correct for Latin and Cyrillic —
/// a stray "a" or "в" is noise — and wrong for Chinese and Japanese, where 水
/// and 火 are whole words. Folding numbers into digit runs makes `3.14` and
/// `314` the same term.
///
/// What it does, in order:
///
/// 1. Tag words with `NSLinguisticTagger` for lemma and lexical class.
/// 2. On the query side, drop closed-class words (pronouns, prepositions,
///    conjunctions, determiners, particles, interjections).
/// 3. Normalise per script: Cyrillic ё→е; Latin diacritic folding plus the
///    Turkish dotted/dotless *i*; Arabic harakat, tatweel and alef variants.
/// 4. Emit the lemma as the primary term and the surface form as a secondary
///    posting when it differs — documents only, never queries.
public struct LinguisticLexicalAnalyzer: LexicalAnalyzer {

    public let analyzerVersion: String

    /// Lexical classes dropped from queries. Kept in documents: a document that
    /// loses its function words loses its length, and BM25 normalises by length.
    private static let queryIgnoredTags: Set<NSLinguisticTag> = [
        .pronoun, .preposition, .conjunction, .determiner, .particle, .interjection,
    ]

    public init() {
        // Bumped whenever tokenisation or normalisation changes. Stored indexes
        // carrying a different version are rebuilt rather than queried.
        self.analyzerVersion = "linguistic.v2"
    }

    public func analyzeDocument(_ text: String, language: String?) -> AnalyzedDocument {
        let result = tokenize(text, language: language, isQuery: false)
        return AnalyzedDocument(terms: result.expanded, tokenCount: result.primary.count)
    }

    public func analyzeQuery(_ text: String, language: String?) -> [String] {
        tokenize(text, language: language, isQuery: true).primary
    }

    // MARK: - Tokenisation

    private struct Tokenization {
        var primary: [String] = []
        var expanded: [String] = []
    }

    private func tokenize(_ text: String, language: String?, isQuery: Bool) -> Tokenization {
        let normalized = TextNormalization.collapseWhitespace(text)
        guard !normalized.isEmpty else { return Tokenization() }

        let nsText = normalized as NSString
        guard nsText.length > 0 else { return Tokenization() }

        let tagger = NSLinguisticTagger(tagSchemes: [.lemma, .lexicalClass], options: 0)
        tagger.string = normalized
        if let language, let orthography = Self.orthography(for: language) {
            // Telling the tagger the language up front is the difference between
            // correct segmentation and character-by-character splitting on a
            // short CJK query, where its own detection has nothing to go on.
            tagger.setOrthography(orthography, range: NSRange(location: 0, length: nsText.length))
        }

        var result = Tokenization()
        result.primary.reserveCapacity(16)
        result.expanded.reserveCapacity(24)

        let options: NSLinguisticTagger.Options = [.omitWhitespace, .omitPunctuation, .joinNames]
        let range = NSRange(location: 0, length: nsText.length)

        tagger.enumerateTags(in: range, unit: .word, scheme: .lemma, options: options) {
            lemmaTag, tokenRange, _ in

            if isQuery {
                let lexicalTag = tagger.tag(
                    at: tokenRange.location, scheme: .lexicalClass, tokenRange: nil,
                    sentenceRange: nil)
                if let lexicalTag, Self.queryIgnoredTags.contains(lexicalTag) { return }
            }

            let raw = nsText.substring(with: tokenRange)
            guard
                let pair = Self.terms(
                    raw: raw, lemma: lemmaTag?.rawValue, includeSurface: !isQuery)
            else { return }

            result.primary.append(pair.primary)
            result.expanded.append(pair.primary)
            if let secondary = pair.secondary { result.expanded.append(secondary) }

            if !isQuery {
                // Dense-script words are also posted character by character.
                // The tagger segments 北极燕鸥 as one token, so a one-character
                // query — which is a whole word in Chinese and a completely
                // ordinary thing to search for — would otherwise match nothing
                // at all. Posting the characters costs a few extra postings per
                // CJK word and is what makes substring retrieval work in a
                // script with no spaces to split on.
                //
                // Document length is unaffected: these are extra postings for a
                // token already counted, not extra tokens.
                result.expanded.append(contentsOf: Self.denseScriptCharacters(pair.primary))
            }
        }

        if result.primary.isEmpty {
            // The tagger emits nothing for some inputs — a bare URL, a run of
            // symbols, a script it has no orthography for. Splitting on
            // non-alphanumerics still recovers the terms.
            let fallback =
                normalized
                .replacingOccurrences(
                    of: #"[^\p{L}\p{N}]+"#, with: " ", options: .regularExpression)
                .split(separator: " ")
                .map { TextNormalization.normalizeToken(String($0)) }
                .filter { Self.isUsableTerm($0) }
            return Tokenization(primary: fallback, expanded: fallback)
        }

        return result
    }

    private static func terms(raw: String, lemma: String?, includeSurface: Bool)
        -> (primary: String, secondary: String?)?
    {
        let normalizedRaw = TextNormalization.normalizeToken(raw)
        let normalizedLemma = TextNormalization.normalizeToken(lemma ?? "")

        // The lemma is preferred, but the tagger returns lemmas for languages it
        // has models for and nothing for the rest, so the surface form is the
        // fallback rather than an error.
        let primary = isUsableTerm(normalizedLemma) ? normalizedLemma : normalizedRaw
        guard isUsableTerm(primary) else { return nil }

        guard includeSurface else { return (primary, nil) }
        if isUsableTerm(normalizedRaw), normalizedRaw != primary {
            return (primary, normalizedRaw)
        }
        return (primary, nil)
    }

    /// Whether a normalised token is worth posting.
    ///
    /// One-character terms are kept when the character carries meaning on its
    /// own — a digit, or a character from a script written without word spacing.
    /// A lone Latin or Cyrillic letter is dropped: it matches nearly every
    /// document and costs a posting list the length of the corpus.
    static func isUsableTerm(_ token: String) -> Bool {
        guard !token.isEmpty else { return false }
        guard token.count == 1 else { return true }
        guard let scalar = token.unicodeScalars.first else { return false }
        if CharacterSet.decimalDigits.contains(scalar) { return true }
        return TextNormalization.isDenseScript(scalar)
    }

    /// Individual characters of a multi-character dense-script token.
    ///
    /// Empty for anything else: splitting "printing" into letters would post
    /// eight terms that match every English document in the corpus.
    static func denseScriptCharacters(_ token: String) -> [String] {
        guard token.count > 1 else { return [] }
        var characters: [String] = []
        for scalar in token.unicodeScalars {
            guard TextNormalization.isDenseScript(scalar) else { return [] }
            characters.append(String(scalar))
        }
        return characters
    }

    private static let orthographyCache = OrthographyCache()

    private static func orthography(for language: String) -> NSOrthography? {
        orthographyCache.orthography(for: language)
    }
}

/// `NSOrthography` construction shows up in profiles of short-query search, and
/// the set of languages a process sees is tiny. One lock, one dictionary.
private final class OrthographyCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cache: [String: NSOrthography] = [:]

    func orthography(for language: String) -> NSOrthography? {
        let base = TextNormalization.baseLanguageCode(language)
        guard !base.isEmpty else { return nil }
        lock.lock()
        defer { lock.unlock() }
        if let cached = cache[base] { return cached }
        guard let script = Self.dominantScript(for: base) else { return nil }
        let orthography = NSOrthography(dominantScript: script, languageMap: [script: [base]])
        cache[base] = orthography
        return orthography
    }

    /// ISO 15924 script for the languages the package is validated against,
    /// plus the common neighbours of each. Unknown languages get no
    /// orthography, which leaves the tagger on its own detection.
    private static func dominantScript(for language: String) -> String? {
        switch language {
        case "en", "de", "fr", "es", "it", "pt", "nl", "tr", "id", "vi", "pl", "cs", "sv", "da",
            "fi", "no", "hu", "ro":
            return "Latn"
        case "ru", "uk", "bg", "sr", "be", "kk", "mk":
            return "Cyrl"
        case "ar", "fa", "ur":
            return "Arab"
        case "zh", "zh-hans", "zh-hant", "yue":
            return "Hans"
        case "ja":
            return "Jpan"
        case "ko":
            return "Kore"
        case "he":
            return "Hebr"
        case "hi", "mr", "ne":
            return "Deva"
        case "th":
            return "Thai"
        case "el":
            return "Grek"
        default:
            return nil
        }
    }
}

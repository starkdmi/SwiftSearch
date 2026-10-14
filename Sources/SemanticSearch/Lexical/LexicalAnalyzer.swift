import Foundation

/// Terms extracted from one document, plus the length BM25 should charge it.
public struct AnalyzedDocument: Sendable, Hashable {
    /// Every term to post, lemma and surface variants included. Repeats are
    /// significant: the term frequency is their count.
    public var terms: [String]
    /// Number of *primary* tokens the text contained.
    ///
    /// Not `terms.count`. Surface variants are extra postings for the same
    /// token, and charging them to the document length would make a document
    /// with heavily inflected words look longer than it is — which is exactly
    /// the quantity BM25's `b` parameter penalises.
    public var tokenCount: Int

    public init(terms: [String], tokenCount: Int) {
        self.terms = terms
        self.tokenCount = tokenCount
    }
}

/// Tokenisation and normalisation for lexical search. Replaceable.
///
/// The analyzer is versioned because its output is baked into the stored
/// posting lists. Change the rules without changing the version and the index
/// silently disagrees with the queries running against it.
public protocol LexicalAnalyzer: Sendable {
    /// Identity of the tokenisation rules. Part of the index fingerprint.
    var analyzerVersion: String { get }

    func analyzeDocument(_ text: String, language: String?) -> AnalyzedDocument

    /// Terms for the query side.
    ///
    /// Returns primary (lemma) tokens only. The document side posts lemma *and*
    /// surface, so a lemma query already reaches both forms; emitting both from
    /// the query as well would score a document twice for one word and only for
    /// words the tagger happened to inflect.
    func analyzeQuery(_ text: String, language: String?) -> [String]
}

extension LexicalAnalyzer {
    public func analyzeDocument(_ text: String) -> AnalyzedDocument {
        analyzeDocument(text, language: nil)
    }

    public func analyzeQuery(_ text: String) -> [String] {
        analyzeQuery(text, language: nil)
    }
}

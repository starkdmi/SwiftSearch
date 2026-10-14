import Foundation
import NaturalLanguage

/// One chunk of a longer text, with the offsets needed to point back at it.
public struct TextChunk: Sendable, Hashable {
    public var ordinal: Int
    public var text: String
    /// Offsets into the *original* string, UTF-16, covering `text` minus any
    /// overlap carried in from the previous chunk.
    ///
    /// Overlap is excluded so that highlighting a hit does not light up the
    /// tail of the chunk before it, and so that concatenating every chunk's
    /// range reconstructs the source exactly once.
    public var range: SourceLocator.CharacterRange
    public var estimatedTokens: Int
    /// UTF-16 characters at the start of `text` that are not this chunk's own
    /// source: overlap carried from the previous chunk and, for Markdown, the
    /// path of headings above it.
    public var overlapPrefixLength: Int

    public init(
        ordinal: Int,
        text: String,
        range: SourceLocator.CharacterRange,
        estimatedTokens: Int,
        overlapPrefixLength: Int = 0
    ) {
        self.ordinal = ordinal
        self.text = text
        self.range = range
        self.estimatedTokens = estimatedTokens
        self.overlapPrefixLength = overlapPrefixLength
    }
}

public struct ChunkerConfiguration: Sendable, Hashable, Codable {
    /// Tokens a chunk aims for.
    public var targetTokens: Int
    /// Hard ceiling. A unit is split mid-sentence rather than exceed it.
    public var maximumTokens: Int
    /// Tokens of the previous chunk repeated at the start of the next, so a
    /// fact that straddles a boundary is retrievable from either side.
    public var overlapTokens: Int
    /// Chunks below this are merged forward instead of emitted on their own.
    public var minimumTokens: Int

    public init(
        targetTokens: Int = 512,
        maximumTokens: Int = 512,
        overlapTokens: Int = 64,
        minimumTokens: Int = 24
    ) {
        self.targetTokens = max(1, targetTokens)
        self.maximumTokens = max(maximumTokens, self.targetTokens)
        // Overlap above half the target makes consecutive chunks mostly
        // duplicates of each other, which inflates the index and gives the same
        // passage several votes in retrieval.
        self.overlapTokens = max(0, min(overlapTokens, self.targetTokens / 2))
        self.minimumTokens = max(0, minimumTokens)
    }

    public static let `default` = ChunkerConfiguration()

    /// Clamp the budget to what a provider will actually accept.
    ///
    /// A 512-token chunk handed to a 256-token provider is a 256-token chunk
    /// with its second half silently discarded, and nothing downstream can tell
    /// that happened.
    public func constrained(toProviderLimit limit: Int) -> ChunkerConfiguration {
        guard limit > 0, limit < maximumTokens else { return self }
        return ChunkerConfiguration(
            targetTokens: Swift.min(targetTokens, limit),
            maximumTokens: limit,
            overlapTokens: overlapTokens,
            minimumTokens: Swift.min(minimumTokens, limit))
    }

    public var fingerprint: String {
        StableHash.of { hasher in
            hasher.append("chunker.v1")
            hasher.append(targetTokens)
            hasher.append(maximumTokens)
            hasher.append(overlapTokens)
            hasher.append(minimumTokens)
        }
    }
}

/// Paragraph- and sentence-aware chunker for plain text.
///
/// Splits on the largest boundary that fits: paragraphs first, then sentences
/// within an oversized paragraph, then words, then — for an unbroken run with
/// none of those, a base64 blob or a Chinese paragraph without punctuation —
/// characters. Every chunk carries the UTF-16 range it occupies in the source,
/// so a hit resolves to a highlightable span rather than to "somewhere in this
/// document".
public struct TextChunker: Sendable {
    public var configuration: ChunkerConfiguration

    public init(configuration: ChunkerConfiguration = .default) {
        self.configuration = configuration
    }

    public func chunk(_ text: String, language: String? = nil) -> [TextChunk] {
        let units = segments(of: text, language: language)
        guard !units.isEmpty else { return [] }

        var chunks: [TextChunk] = []
        var pending: [Segment] = []
        var pendingTokens = 0
        var carriedOverlap: String = ""

        // The overlap carried into the next chunk counts against its budget.
        // Budgeting the body alone lets overlap + body exceed the ceiling —
        // by up to `overlapTokens` on every chunk after the first — and the
        // ceiling is the provider's input limit, so the excess would be cut
        // off at embedding time.
        var overlapCost = 0

        func flush() {
            guard !pending.isEmpty else { return }
            let body = pending.map(\.text).joined(separator: " ")
            let start = pending[0].range.location
            let end = pending[pending.count - 1].range.upperBound
            // A body that alone nearly fills the ceiling leaves less room for
            // overlap than the configuration asks for; the overlap gives way,
            // never the body.
            let room = configuration.maximumTokens - TokenEstimator.estimate(body) - 1
            if !carriedOverlap.isEmpty,
                TokenEstimator.estimate(carriedOverlap) > room
            {
                carriedOverlap = room > 0 ? (Self.tail(of: carriedOverlap, tokens: room) ?? "") : ""
            }
            let full = carriedOverlap.isEmpty ? body : carriedOverlap + " " + body
            chunks.append(
                TextChunk(
                    ordinal: chunks.count,
                    text: full,
                    range: SourceLocator.CharacterRange(
                        location: start, length: max(0, end - start)),
                    estimatedTokens: TokenEstimator.estimate(full),
                    overlapPrefixLength: carriedOverlap.isEmpty
                        ? 0 : (carriedOverlap as NSString).length + 1))

            carriedOverlap = overlapTail(of: pending)
            overlapCost = carriedOverlap.isEmpty ? 0 : TokenEstimator.estimate(carriedOverlap) + 1
            pending.removeAll(keepingCapacity: true)
            pendingTokens = 0
        }

        for unit in units {
            for piece in fit(unit, language: language) {
                let pieceTokens = TokenEstimator.estimate(piece.text)
                let projected = overlapCost + pendingTokens + pieceTokens

                if !pending.isEmpty,
                    // Over the target with enough already to stand alone…
                    (projected > configuration.targetTokens
                        && pendingTokens >= configuration.minimumTokens)
                        // …or over the ceiling, which is not negotiable even
                        // for a chunk below the minimum.
                        || projected > configuration.maximumTokens
                {
                    flush()
                }

                pending.append(piece)
                pendingTokens += pieceTokens

                if overlapCost + pendingTokens >= configuration.maximumTokens { flush() }
            }
        }
        flush()

        return chunks
    }

    // MARK: - Segmentation

    private struct Segment: Sendable {
        var text: String
        var range: SourceLocator.CharacterRange
    }

    /// Paragraphs, in source order, with their offsets.
    private func segments(of text: String, language: String?) -> [Segment] {
        let nsText = text as NSString
        guard nsText.length > 0 else { return [] }

        var paragraphs: [Segment] = []
        var location = 0
        // A blank line ends a paragraph. Single newlines inside one are treated
        // as soft wraps and collapsed, which is what they are in almost every
        // plain-text source that is not poetry.
        let separator = try! NSRegularExpression(pattern: #"\n[ \t]*\n[\s]*"#)
        let matches = separator.matches(
            in: text, options: [], range: NSRange(location: 0, length: nsText.length))

        for match in matches {
            let range = NSRange(location: location, length: match.range.location - location)
            appendParagraph(nsText, range: range, to: &paragraphs)
            location = match.range.upperBound
        }
        appendParagraph(
            nsText, range: NSRange(location: location, length: nsText.length - location),
            to: &paragraphs)

        return paragraphs
    }

    private func appendParagraph(
        _ nsText: NSString, range: NSRange, to paragraphs: inout [Segment]
    ) {
        guard range.length > 0 else { return }
        let raw = nsText.substring(with: range)
        let trimmedRange = Self.trimmedRange(of: raw, offsetBy: range.location)
        guard trimmedRange.length > 0 else { return }
        // Verbatim, not whitespace-collapsed. Every offset below is derived
        // from this string, and collapsing would shorten it relative to the
        // source — after which a chunk's reported range no longer covers the
        // text it holds, which is the one thing a locator has to get right.
        let body = nsText.substring(with: trimmedRange)
        guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        paragraphs.append(
            Segment(
                text: body,
                range: SourceLocator.CharacterRange(
                    location: trimmedRange.location, length: trimmedRange.length)))
    }

    /// Split a segment that is too long for the ceiling.
    private func fit(_ segment: Segment, language: String?) -> [Segment] {
        guard TokenEstimator.estimate(segment.text) > configuration.maximumTokens else {
            return [segment]
        }

        var pieces: [Segment] = []
        for sentence in sentences(of: segment, language: language) {
            if TokenEstimator.estimate(sentence.text) > configuration.maximumTokens {
                pieces.append(contentsOf: hardSplit(sentence))
            } else {
                pieces.append(sentence)
            }
        }
        return pieces.isEmpty ? [segment] : pieces
    }

    /// Sentences via `NLTokenizer`, which knows that 。 ends a Japanese
    /// sentence and that `Dr.` does not end an English one.
    private func sentences(of segment: Segment, language: String?) -> [Segment] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = segment.text
        if let language, let nlLanguage = Self.language(from: language) {
            tokenizer.setLanguage(nlLanguage)
        }

        let nsText = segment.text as NSString
        var output: [Segment] = []
        tokenizer.enumerateTokens(in: segment.text.startIndex..<segment.text.endIndex) {
            range, _ in
            let nsRange = NSRange(range, in: segment.text)
            let piece = nsText.substring(with: nsRange).trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard !piece.isEmpty else { return true }
            output.append(
                Segment(
                    text: piece,
                    range: SourceLocator.CharacterRange(
                        location: segment.range.location + nsRange.location,
                        length: nsRange.length)))
            return true
        }
        return output.isEmpty ? [segment] : output
    }

    /// Last resort for one "sentence" longer than the ceiling.
    ///
    /// Splits on word boundaries where there are any, and on characters where
    /// there are none — the case that matters for CJK, where a paragraph can be
    /// a single unbroken run.
    private func hardSplit(_ segment: Segment) -> [Segment] {
        let nsText = segment.text as NSString
        guard nsText.length > 0 else { return [segment] }

        // Cut positions are derived from a measured characters-per-token ratio
        // rather than by re-estimating a growing prefix on every character,
        // which is quadratic and shows up immediately on the kind of input that
        // reaches this path at all — a 200 KB unbroken run.
        let totalTokens = max(1, TokenEstimator.estimate(segment.text))
        let charactersPerToken = max(1, nsText.length / totalTokens)
        let stride = max(16, configuration.maximumTokens * charactersPerToken)

        var pieces: [Segment] = []
        var start = 0

        while start < nsText.length {
            var cut = min(start + stride, nsText.length)
            if cut < nsText.length {
                // Back off to the last word boundary inside the window. There
                // is none in a dense script, and cutting mid-run is then the
                // correct behaviour rather than a fallback.
                var boundary = cut
                let floor = start + stride / 2
                while boundary > floor {
                    let scalar = Unicode.Scalar(nsText.character(at: boundary - 1))
                    if let scalar, scalar == " " || scalar == "\t" || scalar == "\n" {
                        break
                    }
                    boundary -= 1
                }
                if boundary > floor { cut = boundary }
            }

            let range = NSRange(location: start, length: cut - start)
            let piece = nsText.substring(with: range)
            let trimmed = Self.trimmedRange(of: piece, offsetBy: range.location)
            if trimmed.length > 0 {
                pieces.append(
                    Segment(
                        text: nsText.substring(with: trimmed),
                        range: SourceLocator.CharacterRange(
                            location: segment.range.location + trimmed.location,
                            length: trimmed.length)))
            }
            start = cut
        }

        return pieces.isEmpty ? [segment] : pieces
    }

    /// The tail of the emitted chunk, up to the overlap budget.
    private func overlapTail(of segments: [Segment]) -> String {
        guard configuration.overlapTokens > 0 else { return "" }
        var collected: [String] = []
        var budget = configuration.overlapTokens

        for segment in segments.reversed() {
            let cost = TokenEstimator.estimate(segment.text)
            if cost > budget {
                // A single segment can exceed the whole budget. Carrying it
                // whole would put a paragraph of any length at the head of the
                // next chunk — a near-duplicate in the index, which is the one
                // thing overlap exists to avoid needing.
                if collected.isEmpty, let tail = Self.tail(of: segment.text, tokens: budget) {
                    collected.insert(tail, at: 0)
                }
                break
            }
            collected.insert(segment.text, at: 0)
            budget -= cost
            if budget <= 0 { break }
        }

        return collected.joined(separator: " ")
    }

    private static func tail(of text: String, tokens: Int) -> String? {
        guard tokens > 0 else { return nil }
        var taken: [Substring] = []
        var budget = tokens
        for word in text.split(separator: " ").reversed() {
            let cost = max(TokenEstimator.estimate(String(word)), 1)
            if cost > budget { break }
            taken.insert(word, at: 0)
            budget -= cost
            if budget <= 0 { break }
        }
        var result = taken.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        if result.isEmpty {
            // One unbroken run longer than the budget. One character per token
            // is the estimator's own charge for a dense script and an
            // under-estimate for anything else, so this never overshoots.
            result = String(text.suffix(tokens)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return result.isEmpty ? nil : result
    }

    private static func trimmedRange(of text: String, offsetBy offset: Int) -> NSRange {
        let nsText = text as NSString
        var start = 0
        var end = nsText.length
        let whitespace = CharacterSet.whitespacesAndNewlines
        while start < end,
            let scalar = Unicode.Scalar(nsText.character(at: start)),
            whitespace.contains(scalar)
        { start += 1 }
        while end > start,
            let scalar = Unicode.Scalar(nsText.character(at: end - 1)),
            whitespace.contains(scalar)
        { end -= 1 }
        return NSRange(location: offset + start, length: end - start)
    }

    private static func language(from code: String) -> NLLanguage? {
        let base = TextNormalization.baseLanguageCode(code)
        guard !base.isEmpty else { return nil }
        let language = NLLanguage(rawValue: base)
        return language == .undetermined ? nil : language
    }
}

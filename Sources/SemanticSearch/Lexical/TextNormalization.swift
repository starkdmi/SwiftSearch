import Foundation

/// Script-aware token normalisation shared by the analyzer and the chunker.
public enum TextNormalization {

    /// Trim, then collapse every internal whitespace run to one space.
    public static func collapseWhitespace(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }

    /// Normalise one token to its indexable form.
    ///
    /// Numbers take a separate path. Stripping every non-alphanumeric character
    /// from `3.14` yields `314`, which then collides with the integer 314 and
    /// with `3,14` written in a European locale; the numeric path keeps one
    /// internal separator and normalises it to a full stop.
    public static func normalizeToken(_ value: String) -> String {
        guard !value.isEmpty else { return "" }

        if let numeric = normalizeNumeric(value) { return numeric }

        let lowered = value.lowercased()
        let script = dominantScript(of: lowered)
        let folded: String
        switch script {
        case .cyrillic:
            // Keep letters distinct — й is not и — but fold the ё/е variation
            // that Russian orthography treats as optional.
            folded = lowered.replacingOccurrences(of: "ё", with: "е")
        case .latin:
            folded = foldTurkishDotlessI(lowered)
                .folding(options: [.diacriticInsensitive], locale: nil)
        case .arabic:
            folded = normalizeArabic(lowered)
        case .other:
            folded = lowered
        }

        return folded.replacingOccurrences(
            of: #"[^\p{L}\p{N}]"#, with: "", options: .regularExpression)
    }

    /// Separators a number may be written with. Includes the no-break and
    /// narrow no-break spaces, which is how European locales group thousands.
    private static let numericSeparators: Set<Character> = [
        ".", ",", "\u{00A0}", "\u{202F}", "\u{2009}", " ",
    ]

    /// Canonical form of a numeric token: no sign, no group separators, a full
    /// stop for the decimal point, no trailing zeros in the fraction.
    ///
    /// `1,234.5`, `1 234,5` and `+1234.50` all normalise to `1234.5`.
    ///
    /// Written as a scan rather than a regular expression: the pattern needs
    /// non-ASCII separators, and the escape syntax for those differs between
    /// Swift string literals and ICU — which is the kind of mismatch that
    /// compiles fine and throws at the first call.
    private static func normalizeNumeric(_ value: String) -> String? {
        var body = value
        if body.hasPrefix("+") || body.hasPrefix("-") { body.removeFirst() }
        guard !body.isEmpty else { return nil }

        var sawDigit = false
        var previousWasSeparator = true
        for character in body {
            if character.isNumber, character.isASCII {
                sawDigit = true
                previousWasSeparator = false
                continue
            }
            // A separator must sit between digits: `1,,2`, `,5` and `5,` are
            // not numbers, and treating them as such would fold unrelated
            // tokens together.
            guard numericSeparators.contains(character), !previousWasSeparator else {
                return nil
            }
            previousWasSeparator = true
        }
        guard sawDigit, !previousWasSeparator else { return nil }

        // The last separator is the decimal point unless it is followed by
        // exactly three digits, in which case it is a thousands separator.
        // `1,234` is one thousand two hundred and thirty-four in English and
        // `1.234` is the same number in German; `1,23` is one and a fraction in
        // both. Three digits is the one genuinely ambiguous case and grouping
        // is the more common reading of it.
        var integerPart = body
        var fractionPart = ""
        if let separatorIndex = body.lastIndex(where: { $0 == "." || $0 == "," }) {
            let tail = String(body[body.index(after: separatorIndex)...])
            if tail.count != 3 {
                integerPart = String(body[body.startIndex..<separatorIndex])
                fractionPart = tail
            }
        }

        let digits = integerPart.filter(\.isNumber)
        let trimmedInteger = digits.drop { $0 == "0" }
        var canonical = trimmedInteger.isEmpty ? "0" : String(trimmedInteger)
        let trimmedFraction = String(fractionPart.reversed().drop { $0 == "0" }.reversed())
        if !trimmedFraction.isEmpty { canonical += "." + trimmedFraction }
        return canonical
    }

    enum Script {
        case latin
        case cyrillic
        case arabic
        case other
    }

    static func dominantScript(of value: String) -> Script {
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x0400...0x04FF, 0x0500...0x052F, 0x2DE0...0x2DFF, 0xA640...0xA69F:
                return .cyrillic
            case 0x0600...0x06FF, 0x0750...0x077F, 0x08A0...0x08FF, 0xFB50...0xFDFF,
                0xFE70...0xFEFF:
                return .arabic
            case 0x0041...0x005A, 0x0061...0x007A, 0x00C0...0x024F, 0x1E00...0x1EFF:
                return .latin
            default:
                continue
            }
        }
        return .other
    }

    /// Turkish writes two *i*s. `lowercased()` without a locale maps `I` to
    /// `i` and leaves `ı` alone, so `IŞIK` and `ışık` normalise apart. Folding
    /// both onto `i` is what a Turkish index wants and is a no-op everywhere
    /// else, since `ı` and `İ` occur in essentially no other language.
    private static func foldTurkishDotlessI(_ value: String) -> String {
        guard value.contains("ı") || value.contains("İ") || value.contains("i̇") else {
            return value
        }
        return
            value
            .replacingOccurrences(of: "ı", with: "i")
            .replacingOccurrences(of: "İ", with: "i")
            .replacingOccurrences(of: "i̇", with: "i")
    }

    /// Strip the short vowels and the tatweel, and fold the alef and ya
    /// variants that Arabic writers use interchangeably.
    ///
    /// Without this, a query written with harakat matches nothing in a corpus
    /// written without them, which is most corpora.
    private static func normalizeArabic(_ value: String) -> String {
        var output = String.UnicodeScalarView()
        output.reserveCapacity(value.unicodeScalars.count)
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x064B...0x0652,  // fathatan .. sukun
                0x0670,  // superscript alef
                0x0640,  // tatweel
                0x06D6...0x06ED:  // Quranic annotation marks
                continue
            case 0x0622, 0x0623, 0x0625, 0x0671:  // آ أ إ ٱ
                output.append(Unicode.Scalar(0x0627)!)  // ا
            case 0x0649:  // ى
                output.append(Unicode.Scalar(0x064A)!)  // ي
            case 0x0629:  // ة
                output.append(Unicode.Scalar(0x0647)!)  // ه
            default:
                output.append(scalar)
            }
        }
        return String(output)
    }

    /// Scripts written without spaces between words, where one character is
    /// roughly one token and a one-character term is meaningful.
    public static func isDenseScript(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x2E80...0x9FFF,  // CJK radicals through unified ideographs
            0x3040...0x30FF,  // Hiragana and Katakana
            0xA000...0xA4CF,  // Yi
            0xAC00...0xD7AF,  // Hangul syllables
            0xF900...0xFAFF,  // CJK compatibility ideographs
            0xFF00...0xFF9F,  // halfwidth and fullwidth forms
            0x0E00...0x0E7F,  // Thai
            0x0E80...0x0EFF,  // Lao
            0x1780...0x17FF,  // Khmer
            0x1000...0x109F,  // Myanmar
            0x20000...0x2FA1F:  // CJK extensions B–F
            return true
        default:
            return false
        }
    }

    /// `"pt-BR"` → `"pt"`. Comparisons between language tags in this package
    /// are on the base subtag; a Brazilian document is Portuguese for the
    /// purposes of tokenisation.
    public static func baseLanguageCode(_ raw: String) -> String {
        normalizedLanguageCode(raw)?
            .split(separator: "-", maxSplits: 1).first.map(String.init) ?? raw.lowercased()
    }

    public static func normalizedLanguageCode(_ raw: String?) -> String? {
        guard
            var code = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
            !code.isEmpty
        else { return nil }
        code = code.replacingOccurrences(of: "_", with: "-")
        return code.lowercased()
    }
}

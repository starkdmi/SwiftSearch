import Foundation

/// Tokenizer-free length estimate.
///
/// The chunker runs before any model is loaded — often before one is even
/// chosen — so it cannot ask a tokenizer. The estimate is deliberately
/// conservative: scripts written without word spacing are charged one token per
/// character, because subword vocabularies rarely merge them, and alphabetic
/// scripts are charged the larger of one per word and one per four characters,
/// which tracks BPE closely enough for a chunk budget.
///
/// Providers that *do* have a tokenizer tighten this: the engine clamps a
/// chunk's target to the provider's advertised input limit, and the provider
/// reports real truncation in its output.
public enum TokenEstimator {

    public static func estimate(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }

        var denseCharacters = 0
        var alphabeticCharacters = 0
        var wordBreaks = 0
        var previousWasSeparator = true

        for scalar in text.unicodeScalars {
            if TextNormalization.isDenseScript(scalar) {
                denseCharacters += 1
                previousWasSeparator = true
                continue
            }
            if isSeparator(scalar) {
                previousWasSeparator = true
            } else {
                if previousWasSeparator { wordBreaks += 1 }
                previousWasSeparator = false
                alphabeticCharacters += 1
            }
        }

        return denseCharacters + max(wordBreaks, alphabeticCharacters / 4)
    }

    /// ASCII is answered from a table: this is the chunker's inner loop, and
    /// `CharacterSet.contains(_: Unicode.Scalar)` bridges to `NSCharacterSet`
    /// with a retain and release on every call.
    private static func isSeparator(_ scalar: Unicode.Scalar) -> Bool {
        if scalar.value < 128 { return asciiSeparator[Int(scalar.value)] }
        return CharacterSet.whitespacesAndNewlines.contains(scalar)
            || CharacterSet.punctuationCharacters.contains(scalar)
    }

    private static let asciiSeparator: [Bool] = (0..<128).map { value in
        guard let scalar = Unicode.Scalar(UInt32(value)) else { return false }
        return CharacterSet.whitespacesAndNewlines.contains(scalar)
            || CharacterSet.punctuationCharacters.contains(scalar)
    }
}

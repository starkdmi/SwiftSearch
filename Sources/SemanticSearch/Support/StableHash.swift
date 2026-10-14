import Foundation

/// FNV-1a 64, spelled out rather than taken from `Hasher`.
///
/// `Hasher` is seeded per process, so its output cannot be written to disk and
/// compared on the next launch — which is exactly what a fingerprint has to do.
/// This one is stable across processes, architectures and OS versions.
public struct StableHash: Sendable {
    private static let offsetBasis: UInt64 = 14_695_981_039_346_656_037
    private static let prime: UInt64 = 1_099_511_628_211

    private var state: UInt64 = offsetBasis

    public init() {}

    public mutating func append(_ byte: UInt8) {
        state ^= UInt64(byte)
        state &*= Self.prime
    }

    /// Fields are terminated with a byte that cannot occur in UTF-8 so that
    /// `["ab", "c"]` and `["a", "bc"]` do not hash alike.
    public mutating func append(_ text: String) {
        for byte in text.utf8 { append(byte) }
        append(0xFF)
    }

    public mutating func append(_ data: Data) {
        for byte in data { append(byte) }
        append(0xFE)
    }

    public mutating func append(_ value: Int) {
        append(UInt64(bitPattern: Int64(value)))
    }

    public mutating func append(_ value: UInt32) {
        append(UInt64(value))
    }

    public mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            for byte in bytes { append(byte) }
        }
    }

    public mutating func append(_ value: Bool) {
        append(UInt8(value ? 1 : 0))
    }

    public mutating func append(_ value: Double) {
        append(value.bitPattern)
    }

    public mutating func append(_ values: [String]) {
        append(values.count)
        for value in values { append(value) }
    }

    public var digest: String {
        String(format: "%016llx", state)
    }

    public static func of(_ build: (inout StableHash) -> Void) -> String {
        var hasher = StableHash()
        build(&hasher)
        return hasher.digest
    }
}

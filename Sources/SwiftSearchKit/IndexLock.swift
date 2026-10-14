import Foundation

/// One process owns a library at a time.
///
/// The engine keeps its vector graphs in memory and writes SQLite beside
/// them; a second process writing the same index would leave the first
/// searching a graph that no longer matches the database. So the first to
/// open takes an exclusive `flock` on `lock`, held until it closes or exits —
/// the kernel releases it if the process dies — and a second is refused with
/// the owner's pid.
public final class IndexLock: @unchecked Sendable {
    private let descriptor: Int32

    public struct Held: Error, CustomStringConvertible {
        public var owner: Int32?
        public var description: String {
            "the library is open in another process"
                + (owner.map { " (pid \($0))" } ?? "")
                + " — an MCP server holds it while it runs; use its tools, or stop it first"
        }
    }

    public init(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = open(url.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let owner = (try? String(contentsOf: url, encoding: .utf8)).flatMap {
                Int32($0.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            close(descriptor)
            throw Held(owner: owner)
        }
        ftruncate(descriptor, 0)
        let pid = "\(getpid())\n"
        _ = pid.withCString { write(descriptor, $0, strlen($0)) }
        self.descriptor = descriptor
    }

    private let released = NSLock()
    private var isReleased = false

    /// Let another process have the library; also done on deinit.
    public func release() {
        released.withLock {
            guard !isReleased else { return }
            isReleased = true
            flock(descriptor, LOCK_UN)
            close(descriptor)
        }
    }

    deinit { release() }
}

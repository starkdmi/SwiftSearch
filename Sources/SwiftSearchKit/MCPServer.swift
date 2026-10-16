import Foundation
import MCP
import System

/// The library over MCP on standard input and output.
public enum MCPServer {

    public struct Options: Sendable {
        /// Index new and changed files when the server starts.
        public var syncsOnStart: Bool
        /// Unload the models after this long without a call; `nil` to keep
        /// them. The agent beside the server needs the memory too.
        public var releasesModelsAfter: Duration?

        public init(syncsOnStart: Bool = true, releasesModelsAfter: Duration? = .seconds(300)) {
            self.syncsOnStart = syncsOnStart
            self.releasesModelsAfter = releasesModelsAfter
        }
    }

    public static func run(library: Library, options: Options = Options()) async throws {
        // Standard output is the protocol. Anything else that prints — a model
        // loader's progress, a library's debug line — would corrupt it, so the
        // protocol keeps a private copy of the descriptor and descriptor 1
        // becomes standard error.
        let protocolOutput = dup(STDOUT_FILENO)
        guard protocolOutput >= 0, dup2(STDERR_FILENO, STDOUT_FILENO) >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EBADF)
        }
        let transport = StdioTransport(
            input: .standardInput, output: FileDescriptor(rawValue: protocolOutput))

        let activity = Activity()
        let server = await makeServer(library: library, activity: activity)
        try await server.start(transport: transport)

        if options.syncsOnStart { await library.startSync() }
        let idle = options.releasesModelsAfter.map { limit in
            Task {
                // Which calls the last release came after. A call that began
                // and ended between two polls loaded the models again without
                // either poll seeing it; a flag reset only by a poll that saw
                // activity never released them again.
                var releasedAfter: Int?
                while !Task.isCancelled {
                    try? await Task.sleep(for: min(.seconds(15), limit))
                    guard !activity.inFlight(), activity.idle() >= limit,
                        releasedAfter != activity.calls(), await !library.isSyncing
                    else { continue }
                    let calls = activity.calls()
                    await library.releaseModels()
                    releasedAfter = calls
                }
            }
        }
        await server.waitUntilCompleted()
        idle?.cancel()
        try await library.close()
    }
}

extension MCPServer {
    /// The server with the library's tools, on no transport yet.
    public static func makeServer(library: Library) async -> Server {
        await makeServer(library: library, activity: Activity())
    }

    static func makeServer(library: Library, activity: Activity) async -> Server {
        let server = Server(
            name: "swiftsearch", version: "0.1.1",
            capabilities: .init(tools: .init(listChanged: false)))
        await server.withMethodHandler(ListTools.self) { _ in .init(tools: Tools.all) }
        await server.withMethodHandler(CallTool.self) { parameters in
            activity.begin()
            defer { activity.end() }
            return await Tools.call(
                parameters.name, parameters.arguments ?? [:], library: library)
        }
        return server
    }
}

/// Calls running, and when the last one ended — so models are never
/// released under a call that has been running longer than the idle limit.
final class Activity: @unchecked Sendable {
    private let lock = NSLock()
    private var last = ContinuousClock.now
    private var running = 0
    private var started = 0

    func begin() {
        lock.withLock {
            running += 1
            started += 1
        }
    }
    /// Calls begun so far.
    func calls() -> Int { lock.withLock { started } }
    func end() {
        lock.withLock {
            running -= 1
            last = .now
        }
    }
    func inFlight() -> Bool { lock.withLock { running > 0 } }
    func idle() -> Duration { lock.withLock { ContinuousClock.now - last } }
}

import Foundation

/// A counting semaphore for async code, with two priority lanes.
///
/// The bound that keeps two large models from being resident at once. A plain
/// `DispatchSemaphore` cannot be used here: blocking a cooperative thread pool
/// thread to wait for another task on the same pool is how a Swift concurrency
/// deadlock is spelled.
///
/// Interactive waiters are always served before background ones, and each lane
/// is served in arrival order. An interactive query that arrives while an index
/// build holds the permit therefore waits for exactly one batch — the one in
/// flight — and not for the rest of the build queued behind it, which is why
/// ``SemanticSearchEngine`` takes the permit per batch rather than per job.
///
/// Waiting is cancellable. A cancelled waiter leaves the queue immediately and
/// throws `CancellationError`, so a search the user abandoned does not keep its
/// place in line and then run anyway.
public final class AsyncSemaphore: @unchecked Sendable {

    public enum Priority: Sendable {
        /// A caller someone is waiting on: a search.
        case interactive
        /// Work nobody is watching: indexing, preparation, housekeeping.
        case background
    }

    private typealias Waiter = (id: UInt64, continuation: CheckedContinuation<Void, any Error>)

    private let lock = NSLock()
    private var value: Int
    private var interactive: [Waiter] = []
    private var background: [Waiter] = []
    private var nextWaiterID: UInt64 = 0

    public init(value: Int) {
        self.value = max(0, value)
    }

    /// `NSLock.lock()` is marked unavailable from async contexts, and rightly:
    /// holding it across a suspension would block a cooperative thread. Every
    /// critical section here is synchronous and short, which this wrapper makes
    /// structurally true rather than a convention.
    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    /// Take a permit, waiting behind every earlier waiter of the same or higher
    /// priority.
    ///
    /// - Throws: `CancellationError` if the task is cancelled while waiting. No
    ///   permit is held in that case, so the caller must not `signal`.
    public func wait(priority: Priority = .interactive) async throws {
        let id = withLock { () -> UInt64? in
            if value > 0 {
                value -= 1
                return nil
            }
            let id = nextWaiterID
            nextWaiterID &+= 1
            return id
        }
        guard let id else { return }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                enum Outcome { case acquired, cancelled, queued }
                let outcome = withLock { () -> Outcome in
                    // A permit can have been signalled between the check above
                    // and this closure running, so it is rechecked rather than
                    // assumed.
                    if value > 0 {
                        value -= 1
                        return .acquired
                    }
                    // Checked under the lock so a cancellation that already
                    // happened is not missed by the handler below, which only
                    // finds waiters that are in a queue.
                    if Task.isCancelled { return .cancelled }
                    switch priority {
                    case .interactive: interactive.append((id, continuation))
                    case .background: background.append((id, continuation))
                    }
                    return .queued
                }
                switch outcome {
                case .acquired: continuation.resume()
                case .cancelled: continuation.resume(throwing: CancellationError())
                case .queued: break
                }
            }
        } onCancel: {
            let removed = withLock { () -> CheckedContinuation<Void, any Error>? in
                if let index = interactive.firstIndex(where: { $0.id == id }) {
                    return interactive.remove(at: index).continuation
                }
                if let index = background.firstIndex(where: { $0.id == id }) {
                    return background.remove(at: index).continuation
                }
                // Already resumed with a permit, or not queued yet (the check
                // inside the continuation handles that case).
                return nil
            }
            removed?.resume(throwing: CancellationError())
        }
    }

    public func signal() {
        let waiter = withLock { () -> CheckedContinuation<Void, any Error>? in
            if !interactive.isEmpty { return interactive.removeFirst().continuation }
            if !background.isEmpty { return background.removeFirst().continuation }
            value += 1
            return nil
        }
        // Resumed outside the lock: a resumed task can call `wait` again
        // synchronously, and doing that while holding the lock deadlocks.
        waiter?.resume()
    }

    /// Run `body` holding one permit.
    public func withPermit<T>(
        priority: Priority = .interactive, _ body: () async throws -> T
    ) async throws -> T {
        try await wait(priority: priority)
        defer { signal() }
        return try await body()
    }
}

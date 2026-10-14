import Foundation
import Hub
import MLXLMCommon
import SemanticSearch

@preconcurrency import MLX

/// Process-wide MLX limits.
///
/// MLX's defaults scale with installed RAM, which is generous on a Mac Studio
/// and fatal on a phone: an unbounded cache plus a 2B multimodal model is how
/// an index build gets jetsammed. This caps both, once, from a fraction of the
/// device's own memory.
///
/// The host configures this rather than the library silently changing a
/// process-wide setting. ``apply(_:)`` is therefore not called from any
/// provider's `init`; it runs on the first `prepare()` and can be pre-empted by
/// calling it yourself, or disabled entirely with
/// ``Configuration/hostManaged``.
public enum MLXRuntime {

    public struct Configuration: Sendable, Hashable {
        /// Fraction of device memory MLX may allocate.
        public var memoryFraction: Double
        /// Fraction of the memory limit MLX may keep as a free-buffer cache.
        public var cacheFraction: Double
        /// Absolute overrides, in bytes. Win over the fractions when set.
        public var memoryLimitBytes: Int?
        public var cacheLimitBytes: Int?
        /// Leave MLX's allocator alone. Use when the host process already
        /// configures it.
        public var hostManaged: Bool
        /// Where Hub model downloads are written, one copy each, as
        /// `<directory>/models/<org>/<name>`. `nil` keeps the MLX loaders'
        /// own defaults, which differ by loader — `~/Documents/huggingface` for
        /// embedders, `~/Library/Caches` for language and vision models — and
        /// each keep a second copy in the Hugging Face cache on top.
        public var modelDirectory: URL?

        public init(
            memoryFraction: Double = 0.66,
            cacheFraction: Double = 0.12,
            memoryLimitBytes: Int? = nil,
            cacheLimitBytes: Int? = nil,
            hostManaged: Bool = false,
            modelDirectory: URL? = nil
        ) {
            self.memoryFraction = min(0.95, max(0.25, memoryFraction))
            self.cacheFraction = min(0.95, max(0.01, cacheFraction))
            self.memoryLimitBytes = memoryLimitBytes
            self.cacheLimitBytes = cacheLimitBytes
            self.hostManaged = hostManaged
            self.modelDirectory = modelDirectory
        }

        public static let `default` = Configuration()

        /// Tighter bounds for a phone or an iPad.
        public static let constrainedDevice = Configuration(
            memoryFraction: 0.5, cacheFraction: 0.06)

        /// Change nothing.
        public static let hostConfigured = Configuration(hostManaged: true)

        /// Read overrides from the environment, so a run can be
        /// bounded without a code change.
        public static func fromEnvironment(default base: Configuration = .default)
            -> Configuration
        {
            var configuration = base
            let environment = ProcessInfo.processInfo.environment
            if let raw = environment["SEMANTIC_SEARCH_MLX_MEMORY_FRACTION"],
                let value = Double(raw), value.isFinite
            {
                configuration.memoryFraction = min(0.95, max(0.25, value))
            }
            if let raw = environment["SEMANTIC_SEARCH_MLX_CACHE_FRACTION"],
                let value = Double(raw), value.isFinite
            {
                configuration.cacheFraction = min(0.95, max(0.01, value))
            }
            if let raw = environment["SEMANTIC_SEARCH_MLX_MEMORY_LIMIT_MB"],
                let value = Int(raw), value > 0
            {
                configuration.memoryLimitBytes = value * 1_024 * 1_024
            }
            if let raw = environment["SEMANTIC_SEARCH_MLX_CACHE_LIMIT_MB"],
                let value = Int(raw), value > 0
            {
                configuration.cacheLimitBytes = value * 1_024 * 1_024
            }
            if let raw = environment["SEMANTIC_SEARCH_MLX_HOST_MANAGED"]?.lowercased(),
                ["1", "true", "yes", "on"].contains(raw)
            {
                configuration.hostManaged = true
            }
            if let raw = environment["SEMANTIC_SEARCH_MODEL_DIRECTORY"], !raw.isEmpty {
                configuration.modelDirectory = URL(
                    fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
            }
            return configuration
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var applied = false

    /// Apply the limits, once per process.
    @discardableResult
    public static func apply(_ configuration: Configuration = .fromEnvironment()) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !applied else { return false }
        applied = true
        guard !configuration.hostManaged else { return false }

        let physical = max(minimumPhysicalBytes, GPU.deviceInfo().memorySize)
        let memoryLimit = max(
            minimumMemoryLimitBytes,
            min(
                physical,
                configuration.memoryLimitBytes
                    ?? Int(Double(physical) * configuration.memoryFraction)))
        let cacheLimit = max(
            minimumCacheLimitBytes,
            min(
                memoryLimit,
                configuration.cacheLimitBytes
                    ?? Int(Double(memoryLimit) * configuration.cacheFraction)))

        Memory.memoryLimit = memoryLimit
        Memory.cacheLimit = cacheLimit
        return true
    }

    /// Where a provider's downloads go: ``Configuration/modelDirectory`` and
    /// nowhere else when it is set, otherwise the default mlx-swift-lm 2.x
    /// gave that kind of loader — `~/Documents/huggingface` for text embedders,
    /// `~/Library/Caches` for language and vision models — so a model fetched
    /// before still loads without a download.
    static func downloader(_ configuration: Configuration, for loader: Loader)
        -> any MLXLMCommon.Downloader
    {
        if let directory = configuration.modelDirectory {
            return HubDownloader(
                hub: HubApi(downloadBase: directory, cache: nil, useOfflineMode: false))
        }
        switch loader {
        case .embedder:
            return HubDownloader(hub: HubApi(useOfflineMode: false))
        case .languageModel:
            return HubDownloader(
                hub: HubApi(
                    downloadBase: FileManager.default.urls(
                        for: .cachesDirectory, in: .userDomainMask
                    ).first, useOfflineMode: false))
        }
    }

    enum Loader {
        case embedder
        case languageModel
    }

    static let tokenizerLoader: any MLXLMCommon.TokenizerLoader = TransformersTokenizerLoader()

    /// Drop MLX's free-buffer cache. Called when a provider is released so the
    /// memory actually comes back rather than staying in the allocator.
    public static func releaseCache() {
        Memory.clearCache()
    }

    public static func snapshot() -> (active: Int, cache: Int, peak: Int) {
        let snapshot = Memory.snapshot()
        return (snapshot.activeMemory, snapshot.cacheMemory, snapshot.peakMemory)
    }

    private static let minimumPhysicalBytes = 2 * 1_024 * 1_024 * 1_024
    private static let minimumMemoryLimitBytes = 2 * 1_024 * 1_024 * 1_024
    private static let minimumCacheLimitBytes = 128 * 1_024 * 1_024
}

import AVFoundation
import Foundation
import SemanticSearch

/// Pulls 16 kHz mono float samples out of an audio or video file.
///
/// Reads through `AVAssetReader` on the calling task rather than hopping to a
/// global queue and bridging back through a continuation: that is both
/// simpler and cancellable, which a bridged reader is not — a cancelled
/// transcription there kept reading to the end of the file.
///
/// 16 kHz mono is what Parakeet and every other on-device ASR model this
/// package targets expects. AVFoundation does the resampling and the mix-down
/// inside the reader, which is better than doing either afterwards.
public struct AudioExtractor: Sendable {

    public static let targetSampleRate: Double = 16_000

    /// Whether the file has sound at all. A time-lapse, a screen recording
    /// or a clip exported without audio has none, and has nothing to
    /// transcribe or classify: an empty result, not a failure. Thrown, the
    /// error failed the whole video, frames included.
    public static func hasAudioTrack(_ url: URL) async throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        return try await !AVURLAsset(url: url).loadTracks(withMediaType: .audio).isEmpty
    }
    public static let targetChannelCount: Int = 1

    public init() {}

    /// Samples for the whole asset.
    public func samples(from url: URL) async throws -> [Float] {
        try await samples(from: url, timeRange: nil)
    }

    /// Samples for one time range, or the whole asset when `timeRange` is nil.
    public func samples(from url: URL, timeRange: SourceLocator.TimeRange?) async throws
        -> [Float]
    {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }

        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let track = tracks.first else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "no audio track")
        }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "cannot read audio: \(error.localizedDescription)")
        }

        if let timeRange {
            reader.timeRange = CMTimeRange(
                start: CMTime(seconds: timeRange.start, preferredTimescale: 600),
                duration: CMTime(seconds: timeRange.duration, preferredTimescale: 600))
        }

        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
                AVNumberOfChannelsKey: Self.targetChannelCount,
                AVSampleRateKey: Self.targetSampleRate,
            ])
        output.alwaysCopiesSampleData = false

        guard reader.canAdd(output) else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: "cannot convert this audio track to 16 kHz mono float")
        }
        reader.add(output)

        guard reader.startReading() else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: reader.error?.localizedDescription ?? "reader would not start")
        }

        var samples: [Float] = []
        // Pre-sized from the declared duration. Growing a several-million
        // element array by doubling costs a handful of full copies of the whole
        // recording.
        if let duration = try? await asset.load(.duration).seconds, duration.isFinite {
            let seconds = timeRange?.duration ?? duration
            samples.reserveCapacity(Int(seconds * Self.targetSampleRate))
        }

        while reader.status == .reading {
            if Task.isCancelled {
                reader.cancelReading()
                throw SemanticSearchError.cancelled
            }
            guard let buffer = output.copyNextSampleBuffer() else { break }
            autoreleasepool {
                guard let block = CMSampleBufferGetDataBuffer(buffer) else { return }
                var contiguous = 0
                var total = 0
                var pointer: UnsafeMutablePointer<Int8>?
                let status = CMBlockBufferGetDataPointer(
                    block, atOffset: 0, lengthAtOffsetOut: &contiguous, totalLengthOut: &total,
                    dataPointerOut: &pointer)
                guard status == kCMBlockBufferNoErr, total > 0 else { return }
                let count = total / MemoryLayout<Float>.size

                // The pointer is only valid for `contiguous` bytes. A block
                // buffer can be a chain of several pieces, and reading `total`
                // bytes from the first would run off its end — so the fast path
                // is taken only when the whole buffer is one piece, and the
                // rest is copied out.
                if contiguous == total, let pointer {
                    pointer.withMemoryRebound(to: Float.self, capacity: count) { floats in
                        samples.append(
                            contentsOf: UnsafeBufferPointer(start: floats, count: count))
                    }
                } else {
                    var copied = [Float](repeating: 0, count: count)
                    let copyStatus = copied.withUnsafeMutableBytes { bytes in
                        CMBlockBufferCopyDataBytes(
                            block, atOffset: 0, dataLength: count * MemoryLayout<Float>.size,
                            destination: bytes.baseAddress!)
                    }
                    guard copyStatus == kCMBlockBufferNoErr else { return }
                    samples.append(contentsOf: copied)
                }
            }
        }

        guard reader.status == .completed else {
            if reader.status == .cancelled { throw SemanticSearchError.cancelled }
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent),
                detail: reader.error?.localizedDescription ?? "audio read did not complete")
        }
        guard !samples.isEmpty else {
            throw SemanticSearchError.unsupportedContent(
                itemID: ItemID(url.lastPathComponent), detail: "audio track is empty")
        }
        return samples
    }

    /// Duration of the asset's audio, in seconds.
    public func duration(of url: URL) async throws -> TimeInterval {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SemanticSearchError.mediaUnavailable(url: url, itemID: nil)
        }
        let seconds = try await AVURLAsset(url: url).load(.duration).seconds
        guard seconds.isFinite else { return 0 }
        return seconds
    }
}

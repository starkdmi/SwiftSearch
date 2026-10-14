// Copied from FluidInference/FluidUse (Apache-2.0) and modified for SwiftSearch;
// see NOTICE in this package for the source revision and the list of changes.

@preconcurrency import CoreML
import CryptoKit
import Foundation

/// Downloads the pinned EmbeddingGemma 2 Core ML packages, token table, and tokenizer from Hugging Face into a
/// directory the caller names, verifies every file, and compiles each package once (`<name>.mlmodelc` beside it).
/// Text, audio and vision are separate bundles, so adding audio never recompiles the text model.
public enum EmbeddingGemma2ModelStore {
    public typealias Progress = @Sendable (_ file: String, _ bytes: Int64) -> Void

    public static let repository = "FluidInference/embeddinggemma-2-coreml"
    /// One revision for every bundle. Upstream pins each bundle to the commit that added it (text 70fcf18, audio
    /// c0fefd0, vision f9d567d); every file below has the same checksum at this one, so one name covers all three.
    public static let revision = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"

    private struct Asset {
        let path: String
        let sha256: String
    }

    private static let audioAssets = [
        Asset(
            path: "EmbeddingGemma2Audio.mlpackage/Manifest.json",
            sha256: "65c8e9811516e7916a3d742261c7d5a02d4e43c113b437330ce86f98b3da3514"),
        Asset(
            path: "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "d40896e5c88d62b02d15cc975f27e254c28a5774a1702e09bf5483071623548d"),
        Asset(
            path: "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "41d7b552caabe376e377dcbe1c8c5f9879853bc4f384a161bb1f886f19e84b56"),
    ]

    private static let visionAssets = [
        Asset(
            path: "position_embeddings.f16", sha256: "17de9480acc23638c0634a97ed89e51ed664433256d7d6f109587dcd91c79124"),
        Asset(
            path: "EmbeddingGemma2Vision.mlpackage/Manifest.json",
            sha256: "376afe7b51608637312cd15a0a89b1d7863f9779502aaf5f12fa6b14c4272d8d"),
        Asset(
            path: "EmbeddingGemma2Vision.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0085b5e04a13d000fe5969adeaa13d897f5a3cb7be60c76c8685c023747f89ef"),
        Asset(
            path: "EmbeddingGemma2Vision.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "5b5b6dd8d80aa438bf91ba143fbe90d56a2e25bfcf685a56b990aad71cf75232"),
    ]

    private static let assets = [
        Asset(path: "tokenizer.json", sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
        Asset(path: "embeddings.bf16", sha256: "56ebbdbfd706c827f3fae1d97aa39135c7ddc0f1e6d9b12f8565f84bbfbda877"),
        Asset(
            path: "EmbeddingGemma2Text.mlpackage/Manifest.json",
            sha256: "83eace80b69b46f8415af620ba1b40d7c490d00f7c00ba9dedd2e29da6e9ac19"),
        Asset(
            path: "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            sha256: "0df9aca02807cf490db6d4f1c68607a8c7f486ad23b376d1968fd3be5eb14677"),
        Asset(
            path: "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            sha256: "75b97ab036ca87eae60560c4034a63d604201498dc4b6f6971dd0b1e4558d61f"),
    ]

    /// Ensures the text model, token table and tokenizer exist in `directory`, match their checksums, and are
    /// compiled; returns the directory. With `downloads` false a missing or mismatched file throws instead of being
    /// fetched, for a copy the caller placed there.
    @discardableResult
    public static func ensure(
        in directory: URL, downloads: Bool = true, progress: Progress? = nil
    ) async throws -> URL {
        try await ensure(
            assets: assets, package: "EmbeddingGemma2Text", verifiedName: ".verified-text-\(revision)",
            directory: directory, downloads: downloads, progress: progress)
    }

    /// Ensures the audio model (`EmbeddingGemma2Audio`) exists, matches its checksums, and is compiled; returns the
    /// directory. The text bundle is needed too (`ensure`): audio tokens are embedded by the text model.
    @discardableResult
    public static func ensureAudio(
        in directory: URL, downloads: Bool = true, progress: Progress? = nil
    ) async throws -> URL {
        try await ensure(
            assets: audioAssets, package: "EmbeddingGemma2Audio", verifiedName: ".verified-audio-\(revision)",
            directory: directory, downloads: downloads, progress: progress)
    }

    /// Ensures the vision model (`EmbeddingGemma2Vision`) and its position tables exist, match their checksums, and
    /// are compiled; returns the directory. Image tokens are embedded by the text model, so `ensure` is needed too.
    @discardableResult
    public static func ensureVision(
        in directory: URL, downloads: Bool = true, progress: Progress? = nil
    ) async throws -> URL {
        try await ensure(
            assets: visionAssets, package: "EmbeddingGemma2Vision", verifiedName: ".verified-vision-\(revision)",
            directory: directory, downloads: downloads, progress: progress)
    }

    private static func ensure(
        assets: [Asset], package: String, verifiedName: String, directory: URL, downloads: Bool,
        progress: Progress?
    ) async throws -> URL {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        // Hashing hundreds of MB takes a moment; skip it while every file is as it was when this revision was
        // verified. A file replaced since, such as another revision downloaded over a caller's copy, is hashed again
        // before anything loads it.
        let verified = directory.appendingPathComponent(verifiedName)
        let compiled = directory.appendingPathComponent("\(package).mlmodelc")
        let stamp = compiled.appendingPathComponent("fluiduse-revision")
        func isReady() -> Bool {
            guard let states = fileStates(of: assets, in: directory) else { return false }
            return (try? String(contentsOf: verified, encoding: .utf8)) == states
                && (try? String(contentsOf: stamp, encoding: .utf8)) == revision
        }
        if isReady() { return directory }
        // One ensure at a time in a directory, across providers and processes: two that compile at once swap each
        // other's bundle out, one of them while it loads, and two that download fetch every file twice. Whoever
        // held the lock may have done the work.
        let lock = try await lockDirectory(directory)
        defer { if let lock { close(lock) } }
        if isReady() { return directory }
        if lock != nil { removeLeftovers(of: package, in: directory) }
        try? manager.removeItem(at: verified)
        // What the marker vouches for: each file as it was when it was hashed, not as it is once the package has
        // compiled, minutes later. One replaced in between would otherwise never be hashed.
        var states: String? = ""
        for asset in assets {
            let destination = directory.appendingPathComponent(asset.path)
            if let state = fileState(of: destination), try checksum(of: destination) == asset.sha256 {
                states? += "\(asset.path) \(state)\n"
                continue
            }
            guard downloads else {
                throw EmbeddingGemma2Error.invalidAsset(
                    "\(asset.path) in \(directory.path) is missing or does not match revision \(revision)")
            }
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress?(asset.path, 0)
            let escaped = asset.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? asset.path
            guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)") else {
                throw EmbeddingGemma2Error.invalidAsset("Invalid Hugging Face asset URL")
            }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw EmbeddingGemma2Error.invalidAsset("Download failed for \(asset.path)")
            }
            let actual = try checksum(of: temporary)
            guard actual == asset.sha256 else {
                throw EmbeddingGemma2Error.invalidAsset(
                    "Checksum mismatch for \(asset.path): expected \(asset.sha256), got \(actual)")
            }
            let size = (try manager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            try install(temporary, at: destination)
            // Renamed into place, it is the file just hashed.
            if let state = fileState(of: destination) { states? += "\(asset.path) \(state)\n" } else { states = nil }
            progress?(asset.path, size)
        }
        // Recompile unless the compiled model carries this revision's stamp (a revision bump or an interrupted
        // compile leaves an old or missing stamp). Staged rename, so a reader never sees a half-written bundle.
        if (try? String(contentsOf: stamp, encoding: .utf8)) != revision {
            progress?("\(package).mlmodelc", 0)
            let temporary = try await MLModel.compileModel(at: directory.appendingPathComponent("\(package).mlpackage"))
            let staged = directory.appendingPathComponent("\(package).\(UUID().uuidString).mlmodelc")
            let retired = directory.appendingPathComponent("\(package).\(UUID().uuidString).old")
            // Hundreds of MB each: none is left behind, whichever step below fails.
            defer {
                try? manager.removeItem(at: temporary)
                try? manager.removeItem(at: staged)
                try? manager.removeItem(at: retired)
            }
            try revision.write(
                to: temporary.appendingPathComponent("fluiduse-revision"), atomically: true, encoding: .utf8)
            try manager.moveItem(at: temporary, to: staged)
            if manager.fileExists(atPath: compiled.path) { try manager.moveItem(at: compiled, to: retired) }
            do {
                try manager.moveItem(at: staged, to: compiled)
            } catch {
                // Only without the lock can another ensure install its bundle between the two moves. Stamped with
                // this revision, it was compiled from the same verified package, so it stays.
                guard (try? String(contentsOf: stamp, encoding: .utf8)) == revision else { throw error }
            }
        }
        // Not fatal when it cannot be written: the next ensure hashes again.
        if let states { try? states.write(to: verified, atomically: true, encoding: .utf8) }
        return directory
    }

    /// Every asset's state, a line each, or nil when one is missing: what the `.verified-*` marker records.
    private static func fileStates(of assets: [Asset], in directory: URL) -> String? {
        var lines = ""
        for asset in assets {
            guard let state = fileState(of: directory.appendingPathComponent(asset.path)) else { return nil }
            lines += "\(asset.path) \(state)\n"
        }
        return lines
    }

    /// Size, modification time and inode, or nil when the file is missing. A file written in place has a new
    /// modification time and one renamed over it a new inode, so either is hashed again. `stat` follows a symbolic
    /// link, as reading the file does.
    private static func fileState(of file: URL) -> String? {
        var status = stat()
        guard stat(file.path, &status) == 0 else { return nil }
        let modified = status.st_mtimespec
        return "\(status.st_size) \(modified.tv_sec) \(modified.tv_nsec) \(status.st_ino)"
    }

    /// An exclusive `flock` on `<directory>/.fluiduse-lock`, held until the descriptor is closed; the system drops
    /// it when a process dies. Polled, so a task waiting for another's download or compile holds no thread
    /// meanwhile. nil when the lock file cannot be opened or the volume has no locks: ensure goes on without one.
    private static func lockDirectory(_ directory: URL) async throws -> Int32? {
        let descriptor = open(
            directory.appendingPathComponent(".fluiduse-lock").path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else { return nil }
        while flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
            let code = errno
            guard code == POSIXErrorCode.EWOULDBLOCK.rawValue || code == POSIXErrorCode.EINTR.rawValue else {
                close(descriptor)
                return nil
            }
            do {
                try await Task.sleep(nanoseconds: 200_000_000)
            } catch {
                close(descriptor)
                throw error
            }
        }
        return descriptor
    }

    /// Bundles an ensure killed between its moves left, `<package>.<UUID>.mlmodelc` staged and `<package>.<UUID>.old`
    /// retired. Only under the lock: without it they may be another ensure's, about to be moved into place.
    private static func removeLeftovers(of package: String, in directory: URL) {
        let manager = FileManager.default
        guard let names = try? manager.contentsOfDirectory(atPath: directory.path) else { return }
        for name in names where name.hasPrefix("\(package).") {
            let rest = name.dropFirst(package.count + 1)
            for suffix in [".mlmodelc", ".old"] where rest.hasSuffix(suffix) {
                guard UUID(uuidString: String(rest.dropLast(suffix.count))) != nil else { continue }
                try? manager.removeItem(at: directory.appendingPathComponent(name))
            }
        }
    }

    /// Stage on the destination volume, then atomically replace the file. Concurrent loads use separate staging
    /// files and never remove another load's completed file. (FluidUse's `LayaModelStore.installDownloadedFile`.)
    private static func install(_ temporary: URL, at destination: URL) throws {
        let manager = FileManager.default
        let partial = destination.appendingPathExtension("\(UUID().uuidString).partial")
        defer { try? manager.removeItem(at: partial) }
        try manager.moveItem(at: temporary, to: partial)
        guard rename(partial.path, destination.path) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
    }

    private static func checksum(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

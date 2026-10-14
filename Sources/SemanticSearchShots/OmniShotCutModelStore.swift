import CoreML
import CryptoKit
import Foundation

/// The Core ML conversions of OmniShotCut v1.5 on the Hub, each in its own
/// repository pinned to one revision with every file's SHA-256, downloaded and
/// compiled once into the model directory.
///
/// The packages are compiled beside the download, at a fixed path, rather
/// than into OmniShotCutCoreML's own cache: every model stays in the one
/// directory `SEMANTIC_SEARCH_MODEL_DIRECTORY` names, and the Neural Engine
/// caches its own compilation — seconds on a first load — per compiled model
/// location, so the location must not move.
enum OmniShotCutModelStore {

    /// One of the two conversions OmniShotCutCoreML's `scripts/convert.py`
    /// writes.
    enum Conversion: String, Sendable, CaseIterable {
        /// For the Neural Engine.
        case fp16 = "OmniShotCut-v1.5-fp16"
        /// For the GPU.
        case fp32 = "OmniShotCut-v1.5-fp32"

        /// The Hub repository holding it.
        var repository: String {
            switch self {
            case .fp16: return "starkdmi/OmniShotCut-v1.5-coreml-fp16"
            case .fp32: return "starkdmi/OmniShotCut-v1.5-coreml-fp32"
            }
        }

        /// The revision the checksums pin and the segmenter's fingerprint
        /// names.
        var revision: String {
            switch self {
            case .fp16: return "d6e9f58719ea116a782b094249bfa246c04cd2ae"
            case .fp32: return "247a34532cae791e871c8717f90e4e348f7eadaf"
            }
        }

        /// The package's files at ``revision``.
        var assets: [Asset] {
            switch self {
            case .fp16:
                return [
                    Asset(
                        "Manifest.json", size: 617,
                        "ed949adc620e9ad65f4dafe56f43b529a514a238cbbbad4e9dbf13dbebcc9e1d"),
                    Asset(
                        "Data/com.apple.CoreML/model.mlmodel", size: 327_179,
                        "b3d0c0a6e4b12f9ca036a5381b781da9d484efedbb4537ea9120274bdd85bbae"),
                    Asset(
                        "Data/com.apple.CoreML/weights/weight.bin", size: 107_375_820,
                        "4d821f8b0b2f195567220ec72bff18a218cda402fbb93f4aa27f1d036d24ca8d"),
                ]
            case .fp32:
                return [
                    Asset(
                        "Manifest.json", size: 617,
                        "2fc92553ff05b33b5c4f3b7e85da873d8e6044b26f6a398c4bf20e5acaee4efa"),
                    Asset(
                        "Data/com.apple.CoreML/model.mlmodel", size: 144_306,
                        "d399f99595d7f3d15101cd311fa50aad4a4444ae99b1311105662464f3c8f1a1"),
                    Asset(
                        "Data/com.apple.CoreML/weights/weight.bin", size: 207_347_072,
                        "3fc2f1fd103e7b2771d2c69b6f7788ccf0857b7ce572209d9edea3aecfda7a77"),
                ]
            }
        }
    }

    /// A file of an `.mlpackage`, by its path inside the package.
    struct Asset: Sendable {
        let path: String
        let size: Int
        let sha256: String

        init(_ path: String, size: Int, _ sha256: String) {
            self.path = path
            self.size = size
            self.sha256 = sha256
        }
    }

    /// The file the compiled model carries the revision it was compiled from
    /// in. A compiled model without it, or with another, is compiled again.
    static let stampName = "omnishotcut-revision"

    /// `<base>/models/<org>/<name>` of the conversion's repository.
    static func directory(_ conversion: Conversion, base: URL) -> URL {
        conversion.repository.split(separator: "/").reduce(
            base.appendingPathComponent("models", isDirectory: true)
        ) { $0.appendingPathComponent(String($1), isDirectory: true) }
    }

    /// The compiled `conversion` in its directory under `base`, downloading,
    /// checking and compiling it first if it is not there yet.
    static func compiledModel(_ conversion: Conversion, base: URL) async throws -> URL {
        let directory = directory(conversion, base: base)
        let revision = conversion.revision
        let compiled = directory.appendingPathComponent(
            "\(conversion.rawValue).mlmodelc", isDirectory: true)
        let stamp = compiled.appendingPathComponent(stampName)
        if (try? String(contentsOf: stamp, encoding: .utf8)) == revision { return compiled }

        let package = directory.appendingPathComponent(
            "\(conversion.rawValue).mlpackage", isDirectory: true)
        for asset in conversion.assets {
            let file = package.appendingPathComponent(asset.path)
            if try verified(file, asset) { continue }
            try await download(asset, of: conversion, to: file)
        }

        // `compileModel` writes to `TMPDIR`; the result moves next to the
        // package, where a stale copy from another revision is replaced.
        let temporary = try await MLModel.compileModel(at: package)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try revision.write(
            to: temporary.appendingPathComponent(stampName), atomically: true, encoding: .utf8)
        try? FileManager.default.removeItem(at: compiled)
        do {
            try FileManager.default.moveItem(at: temporary, to: compiled)
        } catch {
            // Another process compiled the same revision first; its copy is
            // as good as this one.
            guard (try? String(contentsOf: stamp, encoding: .utf8)) == revision else {
                throw error
            }
        }
        return compiled
    }

    /// Whether `file` is there with the pinned size and SHA-256.
    private static func verified(_ file: URL, _ asset: Asset) throws -> Bool {
        guard
            let size = try? FileManager.default.attributesOfItem(atPath: file.path)[.size]
                as? Int,
            size == asset.size
        else { return false }
        return try sha256(of: file) == asset.sha256
    }

    private static func download(_ asset: Asset, of conversion: Conversion, to destination: URL)
        async throws
    {
        let path = "\(conversion.rawValue).mlpackage/\(asset.path)"
        func failure(_ detail: String) -> OmniShotCutModelStoreError {
            .download(repository: conversion.repository, path: path, detail: detail)
        }
        let escaped =
            path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        guard
            let url = URL(
                string: "https://huggingface.co/\(conversion.repository)/resolve/"
                    + "\(conversion.revision)/\(escaped)")
        else { throw failure("not a URL") }
        let (temporary, response) = try await URLSession.shared.download(from: url)
        defer { try? FileManager.default.removeItem(at: temporary) }
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw failure("HTTP \(http.statusCode)")
        }
        let digest = try sha256(of: temporary)
        guard digest == asset.sha256 else {
            throw failure("SHA-256 \(digest), expected \(asset.sha256)")
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    /// Read in pieces: the fp32 weights are 198 MB.
    static func sha256(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

enum OmniShotCutModelStoreError: Error, CustomStringConvertible {
    case download(repository: String, path: String, detail: String)

    var description: String {
        switch self {
        case .download(let repository, let path, let detail):
            return "\(repository)/\(path): \(detail)"
        }
    }
}

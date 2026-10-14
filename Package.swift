// swift-tools-version: 6.3

import PackageDescription

// The package is split so that a caller which only needs lexical search, or
// only needs the storage engine, never links MLX. `SemanticSearch` is the
// target every other one depends on, and it depends on nothing but SQLite and
// USearch.
//
// USearch, NumKong and FluidUse's EmbeddingGemma 2 runtime are compiled from
// `Vendor/` as targets of this package (see Vendor/README.md). mlx-swift-lm
// comes from a fork tagged with the patches this package needs until they are
// released upstream.

let numKongSettings: [CSetting] = [
    .define("NK_DYNAMIC_DISPATCH", to: "1"),
    .define("NK_NATIVE_F16", to: "0"),
    .define("NK_NATIVE_BF16", to: "0"),
]

let usearchSettings: [CXXSetting] = [
    .headerSearchPath("../include/"),
    .define("USEARCH_USE_NUMKONG", to: "1"),
    .define("NK_DYNAMIC_DISPATCH", to: "1"),
    .define("NK_NATIVE_F16", to: "0"),
    .define("NK_NATIVE_BF16", to: "0"),
]

let package = Package(
    name: "SwiftSearch",
    platforms: [.macOS(.v14), .iOS(.v18)],
    products: [
        .library(name: "SemanticSearch", targets: ["SemanticSearch"]),
        .library(name: "SemanticSearchMLX", targets: ["SemanticSearchMLX"]),
        .library(name: "SemanticSearchApple", targets: ["SemanticSearchApple"]),
        .library(name: "SemanticSearchFluidAudio", targets: ["SemanticSearchFluidAudio"]),
        .library(name: "SemanticSearchShots", targets: ["SemanticSearchShots"]),
        .library(name: "SemanticSearchCoreML", targets: ["SemanticSearchCoreML"]),
        .library(name: "SwiftSearchKit", targets: ["SwiftSearchKit"]),
        .executable(name: "swiftsearch", targets: ["swiftsearch"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        .package(url: "https://github.com/starkdmi/mlx-swift-lm", exact: "3.32.3-swiftsearch.1"),
        // The runtime takes its downloader and tokenizer from the caller.
        // These are the `HubApi` and tokenizers it used before 3.0, so
        // downloads keep their layout (HubIntegration.swift).
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/starkdmi/OmniShotCutCoreML", from: "0.1.0"),
        .package(url: "https://github.com/FluidInference/FluidAudio", .upToNextMinor(from: "0.12.6")),
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk", .upToNextMinor(from: "0.12.1")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        // MARK: - Core

        .target(
            name: "SemanticSearch",
            dependencies: ["USearch"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Providers

        .target(
            name: "SemanticSearchMLX",
            dependencies: [
                "SemanticSearch",
                // `ImageDecoding.decodeUpright`: the runtime's own decoding of an
                // image URL ignores its EXIF orientation.
                "SemanticSearchApple",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        .target(
            name: "SemanticSearchApple",
            dependencies: ["SemanticSearch"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        .target(
            name: "SemanticSearchFluidAudio",
            dependencies: [
                "SemanticSearch",
                "SemanticSearchApple",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // Shots for video: OmniShotCut on Core ML, whose conversions are
        // downloaded. Separate so a caller that does not index video never
        // links it.
        .target(
            name: "SemanticSearchShots",
            dependencies: [
                "SemanticSearch", "SemanticSearchApple",
                .product(name: "OmniShotCutCoreML", package: "OmniShotCutCoreML"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // EmbeddingGemma 2 on Core ML: FluidInference's fp16 conversion, on the
        // CPU by default although it was converted for the Neural Engine, whose
        // fp16 leaves the reference past 256 tokens (see
        // CoreMLEmbeddingGemma2Provider.Configuration). Kept apart from
        // SemanticSearchMLX so that a caller shipping one never links the
        // other. Its multifunction models need macOS 15 / iOS 18 at run time.
        .target(
            name: "SemanticSearchCoreML",
            dependencies: [
                "SemanticSearch",
                "SemanticSearchApple",
                "FluidUseEmbeddingGemma2",
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - CLI and MCP server

        // Everything but argument parsing: the library's folders, profiles,
        // sync and tools. The CLI and the MCP server are two front ends on it.
        .target(
            name: "SwiftSearchKit",
            dependencies: [
                "SemanticSearch",
                "SemanticSearchApple",
                "SemanticSearchMLX",
                "SemanticSearchFluidAudio",
                "SemanticSearchShots",
                .product(name: "MCP", package: "swift-sdk"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        .executableTarget(
            name: "swiftsearch",
            dependencies: [
                "SwiftSearchKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Vendored

        .target(
            name: "USearchC",
            dependencies: ["CNumKongDispatch"],
            path: "Vendor/USearch/c",
            sources: ["usearch.h", "lib.cpp"],
            publicHeadersPath: ".",
            cxxSettings: usearchSettings
        ),

        .target(
            name: "USearch",
            dependencies: ["USearchC"],
            path: "Vendor/USearch/swift",
            cxxSettings: usearchSettings
        ),

        // Its headers, and one translation unit: Xcode links every C target
        // as an object, and a header-only one fails the link of anything that
        // depends on it with "CNumKong.o cannot be found".
        .target(
            name: "CNumKong",
            path: "Vendor/NumKong/include",
            exclude: ["README.md"],
            publicHeadersPath: ".",
            cSettings: numKongSettings
        ),

        .target(
            name: "CNumKongDispatch",
            dependencies: ["CNumKong"],
            path: "Vendor/NumKong/c",
            publicHeadersPath: ".",
            cSettings: numKongSettings + [.headerSearchPath("../include")]
        ),

        .target(
            name: "FluidUseEmbeddingGemma2",
            path: "Vendor/FluidUseEmbeddingGemma2",
            exclude: ["LICENSE", "NOTICE"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Tests

        // Model-free. Safe under `swift test`.
        .testTarget(
            name: "SemanticSearchTests",
            dependencies: ["SemanticSearch", "SemanticSearchApple", "SemanticSearchShots"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // Model-free: the lexical profile, an in-memory MCP client.
        .testTarget(
            name: "SwiftSearchKitTests",
            dependencies: ["SwiftSearchKit", "SemanticSearch"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // Loads model weights and runs MLX. Built by `xcodebuild
        // build-for-testing` and run with `xcrun xctest`; see README.md.
        .testTarget(
            name: "SemanticSearchIntegrationTests",
            dependencies: [
                "SemanticSearch",
                "SemanticSearchMLX",
                "SemanticSearchApple",
                "SemanticSearchFluidAudio",
                "SemanticSearchShots",
                "SemanticSearchCoreML",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ],
    cxxLanguageStandard: .cxx11
)

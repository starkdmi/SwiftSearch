import Foundation
import SemanticSearch
import SemanticSearchApple
import SemanticSearchFluidAudio
import SemanticSearchMLX
import SemanticSearchShots

/// The engine each profile builds: SwiftSearch's library recipe, wrapped in
/// file metadata.
enum Recipe {

    struct Built: Sendable {
        var configuration: EngineConfiguration
        /// Unloads the models the processors hold. The engine's
        /// `releaseModels` covers its embedders and reranker only; the
        /// captioner and Parakeet belong to the processors and stay loaded
        /// until released here.
        var releaseProcessorModels: @Sendable () async -> Void
        /// Asks again for the places a failed lookup left out; how many were
        /// named. An unchanged file is never processed again, so without
        /// this one refusal would leave it without a place for good.
        var namePendingPlaces: @Sendable (SemanticSearchEngine) async throws -> Int
    }

    static func build(settings: Settings, storeURL: URL, boundary: FolderBoundary) async -> Built {
        let processor: any ContentProcessor
        var configuration: EngineConfiguration
        var release: @Sendable () async -> Void = {}
        // Built only by the profiles that index video frames: the lexical
        // one has no keyframes to take from shots.
        let makeSegmenter = { () -> OmniShotCutSegmenter? in
            guard settings.shots else { return nil }
            return OmniShotCutSegmenter(
                configuration: .init(
                    modelDirectory: settings.shotModel.map { URL(fileURLWithPath: $0) }))
        }
        switch settings.profile {
        case .lexical:
            configuration = EngineConfiguration(
                storeURL: storeURL, spaces: [], reranker: nil, processor: nil)
            processor = ContentProcessorChain([
                await SoundEventProcessor(classifier: AppleSoundClassifier()),
                PDFProcessor(),
                VisionOCRProcessor(),
                TextChunkingProcessor(),
            ])

        case .fast:
            let captioner = ModelProfile.Captioning.captioner()
            let transcription = transcriptionProcessor()
            let segmenter = makeSegmenter()
            configuration = ModelProfile.textConfiguration(
                storeURL: storeURL, queryCaptioner: captioner)
            processor = ContentProcessorChain([
                CombinedContentProcessor([
                    transcription,
                    await CaptioningProcessor(
                        wrapping: await keyframes(segmenter), captioner: captioner),
                    await SoundEventProcessor(classifier: AppleSoundClassifier()),
                ]),
                // A scanned page is a picture: captioned like the others, so
                // it is reached by what it shows as well as by its OCR.
                await CaptioningProcessor(wrapping: PDFProcessor(), captioner: captioner),
                await CaptioningProcessor(wrapping: VisionOCRProcessor(), captioner: captioner),
                TextChunkingProcessor(
                    chunker: .default, providerTokenLimit: ModelProfile.Text.maximumInputTokens),
            ])
            release = {
                await captioner.release()
                await transcription.release()
                await segmenter?.release()
            }

        case .multimodal:
            let transcription = transcriptionProcessor()
            let segmenter = makeSegmenter()
            configuration = ModelProfile.multimodalConfiguration(storeURL: storeURL)
            processor = ContentProcessorChain([
                CombinedContentProcessor([
                    transcription,
                    await keyframes(segmenter),
                    await SoundEventProcessor(classifier: AppleSoundClassifier()),
                ]),
                PDFProcessor(),
                VisionOCRProcessor(),
                TextChunkingProcessor(
                    chunker: .default,
                    providerTokenLimit: ModelProfile.Multimodal.maximumInputTokens),
            ])
            release = {
                await transcription.release()
                await segmenter?.release()
            }
        }

        var placeNamer: (any PlaceNamer)?
        if settings.namesPlaces, settings.recordsLocation, #available(macOS 26.0, iOS 26.0, *) {
            placeNamer = MapKitPlaceNamer()
        }
        let metadata = FileMetadataProcessor(
            wrapping: processor, placeNamer: placeNamer,
            configuration: .init(recordsLocation: settings.recordsLocation))
        configuration.processor = metadata
        // The engine opens the pictures it shows the reranker itself: only
        // those `read_file` would open.
        configuration.mediaAccess = { boundary.allows($0) }
        // One model resident at a time only where two do not fit beside the
        // agent: the multimodal 2B pair is ~4 GB. The fast 0.6B pair is
        // ~0.8 GB, and swapping it loaded both models again for every query.
        // Either way they are released after five idle minutes: the MCP server
        // lives as long as the agent beside it, and holds nothing between its
        // bursts of searches.
        configuration.runtime = RuntimePolicy(
            modelIdleTimeout: idleTimeout,
            sequentialModelResidency: settings.profile == .multimodal)
        return Built(
            configuration: configuration, releaseProcessorModels: release,
            namePendingPlaces: { engine in try await metadata.namePendingPlaces(in: engine) })
    }

    /// Keyframes from OmniShotCut's shots, or a candidate a second when shots
    /// are off.
    /// How long the engine keeps its models with nothing to do.
    static let idleTimeout: Duration = .seconds(300)

    static func keyframes(_ segmenter: OmniShotCutSegmenter?) async -> KeyframeProcessor {
        guard let segmenter else { return KeyframeProcessor() }
        return await KeyframeProcessor(segmenter: segmenter)
    }

    /// Parakeet in the model directory when one is set, as the MLX models are:
    /// `<directory>/models/FluidInference/parakeet-tdt-0.6b-v3-coreml`.
    /// Without it FluidAudio keeps its own cache in Application Support.
    static func transcriptionProcessor() -> TranscriptionProcessor {
        var configuration = TranscriptionProcessor.Configuration.default
        if let directory = MLXRuntime.Configuration.fromEnvironment().modelDirectory {
            configuration.transcriber.modelDirectory = parakeetDirectory(in: directory)
        }
        return TranscriptionProcessor(configuration: configuration)
    }

    static func parakeetDirectory(in modelDirectory: URL) -> URL {
        modelDirectory.appendingPathComponent("models/FluidInference/parakeet-tdt-0.6b-v3-coreml")
    }

    static func searchMode(_ profile: Profile, reranks: Bool = true) -> SearchMode {
        profile == .lexical ? .lexicalOnly : reranks ? .hybrid : .retrievalOnly
    }
}

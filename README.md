# SwiftSearch

On-device hybrid search for Swift. It indexes text, PDFs, pictures, recordings
and videos, and answers a query by fusing BM25 with dense retrieval across
every compatible embedding space, then reranking the best candidates. Every
hit says which representation matched and where: a page, a time range, a
bounding box, a character range. Everything runs locally: no server, no API
key, and no network once the models are downloaded.

The package is a library, plus `swiftsearch`, a command-line tool and MCP
server that keeps an index of your folders for you or for an AI agent.

## Requirements

- macOS 14+ or iOS 18+; Apple silicon for the MLX models
- Xcode 26 / Swift 6.3

## Installation

```swift
.package(url: "https://github.com/starkdmi/SwiftSearch", from: "0.1.0")
```

| Product | Depends on | For |
|---|---|---|
| `SemanticSearch` | SQLite, USearch | Storage, BM25, fusion, the engine; chunking on paragraphs and sentences, Markdown at its headings. No models. |
| `SemanticSearchMLX` | MLX | Qwen3 and Qwen3-VL embedding and reranking, EmbeddingGemma 2, SmolVLM2 captions. |
| `SemanticSearchApple` | Vision, PDFKit, AVFoundation, Speech, SoundAnalysis, ImageIO, MapKit | OCR (document structure on macOS/iOS 26), PDFs, video keyframes, SpeechAnalyzer transcription (macOS/iOS 26), sound events, file metadata and place names. |
| `SemanticSearchFluidAudio` | FluidAudio | Parakeet transcription. |
| `SemanticSearchShots` | Core ML | Shots and transitions for keyframes, with OmniShotCut (downloaded). |
| `SemanticSearchCoreML` | Core ML | EmbeddingGemma 2 on Core ML, macOS 15 / iOS 18, without MLX. |
| `SwiftSearchKit` | all of the above, MCP | The library behind the `swiftsearch` CLI and MCP server. |

`SemanticSearch` alone is a working lexical index: lexical search keeps
working on a device that has downloaded nothing.

## Using the library

```swift
import SemanticSearch
import SemanticSearchMLX

// One space over every text-bearing representation, Qwen3 embedder + reranker.
let engine = try await SemanticSearchEngine(
    configuration: ModelProfile.textConfiguration(
        storeURL: URL(fileURLWithPath: "/path/to/index.sqlite")))

// Preparation is separate from search, so a prepared install works offline.
try await engine.prepare()

try await engine.upsert(
    SearchItem(
        id: "note-1",
        metadata: ["folder": .string("research")],
        representations: [
            ContentRepresentation(
                id: "body", kind: .originalText,
                content: .text("Readers never block the writer in WAL mode."),
                language: "en")
        ]))

let results = try await engine.search(
    "does writing block reads?",
    options: SearchOptions(filter: .equals(key: "folder", value: .string("research"))))

for hit in results.hits {
    print(hit.itemID, hit.kind, hit.locator, hit.evidence.fusedScore)
}
```

`upsert`, `remove`, `search`, `searchInside` and `reindex` are the whole
surface; `ingest` adds the processor pipeline for raw files. For a folder,
`sync` does the bookkeeping: new and changed files are ingested, unchanged ones
cost a `stat`, deleted ones are removed, and a moved or renamed file keeps its
index, recognised by its bytes. `listItems` answers without a query, with the
same filters:

```swift
let report = try await engine.sync(folder: URL(fileURLWithPath: "/Users/me/Photos"))
print(report.added.count, report.updated.count, report.removed.count, report.failed)

let june = try await engine.listItems(
    matching: .and([
        .equals(key: "contentType", value: .string("public.image")),
        .between(key: "contentCreated", lower: .date(juneFirst), upper: .date(juneLast)),
    ]),
    order: .metadata("contentCreated", descending: true), limit: 50)
```

### Raw files

Processors turn files into items. Each adds its own children to a recording
or a video — sentences, frames, sounds — so leave out what you do not need:

```swift
var configuration = ModelProfile.textConfiguration(storeURL: url)
configuration.processor = FileMetadataProcessor(
    wrapping: CombinedContentProcessor([
        TranscriptionProcessor(),                               // Parakeet
        await CaptioningProcessor(
            wrapping: KeyframeProcessor(),                      // 32 keyframes a video
            captioner: ModelProfile.Captioning.captioner()),    // SmolVLM2
        await SoundEventProcessor(classifier: AppleSoundClassifier()),
    ]),
    placeNamer: MapKitPlaceNamer())    // optional: sends coordinates to Apple
```

`FileMetadataProcessor` gives every item of a file filterable metadata —
`contentType`, `fileSize`, `contentCreated`, `latitude`/`longitude`, `place`,
`camera`, `duration`, `title`, `artist`, `album`, `author`, `keywords` — and
the file one line of words: its name, folders, title, tags and place, so
"Eiffel Tower" finds a photo whose pixels never say it. Dates and cameras stay
filters: a photo taken with an iPhone is not a photo of one.

A video is indexed as its speech, its sounds and a fixed number of keyframes,
chosen farthest-point-first (as STIMO does) from the middle of every shot.
`KeyframeProcessor(segmenter: OmniShotCutSegmenter())` finds the shots with
[OmniShotCut v1.5](https://github.com/starkdmi/OmniShotCutCoreML) on Core ML,
downloading its 103 MB conversion on first use, so a keyframe is never a
transition or a fade. Without a segmenter the candidates are one a second.

## How a query runs

1. **The filter** is validated once and pushed into every channel: filtering
   one channel and not another silently changes which one contributes.
2. **BM25** (k1 = 1.2, b = 0.75), with script-aware normalisation and
   language-aware lemmas: single-character CJK queries match, Turkish dotted
   and dotless *i* fold, Arabic harakat are stripped.
3. **Dense retrieval**, 100 candidates from each compatible space. A space
   whose model does not match the query is skipped and reported, never
   silently included.
4. **Weighted RRF, k = 60.** Representations of one item collapse within a
   channel before fusion; raw scores from different models are never compared.
   Pictures and text take turns filling the rerank pool, so neither crowds the
   other out.
5. **Rerank** the best 50 items with their evidence and bounded parent
   context. A frame and its caption go together: a VL reranker sees the
   picture and reads the caption.
6. **Return** the top 10 in the reranker's order.

A short text query can be answered from BM25 alone:
`SearchOptions(quickLexicalAnswer: .default)` takes up to four words and two
terms, skips embedding and reranking, and sets
`diagnostics.answeredLexically`. A query picture can be captioned and searched
as text (`EngineConfiguration.queryCaptioner`). Anything the engine could not
evaluate — a missing file, an unsupported modality, a space that failed to
load — is reported in `SearchDiagnostics`, not dropped.

Reranker scores are not calibrated. No threshold on them means anything.

## Storage

SQLite in WAL mode is authoritative: items, representations, postings,
full-precision vectors and indexing state. The USearch HNSW graph is derived —
f16 vectors, full-precision rescoring — and rebuilt from SQLite whenever it is
missing or stale. Snapshots are published atomically after their delta
commits, so an interrupted update cannot expose a deleted or mismatched item.

Upserts fingerprint content and processor configuration, so re-indexing
unchanged material does no work. Each space records the model, weights,
tokenizer and preprocessing versions, instruction, pooling and normalisation:
vectors from different models are rejected rather than compared, even at equal
dimensions.

## Models

| Profile | Embedding | Reranking | Dimensions |
|---|---|---|---|
| Text | Qwen3-Embedding-0.6B | Qwen3-Reranker-0.6B | 1024 |
| Fast (text + captions) | Qwen3-Embedding-0.6B | Qwen3-Reranker-0.6B | 1024 |
| Multimodal | Qwen3-VL-Embedding-2B | Qwen3-VL-Reranker-2B | 2048 |
| EmbeddingGemma (opt-in) | EmbeddingGemma 2 | either pair | 768 |

The fast profile is the text profile with SmolVLM2-500M captioning each
picture at ingest: pictures are found through their captions and no 2B model
is resident, but it cannot search *by* image. The multimodal profile embeds
pixels and text in one space, and a query can be a picture.

Both Qwen pairs default to 5-bit MLX conversions. Every default model is
pinned to a Hub commit, and an index records it: a reference is
`ModelReference("<org>/<name>@<revision>")`, and a new revision opens a new
space rather than mixing its vectors with the old ones. Downloads go where the MLX
loaders put them, unless `MLXRuntime.Configuration.modelDirectory` (or
`SEMANTIC_SEARCH_MODEL_DIRECTORY`) names one directory for all of them.
`MLXRuntime.Configuration.fromEnvironment()` also reads MLX memory bounds:
`SEMANTIC_SEARCH_MLX_MEMORY_LIMIT_MB`, `_CACHE_LIMIT_MB`, `_MEMORY_FRACTION`,
`_CACHE_FRACTION`, and `_HOST_MANAGED` to leave them to the app.

**EmbeddingGemma 2** puts text, images and video in one space (Matryoshka at
512, 256 and 128) and keeps the Qwen3 rerankers:

```swift
// MLX, bf16. vision: false keeps the text tower alone, in the same space.
var configuration = ModelProfile.EmbeddingGemma.configuration(storeURL: url, vision: true)

// Or FluidInference's Core ML conversion: text and images, 512 tokens.
import SemanticSearchCoreML

if #available(macOS 15, iOS 18, *) {
    configuration.spaces[0].id = "embeddinggemma2-coreml"
    configuration.spaces[0].provider = CoreMLEmbeddingGemma2Provider(
        configuration: .init(vision: true))
}
```

The Core ML provider runs on the CPU by default: on the Neural Engine, fp16
pulls text past 256 tokens away from the reference (cosine 0.962 against
transformers in fp32, and not repeatable run to run; 0.9997 on the CPU). Its
vectors are not the MLX provider's, so it needs a space of its own.

## The `swiftsearch` CLI and MCP server

```bash
make install        # → ~/.local/lib/swiftsearch/swiftsearch
```

It has to be built with xcodebuild: MLX's Metal library is a bundle beside
the binary, which `swift build` does not produce.

```bash
S=~/.local/lib/swiftsearch/swiftsearch
$S add ~/Documents/Notes           # add a folder and index it
$S search "the receipt for the bike repair"
$S search "kids on a beach at sunset" --type image --after 2024-06-01
$S list --type video --order newest
$S info ~/Pictures/Trips/IMG_0412.HEIC
$S sync                            # pick up new, changed and deleted files
$S status
```

| Profile | Models | Downloads | Pictures found by |
|---|---|---|---|
| `lexical` | none | — | their words: OCR, tags, file names |
| `fast` (default) | Qwen3-Embedding/Reranker 0.6B, SmolVLM2-500M, Parakeet v3 | ~2 GB | captions |
| `multimodal` | Qwen3-VL-Embedding/Reranker 2B, Parakeet v3 | ~5 GB | their pixels; a picture can be the query |

```bash
$S config --profile multimodal     # each profile keeps its own index
$S config --place-names on         # name places with MapKit: sends coordinates to Apple
$S config --location off           # keep coordinates out of the index
$S config --snippets 0             # answers carry paths and times, no quoted text
$S config --model-directory /Volumes/SSD/models
$S config --shots off              # keyframes from one candidate a second
```

The library lives in `~/Library/Application Support/SwiftSearch`, or wherever
`--home` or `SWIFTSEARCH_HOME` points; it is created readable by you alone.

### Agents

`swiftsearch mcp` speaks MCP on stdin/stdout, indexes new and changed files
when it starts, and unloads its models after five idle minutes.

```bash
claude mcp add swiftsearch -- ~/.local/lib/swiftsearch/swiftsearch mcp
```

| Tool | Does |
|---|---|
| `search` | files by meaning, with where each match is: a page, or seconds into a recording; filters by type, folder, date; `path` searches inside one file |
| `list_files` | files without a query: "photos from June", "every PDF in Invoices" |
| `file_info` | a file's metadata and what was extracted from it |
| `read_file` | one indexed file: a picture, a video frame at a second, a transcript between two seconds, a PDF page, a text file |
| `library_status` | folders, counts by type, indexing progress |
| `sync_library` | index new and changed files in the background |

`read_file` only opens indexed files inside the library's folders. That is
not a sandbox: an agent that can run shell commands can still open the folder,
so deny it in the agent's own permissions as well.

One process owns a library. While the server runs, `swiftsearch search` is
refused with the server's pid, and a second agent session starting the same
server sees only "Connection closed". Give each its own `--home`.

## Development

```bash
make test               # model-free: swift test
make test-integration   # xcodebuild build-for-testing, then xcrun xctest
```

Do not use `swift test` for anything that runs MLX: SwiftPM's test bundle does
not carry MLX's Metal library, and the first kernel fails with `Failed to load
the default metallib`, which looks like a broken model rather than a broken
build. Every model-backed test is gated on an environment variable, so
`make test-integration` passes on a machine that has downloaded nothing:

| Variable | Enables |
|---|---|
| `SEMANTIC_SEARCH_RUN_INTEGRATION_TESTS` | Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B (~1 GB) |
| `SEMANTIC_SEARCH_RUN_MULTIMODAL_TESTS` | Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B (~4 GB) |
| `SEMANTIC_SEARCH_RUN_ASR_TESTS` | Parakeet TDT v3 (~600 MB) |
| `SEMANTIC_SEARCH_RUN_CAPTION_TESTS` | SmolVLM2-500M (~500 MB) |
| `SEMANTIC_SEARCH_RUN_GEOCODING_TESTS` | MapKit reverse geocoding (macOS 26; network) |
| `SEMANTIC_SEARCH_RUN_EMBEDDINGGEMMA_TESTS` | EmbeddingGemma 2 on MLX (1.49 GB) |
| `SEMANTIC_SEARCH_RUN_COREML_EMBEDDINGGEMMA_TESTS` | EmbeddingGemma 2 on Core ML (macOS 15; ~1.5 GB) |

`SEMANTIC_SEARCH_MODEL_DIRECTORY` keeps every download in one place.
`SEMANTIC_SEARCH_TEXT_EMBEDDING_MODEL`, `_TEXT_RERANKER_MODEL`,
`_VL_EMBEDDING_MODEL`, `_VL_RERANKER_MODEL`, `_CAPTION_MODEL` and
`_EMBEDDINGGEMMA_MODEL` point the tests at another Hub id (`@revision` to pin
it) or a local conversion; `SEMANTIC_SEARCH_COREML_EMBEDDINGGEMMA_DIRECTORY` at a complete
local copy of the Core ML conversion.

Vendored code and the mlx-swift-lm fork are described in
[Vendor/README.md](Vendor/README.md).

## License

MIT, see [LICENSE](LICENSE). Vendored code keeps its own licence: USearch,
NumKong and FluidUse are Apache-2.0, and the FluidUse image resize is a port
of Pillow's (MIT-CMU).

The models are downloaded, not distributed with the package, under their own
terms:

| Model | Licence |
|---|---|
| Qwen3-Embedding, Qwen3-Reranker, Qwen3-VL-Embedding, Qwen3-VL-Reranker | Apache-2.0 |
| SmolVLM2-500M-Video-Instruct | Apache-2.0 |
| EmbeddingGemma 2 | Apache-2.0, with Google's Gemma Prohibited Use Policy |
| Parakeet TDT 0.6B v3 | CC-BY-4.0 |
| OmniShotCut v1.5 | MIT |

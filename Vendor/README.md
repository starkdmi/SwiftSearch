# Vendored code

Third-party sources compiled as targets of this package. Each directory keeps
its upstream licence; changed files say so at the top.

## USearch

[unum-cloud/usearch](https://github.com/unum-cloud/usearch) 2.26.2, Apache-2.0:
`include/`, `c/` (`lib.cpp`, `usearch.h`, `module.modulemap`) and the three
Swift sources in `swift/`, unchanged. The targets are declared in this
package's manifest instead of USearch's, so that its NumKong is the copy
below.

## NumKong

[ashvardanian/NumKong](https://github.com/ashvardanian/NumKong) 7.8.2,
Apache-2.0: `include/` and `c/`, unchanged, plus one added file,
`include/numkong_xcode_shim.c`.

`CNumKong` is a header-only C target. SwiftPM builds it, but Xcode expects an
object file for every C target and fails to link any app, executable or test
bundle that depends on it: `Build input file cannot be found: '…/CNumKong.o'`
([swiftlang/swift-package-manager#5706](https://github.com/swiftlang/swift-package-manager/issues/5706),
[ashvardanian/NumKong#353](https://github.com/ashvardanian/NumKong/issues/353)).
It still happens with USearch 2.26.4, NumKong 7.8.5 and Xcode 26.6. The shim is
one translation unit defining one symbol. When upstream gives `CNumKong` a
source file, this copy and USearch's can go, and USearch can be a regular
dependency again.

## FluidUseEmbeddingGemma2

Six files of [FluidInference/FluidUse](https://github.com/FluidInference/FluidUse),
its Swift runtime for the Core ML conversion of EmbeddingGemma 2, Apache-2.0:
five from `main` at `e974b38`, `EmbeddingGemma2Vision.swift` from
[FluidUse#32](https://github.com/FluidInference/FluidUse/pull/32) at `8d7c014`.
Only `SemanticSearchCoreML` uses it. FluidUse itself cannot be a dependency:
it requires a FluidAudio newer than this package's, imports AppKit, and no tag
contains EmbeddingGemma 2.

Changes, each marked in the files:

1. FluidUse#32's Manager and ModelStore hunks, which the vision file calls.
2. Loading from a directory the caller names, at one pinned revision for all
   bundles, every file checked against its SHA-256; a file changed after its
   check is hashed again; one `ensure` at a time per directory (`flock`).
3. Only the image budgets asked for are loaded, defaulting to the reference's
   280 tokens; the packed function only when asked, since it computes rotary
   angles in fp16 and a text's vector would depend on its batch.
4. Matching the reference: images resized as transformers' PIL backend does (a
   port of Pillow's resampler, under its MIT-CMU licence — see `NOTICE`); the
   tokenizer's added tokens split out of text; media placeholders typed in
   text looked up as padding; untruncated token ids, with embedding of id
   sequences the caller built; one sequence of text followed by images.
5. Public marker and placeholder ids and the audio token count.

## mlx-swift-lm

Not vendored: the package depends on
[starkdmi/mlx-swift-lm](https://github.com/starkdmi/mlx-swift-lm) at
`3.32.3-swiftsearch.1`, which is upstream 3.32.3 plus four commits:

1. Qwen3-VL's hidden states in `LMOutput`, which the multimodal embedder pools.
2. Qwen-VL's image placeholders encoded without special tokens.
   Qwen3-VL-Embedding's tokenizer appends `<|endoftext|>` to every encoding,
   so no placeholder matched and every image input failed.
3. EmbeddingGemma 2, from
   [ml-explore/mlx-swift-lm#684](https://github.com/ml-explore/mlx-swift-lm/pull/684)
   at `afc64c6`, squashed.
4. `EmbeddingGemma2.sanitize` non-throwing. A throwing method cannot witness
   3.32.3's non-throwing requirement, and Swift silently takes the protocol's
   no-op default, loading the checkpoint in PyTorch's layout.

An app that also depends on upstream mlx-swift-lm cannot resolve both: they
share a package identity. The fork goes when an upstream release carries what
these commits do.


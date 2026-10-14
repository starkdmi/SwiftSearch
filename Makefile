# Anything that runs MLX has to be built by xcodebuild: SwiftPM does not bundle
# MLX's compiled Metal library, and the first kernel launch from a `swift build`
# product fails with "Failed to load the default metallib".

SCHEME      ?= SwiftSearch-Package
DESTINATION ?= platform=macOS,arch=arm64
DERIVED     ?= .derivedData
PREFIX      ?= $(HOME)/.local/lib/swiftsearch
DEBUG        = $(DERIVED)/Build/Products/Debug
RELEASE      = $(DERIVED)/Build/Products/Release
INTEGRATION  = $(DEBUG)/SemanticSearchIntegrationTests.xctest

.PHONY: test build-tests test-integration test-models build install

# Model-free: the engine, the processors, the CLI's lexical profile.
test:
	swift test --filter SemanticSearchTests
	swift test --filter SwiftSearchKitTests

build-tests:
	xcodebuild build-for-testing -scheme $(SCHEME) \
		-destination '$(DESTINATION)' -derivedDataPath "$(DERIVED)"

# Every model-backed case is gated on an environment variable (README.md), so
# this passes on a machine that has downloaded nothing.
test-integration: build-tests
	xcrun xctest "$(INTEGRATION)"

test-models: build-tests
	SEMANTIC_SEARCH_RUN_INTEGRATION_TESTS=1 \
	SEMANTIC_SEARCH_RUN_MULTIMODAL_TESTS=1 \
	SEMANTIC_SEARCH_RUN_ASR_TESTS=1 \
	SEMANTIC_SEARCH_RUN_CAPTION_TESTS=1 \
		xcrun xctest "$(INTEGRATION)"

build:
	xcodebuild build -scheme swiftsearch -configuration Release \
		-destination '$(DESTINATION)' -derivedDataPath "$(DERIVED)"

# The binary and the resource bundles beside it, which it loads from there.
install: build
	mkdir -p "$(PREFIX)"
	cp -R "$(RELEASE)/swiftsearch" "$(RELEASE)"/*.bundle "$(PREFIX)/"
	@echo "installed $(PREFIX)/swiftsearch"

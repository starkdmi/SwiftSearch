/*
 * Link shim for `xcodebuild`.
 *
 * `CNumKong` is a header-only target: `include/` holds nothing but headers and
 * a module map, and SwiftPM is happy to build it without emitting an object.
 * Xcode's SwiftPM integration is not — it creates a static library for the
 * target regardless and then fails to link any app, executable or test bundle
 * that depends on it, with
 *
 *   Build input file cannot be found: '.../Debug/CNumKong.o'
 *
 * (swiftlang/swift-package-manager#5706, ashvardanian/NumKong#353).
 *
 * One translation unit is enough to give Xcode the object it insists on. It
 * defines a single symbol rather than nothing at all, because an empty
 * translation unit is not valid C and some toolchains discard the resulting
 * archive member.
 */

int nk_xcode_link_shim(void);

int nk_xcode_link_shim(void) { return 0; }

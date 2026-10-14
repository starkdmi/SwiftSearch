import Foundation

/// A Markdown document cut at its headings.
///
/// Each section is the text under one heading, up to the next heading of any
/// level, with the path of headings above it — the layout of LangChain's
/// `MarkdownHeaderTextSplitter` and LlamaIndex's `MarkdownNodeParser`. A
/// chunk of a long section otherwise loses what it is about: the third
/// paragraph under "Install › macOS" never says "macOS".
///
/// Headings are CommonMark's: ATX (`## Title`, up to three spaces of indent,
/// optional closing `#`s) and setext (a paragraph line underlined with `=` or
/// `-`). Nothing inside a fenced code block or a leading YAML front matter
/// block is a heading — a `# comment` in a shell snippet is not a section.
/// Offsets are UTF-16 into the source, as ``TextChunk`` ranges are, so a hit
/// still points at the exact span in the file.
struct MarkdownSections {

    struct Section: Equatable {
        /// Headings from the top level down to this section's own; empty for
        /// text before the first heading.
        var headings: [String]
        /// The section in the source, its heading line included.
        var range: SourceLocator.CharacterRange
        /// Whether anything but the heading is in it.
        var hasBody: Bool
    }

    static func split(_ text: String) -> [Section] {
        let lines = Self.lines(of: text)
        guard !lines.isEmpty else { return [] }

        // Where each heading starts, its level and title.
        var headings: [(line: Int, start: Int, level: Int, title: String)] = []
        var fence: (character: Character, length: Int)?
        var index = 0
        if lines[0].text.trimmingCharacters(in: .whitespaces) == "---" {
            // Front matter runs to the next `---` or `...` line.
            index = 1
            while index < lines.count {
                let trimmed = lines[index].text.trimmingCharacters(in: .whitespaces)
                index += 1
                if trimmed == "---" || trimmed == "..." { break }
            }
        }
        var previousIsParagraph = false
        while index < lines.count {
            let line = lines[index].text
            if let open = fence {
                if let close = Self.fenceMarker(line), close.character == open.character,
                    close.length >= open.length, close.info.isEmpty
                {
                    fence = nil
                }
                previousIsParagraph = false
            } else if let marker = Self.fenceMarker(line) {
                fence = (marker.character, marker.length)
                previousIsParagraph = false
            } else if let (level, title) = Self.atxHeading(line) {
                headings.append((index, lines[index].start, level, title))
                previousIsParagraph = false
            } else if previousIsParagraph, let level = Self.setextUnderline(line) {
                let title = lines[index - 1].text.trimmingCharacters(in: .whitespaces)
                headings.append((index - 1, lines[index - 1].start, level, title))
                previousIsParagraph = false
            } else {
                previousIsParagraph = Self.isParagraphText(line)
            }
            index += 1
        }

        let length = (text as NSString).length
        var sections: [Section] = []
        if let first = headings.first, first.start > 0 {
            let lead = Self.hasText(text, from: 0, to: first.start)
            if lead {
                sections.append(
                    Section(
                        headings: [],
                        range: SourceLocator.CharacterRange(location: 0, length: first.start),
                        hasBody: true))
            }
        } else if headings.isEmpty {
            return [
                Section(
                    headings: [], range: SourceLocator.CharacterRange(location: 0, length: length),
                    hasBody: Self.hasText(text, from: 0, to: length))
            ]
        }

        var path: [(level: Int, title: String)] = []
        for (position, heading) in headings.enumerated() {
            path.removeAll { $0.level >= heading.level }
            path.append((heading.level, heading.title))
            let end = position + 1 < headings.count ? headings[position + 1].start : length
            // The body starts after the heading's line — or, for setext, its
            // underline.
            let isSetext = Self.atxHeading(lines[heading.line].text) == nil
            let bodyLine = heading.line + (isSetext ? 2 : 1)
            let bodyStart = bodyLine < lines.count ? lines[bodyLine].start : length
            sections.append(
                Section(
                    headings: path.map(\.title),
                    range: SourceLocator.CharacterRange(
                        location: heading.start, length: end - heading.start),
                    hasBody: bodyStart < end && Self.hasText(text, from: bodyStart, to: end)))
        }
        return sections
    }

    // MARK: - Lines

    private struct Line {
        var text: String
        /// UTF-16 offset of the line's first character.
        var start: Int
    }

    private static func lines(of text: String) -> [Line] {
        let ns = text as NSString
        var lines: [Line] = []
        var location = 0
        while location < ns.length {
            var lineEnd = 0
            var contentsEnd = 0
            ns.getLineStart(
                nil, end: &lineEnd, contentsEnd: &contentsEnd,
                for: NSRange(location: location, length: 0))
            lines.append(
                Line(
                    text: ns.substring(
                        with: NSRange(location: location, length: contentsEnd - location)),
                    start: location))
            location = lineEnd
        }
        return lines
    }

    private static func hasText(_ text: String, from start: Int, to end: Int) -> Bool {
        guard end > start else { return false }
        return !(text as NSString).substring(with: NSRange(location: start, length: end - start))
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Up to three spaces of indentation, as CommonMark allows before a block
    /// marker; `nil` for four or more (an indented code block) or a tab.
    private static func stripIndent(_ line: String) -> Substring? {
        var rest = Substring(line)
        var spaces = 0
        while rest.first == " " {
            spaces += 1
            rest = rest.dropFirst()
        }
        guard spaces <= 3, rest.first != "\t" else { return nil }
        return rest
    }

    // MARK: - Blocks

    static func atxHeading(_ line: String) -> (level: Int, title: String)? {
        guard let rest = stripIndent(line) else { return nil }
        let hashes = rest.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes) else { return nil }
        let after = rest.dropFirst(hashes)
        guard after.isEmpty || after.first == " " || after.first == "\t" else { return nil }
        var title = after.trimmingCharacters(in: .whitespaces)
        // An optional closing sequence of `#`s, preceded by a space.
        if let range = title.range(of: #"(^|[ \t])#+$"#, options: .regularExpression) {
            title = String(title[..<range.lowerBound]).trimmingCharacters(in: .whitespaces)
        }
        return (hashes, title)
    }

    static func setextUnderline(_ line: String) -> Int? {
        guard let rest = stripIndent(line) else { return nil }
        let trimmed = rest.trimmingCharacters(in: .whitespaces)
        guard let first = trimmed.first, first == "=" || first == "-",
            trimmed.allSatisfy({ $0 == first })
        else { return nil }
        return first == "=" ? 1 : 2
    }

    static func fenceMarker(_ line: String) -> (character: Character, length: Int, info: String)? {
        guard let rest = stripIndent(line), let first = rest.first, first == "`" || first == "~"
        else { return nil }
        let length = rest.prefix { $0 == first }.count
        guard length >= 3 else { return nil }
        let info = rest.dropFirst(length).trimmingCharacters(in: .whitespaces)
        // A backtick fence's info string cannot contain a backtick.
        if first == "`", info.contains("`") { return nil }
        return (first, length, info)
    }

    /// A line that can carry a setext underline: text, not a blank line, a
    /// list item, a quote, a table rule or a thematic break.
    private static func isParagraphText(_ line: String) -> Bool {
        guard let rest = stripIndent(line) else { return false }
        let trimmed = rest.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return false }
        if trimmed.hasPrefix(">") { return false }
        if trimmed.range(of: #"^([-+*]|\d{1,9}[.)])([ \t]|$)"#, options: .regularExpression)
            != nil
        {
            return false
        }
        if trimmed.range(of: #"^([-*_][ \t]*){3,}$"#, options: .regularExpression) != nil {
            return false
        }
        return true
    }
}

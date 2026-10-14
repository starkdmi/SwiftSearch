import Foundation

/// Translates a ``SearchFilter`` into SQL over the `items` and
/// `representations` tables.
///
/// Pushdown matters for two reasons. It keeps the filter identical across
/// retrieval channels — the resolved slot set is handed to BM25 and to every
/// vector index unchanged, so no channel can be filtered differently from
/// another. And it keeps a selective filter from having to be applied *after*
/// retrieval, which is the failure mode where a filter matching 1% of the
/// corpus leaves a 100-candidate dense channel with one survivor.
///
/// Metadata is stored as a JSON object of `{"type": …, "value": …}` pairs, which
/// `json_extract` addresses directly. Unsupported shapes fall back to nothing
/// rather than to everything: a filter the compiler cannot express must not
/// silently widen the result set.
struct FilterCompiler {

    enum Binding {
        case text(String)
        case int(Int64)
        case double(Double)
    }

    struct Compiled {
        var whereClause: String
        /// The `WITH` clause to put before the query, or empty.
        var cte: String
        /// In placeholder order: the CTE's, then the where clause's.
        var bindings: [Binding]
    }

    static func compile(_ filter: SearchFilter) -> Compiled? {
        guard !filter.isUnconstrained else { return nil }
        var bindings: [Binding] = []
        var parents: [String] = []
        let clause = expression(filter, bindings: &bindings, parents: &parents)
        guard !parents.isEmpty else {
            return Compiled(whereClause: clause, cte: "", bindings: bindings)
        }
        let roots = Array(Set(parents)).sorted()
        return Compiled(
            whereClause: clause, cte: subtreeCTE(roots: roots.count),
            bindings: roots.map { .text($0) } + bindings)
    }

    /// Every item under the parents a filter names, with the parent it is
    /// under — the parent included, as `withinParent` includes it.
    ///
    /// Walked down from those parents through `items_parent`, so it costs
    /// the size of their subtrees; `UNION` rather than `UNION ALL`, so a
    /// parent cycle a buggy caller wrote ends instead of recursing forever.
    /// The closure of the whole item tree, which this replaced, cost the
    /// whole index on every call — once per file in a sync that moves or
    /// refreshes files, quadratic in the library. Seeded from the ids
    /// themselves, not from `items`: children name a parent that need not be
    /// stored, and `SearchFilter.matches` follows `parent_id` all the same.
    static func subtreeCTE(roots: Int) -> String {
        let placeholders = Array(repeating: "(?)", count: roots).joined(separator: ",")
        return """
            WITH RECURSIVE subtree(id, root) AS (
                SELECT column1, column1 FROM (VALUES \(placeholders))
                UNION
                SELECT items.id, subtree.root
                FROM items JOIN subtree ON items.parent_id = subtree.id
            )
            """
    }

    private static func expression(
        _ filter: SearchFilter, bindings: inout [Binding], parents: inout [String]
    ) -> String {
        switch filter {
        case .all:
            return "1"
        case .none:
            return "0"

        case .equals(let key, let value):
            return matchClause(key: key, value: value, bindings: &bindings)
        case .notEquals(let key, let value):
            return negated(matchClause(key: key, value: value, bindings: &bindings))
        case .anyOf(let key, let values):
            guard !values.isEmpty else { return "0" }
            let parts = values.map { matchClause(key: key, value: $0, bindings: &bindings) }
            return "(" + parts.joined(separator: " OR ") + ")"

        case .exists(let key):
            return metadataCall("json_extract", key: key, suffix: ".value", bindings: &bindings)
                + " IS NOT NULL"

        case .lessThan(let key, let value):
            return comparison(key: key, value: value, op: "<", bindings: &bindings)
        case .greaterThan(let key, let value):
            return comparison(key: key, value: value, op: ">", bindings: &bindings)
        case .between(let key, let lower, let upper):
            let low = comparison(key: key, value: lower, op: ">=", bindings: &bindings)
            let high = comparison(key: key, value: upper, op: "<=", bindings: &bindings)
            return "(\(low) AND \(high))"

        case .withinParent(let parent):
            parents.append(parent.rawValue)
            bindings.append(.text(parent.rawValue))
            // `IN`, not a correlated `EXISTS`: SQLite can drive the query
            // from the subtree through the items' primary key instead of
            // testing every item.
            return "i.id IN (SELECT id FROM subtree WHERE root = ?)"

        case .hasItemID(let ids):
            guard !ids.isEmpty else { return "0" }
            let sorted = ids.map(\.rawValue).sorted()
            for id in sorted { bindings.append(.text(id)) }
            let placeholders = Array(repeating: "?", count: sorted.count).joined(separator: ",")
            return "i.id IN (\(placeholders))"

        case .hasKind(let kinds):
            guard !kinds.isEmpty else { return "0" }
            let list = kinds.map { "'\($0.rawValue)'" }.sorted().joined(separator: ",")
            return "r.kind IN (\(list))"

        case .and(let filters):
            guard !filters.isEmpty else { return "1" }
            let parts = filters.map {
                expression($0, bindings: &bindings, parents: &parents)
            }
            return "(" + parts.joined(separator: " AND ") + ")"

        case .or(let filters):
            guard !filters.isEmpty else { return "0" }
            let parts = filters.map {
                expression($0, bindings: &bindings, parents: &parents)
            }
            return "(" + parts.joined(separator: " OR ") + ")"

        case .not(let inner):
            return negated(
                expression(inner, bindings: &bindings, parents: &parents))
        }
    }

    /// The complement of `clause`: every row it does not match.
    ///
    /// Plain `NOT` is not that. A clause over a key an item lacks evaluates to
    /// NULL, `NOT NULL` is NULL, and the row is excluded by both the filter and
    /// its negation — so `notEquals("status", "done")` would silently drop
    /// every item with no status. Collapsing NULL to false first makes
    /// `f` and `.not(f)` partition the corpus.
    private static func negated(_ clause: String) -> String {
        "NOT COALESCE((\(clause)), 0)"
    }

    /// Equality, with the same meaning as ``MetadataValue/contains(_:)``.
    ///
    /// Values of different types are never equal — `.integer(5)` does not
    /// match a stored `.double(5)` — so every branch checks the stored type
    /// tag. A string also matches membership in a stored `stringList`, so a
    /// filter written against a scalar keeps working when the field becomes a
    /// list; a `stringList` matches only an identical list.
    ///
    /// Bindings are appended in the order their placeholders appear in the
    /// returned SQL, and `metadataCall` may add one of its own.
    private static func matchClause(
        key: String, value: MetadataValue, bindings: inout [Binding]
    ) -> String {
        switch value {
        case .string(let text):
            let type = typeTest(key: key, types: ["string"], bindings: &bindings)
            let extracted = metadataCall(
                "json_extract", key: key, suffix: ".value", bindings: &bindings)
            bindings.append(.text(text))
            let array = arrayExpression(key: key, bindings: &bindings)
            bindings.append(.text(text))
            return """
                ((\(type) AND \(extracted) = ?) OR EXISTS (
                    SELECT 1 FROM json_each(\(array))
                    WHERE json_each.value = ?))
                """
        case .integer(let number):
            return typedEquality(key: key, type: "integer", .int(number), bindings: &bindings)
        case .double(let number):
            return typedEquality(key: key, type: "double", .double(number), bindings: &bindings)
        case .boolean(let flag):
            // JSON booleans come back from `json_extract` as 0 and 1.
            return typedEquality(key: key, type: "boolean", .int(flag ? 1 : 0), bindings: &bindings)
        case .date(let date):
            return typedEquality(
                key: key, type: "date", .text(MetadataDateFormat.string(from: date)),
                bindings: &bindings)
        case .stringList(let values):
            // Element by element, in order: `json_extract` renders arrays
            // with SQLite's own escaping, so comparing serialised text would
            // depend on two encoders agreeing character for character.
            let type = typeTest(key: key, types: ["stringList"], bindings: &bindings)
            let length = metadataCall(
                "json_array_length", key: key, suffix: ".value", bindings: &bindings)
            bindings.append(.int(Int64(values.count)))
            var parts = [type, "\(length) = ?"]
            for (index, entry) in values.enumerated() {
                let element = metadataCall(
                    "json_extract", key: key, suffix: ".value[\(index)]", bindings: &bindings)
                bindings.append(.text(entry))
                parts.append("\(element) = ?")
            }
            return "(" + parts.joined(separator: " AND ") + ")"
        }
    }

    private static func typedEquality(
        key: String, type: String, _ binding: Binding, bindings: inout [Binding]
    ) -> String {
        let test = typeTest(key: key, types: [type], bindings: &bindings)
        let extracted = metadataCall(
            "json_extract", key: key, suffix: ".value", bindings: &bindings)
        bindings.append(binding)
        return "(\(test) AND \(extracted) = ?)"
    }

    /// True when the stored value under `key` has one of the given type tags.
    private static func typeTest(key: String, types: [String], bindings: inout [Binding])
        -> String
    {
        let tag = metadataCall("json_extract", key: key, suffix: ".type", bindings: &bindings)
        let list = types.map { "'\($0)'" }.joined(separator: ",")
        return "\(tag) IN (\(list))"
    }

    /// A range test, with the same meaning as ``MetadataValue/comparable(to:)``:
    /// numbers compare with numbers (integer and double mixed), dates with
    /// dates, and nothing else has an order. A range over strings, booleans or
    /// lists, or across types, matches nothing — the alternative is SQLite's
    /// cross-type ordering, under which every number is less than every string.
    private static func comparison(
        key: String, value: MetadataValue, op: String, bindings: inout [Binding]
    ) -> String {
        let types: [String]
        let binding: Binding
        switch value {
        case .integer(let number):
            types = ["integer", "double"]
            binding = .int(number)
        case .double(let number):
            types = ["integer", "double"]
            binding = .double(number)
        case .date(let date):
            // ISO-8601 in UTC with fixed width orders correctly as text.
            types = ["date"]
            binding = .text(MetadataDateFormat.string(from: date))
        case .string, .boolean, .stringList:
            return "0"
        }
        let test = typeTest(key: key, types: types, bindings: &bindings)
        let extracted = metadataCall(
            "json_extract", key: key, suffix: ".value", bindings: &bindings)
        bindings.append(binding)
        return "(\(test) AND \(extracted) \(op) ?)"
    }

    /// The value under `key` when it is a JSON array, and an empty array
    /// otherwise.
    ///
    /// `json_each` raises "malformed JSON" on a scalar, and SQLite makes no
    /// promise about short-circuiting `OR` around a subquery — so the guard has
    /// to be inside the argument rather than beside it. Without this, any
    /// equality filter against a plain string or number fails outright.
    private static func arrayExpression(key: String, bindings: inout [Binding]) -> String {
        let type = metadataCall("json_type", key: key, suffix: ".value", bindings: &bindings)
        let value = metadataCall("json_extract", key: key, suffix: ".value", bindings: &bindings)
        return "CASE WHEN \(type) = 'array' THEN \(value) ELSE '[]' END"
    }

    /// The value under `key`, for ordering a listing by it.
    static func metadataValue(key: String, bindings: inout [Binding]) -> String {
        metadataCall("json_extract", key: key, suffix: ".value", bindings: &bindings)
    }

    /// `function` applied to the metadata entry for `key`, at `suffix` within
    /// it.
    ///
    /// The fast form embeds the key in a JSON path literal, `$."key".value`,
    /// which SQLite resolves without scanning the object. A path label has no
    /// escape for `"`, though, so a key containing one cannot be written that
    /// way at all — it would produce a malformed path and fail the whole
    /// query. Those keys are looked up through `json_each` instead, with the
    /// key bound as a parameter: slower, and correct for any key.
    private static func metadataCall(
        _ function: String, key: String, suffix: String, bindings: inout [Binding]
    ) -> String {
        if !key.contains("\"") {
            // Single quotes are the only character that can end the SQL
            // string literal the path sits in.
            let escaped = key.replacingOccurrences(of: "'", with: "''")
            return "\(function)(i.metadata, '$.\"\(escaped)\"\(suffix)')"
        }
        bindings.append(.text(key))
        return "(SELECT \(function)(entry.value, '$\(suffix)') "
            + "FROM json_each(i.metadata) AS entry WHERE entry.key = ?)"
    }
}

enum MetadataDateFormat {
    /// Matches `JSONEncoder.DateEncodingStrategy.iso8601`, which is what the
    /// store writes metadata with. A filter comparing a date has to produce the
    /// same string the encoder did, character for character, or the comparison
    /// is against a value that is not in the column.
    ///
    /// Built per call rather than shared: `ISO8601DateFormatter` is a mutable
    /// class with no documented thread safety, and date filters are rare enough
    /// that the allocation never shows up.
    static func string(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }
}

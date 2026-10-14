import Foundation

extension SearchStore {

    /// Slots a filter admits, or `nil` when the filter admits everything.
    ///
    /// Returning `nil` rather than every slot matters at scale: the common case
    /// is no filter at all, and materialising 100k slot numbers to say "all of
    /// them" would cost more than the query it precedes.
    func resolveFilter(_ filter: SearchFilter) throws -> Set<Int>? {
        guard let compiled = FilterCompiler.compile(filter) else { return nil }

        let sql = """
            \(compiled.cte)
            SELECT r.slot
            FROM representations r
            JOIN items i ON i.id = r.item_id
            WHERE \(compiled.whereClause)
            """

        return try withStatement(sql) { statement in
            Self.bind(compiled.bindings, to: statement)
            var slots: Set<Int> = []
            try statement.forEachRow { slots.insert($0.int(0)) }
            return slots
        }
    }

    static func bind(_ bindings: [FilterCompiler.Binding], to statement: SQLiteStatement) {
        for (offset, binding) in bindings.enumerated() {
            let index = Int32(offset + 1)
            switch binding {
            case .text(let value): statement.bind(index, value)
            case .int(let value): statement.bind(index, value)
            case .double(let value): statement.bind(index, value)
            }
        }
    }

    /// Items a filter admits, a page at a time, and how many there are.
    ///
    /// An item is admitted when the filter holds for it with any of its
    /// representations — as search would find it through any of them — or,
    /// with none, for the item alone: metadata, id and parent, no kind.
    func listItems(
        filter: SearchFilter, topLevelOnly: Bool, order: ItemListing.Order, limit: Int,
        offset: Int
    ) throws -> (items: [StoredItem], total: Int) {
        var bindings: [FilterCompiler.Binding] = []
        var conditions: [String] = []
        var ancestry = ""
        if let compiled = FilterCompiler.compile(filter) {
            conditions.append("(\(compiled.whereClause))")
            bindings = compiled.bindings
            ancestry = compiled.cte
        }
        if topLevelOnly { conditions.append("i.parent_id IS NULL") }
        let from = """
            FROM items i LEFT JOIN representations r ON r.item_id = i.id
            WHERE \(conditions.isEmpty ? "1" : conditions.joined(separator: " AND "))
            """

        let total = try withStatement("\(ancestry) SELECT COUNT(DISTINCT i.id) \(from)") {
            statement in
            Self.bind(bindings, to: statement)
            return try statement.step() ? statement.int(0) : 0
        }

        var pageBindings = bindings
        let ordering: String
        switch order {
        case .id:
            ordering = "i.id"
        case .metadata(let key, let descending):
            // Items without the key last, whichever the direction; the id
            // breaks ties so pages never overlap.
            let isMissing = FilterCompiler.metadataValue(key: key, bindings: &pageBindings)
            let value = FilterCompiler.metadataValue(key: key, bindings: &pageBindings)
            ordering = "(\(isMissing) IS NULL), \(value) \(descending ? "DESC" : "ASC"), i.id"
        }
        pageBindings.append(.int(Int64(max(0, limit))))
        pageBindings.append(.int(Int64(max(0, offset))))
        let sql = """
            \(ancestry)
            SELECT i.id, i.parent_id, i.ordinal, i.locator, i.metadata,
                   i.content_fingerprint, i.processor_fingerprint
            \(from)
            GROUP BY i.id
            ORDER BY \(ordering)
            LIMIT ? OFFSET ?
            """
        let items = try withStatement(sql) { statement in
            Self.bind(pageBindings, to: statement)
            var items: [StoredItem] = []
            try statement.forEachRow { row in
                if let item = try decodeItem(row) { items.append(item) }
            }
            return items
        }
        return (items, total)
    }

    /// Items without a parent whose id starts with `prefix`: the files a
    /// folder sync put in, by their URLs.
    func topLevelItemIDs(withPrefix prefix: String) throws -> [ItemID] {
        try withStatement(
            "SELECT id FROM items WHERE parent_id IS NULL AND substr(id, 1, ?) = ? ORDER BY id"
        ) { statement in
            // SQLite's substr counts code points on text.
            statement.bind(1, prefix.unicodeScalars.count)
            statement.bind(2, prefix)
            var ids: [ItemID] = []
            try statement.forEachRow { row in
                if let id = row.string(0) { ids.append(ItemID(id)) }
            }
            return ids
        }
    }

    /// Ancestors of an item, nearest first. Used to evaluate `withinParent` on
    /// candidates the SQL path did not pre-filter.
    func ancestors(of itemID: ItemID, limit: Int = 32) throws -> [ItemID] {
        try withStatement(
            """
            WITH RECURSIVE chain(id, depth) AS (
                SELECT parent_id, 0 FROM items WHERE id = ?
                UNION ALL
                SELECT items.parent_id, chain.depth + 1
                FROM chain JOIN items ON items.id = chain.id
                WHERE items.parent_id IS NOT NULL AND chain.depth < ?
            )
            SELECT id FROM chain WHERE id IS NOT NULL
            """
        ) { statement in
            statement.bind(1, itemID.rawValue)
            statement.bind(2, limit)
            var result: [ItemID] = []
            try statement.forEachRow { row in
                if let id = row.string(0) { result.append(ItemID(id)) }
            }
            return result
        }
    }
}

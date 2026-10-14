import Foundation

/// One labelled query in an evaluation set.
public struct EvaluationQuery: Sendable, Codable {
    public var id: String
    public var text: String
    /// Slice this query belongs to — a language, a modality, a difficulty band.
    /// The release gate is per-slice as well as overall, because an average
    /// that holds while Arabic collapses is not a passing result.
    public var slice: String
    /// Relevance per item. Absent means not relevant.
    public var relevance: [String: Int]
    /// Applied when the query runs, so a slice can test filtering as well as
    /// ranking.
    public var filterKey: String?
    public var filterValue: String?

    public init(
        id: String,
        text: String,
        slice: String = "default",
        relevance: [String: Int],
        filterKey: String? = nil,
        filterValue: String? = nil
    ) {
        self.id = id
        self.text = text
        self.slice = slice
        self.relevance = relevance
        self.filterKey = filterKey
        self.filterValue = filterValue
    }

    public func gain(for itemID: ItemID) -> Int {
        relevance[itemID.rawValue] ?? 0
    }

    public var relevantItemIDs: Set<String> {
        Set(relevance.filter { $0.value > 0 }.keys)
    }
}

/// Ranking metrics.
///
/// Three, deliberately. nDCG@10 is the release gate because it is sensitive to
/// where in the top ten an answer lands; MRR catches the "is the first result
/// right" question that nDCG averages away; recall@k is what the ANN gate is
/// measured in, and is about the *candidate pool* rather than the final order.
public enum RetrievalMetrics {

    /// Normalised discounted cumulative gain at `k`.
    ///
    /// Graded relevance with the `2^gain − 1` numerator, so a query with one
    /// highly relevant and three marginal answers is not scored the same as one
    /// with four marginal ones.
    public static func ndcg(ranked: [ItemID], query: EvaluationQuery, k: Int) -> Double {
        guard k > 0 else { return 0 }
        var dcg = 0.0
        for (index, itemID) in ranked.prefix(k).enumerated() {
            let gain = query.gain(for: itemID)
            guard gain > 0 else { continue }
            dcg += (pow(2, Double(gain)) - 1) / log2(Double(index + 2))
        }

        let ideal = query.relevance.values.filter { $0 > 0 }.sorted(by: >).prefix(k)
        var idcg = 0.0
        for (index, gain) in ideal.enumerated() {
            idcg += (pow(2, Double(gain)) - 1) / log2(Double(index + 2))
        }

        // No relevant documents at all: the query contributes nothing rather
        // than a perfect or a zero score, either of which would move an average
        // for a reason that has nothing to do with the system.
        guard idcg > 0 else { return .nan }
        return dcg / idcg
    }

    /// Reciprocal rank of the first relevant result, or 0 if there is none.
    public static func reciprocalRank(ranked: [ItemID], query: EvaluationQuery) -> Double {
        for (index, itemID) in ranked.enumerated() where query.gain(for: itemID) > 0 {
            return 1 / Double(index + 1)
        }
        return 0
    }

    public static func recall(ranked: [ItemID], query: EvaluationQuery, k: Int) -> Double {
        let relevant = query.relevantItemIDs
        guard !relevant.isEmpty else { return .nan }
        let retrieved = Set(ranked.prefix(k).map(\.rawValue))
        return Double(retrieved.intersection(relevant).count) / Double(relevant.count)
    }

    /// Overlap between an approximate neighbour list and the exact one, by slot.
    ///
    /// Use ``annRecall(approximate:exact:k:)`` for the release gate: on a
    /// corpus with near-duplicate vectors this measure reports a shortfall for
    /// ties at the k boundary, which is a disagreement about *which* of several
    /// equally-good neighbours to return rather than a failure to find them.
    public static func recall(approximate: [Int], exact: [Int], k: Int) -> Double {
        let truth = Set(exact.prefix(k))
        guard !truth.isEmpty else { return 1 }
        let found = Set(approximate.prefix(k))
        return Double(found.intersection(truth).count) / Double(truth.count)
    }

    /// Tie-aware ANN recall: the measurement the ANN index is held to.
    ///
    /// A returned neighbour counts when it is in the exact top-k *or* when its
    /// score equals the k-th exact score. Without that second clause the metric
    /// punishes a graph for the one thing it cannot get right: when a dozen
    /// documents sit at exactly the same distance and only some of them fit in
    /// the top hundred, every implementation picks a different subset and every
    /// one of them is correct.
    ///
    /// The effect is not hypothetical. On a corpus of near-duplicate documents
    /// this is the difference between a measured 0.989 that no amount of
    /// connectivity or rescoring moves, and the 1.0 the graph actually earns.
    public static func annRecall(
        approximate: [VectorMatch], exact: [VectorMatch], k: Int, tolerance: Float = 1e-5
    ) -> Double {
        let truth = Array(exact.prefix(k))
        guard !truth.isEmpty else { return 1 }
        let truthSlots = Set(truth.map(\.slot))
        let boundary = truth[truth.count - 1].score

        var matched = 0
        for match in approximate.prefix(k) {
            if truthSlots.contains(match.slot) || match.score >= boundary - tolerance {
                matched += 1
            }
        }
        return Double(min(matched, truth.count)) / Double(truth.count)
    }

    /// Mean over values, skipping the `NaN`s that queries with no relevant
    /// documents contribute.
    public static func mean(_ values: [Double]) -> Double {
        let usable = values.filter { !$0.isNaN }
        guard !usable.isEmpty else { return .nan }
        return usable.reduce(0, +) / Double(usable.count)
    }
}

/// Metrics for one run, overall and per slice.
public struct EvaluationReport: Sendable, Codable {
    public struct Slice: Sendable, Codable {
        public var name: String
        public var queryCount: Int
        public var ndcg10: Double
        public var mrr: Double
        public var recall10: Double

        public init(
            name: String, queryCount: Int, ndcg10: Double, mrr: Double, recall10: Double
        ) {
            self.name = name
            self.queryCount = queryCount
            self.ndcg10 = ndcg10
            self.mrr = mrr
            self.recall10 = recall10
        }
    }

    public var label: String
    public var overall: Slice
    public var slices: [Slice]
    /// Median and p95 wall clock per query, in milliseconds.
    public var medianLatencyMilliseconds: Double
    public var p95LatencyMilliseconds: Double
    /// Queries whose search the engine reported degraded, per reason: a space
    /// it skipped or answered only in part (``SearchDiagnostics/skippedSpaces``,
    /// ``SearchDiagnostics/partiallyAnsweredSpaces``). Their rankings are
    /// scored as they came back, so without this a provider that returned no
    /// query vector would read as a model that found nothing. Empty when every
    /// query was searched in full; `nil` in reports recorded before it was
    /// counted.
    public var degradedQueries: [String: Int]?

    public init(
        label: String, overall: Slice, slices: [Slice],
        medianLatencyMilliseconds: Double = 0, p95LatencyMilliseconds: Double = 0
    ) {
        self.label = label
        self.overall = overall
        self.slices = slices
        self.medianLatencyMilliseconds = medianLatencyMilliseconds
        self.p95LatencyMilliseconds = p95LatencyMilliseconds
    }

    /// Build a report from per-query results.
    public static func make(
        label: String,
        results: [(query: EvaluationQuery, ranked: [ItemID], duration: Duration)]
    ) -> EvaluationReport {
        func slice(_ name: String, _ entries: [(EvaluationQuery, [ItemID], Duration)]) -> Slice {
            Slice(
                name: name,
                queryCount: entries.count,
                ndcg10: RetrievalMetrics.mean(
                    entries.map { RetrievalMetrics.ndcg(ranked: $0.1, query: $0.0, k: 10) }),
                mrr: RetrievalMetrics.mean(
                    entries.map { RetrievalMetrics.reciprocalRank(ranked: $0.1, query: $0.0) }),
                recall10: RetrievalMetrics.mean(
                    entries.map { RetrievalMetrics.recall(ranked: $0.1, query: $0.0, k: 10) }))
        }

        let tuples = results.map { ($0.query, $0.ranked, $0.duration) }
        let grouped = Dictionary(grouping: tuples) { $0.0.slice }
        let milliseconds = results
            .map { Double($0.duration.components.seconds) * 1_000
                + Double($0.duration.components.attoseconds) / 1e15 }
            .sorted()

        return EvaluationReport(
            label: label,
            overall: slice("overall", tuples),
            slices: grouped.keys.sorted().map { slice($0, grouped[$0]!) },
            medianLatencyMilliseconds: milliseconds.isEmpty
                ? 0 : milliseconds[milliseconds.count / 2],
            p95LatencyMilliseconds: milliseconds.isEmpty
                ? 0 : milliseconds[min(milliseconds.count - 1, Int(Double(milliseconds.count) * 0.95))])
    }

    /// Result of comparing this report to a recorded baseline.
    public struct GateResult: Sendable {
        public var passed: Bool
        public var failures: [String]

        public init(passed: Bool, failures: [String]) {
            self.passed = passed
            self.failures = failures
        }
    }

    /// The regression gate: nDCG@10 and MRR must retain at least `ratio` of
    /// the baseline, overall *and in every slice the baseline established*.
    ///
    /// A slice present in the baseline and missing here is a failure, not a
    /// pass by omission — dropping the Arabic queries is not a way to keep the
    /// Arabic numbers.
    public func gate(against baseline: EvaluationReport, ratio: Double = 0.98) -> GateResult {
        var failures: [String] = []

        func compare(_ name: String, _ metric: String, current: Double, base: Double) {
            guard !base.isNaN, base > 0 else { return }
            guard !current.isNaN else {
                failures.append("\(name)/\(metric): no measurement in this run")
                return
            }
            let retained = current / base
            if retained < ratio {
                failures.append(
                    String(
                        format: "%@/%@: %.4f vs baseline %.4f (%.1f%% of baseline, need %.1f%%)",
                        name, metric, current, base, retained * 100, ratio * 100))
            }
        }

        compare("overall", "ndcg@10", current: overall.ndcg10, base: baseline.overall.ndcg10)
        compare("overall", "mrr", current: overall.mrr, base: baseline.overall.mrr)

        let bySlice = Dictionary(uniqueKeysWithValues: slices.map { ($0.name, $0) })
        for baseSlice in baseline.slices {
            guard let current = bySlice[baseSlice.name] else {
                failures.append("\(baseSlice.name): slice is missing from this run")
                continue
            }
            compare(baseSlice.name, "ndcg@10", current: current.ndcg10, base: baseSlice.ndcg10)
            compare(baseSlice.name, "mrr", current: current.mrr, base: baseSlice.mrr)
        }

        return GateResult(passed: failures.isEmpty, failures: failures)
    }
}

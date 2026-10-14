import CoreLocation
import Foundation
import MapKit
import SemanticSearch

/// Names places with MapKit's reverse geocoder: "Eiffel Tower, Paris, France".
///
/// Online: each coordinate goes to Apple's servers, which is why
/// ``FileMetadataProcessor`` takes no place namer unless given one. The name
/// is what is at the point — a landmark, or else the street — then the city
/// with its context and the country, each left out when an earlier part
/// already says it, in `locale`'s language or the system's. MapKit's own
/// names: Lisbon is "Lisboa" in English too. `cityWithContext` can be empty
/// where `cityName` is not (Lisbon again), so the city falls back to it.
///
/// Apple rate-limits geocoding and does not document the limit; it has been
/// measured at 50 requests a minute per app, refused with
/// `MKError.loadingThrottled` and the seconds until the window resets
/// (developer forums, thread 795522). Apple's advice is to defer the next
/// request, and to reuse the answer for a location already asked about. So a
/// refused request waits for the reset — the error's own figure, or the
/// 60 s window when it gives none — and is asked again; and answers are kept
/// per coordinate, rounded to three decimals (about 100 m, well inside a
/// city), for the life of the namer. Photos taken together cost one request.
@available(macOS 26.0, iOS 26.0, *)
public actor MapKitPlaceNamer: PlaceNamer {

    public struct Configuration: Sendable, Hashable {
        /// Language of the names; `nil` for the system's.
        public var locale: Locale?
        /// Requests before giving up on a coordinate the server keeps refusing.
        public var maximumAttempts: Int

        public init(locale: Locale? = nil, maximumAttempts: Int = 3) {
            self.locale = locale
            self.maximumAttempts = max(1, maximumAttempts)
        }
    }

    public nonisolated let identifier = "mapkit.reverse-geocoding"
    public nonisolated let configuration: Configuration
    private var names: [Key: String?] = [:]

    private struct Key: Hashable {
        var latitude: Int
        var longitude: Int
        init(_ latitude: Double, _ longitude: Double) {
            self.latitude = Int((latitude * 1000).rounded())
            self.longitude = Int((longitude * 1000).rounded())
        }
    }

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    public nonisolated var configurationFingerprint: String {
        StableHash.of { hasher in
            hasher.append("mapkit-place.v1")
            hasher.append("name-city-region.v2")
            hasher.append(configuration.locale?.identifier ?? "system")
        }
    }

    public func placeName(latitude: Double, longitude: Double) async throws -> String? {
        let key = Key(latitude, longitude)
        if let name = names[key] { return name }
        var attempt = 1
        while true {
            do {
                let name = try await Self.lookup(
                    latitude: latitude, longitude: longitude, locale: configuration.locale)
                names[key] = .some(name)
                return name
            } catch let error as MKError where error.code == .loadingThrottled {
                guard attempt < configuration.maximumAttempts else { throw error }
                attempt += 1
                try await Task.sleep(for: .seconds(Self.secondsUntilReset(error) ?? 60))
                if let name = names[key] { return name }
            }
        }
    }

    @MainActor
    private static func lookup(
        latitude: Double, longitude: Double, locale: Locale?
    ) async throws -> String? {
        guard
            let request = MKReverseGeocodingRequest(
                location: CLLocation(latitude: latitude, longitude: longitude))
        else { return nil }
        if let locale { request.preferredLocale = locale }
        let items = try await request.mapItems
        guard let item = items.first else { return nil }
        let address = item.addressRepresentations
        let city = address?.cityWithContext(.full).flatMap { $0.isEmpty ? nil : $0 }
        return joined([item.name, city ?? address?.cityName, address?.regionName])
    }

    /// The parts in order, each left out when another already says it.
    static func joined(_ candidates: [String?]) -> String? {
        var parts: [String] = []
        for part in candidates {
            guard let part = part?.trimmingCharacters(in: .whitespacesAndNewlines),
                !part.isEmpty,
                !parts.contains(where: { $0.localizedCaseInsensitiveContains(part) })
            else { continue }
            // And the other way: at sea the name is "Mediterranean Sea" and
            // the city with its context "Mediterranean Sea, Türkiye".
            parts.removeAll { part.localizedCaseInsensitiveContains($0) }
            parts.append(part)
        }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }

    /// The reset time MapKit passes along from GeoServices. Not public API —
    /// hence optional, with the window length as the fallback.
    static func secondsUntilReset(_ error: MKError) -> Double? {
        let details = (error as NSError).userInfo["MKErrorGEOErrorUserInfo"] as? [String: Any]
        guard let seconds = (details?["timeUntilReset"] as? NSNumber)?.doubleValue,
            seconds > 0
        else { return nil }
        return seconds
    }
}

//
//  MoodData.swift
//  InSite
//
//  Created by Anand Parikh on 10/8/25.
//

import Foundation
import FirebaseFirestore

// MARK: - Mood event (what the UI saves)
struct MoodEventRecord: StreamRecord {
    static let kind: DataKind = .mood
    static let cadence: Cadence = .event

    let id: String              // UUID
    let timestamp: Date         // client timestamp
    let valence: Double         // [-1, 1]
    let arousal: Double         // [-1, 1]

    var documentId: String { id }
    var payload: [String: Any] {
        [
            "clientTs": Timestamp(date: timestamp),
            "serverTs": FieldValue.serverTimestamp(),
            "valence": valence,
            "arousal": arousal
        ]
    }
}

// OPTIONAL: if you want to persist hourly mood ctx to Firestore too
struct MoodHourlyCtxRecord: StreamRecord {
    static let kind: DataKind = .mood
    static let cadence: Cadence = .hourly

    let hour: Date
    let valence: Double?
    let arousal: Double?
    let quad_posPos: Int?
    let quad_posNeg: Int?
    let quad_negPos: Int?
    let quad_negNeg: Int?
    let hoursSinceMood: Double?

    var documentId: String { isoHourId(hour) }
    var payload: [String: Any] {
        var d: [String: Any] = ["hourUtc": isoHourId(hour)]
        func put(_ k: String, _ v: Any?) { if let v = v { d[k] = v } }
        put("valence", valence); put("arousal", arousal)
        put("quad_posPos", quad_posPos); put("quad_posNeg", quad_posNeg)
        put("quad_negPos", quad_negPos); put("quad_negNeg", quad_negNeg)
        put("hoursSinceMood", hoursSinceMood)
        return d
    }
}


final class MoodCache {
    static let shared = MoodCache()
    private let key = "mood_points_v1"
    private init() {}

    func add(_ m: MoodPoint) {
        var all = load()
        all.append(m)
        if let data = try? JSONEncoder().encode(all) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    func load() -> [MoodPoint] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let arr = try? JSONDecoder().decode([MoodPoint].self, from: data) else { return [] }
        return arr
    }
}

func buildMoodCTXByHour(
    span: ClosedRange<Date>,
    events: [MoodPoint],
    maxCarryHours: Double = 24
) -> [Date: MoodCTX] {
    let sortedEvents = events.sorted { $0.timestamp < $1.timestamp }
    guard !sortedEvents.isEmpty else { return [:] }

    var utcCalendar = Calendar(identifier: .gregorian)
    utcCalendar.timeZone = TimeZone(secondsFromGMT: 0)!

    func floorHour(_ date: Date) -> Date {
        let components = utcCalendar.dateComponents([.year, .month, .day, .hour], from: date)
        return utcCalendar.date(from: components) ?? date
    }

    func quadrant(for event: MoodPoint) -> (Int, Int, Int, Int) {
        let positiveValence = event.valence >= 0
        let positiveArousal = event.arousal >= 0

        switch (positiveValence, positiveArousal) {
        case (true, true):
            return (1, 0, 0, 0)
        case (true, false):
            return (0, 1, 0, 0)
        case (false, true):
            return (0, 0, 1, 0)
        case (false, false):
            return (0, 0, 0, 1)
        }
    }

    var contextByHour: [Date: MoodCTX] = [:]
    var eventIndex = 0
    var latestEvent: MoodPoint?
    var hour = floorHour(span.lowerBound)
    let end = floorHour(span.upperBound)

    while hour <= end {
        while eventIndex < sortedEvents.count, sortedEvents[eventIndex].timestamp <= hour {
            latestEvent = sortedEvents[eventIndex]
            eventIndex += 1
        }

        if let latestEvent {
            let hoursSinceMood = hour.timeIntervalSince(latestEvent.timestamp) / 3600
            if hoursSinceMood <= maxCarryHours {
                let quad = quadrant(for: latestEvent)
                contextByHour[hour] = MoodCTX(
                    hourStartUtc: hour,
                    valence: latestEvent.valence,
                    arousal: latestEvent.arousal,
                    quad_posPos: quad.0,
                    quad_posNeg: quad.1,
                    quad_negPos: quad.2,
                    quad_negNeg: quad.3,
                    hoursSinceMood: max(0, hoursSinceMood)
                )
            }
        }

        guard let nextHour = utcCalendar.date(byAdding: .hour, value: 1, to: hour) else { break }
        hour = nextHour
    }

    return contextByHour
}

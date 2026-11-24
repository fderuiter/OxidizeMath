use egui::Id;

use crate::poi::{CriticalPoint, CriticalPointKind};

#[test]
fn critical_point_kinds_and_records_roundtrip_json() {
    let kinds = [
        CriticalPointKind::Root,
        CriticalPointKind::YIntercept,
        CriticalPointKind::LocalMax,
        CriticalPointKind::LocalMin,
        CriticalPointKind::Intersection,
    ];
    for kind in kinds {
        let encoded = serde_json::to_string(&kind).expect("serialize CriticalPointKind");
        let decoded: CriticalPointKind =
            serde_json::from_str(&encoded).expect("deserialize CriticalPointKind");
        assert_eq!(decoded, kind);
    }

    let points = [
        CriticalPoint {
            kind: CriticalPointKind::LocalMax,
            point: [1.5, 4.25],
            primary_series: "Trajectory A".to_owned(),
            secondary_series: None,
            item_id: Id::new("series_a"),
        },
        CriticalPoint {
            kind: CriticalPointKind::Intersection,
            point: [2.0, 0.0],
            primary_series: "Trajectory A".to_owned(),
            secondary_series: Some("Threshold".to_owned()),
            item_id: Id::new("series_a_x_threshold"),
        },
    ];
    for cp in points {
        let encoded = serde_json::to_string(&cp).expect("serialize CriticalPoint");
        let decoded: CriticalPoint =
            serde_json::from_str(&encoded).expect("deserialize CriticalPoint");
        assert_eq!(decoded, cp);
    }
}

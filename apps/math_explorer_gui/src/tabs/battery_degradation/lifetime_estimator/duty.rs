//! Compact `URL_JSON_BOUNDS`-safe duty-mission segment types (`1..=8` segments)
//! supporting commute, fast-charge, and zero-cycle calendar storage phases.

// [cite:battery_degradation]

use math_explorer::applied::battery_degradation::{
    DutySegment, MAX_C_RATE, MAX_DUTY_SEGMENTS, MAX_TEMPERATURE_C, MIN_C_RATE, MIN_TEMPERATURE_C,
};

/// Canonical duty-segment template kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DutyPhaseKind {
    /// Daily commute cycling phase.
    #[default]
    Commute,
    /// High-power highway fast-charging trip phase.
    FastCharge,
    /// Zero-cycle idle storage phase (`daily_cycles = 0.0`) integrating only $\sqrt{t}$ SEI fade.
    Storage,
}

impl DutyPhaseKind {
    /// Ordered list of segment kinds for UI selectors.
    pub const ALL: [Self; 3] = [Self::Commute, Self::FastCharge, Self::Storage];

    /// Short display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Commute => "Commute",
            Self::FastCharge => "Fast-Charge",
            Self::Storage => "Storage (Idle)",
        }
    }
}

/// Serializable `1..=8` duty segment specification kept compact for `#state=` URL sharing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DutySegmentSpec {
    /// Operating mode label.
    pub kind: DutyPhaseKind,
    /// Segment duration in days (`0.01..=365.25`).
    pub days: f64,
    /// Depth of discharge in percent (`0.0..=100.0`).
    pub dod: f64,
    /// Daily cycles (`0.0..=10.0`; forced to `0.0` for `Storage`).
    pub cycles: f64,
    /// Segment temperature in degrees Celsius (`-20.0..=60.0`).
    pub temp_c: f64,
    /// Segment C-rate (`0.05..=10.0`).
    pub c_rate: f64,
}

impl DutySegmentSpec {
    /// Constructs default operating parameters for `kind`.
    #[must_use]
    pub fn from_kind(kind: DutyPhaseKind) -> Self {
        match kind {
            DutyPhaseKind::Commute => Self {
                kind,
                days: 5.0,
                dod: 50.0,
                cycles: 1.0,
                temp_c: 25.0,
                c_rate: 0.5,
            },
            DutyPhaseKind::FastCharge => Self {
                kind,
                days: 1.0,
                dod: 80.0,
                cycles: 1.5,
                temp_c: 38.0,
                c_rate: 2.5,
            },
            DutyPhaseKind::Storage => Self {
                kind,
                days: 1.0,
                dod: 0.0,
                cycles: 0.0,
                temp_c: 22.0,
                c_rate: 0.5,
            },
        }
    }

    /// Enforces domain bounds and keeps `Storage` segments at `0.0` daily cycles.
    pub fn clamp_in_place(&mut self) {
        self.days = self.days.clamp(0.01, 365.25);
        self.temp_c = self.temp_c.clamp(MIN_TEMPERATURE_C, MAX_TEMPERATURE_C);
        self.c_rate = self.c_rate.clamp(MIN_C_RATE, MAX_C_RATE);
        if self.kind == DutyPhaseKind::Storage {
            self.cycles = 0.0;
            self.dod = 0.0;
        } else {
            self.dod = self.dod.clamp(0.0, 100.0);
            self.cycles = self.cycles.clamp(0.0, 10.0);
        }
    }

    /// Converts this UI segment into a verified [`DutySegment`].
    #[must_use]
    pub fn to_domain(&self) -> DutySegment {
        let cyc = if self.kind == DutyPhaseKind::Storage {
            0.0
        } else {
            self.cycles
        };
        let dod = if self.kind == DutyPhaseKind::Storage {
            0.0
        } else {
            self.dod
        };
        DutySegment::new(self.days, dod, cyc, self.temp_c, self.c_rate)
    }
}

/// Validates decoded duty segments against `1..=8` count and finite physical bounds.
pub fn validate_duty_segments(segments: &[DutySegmentSpec]) -> Result<(), String> {
    if segments.len() > MAX_DUTY_SEGMENTS {
        return Err(format!(
            "Duty schedule exceeds maximum of {MAX_DUTY_SEGMENTS} segments"
        ));
    }
    for (idx, seg) in segments.iter().enumerate() {
        let ok = seg.days.is_finite()
            && (0.01..=365.25).contains(&seg.days)
            && seg.dod.is_finite()
            && (0.0..=100.0).contains(&seg.dod)
            && seg.cycles.is_finite()
            && (0.0..=10.0).contains(&seg.cycles)
            && seg.temp_c.is_finite()
            && (MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C).contains(&seg.temp_c)
            && seg.c_rate.is_finite()
            && (MIN_C_RATE..=MAX_C_RATE).contains(&seg.c_rate);
        if !ok {
            return Err(format!(
                "Duty segment {} has out-of-bounds parameters",
                idx + 1
            ));
        }
    }
    Ok(())
}

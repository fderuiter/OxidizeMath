//! Bounded JSON state persistence, recovery, and starter preset payload for `PharmacokineticsTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AbsorptionMode, EntityMode, PharmacokineticsTool, ReleaseMode, MAX_DOSES, MAX_DOSE_MG,
    MAX_HORIZON_H, MAX_INTERVAL_H, MAX_KA, MAX_KE, MAX_LAG_H, MAX_MEC, MAX_MTC, MAX_SAMPLE_POINTS,
    MAX_TARGET_TMAX_H, MAX_VOLUME_L, MIN_HORIZON_H, MIN_INTERVAL_H, MIN_RATE, MIN_SAMPLE_POINTS,
    MIN_VOLUME_L,
};
use crate::framework::{InteractiveTool, TraceSnapshotManager};
use std::hash::{Hash, Hasher};

#[derive(serde::Serialize)]
struct PharmacokineticsStateRef<'a> {
    dose_mg: f64,
    bioavailability: f64,
    ka: f64,
    ke: f64,
    volume_l: f64,
    target_t_max: f64,
    entity_mode: EntityMode,
    release_mode: ReleaseMode,
    absorption_mode: AbsorptionMode,
    use_loading_dose: bool,
    loading_dose_mg: f64,
    interval_hours: f64,
    n_doses: usize,
    missed_slots: &'a [usize],
    horizon_hours: f64,
    sample_points: usize,
    lag_time_hours: f64,
    f1: f64,
    f2: f64,
    f_d: f64,
    f_l: f64,
    ka_l: f64,
    ke_l: f64,
    mec: f64,
    mtc: f64,
    log10_y: bool,
    #[serde(skip_serializing_if = "TraceSnapshotManager::is_empty")]
    snapshot_manager: &'a TraceSnapshotManager,
}

#[derive(serde::Serialize)]
struct PharmacokineticsRecoveryRef<'a> {
    dose_mg: f64,
    bioavailability: f64,
    ka: f64,
    ke: f64,
    volume_l: f64,
    target_t_max: f64,
    entity_mode: EntityMode,
    release_mode: ReleaseMode,
    absorption_mode: AbsorptionMode,
    use_loading_dose: bool,
    loading_dose_mg: f64,
    interval_hours: f64,
    n_doses: usize,
    missed_slots: &'a [usize],
    horizon_hours: f64,
    sample_points: usize,
    lag_time_hours: f64,
    f1: f64,
    f2: f64,
    f_d: f64,
    f_l: f64,
    ka_l: f64,
    ke_l: f64,
    mec: f64,
    mtc: f64,
    log10_y: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PharmacokineticsState {
    dose_mg: f64,
    bioavailability: f64,
    ka: f64,
    ke: f64,
    volume_l: f64,
    target_t_max: f64,
    entity_mode: EntityMode,
    release_mode: ReleaseMode,
    absorption_mode: AbsorptionMode,
    use_loading_dose: bool,
    loading_dose_mg: f64,
    interval_hours: f64,
    n_doses: usize,
    missed_slots: Vec<usize>,
    horizon_hours: f64,
    sample_points: usize,
    lag_time_hours: f64,
    f1: f64,
    f2: f64,
    f_d: f64,
    f_l: f64,
    ka_l: f64,
    ke_l: f64,
    mec: f64,
    mtc: f64,
    log10_y: bool,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
}

fn sanitize_bounded(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

impl PharmacokineticsTool {
    fn apply_deserialized_state(&mut self, s: PharmacokineticsState, preserve_snapshots: bool) {
        self.dose_mg = s.dose_mg;
        self.bioavailability = s.bioavailability;
        self.ka = s.ka;
        self.ke = s.ke;
        self.volume_l = s.volume_l;
        self.target_t_max = s.target_t_max;
        self.entity_mode = s.entity_mode;
        self.release_mode = s.release_mode;
        self.absorption_mode = s.absorption_mode;
        self.use_loading_dose = s.use_loading_dose;
        self.loading_dose_mg = s.loading_dose_mg;
        self.interval_hours = s.interval_hours;
        self.n_doses = s.n_doses;
        self.missed_slots = s.missed_slots;
        self.horizon_hours = s.horizon_hours;
        self.sample_points = s.sample_points;
        self.lag_time_hours = s.lag_time_hours;
        self.f1 = s.f1;
        self.f2 = s.f2;
        self.f_d = s.f_d;
        self.f_l = s.f_l;
        self.ka_l = s.ka_l;
        self.ke_l = s.ke_l;
        self.mec = s.mec;
        self.mtc = s.mtc;
        self.log10_y = s.log10_y;
        if !preserve_snapshots {
            self.snapshot_manager = s.snapshot_manager;
        }
    }

    fn validate_state_bounds(s: &PharmacokineticsState) -> Result<(), String> {
        let floats_finite = [
            s.dose_mg,
            s.bioavailability,
            s.ka,
            s.ke,
            s.volume_l,
            s.target_t_max,
            s.loading_dose_mg,
            s.interval_hours,
            s.horizon_hours,
            s.lag_time_hours,
            s.f1,
            s.f2,
            s.f_d,
            s.f_l,
            s.ka_l,
            s.ke_l,
            s.mec,
            s.mtc,
        ]
        .into_iter()
        .all(f64::is_finite);
        if !floats_finite {
            return Err("Pharmacokinetics parameters must be finite".into());
        }
        Self::validate_state_ranges(s)
    }

    fn validate_state_ranges(s: &PharmacokineticsState) -> Result<(), String> {
        let in_bounds = (0.0..=MAX_DOSE_MG).contains(&s.dose_mg)
            && (0.0..=1.0).contains(&s.bioavailability)
            && (MIN_RATE..=MAX_KA).contains(&s.ka)
            && (MIN_RATE..=MAX_KE).contains(&s.ke)
            && (MIN_VOLUME_L..=MAX_VOLUME_L).contains(&s.volume_l)
            && (0.0..=MAX_TARGET_TMAX_H).contains(&s.target_t_max)
            && (0.0..=MAX_DOSE_MG).contains(&s.loading_dose_mg)
            && (MIN_INTERVAL_H..=MAX_INTERVAL_H).contains(&s.interval_hours)
            && (1..=MAX_DOSES).contains(&s.n_doses)
            && (MIN_HORIZON_H..=MAX_HORIZON_H).contains(&s.horizon_hours)
            && (MIN_SAMPLE_POINTS..=MAX_SAMPLE_POINTS).contains(&s.sample_points)
            && (0.0..=MAX_LAG_H).contains(&s.lag_time_hours)
            && (0.0..=1.0).contains(&s.f1)
            && (0.0..=1.0).contains(&s.f2)
            && (0.0..=1.0).contains(&s.f_d)
            && (0.0..=1.0).contains(&s.f_l)
            && (MIN_RATE..=MAX_KA).contains(&s.ka_l)
            && (MIN_RATE..=MAX_KE).contains(&s.ke_l)
            && (0.0..=MAX_MEC).contains(&s.mec)
            && (0.01..=MAX_MTC).contains(&s.mtc)
            && s.mtc > s.mec;
        if !in_bounds {
            return Err("Pharmacokinetics parameters out of allowed physical bounds".into());
        }
        if s.missed_slots.len() > s.n_doses || s.missed_slots.iter().any(|&idx| idx >= s.n_doses) {
            return Err("Missed dose slot index exceeds scheduled dose count".into());
        }
        Ok(())
    }

    pub(crate) fn load_pk_state(
        &mut self,
        state: &str,
        preserve_snapshots: bool,
    ) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let mut s: PharmacokineticsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid pharmacokinetics state: {e}"))?;
        Self::validate_state_bounds(&s)?;
        s.missed_slots.sort_unstable();
        s.missed_slots.dedup();

        let mut candidate = Self::default();
        if preserve_snapshots {
            candidate.snapshot_manager = self.snapshot_manager.clone();
        }
        candidate.apply_deserialized_state(s, preserve_snapshots);
        candidate.recalculate();
        if let Some(err) = candidate.solver_error {
            return Err(err);
        }
        if !candidate.has_finite_parameters() {
            return Err("Loaded pharmacokinetics state produced non-finite values".into());
        }
        *self = candidate;
        Ok(())
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.entity_mode.hash(&mut h);
        self.release_mode.hash(&mut h);
        self.absorption_mode.hash(&mut h);
        for val in [
            self.dose_mg,
            self.bioavailability,
            self.ka,
            self.ke,
            self.volume_l,
            self.target_t_max,
            self.loading_dose_mg,
            self.interval_hours,
            self.horizon_hours,
            self.lag_time_hours,
            self.f1,
            self.f2,
            self.f_d,
            self.f_l,
            self.ka_l,
            self.ke_l,
            self.mec,
            self.mtc,
        ] {
            val.to_bits().hash(&mut h);
        }
        self.use_loading_dose.hash(&mut h);
        self.n_doses.hash(&mut h);
        self.missed_slots.hash(&mut h);
        self.sample_points.hash(&mut h);
        self.log10_y.hash(&mut h);
        self.snapshot_manager.len().hash(&mut h);
        self.snapshot_manager.history_len().hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn check_finite_parameters(&self) -> bool {
        let params_ok = [
            self.dose_mg,
            self.bioavailability,
            self.ka,
            self.ke,
            self.volume_l,
            self.target_t_max,
            self.loading_dose_mg,
            self.interval_hours,
            self.horizon_hours,
            self.lag_time_hours,
            self.f1,
            self.f2,
            self.f_d,
            self.f_l,
            self.ka_l,
            self.ke_l,
            self.mec,
            self.mtc,
            self.effective_ka,
        ]
        .into_iter()
        .all(f64::is_finite);
        params_ok
            && (0.0..=1.0).contains(&self.bioavailability)
            && self.dose_mg >= 0.0
            && self.loading_dose_mg >= 0.0
            && self.ka > 0.0
            && self.ke > 0.0
            && self.volume_l > 0.0
            && self.target_t_max >= 0.0
            && self.interval_hours > 0.0
            && self.horizon_hours > 0.0
            && (1..=MAX_DOSES).contains(&self.n_doses)
            && (MIN_SAMPLE_POINTS..=MAX_SAMPLE_POINTS).contains(&self.sample_points)
            && self.mec >= 0.0
            && self.mtc > self.mec
            && self.metrics.is_finite()
            && self
                .total_points
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
            && self
                .plot_total_points
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
    }

    pub(crate) fn clamp_all_parameters(&mut self) -> bool {
        self.dose_mg = sanitize_bounded(self.dose_mg, 0.0, MAX_DOSE_MG, 100.0);
        self.bioavailability = sanitize_bounded(self.bioavailability, 0.0, 1.0, 0.85);
        self.ka = sanitize_bounded(self.ka, MIN_RATE, MAX_KA, 1.2);
        self.ke = sanitize_bounded(self.ke, MIN_RATE, MAX_KE, 0.10);
        self.volume_l = sanitize_bounded(self.volume_l, MIN_VOLUME_L, MAX_VOLUME_L, 40.0);
        self.target_t_max = sanitize_bounded(self.target_t_max, 0.1, MAX_TARGET_TMAX_H, 2.5);
        self.loading_dose_mg = sanitize_bounded(self.loading_dose_mg, 0.0, MAX_DOSE_MG, 200.0);
        self.interval_hours =
            sanitize_bounded(self.interval_hours, MIN_INTERVAL_H, MAX_INTERVAL_H, 12.0);
        self.n_doses = self.n_doses.clamp(1, MAX_DOSES);
        self.missed_slots.retain(|&idx| idx < self.n_doses);
        self.missed_slots.sort_unstable();
        self.missed_slots.dedup();
        self.horizon_hours =
            sanitize_bounded(self.horizon_hours, MIN_HORIZON_H, MAX_HORIZON_H, 72.0);
        self.sample_points = self
            .sample_points
            .clamp(MIN_SAMPLE_POINTS, MAX_SAMPLE_POINTS);
        self.lag_time_hours = sanitize_bounded(self.lag_time_hours, 0.0, MAX_LAG_H, 4.0);
        self.f1 = sanitize_bounded(self.f1, 0.0, 1.0, 0.5);
        self.f2 = sanitize_bounded(self.f2, 0.0, 1.0, 0.5);
        self.f_d = sanitize_bounded(self.f_d, 0.0, 1.0, 0.75);
        self.f_l = sanitize_bounded(self.f_l, 0.0, 1.0, 0.25);
        self.ka_l = sanitize_bounded(self.ka_l, MIN_RATE, MAX_KA, 1.0);
        self.ke_l = sanitize_bounded(self.ke_l, MIN_RATE, MAX_KE, 0.06);
        self.mec = sanitize_bounded(self.mec, 0.0, MAX_MEC, 1.5);
        let min_mtc = (self.mec + 0.1).min(MAX_MTC);
        self.mtc = sanitize_bounded(self.mtc, min_mtc, MAX_MTC, (self.mec + 3.5).min(MAX_MTC));
        if self.absorption_mode == AbsorptionMode::SolveFromTMax {
            self.absorption_mode = AbsorptionMode::DirectKa;
        }
        self.recalculate();
        true
    }

    pub(crate) fn serialize_full_state(&self) -> Option<String> {
        serde_json::to_string(&PharmacokineticsStateRef {
            dose_mg: self.dose_mg,
            bioavailability: self.bioavailability,
            ka: self.ka,
            ke: self.ke,
            volume_l: self.volume_l,
            target_t_max: self.target_t_max,
            entity_mode: self.entity_mode,
            release_mode: self.release_mode,
            absorption_mode: self.absorption_mode,
            use_loading_dose: self.use_loading_dose,
            loading_dose_mg: self.loading_dose_mg,
            interval_hours: self.interval_hours,
            n_doses: self.n_doses,
            missed_slots: &self.missed_slots,
            horizon_hours: self.horizon_hours,
            sample_points: self.sample_points,
            lag_time_hours: self.lag_time_hours,
            f1: self.f1,
            f2: self.f2,
            f_d: self.f_d,
            f_l: self.f_l,
            ka_l: self.ka_l,
            ke_l: self.ke_l,
            mec: self.mec,
            mtc: self.mtc,
            log10_y: self.log10_y,
            snapshot_manager: &self.snapshot_manager,
        })
        .ok()
    }

    pub(crate) fn serialize_recovery_state(&self) -> Option<String> {
        serde_json::to_string(&PharmacokineticsRecoveryRef {
            dose_mg: self.dose_mg,
            bioavailability: self.bioavailability,
            ka: self.ka,
            ke: self.ke,
            volume_l: self.volume_l,
            target_t_max: self.target_t_max,
            entity_mode: self.entity_mode,
            release_mode: self.release_mode,
            absorption_mode: self.absorption_mode,
            use_loading_dose: self.use_loading_dose,
            loading_dose_mg: self.loading_dose_mg,
            interval_hours: self.interval_hours,
            n_doses: self.n_doses,
            missed_slots: &self.missed_slots,
            horizon_hours: self.horizon_hours,
            sample_points: self.sample_points,
            lag_time_hours: self.lag_time_hours,
            f1: self.f1,
            f2: self.f2,
            f_d: self.f_d,
            f_l: self.f_l,
            ka_l: self.ka_l,
            ke_l: self.ke_l,
            mec: self.mec,
            mtc: self.mtc,
            log10_y: self.log10_y,
        })
        .ok()
    }
}

/// Starter preset JSON payload for `PharmacokineticsTool`.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Pharmacokinetics (ADME)",
    "domain": "clinical_trials",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "dose_mg": 150.0,
    "bioavailability": 0.9,
    "ka": 1.4,
    "ke": 0.11552453009332421,
    "volume_l": 35.0,
    "target_t_max": 2.2,
    "entity_mode": "SingleEntity",
    "release_mode": "ImmediateRelease",
    "absorption_mode": "DirectKa",
    "use_loading_dose": true,
    "loading_dose_mg": 250.0,
    "interval_hours": 12.0,
    "n_doses": 6,
    "missed_slots": [2],
    "horizon_hours": 72.0,
    "sample_points": 501,
    "lag_time_hours": 4.0,
    "f1": 0.5,
    "f2": 0.5,
    "f_d": 0.75,
    "f_l": 0.25,
    "ka_l": 1.0,
    "ke_l": 0.06,
    "mec": 2.0,
    "mtc": 6.5,
    "log10_y": false
  }
}"#;

// [cite:clinical_trials_statistics]

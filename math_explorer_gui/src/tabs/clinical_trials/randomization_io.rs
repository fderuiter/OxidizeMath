//! Strata parsing, cumulative arm-balance series builders, and CSV/JSON schedule download
//! helpers for [`super::randomization::RandomizationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::mpsc::{channel, Receiver};

use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use math_explorer::applied::clinical_trials::design::{Group, Patient};

const SCHEDULE_CSV_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Randomization Schedule (*.csv)",
    extensions: &["csv"],
    fallback_filename: "randomization_schedule.csv",
    write_error_prefix: "Failed to save schedule CSV",
};

const SCHEDULE_JSON_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Randomization Schedule (*.json)",
    extensions: &["json"],
    fallback_filename: "randomization_schedule.json",
    write_error_prefix: "Failed to save schedule JSON",
};

/// Single allocated subject row in a trial randomization schedule.
#[derive(Debug, Clone)]
pub struct ScheduleRow {
    /// 1-indexed enrollment sequence number.
    pub subject_index: usize,
    /// Subject identifier (e.g., `"S0001"`).
    pub patient_id: String,
    /// Prognostic stratum label (`"Unstratified"` for simple/block modes).
    pub stratum: String,
    /// Assigned study arm.
    pub group: Group,
}

impl ScheduleRow {
    /// Human-readable arm label (`"Treatment"` or `"Control"`).
    #[must_use]
    pub const fn group_label(&self) -> &'static str {
        match self.group {
            Group::Treatment => "Treatment",
            Group::Control => "Control",
        }
    }
}

/// Parses comma- or newline-delimited `'StratumName:Count'` specifications into `Patient` records.
pub fn parse_strata_patients(spec: &str) -> Result<Vec<Patient>, String> {
    let mut patients = Vec::new();
    for raw_token in spec.split([',', '\n']) {
        let token = raw_token.trim();
        if token.is_empty() {
            continue;
        }
        let (name_raw, count_raw) = token.split_once(':').ok_or_else(|| {
            format!("Invalid stratum entry '{token}' (expected 'StratumName:Count')")
        })?;
        let stratum = name_raw.trim();
        if stratum.is_empty() || stratum.len() > 64 {
            return Err("Stratum name must be 1..=64 characters".into());
        }
        let count = count_raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid count in stratum '{token}'"))?;
        if count == 0 {
            return Err(format!("Stratum '{stratum}' count must be at least 1"));
        }
        if patients.len().saturating_add(count) > math_commons::registry::MAX_ITERATIONS {
            return Err(format!(
                "Total stratified subjects exceed maximum limit ({})",
                math_commons::registry::MAX_ITERATIONS
            ));
        }
        for _ in 0..count {
            let idx = patients.len() + 1;
            patients.push(Patient {
                id: format!("S{idx:04}"),
                stratum: stratum.to_string(),
            });
        }
    }
    if patients.is_empty() {
        return Err("At least one stratum ('Name:Count') is required".into());
    }
    Ok(patients)
}

/// Cumulative `(treatment, control, imbalance)` coordinate trajectories.
pub type BalanceSeriesTuple = (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<[f64; 2]>);

/// Builds `(treatment_cumulative, control_cumulative, imbalance)` series for plotting and export.
#[must_use]
pub fn build_cumulative_series(schedule: &[ScheduleRow]) -> BalanceSeriesTuple {
    let mut trt_pts = Vec::with_capacity(schedule.len() + 1);
    let mut ctrl_pts = Vec::with_capacity(schedule.len() + 1);
    let mut diff_pts = Vec::with_capacity(schedule.len() + 1);
    let mut t_acc = 0.0;
    let mut c_acc = 0.0;
    trt_pts.push([0.0, 0.0]);
    ctrl_pts.push([0.0, 0.0]);
    diff_pts.push([0.0, 0.0]);
    for row in schedule {
        match row.group {
            Group::Treatment => t_acc += 1.0,
            Group::Control => c_acc += 1.0,
        }
        let x = row.subject_index as f64;
        trt_pts.push([x, t_acc]);
        ctrl_pts.push([x, c_acc]);
        diff_pts.push([x, t_acc - c_acc]);
    }
    (trt_pts, ctrl_pts, diff_pts)
}

/// Formats a seeded subject allocation schedule as CSV.
#[must_use]
pub fn format_schedule_csv(schedule: &[ScheduleRow], seed: u64) -> String {
    use std::fmt::Write;
    let mut out = String::from("subject_index,patient_id,stratum,group,seed\n");
    for row in schedule {
        let safe_stratum = row.stratum.replace(',', ";");
        let _ = writeln!(
            &mut out,
            "{},{},{},{},{}",
            row.subject_index,
            row.patient_id,
            safe_stratum,
            row.group_label(),
            seed
        );
    }
    out
}

/// Formats a seeded subject allocation schedule as pretty-printed JSON.
#[must_use]
pub fn format_schedule_json(schedule: &[ScheduleRow], seed: u64, block_size: usize) -> String {
    let items: Vec<serde_json::Value> = schedule
        .iter()
        .map(|r| {
            serde_json::json!({
                "subject_index": r.subject_index,
                "patient_id": r.patient_id,
                "stratum": r.stratum,
                "group": r.group_label(),
            })
        })
        .collect();
    let payload = serde_json::json!({
        "seed": seed,
        "block_size": block_size,
        "total_subjects": schedule.len(),
        "schedule": items,
    });
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
}

/// Non-blocking save-dialog channel wrapper for schedule CSV and JSON downloads.
#[derive(Default)]
pub struct ScheduleSaveIo {
    save_rx: Option<Receiver<Result<String, String>>>,
    /// Latest save status (`is_error`, `message`).
    pub status: Option<(bool, String)>,
}

impl ScheduleSaveIo {
    /// Saves schedule CSV content via `spawn_save_as_file_dialog`.
    pub fn save_csv(&mut self, content: String) {
        self.trigger(SCHEDULE_CSV_CONFIG, "randomization_schedule.csv", content);
    }

    /// Saves schedule JSON content via `spawn_save_as_file_dialog`.
    pub fn save_json(&mut self, content: String) {
        self.trigger(SCHEDULE_JSON_CONFIG, "randomization_schedule.json", content);
    }

    fn trigger(&mut self, cfg: SaveAsDialogConfig, filename: &str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, cfg, filename, content, |path, fallback| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| fallback.to_string())
        });
    }

    /// Polls pending save channel for completion status.
    pub fn poll(&mut self) {
        if let Some(Ok(res)) = self.save_rx.as_ref().map(Receiver::try_recv) {
            self.save_rx = None;
            match res {
                Ok(name) => self.status = Some((false, format!("Saved {name}"))),
                Err(e) => self.status = Some((true, e)),
            }
        }
    }
}

// [cite:clinical_trials_statistics]

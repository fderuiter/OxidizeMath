//! Per-subject allocation schedule records, cumulative arm-balance plots, and `.csv`/`.json`
//! exporters shared by [`super::randomization::RandomizationTool`] and [`super::simpson_paradox::SimpsonParadoxLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::clinical_io::ClinicalIoState;
use super::simpson_paradox::TableCounts;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints};
use math_explorer::applied::clinical_trials::design::{Group, Patient};
use std::collections::HashMap;
use std::fmt::Write as _;

/// Default deterministic RNG seed for trial subject allocation.
pub const DEFAULT_RNG_SEED: u64 = 42;

/// Default multi-stratum enrollment specification.
pub const DEFAULT_STRATA_SPEC: &str = "Age<65:12, Age>=65:12";

/// Returns [`DEFAULT_RNG_SEED`] for `serde(default)`.
#[must_use]
pub const fn default_seed() -> u64 {
    DEFAULT_RNG_SEED
}

/// Returns `true` when `seed` equals [`DEFAULT_RNG_SEED`].
#[must_use]
pub const fn is_default_seed(seed: &u64) -> bool {
    *seed == DEFAULT_RNG_SEED
}

/// Returns [`DEFAULT_STRATA_SPEC`] as an owned `String` for `serde(default)`.
#[must_use]
pub fn default_strata_spec() -> String {
    DEFAULT_STRATA_SPEC.to_string()
}

/// Returns `true` when `spec` matches [`DEFAULT_STRATA_SPEC`].
#[must_use]
pub fn is_default_strata_spec(spec: &str) -> bool {
    spec == DEFAULT_STRATA_SPEC
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

/// Single subject-level randomization assignment retained for audit and `.csv`/`.json` export.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SubjectAllocationRecord {
    /// 1-indexed enrollment sequence number.
    pub sequence: usize,
    /// Subject identifier (e.g., `"SUBJ-0001"` or stratified `"M-0"` / `"S-0"`).
    pub subject_id: String,
    /// Stratum or permuted block label (e.g., `"Unstratified"`, `"Block-1"`, `"Mild"`, `"Severe"`).
    pub stratum: String,
    /// Assigned clinical study arm (`"Treatment"` or `"Control"`).
    pub arm: String,
}

#[derive(serde::Serialize)]
struct ScheduleJsonEnvelope<'a> {
    randomization_mode: &'a str,
    seed_counter: u64,
    total_subjects: usize,
    treatment_subjects: usize,
    control_subjects: usize,
    schedule: &'a [SubjectAllocationRecord],
}

fn group_label(group: Group) -> &'static str {
    match group {
        Group::Treatment => "Treatment",
        Group::Control => "Control",
    }
}

/// Builds subject allocation records for Simple or Permuted Block assignments.
#[must_use]
pub fn build_unstratified_schedule(
    assignments: &[Group],
    block_size: Option<usize>,
) -> Vec<SubjectAllocationRecord> {
    assignments
        .iter()
        .enumerate()
        .map(|(idx, &grp)| {
            let stratum = match block_size {
                Some(bs) if bs > 0 => format!("Block-{}", idx / bs + 1),
                _ => "Unstratified".to_string(),
            };
            SubjectAllocationRecord {
                sequence: idx + 1,
                subject_id: format!("SUBJ-{:04}", idx + 1),
                stratum,
                arm: group_label(grp).to_string(),
            }
        })
        .collect()
}

/// Retains exact per-patient IDs (`M-i`, `S-i`) from `StratifiedRandomizer::assign_stratified`.
#[must_use]
pub fn build_stratified_schedule_from_map(
    patients: &[Patient],
    assigned: &HashMap<String, Group>,
) -> Vec<SubjectAllocationRecord> {
    patients
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let grp = assigned.get(&p.id).copied().unwrap_or(Group::Control);
            SubjectAllocationRecord {
                sequence: idx + 1,
                subject_id: p.id.clone(),
                stratum: p.stratum.clone(),
                arm: group_label(grp).to_string(),
            }
        })
        .collect()
}

/// Synthesizes deterministic per-patient Mild/Severe records matching 2x2 cell counts prior to re-randomization.
#[must_use]
pub fn build_deterministic_stratum_schedule(
    mild_counts: TableCounts,
    severe_counts: TableCounts,
) -> Vec<SubjectAllocationRecord> {
    let mild_t = (mild_counts.treat_event + mild_counts.treat_no_event) as usize;
    let mild_c = (mild_counts.ctrl_event + mild_counts.ctrl_no_event) as usize;
    let sev_t = (severe_counts.treat_event + severe_counts.treat_no_event) as usize;
    let sev_c = (severe_counts.ctrl_event + severe_counts.ctrl_no_event) as usize;

    let mut out = Vec::with_capacity(mild_t + mild_c + sev_t + sev_c);
    let mut seq = 1usize;
    for i in 0..(mild_t + mild_c) {
        let arm = if i < mild_t { "Treatment" } else { "Control" };
        out.push(SubjectAllocationRecord {
            sequence: seq,
            subject_id: format!("M-{i}"),
            stratum: "Mild".to_string(),
            arm: arm.to_string(),
        });
        seq += 1;
    }
    for i in 0..(sev_t + sev_c) {
        let arm = if i < sev_t { "Treatment" } else { "Control" };
        out.push(SubjectAllocationRecord {
            sequence: seq,
            subject_id: format!("S-{i}"),
            stratum: "Severe".to_string(),
            arm: arm.to_string(),
        });
        seq += 1;
    }
    out
}

/// Builds cumulative Treatment, Control, and `Treatment - Control` imbalance trajectories.
#[must_use]
pub fn build_cumulative_balance_datasets(
    schedule: &[SubjectAllocationRecord],
) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut treat_pts = Vec::with_capacity(schedule.len() + 1);
    let mut ctrl_pts = Vec::with_capacity(schedule.len() + 1);
    let mut diff_pts = Vec::with_capacity(schedule.len() + 1);
    treat_pts.push([0.0, 0.0]);
    ctrl_pts.push([0.0, 0.0]);
    diff_pts.push([0.0, 0.0]);

    let (mut cum_t, mut cum_c) = (0usize, 0usize);
    for rec in schedule {
        if rec.arm == "Treatment" {
            cum_t += 1;
        } else {
            cum_c += 1;
        }
        let x = rec.sequence as f64;
        treat_pts.push([x, cum_t as f64]);
        ctrl_pts.push([x, cum_c as f64]);
        diff_pts.push([x, cum_t as f64 - cum_c as f64]);
    }

    vec![
        ("Cumulative Treatment".to_string(), treat_pts),
        ("Cumulative Control".to_string(), ctrl_pts),
        ("Imbalance (Treatment - Control)".to_string(), diff_pts),
    ]
}

/// Formats a subject allocation schedule as UTF-8 CSV.
#[must_use]
pub fn format_schedule_csv(
    mode_label: &str,
    seed_counter: u64,
    schedule: &[SubjectAllocationRecord],
) -> String {
    let mut out = String::with_capacity(schedule.len().saturating_mul(44) + 128);
    let _ = writeln!(out, "sequence,subject_id,stratum,arm,mode,seed_counter");
    for rec in schedule {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{}",
            rec.sequence, rec.subject_id, rec.stratum, rec.arm, mode_label, seed_counter
        );
    }
    out
}

/// Formats a subject allocation schedule as pretty-printed UTF-8 JSON.
#[must_use]
pub fn format_schedule_json(
    mode_label: &str,
    seed_counter: u64,
    schedule: &[SubjectAllocationRecord],
) -> String {
    let treat_n = schedule.iter().filter(|r| r.arm == "Treatment").count();
    let envelope = ScheduleJsonEnvelope {
        randomization_mode: mode_label,
        seed_counter,
        total_subjects: schedule.len(),
        treatment_subjects: treat_n,
        control_subjects: schedule.len().saturating_sub(treat_n),
        schedule,
    };
    serde_json::to_string_pretty(&envelope).unwrap_or_else(|_| "{}".to_string())
}

/// Renders the cumulative arm balance & imbalance `egui_plot::Plot`.
pub fn render_cumulative_balance_plot(
    ui: &mut egui::Ui,
    plot_id: &str,
    schedule: &[SubjectAllocationRecord],
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let series = build_cumulative_balance_datasets(schedule);
    Plot::new(plot_id)
        .height(240.0)
        .legend(Legend::default())
        .x_axis_label("Enrollment Sequence")
        .y_axis_label("Subjects / Arm Imbalance")
        .show(ui, |plot_ui| {
            for (idx, (name, pts)) in series.into_iter().enumerate() {
                plot_ui.line(
                    Line::new(name, PlotPoints::new(pts))
                        .color(palette.series_color(idx * 2))
                        .width(2.2_f32),
                );
            }
            plot_ui.hline(
                HLine::new("Exact Balance (Δ = 0)", 0.0)
                    .color(palette.muted_text)
                    .style(LineStyle::Dashed { length: 6.0 }),
            );
        });
}

/// Renders `.csv` / `.json` download and clipboard actions for a subject allocation schedule.
pub fn render_schedule_export_bar(
    io_state: &mut ClinicalIoState,
    ui: &mut egui::Ui,
    mode_label: &str,
    seed_counter: u64,
    schedule: &[SubjectAllocationRecord],
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Allocation Schedule Exports:").strong());
        if ui
            .button("⬇ Schedule (.csv)")
            .accessible_hover_text("Download per-patient allocation schedule as CSV")
            .clicked()
        {
            let csv = format_schedule_csv(mode_label, seed_counter, schedule);
            io_state.trigger_save_csv("allocation_schedule.csv", csv);
        }
        if ui
            .button("⬇ Schedule (.json)")
            .accessible_hover_text("Download per-patient allocation schedule as JSON")
            .clicked()
        {
            let json = format_schedule_json(mode_label, seed_counter, schedule);
            io_state.trigger_save_json("allocation_schedule.json", json);
        }
        if ui
            .button("📋 Copy CSV")
            .accessible_hover_text("Copy subject allocation CSV schedule to clipboard")
            .clicked()
        {
            ui.ctx()
                .copy_text(format_schedule_csv(mode_label, seed_counter, schedule));
            io_state.record_info(format!(
                "Copied {} subject assignments to clipboard",
                schedule.len()
            ));
        }
    });
}

/// Renders the per-subject roster summary and scrollable enrollment sequence list.
pub fn render_roster_summary(
    ui: &mut egui::Ui,
    error_message: Option<&str>,
    roster: &[SubjectAllocationRecord],
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    if let Some(err) = error_message {
        ui.colored_label(palette.error, format!("Error: {err}"));
        return;
    }
    let trt_n = roster.iter().filter(|r| r.arm == "Treatment").count();
    let ctrl_n = roster.len().saturating_sub(trt_n);
    render_copyable_metric(ui, "Treatment / Control", &format!("{trt_n} / {ctrl_n}"));
    egui::ScrollArea::vertical()
        .id_salt("randomization_roster_scroll")
        .max_height(200.0)
        .show(ui, |ui| {
            for rec in roster {
                ui.label(format!(
                    "#{} | {} ({}): {}",
                    rec.sequence, rec.subject_id, rec.stratum, rec.arm
                ));
            }
        });
}

/// Returns mode-level theory descriptions for `RandomizationTool`.
#[must_use]
pub fn randomization_available_descriptions() -> HashMap<String, String> {
    [
        (
            "Simple",
            "Independent coin-toss allocation with potential small-sample imbalance",
        ),
        (
            "Block",
            "Permuted block allocation ensuring exact balance at block boundaries",
        ),
        (
            "Stratified",
            "Within-stratum permuted block randomization preventing Simpson's paradox reversals",
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Starter preset JSON payload for permuted block randomization.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Permuted Block Randomization",
    "domain": "clinical_trials",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "randomization_type": "Block",
    "n_subjects": 24,
    "block_size": 4
  }
}"#;

// [cite:clinical_trials_statistics]

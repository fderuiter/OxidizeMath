#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::TraceSnapshotManager;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Scalar key performance indicator captured alongside a [`super::TraceSnapshot`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioKpi {
    pub name: String,
    pub value: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub unit: String,
}

impl ScenarioKpi {
    pub fn new(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
        }
    }
}

/// Comparison row contrasting a scenario's scalar KPI against the reference baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct KpiComparisonRow {
    pub scenario_label: String,
    pub kpi_name: String,
    pub unit: String,
    pub value: f64,
    pub baseline_value: f64,
    pub abs_delta: f64,
    pub pct_delta: Option<f64>,
    pub is_reference: bool,
}

impl TraceSnapshotManager {
    /// Computes KPI delta rows across visible pinned baselines, ghost runs, and optional live KPIs.
    pub fn kpi_comparison_rows(&self, current_kpis: &[ScenarioKpi]) -> Vec<KpiComparisonRow> {
        let mut scenarios: Vec<(String, &[ScenarioKpi])> = Vec::new();
        for snap in &self.snapshots {
            if snap.visible && !snap.kpis.is_empty() {
                scenarios.push((snap.display_label(), &snap.kpis));
            }
        }
        for run in &self.history_runs {
            if run.visible && !run.kpis.is_empty() {
                scenarios.push((run.display_label(), &run.kpis));
            }
        }
        if !current_kpis.is_empty() && !scenarios.is_empty() {
            scenarios.push(("Current".to_string(), current_kpis));
        }
        let mut ref_by_name: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        let mut rows = Vec::new();
        for (scen_idx, (scenario_label, kpis)) in scenarios.iter().enumerate() {
            for kpi in *kpis {
                if !kpi.value.is_finite() {
                    continue;
                }
                let &mut (ref_idx, base_val) = ref_by_name
                    .entry(kpi.name.as_str())
                    .or_insert((scen_idx, kpi.value));
                let is_reference = scen_idx == ref_idx;
                let abs_delta = if is_reference {
                    0.0
                } else {
                    kpi.value - base_val
                };
                let pct_delta = if is_reference {
                    (base_val.abs() > 1e-12).then_some(0.0)
                } else if base_val.abs() > 1e-12 {
                    Some((abs_delta / base_val.abs()) * 100.0)
                } else {
                    None
                };
                rows.push(KpiComparisonRow {
                    scenario_label: scenario_label.clone(),
                    kpi_name: kpi.name.clone(),
                    unit: kpi.unit.clone(),
                    value: kpi.value,
                    baseline_value: base_val,
                    abs_delta,
                    pct_delta,
                    is_reference,
                });
            }
        }
        rows
    }

    /// Formats [`Self::kpi_comparison_rows`] as RFC-4180 CSV text.
    pub fn export_kpi_csv(&self, current_kpis: &[ScenarioKpi]) -> String {
        let mut csv = String::from("scenario,kpi,unit,value,baseline,abs_delta,pct_delta\n");
        for row in self.kpi_comparison_rows(current_kpis) {
            let pct_str = row
                .pct_delta
                .map_or_else(String::new, |p| format!("{p:.4}"));
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},{:.6},{pct_str}\n",
                escape_kpi_csv_field(&row.scenario_label),
                escape_kpi_csv_field(&row.kpi_name),
                escape_kpi_csv_field(&row.unit),
                row.value,
                row.baseline_value,
                row.abs_delta,
            ));
        }
        csv
    }
}

fn escape_kpi_csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

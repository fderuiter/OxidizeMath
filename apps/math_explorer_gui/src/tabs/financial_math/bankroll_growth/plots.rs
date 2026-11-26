//! Dataset and figure export builders for [`super::BankrollGrowthTool`].

use super::super::bankroll_ensemble::BankrollLabMode;
use super::BankrollGrowthTool;
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};

fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

fn polyline(name: &str, pts: &[PlotPoint], color: egui::Color32, width: f32) -> ExportPrimitive {
    ExportPrimitive::Polyline {
        name: name.to_string(),
        points: to_pairs(pts).into(),
        color,
        stroke_width: width,
        closed: false,
        fill: None,
    }
}

impl BankrollGrowthTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.mode {
            BankrollLabMode::KellyStrategies => vec![
                ("Full Kelly".into(), to_pairs(&self.full_kelly_points)),
                ("Half Kelly".into(), to_pairs(&self.half_kelly_points)),
                ("Quarter Kelly".into(), to_pairs(&self.quarter_kelly_points)),
                (
                    "Drawdown-Constrained Kelly".into(),
                    to_pairs(&self.risk_constrained_points),
                ),
                (
                    "Active Allocation".into(),
                    to_pairs(&self.active_stake_points),
                ),
            ],
            BankrollLabMode::VolatilityDrag => vec![
                (
                    "Expected Wealth".into(),
                    to_pairs(&self.ensemble_lab.expected_points),
                ),
                (
                    "Analytical Median".into(),
                    to_pairs(&self.ensemble_lab.median_points),
                ),
                (
                    "Ensemble Median".into(),
                    to_pairs(&self.ensemble_lab.empirical_median_points),
                ),
                (
                    "Drawdown-Constrained Median".into(),
                    to_pairs(&self.ensemble_lab.drawdown_median_points),
                ),
                (
                    "5th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p05_points),
                ),
                (
                    "25th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p25_points),
                ),
                (
                    "75th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p75_points),
                ),
                (
                    "95th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p95_points),
                ),
                (
                    "Drawdown Barrier Floor".into(),
                    to_pairs(&self.ensemble_lab.barrier_points),
                ),
                (
                    "Underwater Drawdown (%)".into(),
                    to_pairs(&self.ensemble_lab.underwater_points),
                ),
            ],
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.mode {
            BankrollLabMode::KellyStrategies => vec![
                polyline("Full Kelly", &self.full_kelly_points, pal.series[3], 2.0),
                polyline("Half Kelly", &self.half_kelly_points, pal.series[5], 2.0),
                polyline(
                    "Quarter Kelly",
                    &self.quarter_kelly_points,
                    pal.series[2],
                    2.0,
                ),
                polyline(
                    "Drawdown-Constrained Kelly",
                    &self.risk_constrained_points,
                    pal.series[1],
                    2.0,
                ),
            ],
            BankrollLabMode::VolatilityDrag => vec![
                polyline(
                    "95th Percentile",
                    &self.ensemble_lab.p95_points,
                    pal.series[2],
                    1.5,
                ),
                polyline(
                    "5th Percentile",
                    &self.ensemble_lab.p05_points,
                    pal.warning,
                    1.5,
                ),
                polyline(
                    "Ensemble Median",
                    &self.ensemble_lab.empirical_median_points,
                    pal.series[5],
                    2.2,
                ),
                polyline(
                    "Drawdown-Constrained Median",
                    &self.ensemble_lab.drawdown_median_points,
                    pal.series[0],
                    1.8,
                ),
                polyline(
                    "Analytical Median",
                    &self.ensemble_lab.median_points,
                    pal.series[3],
                    1.8,
                ),
            ],
        }
    }
}

// [cite:clinical_trials_statistics]

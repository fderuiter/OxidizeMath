//! Dataset and offscreen `ExportPrimitive` builders for [`super::WinRatioStressTesterTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::ExportPrimitive;

use super::ui::km_step_points;
use super::{ChartMode, WinRatioStressTesterTool};

impl WinRatioStressTesterTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.chart_mode {
            ChartMode::TierWaterfall => {
                let (summary, tiers) = self.active_summary_and_tiers();
                let denom = (summary.total_pairs.max(1)) as f64 / 100.0;
                vec![
                    (
                        "Treatment Wins (%)".to_string(),
                        vec![
                            [1.0, (tiers[0].wins as f64) / denom],
                            [2.0, (tiers[1].wins as f64) / denom],
                        ],
                    ),
                    (
                        "Treatment Losses (%)".to_string(),
                        vec![
                            [1.0, (tiers[0].losses as f64) / denom],
                            [2.0, (tiers[1].losses as f64) / denom],
                        ],
                    ),
                    (
                        "Unresolved Ties (%)".to_string(),
                        vec![[3.0, (tiers[1].ties_remaining as f64) / denom]],
                    ),
                ]
            }
            ChartMode::CohortSurvival if self.uploaded_cohort.is_some() => {
                let c = self.uploaded_cohort.as_ref().unwrap();
                vec![
                    ("Treatment Survival".to_string(), km_step_points(&c.trt_km)),
                    ("Control Survival".to_string(), km_step_points(&c.ctrl_km)),
                ]
            }
            _ => {
                let rep = self.report();
                vec![
                    (
                        "Empirical Power".to_string(),
                        rep.points
                            .iter()
                            .map(|p| [p.n_per_arm as f64, p.empirical_power])
                            .collect(),
                    ),
                    (
                        "Composite Tie Fraction".to_string(),
                        rep.points
                            .iter()
                            .map(|p| [p.n_per_arm as f64, p.tie_fraction])
                            .collect(),
                    ),
                ]
            }
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        match self.chart_mode {
            ChartMode::TierWaterfall => self.build_waterfall_primitives(),
            ChartMode::CohortSurvival if self.uploaded_cohort.is_some() => {
                let c = self.uploaded_cohort.as_ref().unwrap();
                let palette = crate::app::palette::DARK_PALETTE;
                vec![
                    ExportPrimitive::Polyline {
                        name: "Treatment Survival".to_string(),
                        points: km_step_points(&c.trt_km),
                        color: palette.accent,
                        stroke_width: 2.5,
                        closed: false,
                        fill: None,
                    },
                    ExportPrimitive::Polyline {
                        name: "Control Survival".to_string(),
                        points: km_step_points(&c.ctrl_km),
                        color: palette.error,
                        stroke_width: 2.2,
                        closed: false,
                        fill: None,
                    },
                ]
            }
            _ => self.build_power_primitives(),
        }
    }

    fn build_waterfall_primitives(&self) -> Vec<ExportPrimitive> {
        let (summary, tiers) = self.active_summary_and_tiers();
        let denom = (summary.total_pairs.max(1)) as f64 / 100.0;
        let palette = crate::app::palette::DARK_PALETTE;
        let mk_bar = |name: &str, x: f64, h: f64, fill: egui::Color32| ExportPrimitive::Bar {
            name: name.to_string(),
            min: [x - 0.14, 0.0],
            max: [x + 0.14, h.max(0.25)],
            fill,
            stroke_color: fill,
            stroke_width: 1.0,
        };
        vec![
            mk_bar(
                "Tier 1 Fatal Wins (%)",
                1.0,
                (tiers[0].wins as f64) / denom,
                palette.success,
            ),
            mk_bar(
                "Tier 1 Fatal Losses (%)",
                1.34,
                (tiers[0].losses as f64) / denom,
                palette.error,
            ),
            mk_bar(
                "Tier 2 Non-Fatal Wins (%)",
                2.0,
                (tiers[1].wins as f64) / denom,
                palette.success,
            ),
            mk_bar(
                "Tier 2 Non-Fatal Losses (%)",
                2.34,
                (tiers[1].losses as f64) / denom,
                palette.error,
            ),
            mk_bar(
                "Unresolved Ties (%)",
                3.0,
                (tiers[1].ties_remaining as f64) / denom,
                palette.warning,
            ),
        ]
    }

    fn build_power_primitives(&self) -> Vec<ExportPrimitive> {
        let rep = self.report();
        let palette = crate::app::palette::DARK_PALETTE;
        let pwr_pts: Vec<[f64; 2]> = rep
            .points
            .iter()
            .map(|p| [p.n_per_arm as f64, p.empirical_power])
            .collect();
        let tie_pts: Vec<[f64; 2]> = rep
            .points
            .iter()
            .map(|p| [p.n_per_arm as f64, p.tie_fraction])
            .collect();
        vec![
            ExportPrimitive::Polyline {
                name: "Empirical Power".to_string(),
                points: pwr_pts.clone(),
                color: palette.accent,
                stroke_width: 2.5,
                closed: false,
                fill: None,
            },
            ExportPrimitive::Points {
                name: "Sweep Grid Points".to_string(),
                points: pwr_pts,
                color: palette.accent,
                radius: 4.0,
            },
            ExportPrimitive::Polyline {
                name: "Composite Tie Fraction".to_string(),
                points: tie_pts,
                color: palette.warning,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            },
        ]
    }
}

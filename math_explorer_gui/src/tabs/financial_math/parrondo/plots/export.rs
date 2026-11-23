//! Offscreen dataset and plot primitive export builders for [`super::super::ParrondoParadoxTool`].
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::{ParrondoParadoxTool, ParrondoViewMode};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};

impl ParrondoParadoxTool {
    pub(in super::super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.view_mode {
            ParrondoViewMode::SplitOverview | ParrondoViewMode::PhaseWedge => {
                let mut out = vec![
                    ("Paradox Wedge".into(), to_pairs(&self.paradox_scatter)),
                    (
                        "Non-Paradox Grid".into(),
                        to_pairs(&self.non_paradox_scatter),
                    ),
                    (
                        "Active Operating Point".into(),
                        vec![[self.p_a_base - self.epsilon, self.epsilon]],
                    ),
                ];
                if !self.boundary_schedule.is_empty() {
                    out.push(("E[dW_sched] = 0".into(), to_pairs(&self.boundary_schedule)));
                }
                out
            }
            ParrondoViewMode::WealthTrajectories => vec![
                ("Game A Ensemble".into(), to_pairs(&self.traj_a)),
                ("Game B Ensemble".into(), to_pairs(&self.traj_b)),
                ("50/50 Mixture Ensemble".into(), to_pairs(&self.traj_mix)),
                ("Schedule Ensemble".into(), to_pairs(&self.traj_schedule)),
                (
                    "Schedule Exact Drift".into(),
                    to_pairs(&self.drift_schedule),
                ),
            ],
            ParrondoViewMode::TrapOccupancy => {
                let Some(ref r) = self.report else {
                    return Vec::new();
                };
                let m = self.modulus.min(r.pi_b.len());
                vec![
                    (
                        "pi_B".into(),
                        (0..m).map(|s| [s as f64, r.pi_b[s]]).collect(),
                    ),
                    (
                        "pi_mix".into(),
                        (0..m).map(|s| [s as f64, r.pi_mix[s]]).collect(),
                    ),
                    (
                        "pi_schedule".into(),
                        (0..m).map(|s| [s as f64, r.pi_schedule_mean[s]]).collect(),
                    ),
                ]
            }
        }
    }

    pub(in super::super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.view_mode {
            ParrondoViewMode::SplitOverview | ParrondoViewMode::PhaseWedge => {
                self.export_phase_wedge_primitives(&pal)
            }
            ParrondoViewMode::WealthTrajectories => vec![
                polyline("Game A Ensemble", &self.traj_a, pal.series[0], 1.8),
                polyline("Game B Ensemble", &self.traj_b, pal.series[1], 1.8),
                polyline("50/50 Mix Ensemble", &self.traj_mix, pal.series[2], 1.8),
                polyline("Schedule Ensemble", &self.traj_schedule, pal.series[3], 2.4),
                polyline("Schedule Drift", &self.drift_schedule, pal.success, 1.6),
            ],
            ParrondoViewMode::TrapOccupancy => self.export_occupancy_primitives(&pal),
        }
    }

    fn export_phase_wedge_primitives(&self, pal: &SemanticPalette) -> Vec<ExportPrimitive> {
        let mut prims = vec![
            scatter_prim(
                "Non-Paradox Grid",
                to_pairs(&self.non_paradox_scatter),
                pal.series[5],
                2.2,
            ),
            scatter_prim(
                "Paradox Wedge",
                to_pairs(&self.paradox_scatter),
                pal.success,
                4.2,
            ),
            scatter_prim(
                "Active Operating Point",
                vec![[self.p_a_base - self.epsilon, self.epsilon]],
                pal.accent,
                6.0,
            ),
        ];
        for (name, pts, col) in [
            ("E[dW_A] = 0", &self.boundary_game_a, pal.series[0]),
            ("E[dW_B] = 0", &self.boundary_game_b, pal.series[1]),
            ("E[dW_sched] = 0", &self.boundary_schedule, pal.series[3]),
        ] {
            if pts.len() >= 2 {
                prims.push(polyline(name, pts, col, 2.0));
            }
        }
        prims
    }

    fn export_occupancy_primitives(&self, pal: &SemanticPalette) -> Vec<ExportPrimitive> {
        let Some(ref r) = self.report else {
            return Vec::new();
        };
        let m = self.modulus.min(r.pi_b.len());
        let mut prims = Vec::with_capacity(m * 3);
        for s in 0..m {
            let x = s as f64;
            for (name, dx, val, col) in [
                ("pi_B", -0.22, r.pi_b[s], pal.series[1]),
                ("pi_mix", 0.0, r.pi_mix[s], pal.series[2]),
                ("pi_schedule", 0.22, r.pi_schedule_mean[s], pal.series[3]),
            ] {
                prims.push(ExportPrimitive::Bar {
                    name: format!("{name}[{s}]"),
                    min: [x + dx - 0.09, 0.0],
                    max: [x + dx + 0.09, val],
                    fill: col,
                    stroke_color: pal.accent,
                    stroke_width: 1.0,
                });
            }
        }
        prims
    }
}

pub(super) fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

fn scatter_prim(
    name: &str,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
    radius: f32,
) -> ExportPrimitive {
    ExportPrimitive::Points {
        name: name.into(),
        points,
        color,
        radius,
    }
}

fn polyline(name: &str, pts: &[PlotPoint], color: egui::Color32, width: f32) -> ExportPrimitive {
    ExportPrimitive::Polyline {
        name: name.to_string(),
        points: to_pairs(pts),
        color,
        stroke_width: width,
        closed: false,
        fill: None,
    }
}

// [cite:clinical_trials_statistics]

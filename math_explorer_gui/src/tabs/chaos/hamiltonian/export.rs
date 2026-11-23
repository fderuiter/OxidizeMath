//! Tabular CSV/JSON dataset and vector SVG/PNG primitive assembly for [`HamiltonianChaosLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HamiltonianChaosLab, HamiltonianModelKind};
use egui_plot::ExportPrimitive;
use std::f64::consts::{PI, TAU};

impl HamiltonianChaosLab {
    pub(crate) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        for (idx, slot) in self.orbits.iter().enumerate() {
            if !slot.section_points.is_empty() {
                datasets.push((
                    format!("Poincare Orbit {} (q, p)", idx + 1),
                    slot.section_points.clone(),
                ));
            }
        }
        if self.show_rk4_ghost && !self.ghost.symplectic_drift.is_empty() {
            datasets.push((
                "Symplectic Relative Energy Error (%)".to_string(),
                self.ghost.symplectic_drift.clone(),
            ));
            datasets.push((
                "RungeKutta4 Relative Energy Error (%)".to_string(),
                self.ghost.rk4_drift.clone(),
            ));
        }
        datasets
    }

    pub(crate) fn collect_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut prims = Vec::new();
        let boundary = self.compute_hill_boundary_curve();
        if boundary.len() >= 3 {
            prims.push(ExportPrimitive::Polyline {
                name: "Hill Energy Shell Boundary".to_string(),
                points: boundary.into(),
                color: palette.info,
                stroke_width: 1.5_f32,
                closed: true,
                fill: None,
            });
        }
        for (idx, slot) in self.orbits.iter().enumerate() {
            if !slot.section_points.is_empty() {
                prims.push(ExportPrimitive::Points {
                    name: format!("Orbit {} Section", idx + 1),
                    points: egui_plot::SharedPoints::from_slice(&slot.section_points),
                    color: palette.series_color(slot.color_index),
                    radius: 2.2_f32,
                });
            }
        }
        if self.show_rk4_ghost && !self.ghost.section_points.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "RungeKutta4 Ghost Drift".to_string(),
                points: egui_plot::SharedPoints::from_slice(&self.ghost.section_points),
                color: palette.error,
                radius: 2.0_f32,
            });
        }
        prims
    }

    /// Computes closed `(q, p)` boundary curve of the energetically allowed Poincaré region.
    #[must_use]
    pub fn compute_hill_boundary_curve(&self) -> Vec<[f64; 2]> {
        let samples = 96;
        let mut upper = Vec::with_capacity(samples * 2);
        let mut lower = Vec::with_capacity(samples);
        match self.model {
            HamiltonianModelKind::HenonHeiles => {
                for i in 0..=samples {
                    let q2 = -0.55 + (i as f64 / samples as f64) * 1.40;
                    let max_p2_sq = 2.0 * (self.energy - self.henon.potential_xy(0.0, q2));
                    if max_p2_sq >= 0.0 {
                        let p2 = max_p2_sq.sqrt();
                        upper.push([q2, p2]);
                        lower.push([q2, -p2]);
                    }
                }
            }
            HamiltonianModelKind::DoublePendulum => {
                for i in 0..=samples {
                    let th2 = -PI + (i as f64 / samples as f64) * TAU;
                    let t_req = self.energy - self.pendulum.potential_energy(0.0, th2);
                    if t_req >= 0.0 {
                        let s2 = th2.sin();
                        let eff = self.pendulum.m1 + self.pendulum.m2 * s2 * s2;
                        let p2_max =
                            (2.0 * self.pendulum.m2 * self.pendulum.l2.powi(2) * t_req * eff
                                / self.pendulum.m1)
                                .sqrt();
                        upper.push([th2, p2_max]);
                        lower.push([th2, -p2_max]);
                    }
                }
            }
            HamiltonianModelKind::FigureEightThreeBody => return Vec::new(),
        }
        lower.reverse();
        upper.extend(lower);
        upper
    }
}

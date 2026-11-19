#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ReplicatorDynamicsTool;
use eframe::egui;
use egui_plot::export::EXPORT_PALETTE;
use egui_plot::{ExportPrimitive, ExportSubplotPanel};

type PanelData = (Vec<(String, Vec<[f64; 2]>)>, Vec<ExportPrimitive>);

impl ReplicatorDynamicsTool {
    pub(super) fn step_export_animation(&mut self) {
        if self.trajectory.is_empty() {
            return;
        }
        let step_dt = (self.time_horizon / 20.0).max(self.dt);
        self.playback_time += step_dt;
        if self.playback_time > self.time_horizon {
            self.playback_time = 0.0;
        }
    }

    pub(super) fn cold_export_subplot_panels(&self) -> Vec<ExportSubplotPanel> {
        let captured = egui_plot::get_last_plot_subplots();
        let find_panel = |id: &str| captured.iter().find(|p| p.id == id);
        let (traj_ds, traj_prims) = self.build_trajectory_panel_data();
        let (dist_ds, dist_prims) = self.build_distribution_panel_data();

        let top_meta = find_panel("replicator_trajectories");
        let bot_meta = find_panel("population_distribution");
        vec![
            ExportSubplotPanel {
                id: "replicator_trajectories".to_string(),
                screen_rect: top_meta
                    .map(|p| p.screen_rect)
                    .filter(|r| r.width() > 1.0 && r.height() > 1.0)
                    .unwrap_or_else(|| {
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 380.0))
                    }),
                log_x: top_meta.is_some_and(|p| p.log_x),
                log_y: top_meta.is_some_and(|p| p.log_y),
                datasets: traj_ds,
                primitives: traj_prims,
            },
            ExportSubplotPanel {
                id: "population_distribution".to_string(),
                screen_rect: bot_meta
                    .map(|p| p.screen_rect)
                    .filter(|r| r.width() > 1.0 && r.height() > 1.0)
                    .unwrap_or_else(|| {
                        egui::Rect::from_min_max(egui::pos2(0.0, 400.0), egui::pos2(800.0, 600.0))
                    }),
                log_x: bot_meta.is_some_and(|p| p.log_x),
                log_y: bot_meta.is_some_and(|p| p.log_y),
                datasets: dist_ds,
                primitives: dist_prims,
            },
        ]
    }

    fn build_trajectory_panel_data(&self) -> PanelData {
        let mut traj_ds = Vec::new();
        let mut traj_prims = Vec::new();
        if self.trajectory.is_empty() {
            return (traj_ds, traj_prims);
        }
        let n_strat = self.trajectory[0].1.len();
        for i in 0..n_strat {
            let strat = self
                .strategy_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("Strategy {i}"));
            let label = format!("replicator_trajectories: {strat}");
            let pts: Vec<[f64; 2]> = self
                .trajectory
                .iter()
                .map(|(t, state)| [*t, state[i]])
                .collect();
            traj_ds.push((label.clone(), pts.clone()));
            traj_prims.push(ExportPrimitive::Polyline {
                name: label,
                points: pts,
                color: EXPORT_PALETTE[i % EXPORT_PALETTE.len()],
                stroke_width: 2.2,
                closed: false,
                fill: None,
            });
        }
        traj_prims.push(ExportPrimitive::Polyline {
            name: "replicator_trajectories: Current Time".to_string(),
            points: vec![[self.playback_time, 0.0], [self.playback_time, 1.0]],
            color: egui::Color32::from_rgb(220, 224, 232),
            stroke_width: 1.5,
            closed: false,
            fill: None,
        });
        (traj_ds, traj_prims)
    }

    fn build_distribution_panel_data(&self) -> PanelData {
        let mut dist_ds = Vec::new();
        let mut dist_prims = Vec::new();
        if self.trajectory.is_empty() {
            return (dist_ds, dist_prims);
        }
        let idx = ((self.playback_time / self.dt).round() as usize)
            .min(self.trajectory.len().saturating_sub(1));
        let (_, state) = &self.trajectory[idx];
        let mut bar_pts = Vec::with_capacity(state.len());
        for (i, &val) in state.iter().enumerate() {
            let strat = self
                .strategy_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("Strategy {i}"));
            let color = EXPORT_PALETTE[i % EXPORT_PALETTE.len()];
            bar_pts.push([i as f64, val]);
            dist_prims.push(ExportPrimitive::Bar {
                name: format!("population_distribution: {strat}"),
                min: [i as f64 - 0.25, 0.0],
                max: [i as f64 + 0.25, val],
                fill: color.linear_multiply(0.75),
                stroke_color: color,
                stroke_width: 1.5,
            });
        }
        dist_ds.push(("population_distribution: Distribution".to_string(), bar_pts));
        (dist_ds, dist_prims)
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        for panel in self.cold_export_subplot_panels() {
            out.extend(panel.datasets);
        }
        out
    }
}

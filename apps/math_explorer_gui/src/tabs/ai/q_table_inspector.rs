//! Background Q-Table and Bellman optimality inspector with live maze sculpting and TD sonification.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::rl_sculpt::{
    apply_rl_brush_stroke, build_rl_sculpt_payload, new_rl_brush_kit, render_rl_sculpt_toolbar,
    solve_bellman_with_policy, RlMazePreset, RlSolverMode,
};
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::audio::{render_sonification_panel, SonificationFrame};
use crate::framework::{
    InputMode, InteractionContext, InteractiveTool, PlaybackAction, PlaybackTransport,
    SpatialBrushKit,
};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, Move};
use math_explorer::ai::reinforcement_learning::{move_to_index, DenseRlWorkspace, SculptedGridMdp};

#[path = "q_table_inspector/canvas.rs"]
mod canvas;
#[path = "q_table_inspector/export.rs"]
mod export;
#[path = "q_table_inspector/preset.rs"]
mod preset;
#[path = "q_table_inspector/runner.rs"]
mod runner;
#[cfg(test)]
#[path = "q_table_inspector/tests.rs"]
mod tests;

pub use runner::*;

/// Scalar heatmap layer displayed in [`QTableInspectorTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) enum HeatmapView {
    MaxQ,
    Up,
    Down,
    Left,
    Right,
    BellmanVStar,
    PolicyVPi,
    BellmanResidual,
}

impl HeatmapView {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::MaxQ => "Max Q(s, a)",
            Self::Up => "Move Up Q",
            Self::Down => "Move Down Q",
            Self::Left => "Move Left Q",
            Self::Right => "Move Right Q",
            Self::BellmanVStar => "Bellman Optimal V*(s)",
            Self::PolicyVPi => "Policy Value V_π(s)",
            Self::BellmanResidual => "Residual |V - V*|",
        }
    }
}

/// Interactive Q-table & Bellman value-function inspector backed by a worker thread.
pub struct QTableInspectorTool {
    pub(super) mdp: SculptedGridMdp,
    pub(super) solver_mode: RlSolverMode,
    pub(super) maze_preset: RlMazePreset,
    pub(super) brush_kit: SpatialBrushKit,
    pub(super) latest_snap: QTableSnapshot,
    pub(super) episodes_trained: u32,
    pub(super) view: HeatmapView,
    pub(super) latest_audio: SonificationFrame,
    pub(super) revision: u64,
    pub(crate) worker: BackgroundWorker<SculptedQTableRunner>,
}

impl Default for QTableInspectorTool {
    fn default() -> Self {
        let mdp = SculptedGridMdp::default_with_traps(5, 5, 0.9);
        let runner = SculptedQTableRunner::from_mdp(mdp.clone(), RlSolverMode::TabularQ);
        let latest_snap = runner.build_q_snapshot();
        let worker = BackgroundWorker::new(runner);
        let mut tool = Self {
            mdp,
            solver_mode: RlSolverMode::TabularQ,
            maze_preset: RlMazePreset::BaselineTraps,
            brush_kit: new_rl_brush_kit(),
            latest_snap,
            episodes_trained: 0,
            view: HeatmapView::MaxQ,
            latest_audio: SonificationFrame::silent(),
            revision: 0,
            worker,
        };
        tool.sync_from_latest();
        tool
    }
}

fn extract_snapshot_parts(
    snapshot: &crate::async_sim::StateSnapshot,
) -> (Option<SonificationFrame>, Option<QTableSnapshot>) {
    let audio = if snapshot.sonification.is_audible() {
        Some(snapshot.sonification)
    } else {
        None
    };
    let data = snapshot
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<QTableSnapshot>())
        .cloned();
    (audio, data)
}

impl QTableInspectorTool {
    pub(super) fn sync_from_latest(&mut self) {
        let parts = self.worker.latest_snapshot().map(extract_snapshot_parts);
        if let Some((audio, data)) = parts {
            self.apply_snapshot_parts(audio, data);
        }
    }

    fn apply_snapshot_parts(
        &mut self,
        audio: Option<SonificationFrame>,
        data: Option<QTableSnapshot>,
    ) {
        if let Some(frame) = audio {
            self.latest_audio = frame;
        }
        if let Some(snap) = data {
            self.episodes_trained = snap.episodes_trained;
            self.mdp = snap.mdp.clone();
            self.solver_mode = snap.solver_mode;
            self.latest_snap = snap;
        }
    }

    fn refresh_snapshot_bellman(&mut self) {
        let (w, h) = (self.mdp.width as usize, self.mdp.height as usize);
        let mut temp_ws = DenseRlWorkspace::new(w, h, 100, None);
        if self.latest_snap.q_table.len() == temp_ws.q_table().len() {
            temp_ws
                .q_table_mut()
                .copy_from_slice(&self.latest_snap.q_table);
            temp_ws
                .state_values_mut()
                .copy_from_slice(&self.latest_snap.state_values);
        }
        self.latest_snap.mdp = self.mdp.clone();
        self.latest_snap.solver_mode = self.solver_mode;
        self.latest_snap.bellman = solve_bellman_with_policy(&self.mdp, &temp_ws, self.solver_mode);
    }

    fn sync_sculpt_to_worker(&mut self) {
        self.refresh_snapshot_bellman();
        let payload = build_rl_sculpt_payload(&self.mdp, self.solver_mode);
        let _ = self
            .worker
            .send_command(SimCommand::SyncSculptState(payload));
        self.revision = self.revision.wrapping_add(1);
    }

    /// Resets the background RL agent in place or respawns a disconnected worker.
    pub fn reset(&mut self) {
        let mdp_clone = self.mdp.clone();
        let solver = self.solver_mode;
        self.worker
            .reset_or_respawn(move || SculptedQTableRunner::from_mdp(mdp_clone, solver));
        let payload = build_rl_sculpt_payload(&self.mdp, self.solver_mode);
        let _ = self
            .worker
            .send_command(SimCommand::SyncSculptState(payload));
        let ws =
            DenseRlWorkspace::new(self.mdp.width as usize, self.mdp.height as usize, 100, None);
        self.latest_snap = QTableSnapshot {
            episodes_trained: 0,
            mdp: self.mdp.clone(),
            solver_mode: self.solver_mode,
            q_table: ws.q_table().to_vec(),
            state_values: ws.state_values().to_vec(),
            greedy_actions: vec![None; ws.q_table().len()],
            bellman: solve_bellman_with_policy(&self.mdp, &ws, self.solver_mode),
        };
        self.episodes_trained = 0;
        self.latest_audio = SonificationFrame::silent();
        self.sync_from_latest();
    }

    #[cfg(test)]
    pub fn train(&mut self, episodes: u32) {
        let target = self.episodes_trained + episodes;
        let _ = self
            .worker
            .send_command(SimCommand::TrainEpisodes(episodes as usize));
        for _ in 0..200 {
            let parts = self.worker.update().map(extract_snapshot_parts);
            if let Some((audio, data)) = parts {
                self.apply_snapshot_parts(audio, data);
                if self.episodes_trained >= target {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    pub(super) fn get_q(&self, state: &GridState, action: &Move) -> f64 {
        if !self.mdp.in_bounds(*state) {
            return 0.0;
        }
        let idx = flatten_2d_index(state.x as usize, state.y as usize, self.mdp.width as usize);
        self.latest_snap
            .q_table
            .get(idx)
            .map_or(0.0, |row| row[move_to_index(*action)])
    }

    pub(super) fn get_q_value_for_view(&self, state: &GridState) -> f64 {
        if !self.mdp.in_bounds(*state) {
            return 0.0;
        }
        let idx = flatten_2d_index(state.x as usize, state.y as usize, self.mdp.width as usize);
        match self.view {
            HeatmapView::MaxQ => match self.solver_mode {
                RlSolverMode::TabularQ => self.latest_snap.q_table.get(idx).map_or(0.0, |row| {
                    row.iter().copied().fold(0.0_f64.min(row[0]), f64::max)
                }),
                RlSolverMode::PolicyGradient => self
                    .latest_snap
                    .state_values
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0),
            },
            HeatmapView::Up => self.get_q(state, &Move::Up),
            HeatmapView::Down => self.get_q(state, &Move::Down),
            HeatmapView::Left => self.get_q(state, &Move::Left),
            HeatmapView::Right => self.get_q(state, &Move::Right),
            HeatmapView::BellmanVStar => self.latest_snap.bellman.v_star_at(*state),
            HeatmapView::PolicyVPi => self.latest_snap.bellman.v_pi_at(*state),
            HeatmapView::BellmanResidual => {
                let emp = match self.solver_mode {
                    RlSolverMode::TabularQ => self
                        .latest_snap
                        .q_table
                        .get(idx)
                        .map_or(0.0, |row| row.iter().copied().fold(row[0], f64::max)),
                    RlSolverMode::PolicyGradient => self
                        .latest_snap
                        .state_values
                        .get(idx)
                        .copied()
                        .unwrap_or(0.0),
                };
                (emp - self.latest_snap.bellman.v_star_at(*state)).abs()
            }
        }
    }

    fn handle_transport_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
        match action {
            PlaybackAction::Play => {
                let _ = self.worker.send_command(SimCommand::TrainEpisodes(100));
            }
            PlaybackAction::Pause => {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                let _ = self.worker.send_command(SimCommand::Step(1));
            }
            PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context, legend_size: f32) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "q_table_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.draw_training_transport(ui, ctx);
                ui.separator();
                ui.heading("Maze Sculpting");
                if render_rl_sculpt_toolbar(
                    ui,
                    "q_table_inspector",
                    &mut self.mdp,
                    &mut self.brush_kit,
                    &mut self.maze_preset,
                ) {
                    self.sync_sculpt_to_worker();
                }
                if ui
                    .add(egui::Slider::new(&mut self.mdp.gamma, 0.0..=1.0).text("Discount (γ)"))
                    .changed()
                {
                    self.sync_sculpt_to_worker();
                }
                self.draw_view_and_legend(ui, legend_size);
                ui.separator();
                render_sonification_panel(
                    ui,
                    "q_table_inspector",
                    &self.latest_audio,
                    !self.worker.state().is_running(),
                );
            });
        });
    }

    fn draw_training_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Training Controls");
        if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
            render_worker_fault_banner(
                ui,
                self.worker.state(),
                self.worker.diagnostic_report(),
                false,
            )
        {
            self.reset();
            ctx.request_repaint();
            ui.separator();
        }

        let state = self.worker.state();
        ui.label(format!("Episodes Trained: {}", self.episodes_trained));
        if state.is_running() {
            let info = crate::app::SemanticPalette::from_visuals(ui.visuals()).info;
            ui.colored_label(info, "⏳ Training in background...");
        }

        ui.horizontal(|ui| {
            ui.label("Solver:");
            egui::ComboBox::from_id_salt("q_table_solver")
                .selected_text(self.solver_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [RlSolverMode::TabularQ, RlSolverMode::PolicyGradient] {
                        if ui
                            .selectable_value(&mut self.solver_mode, mode, mode.label())
                            .changed()
                        {
                            self.sync_sculpt_to_worker();
                        }
                    }
                });
        });

        let canvas_focused = self
            .brush_kit
            .focused_plot_id()
            .is_some_and(|id| ui.ctx().memory(|m| m.has_focus(id)));
        if let Some(action) =
            PlaybackTransport::from_worker_state("q_table_transport", "Q-Table Inspector", state)
                .play_label("▶ Train 100")
                .step_label("⏭ Step Episode")
                .can_reset(true)
                .allow_space_shortcut(!canvas_focused)
                .show(ui)
        {
            self.handle_transport_action(action, ctx);
        }

        if ui
            .add_enabled(state.is_ready(), egui::Button::new("▶ Train 1000 Episodes"))
            .accessible_hover_text("Train the agent for 1000 episodes in the background")
            .clicked()
        {
            let _ = self.worker.send_command(SimCommand::TrainEpisodes(1000));
        }
    }
}

fn lerp_q_channel(a: u8, b: u8, t: f64) -> u8 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * t.clamp(0.0, 1.0)).round() as u8
}

pub(crate) fn q_value_cell_color(
    normalized: f64,
    palette: &crate::app::SemanticPalette,
) -> egui::Color32 {
    let low = palette.axis_z;
    let high = palette.accent;
    egui::Color32::from_rgb(
        lerp_q_channel(low.r(), high.r(), normalized),
        lerp_q_channel(low.g(), high.g(), normalized),
        lerp_q_channel(low.b(), high.b(), normalized),
    )
}

impl InteractiveTool for QTableInspectorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Q-Table Inspector"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot()
    }
    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_q_table_primitives()
    }
    fn export_tables(&self) -> Vec<crate::framework::ExportTable> {
        self.build_q_table_export_tables()
    }
    fn step_simulation(&mut self) {
        self.step_q_table_episode();
    }
    fn domain(&self) -> &'static str {
        "ai"
    }
    fn set_suspended(&mut self, suspended: bool) {
        self.worker.set_suspended(suspended);
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn on_brush(&mut self, ctx: &InteractionContext) {
        let Some(stroke) = ctx.brush_stroke else {
            return;
        };
        if apply_rl_brush_stroke(&mut self.mdp, &stroke) {
            self.refresh_snapshot_bellman();
            let _ = self
                .worker
                .send_command(SimCommand::ApplyBrushStroke(stroke));
            self.revision = self.revision.wrapping_add(1);
        }
    }
    fn save_state(&self) -> Option<String> {
        self.save_preset_state()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.try_load_preset_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let parts = self.worker.update().map(extract_snapshot_parts);
        if let Some((audio, data)) = parts {
            self.apply_snapshot_parts(audio, data);
        }
        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let legend_size = if input_mode == InputMode::Touch {
            44.0
        } else {
            15.0
        };
        self.draw_side_panel(ui, &ctx, legend_size);
        self.draw_heatmap(ui, 60.0);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "QTableInspectorTool",
        display_name: "Q-Table Inspector",
        domain: "ai",
        tags: &[
            "q-learning",
            "bellman",
            "temporal difference",
            "action-value",
            "reinforcement learning",
            "rl",
            "sonification",
        ],
        build: || Box::new(QTableInspectorTool::default()),
    }
}

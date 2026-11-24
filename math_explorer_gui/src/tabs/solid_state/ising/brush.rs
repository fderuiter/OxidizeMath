#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    apply_pinned_rle_to_lattice, ising_spin_color, DropletSeed, IsingBrushMode,
    IsingInteractionMode, IsingModelTool, DEFAULT_ISING_SIZE,
};
use crate::async_sim::SimCommand;
use crate::framework::{
    encode_u8_grid, BrushStencil, InteractionContext, RasterBrushStroke, SymmetryMode,
};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use nalgebra::DMatrix;

impl IsingModelTool {
    pub(crate) fn sync_brush_fields_before_frame(&mut self) {
        let max_r = if self.active_challenge.is_some() {
            10
        } else {
            99
        };
        if self.brush_radius != self.last_synced_brush_radius {
            self.brush_state.radius = self.brush_radius.clamp(0, max_r) as usize;
            self.last_synced_brush_radius = self.brush_radius;
        } else if self.brush_state.radius > max_r as usize {
            self.brush_state.radius = max_r as usize;
        }
        if self.brush_cooperator != self.last_synced_brush_cooperator {
            self.brush_mode = if self.brush_cooperator {
                IsingBrushMode::SpinUp
            } else {
                IsingBrushMode::SpinDown
            };
            self.last_synced_brush_cooperator = self.brush_cooperator;
        }
        if self.active_challenge.is_some() {
            self.brush_state.stencil = BrushStencil::Circle;
            self.brush_state.symmetry = SymmetryMode::None;
            if !matches!(
                self.brush_mode,
                IsingBrushMode::SpinUp | IsingBrushMode::SpinDown
            ) {
                self.brush_mode = IsingBrushMode::SpinUp;
            }
        }
    }

    pub(crate) fn sync_brush_fields_from_palette(&mut self) {
        let max_r = if self.active_challenge.is_some() {
            10
        } else {
            99
        };
        self.brush_state.radius = self.brush_state.radius.min(max_r as usize);
        self.brush_radius = (self.brush_state.radius as i32).clamp(0, max_r);
        self.last_synced_brush_radius = self.brush_radius;
        match self.brush_mode {
            IsingBrushMode::SpinUp | IsingBrushMode::PinUp => self.brush_cooperator = true,
            IsingBrushMode::SpinDown | IsingBrushMode::PinDown => self.brush_cooperator = false,
            IsingBrushMode::Unpin => {}
        }
        self.last_synced_brush_cooperator = self.brush_cooperator;
    }

    /// Applies a [`RasterBrushStroke`] to spin domains, nucleation seeds, or the boundary pin mask.
    pub fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke, ctx: Option<&egui::Context>) {
        let effective_mode = if stroke.secondary {
            match self.brush_mode {
                IsingBrushMode::SpinUp => IsingBrushMode::SpinDown,
                IsingBrushMode::SpinDown => IsingBrushMode::SpinUp,
                IsingBrushMode::PinUp | IsingBrushMode::PinDown | IsingBrushMode::Unpin => {
                    IsingBrushMode::Unpin
                }
            }
        } else {
            self.brush_mode
        };
        if self.active_challenge.is_some() {
            let spin = match effective_mode {
                IsingBrushMode::SpinDown | IsingBrushMode::PinDown | IsingBrushMode::Unpin => -1,
                IsingBrushMode::SpinUp | IsingBrushMode::PinUp => 1,
            };
            self.record_droplet_seed(DropletSeed {
                cx: (stroke.focus_cell.0 as i32).clamp(0, (DEFAULT_ISING_SIZE as i32) - 1),
                cy: (stroke.focus_cell.1 as i32).clamp(0, (DEFAULT_ISING_SIZE as i32) - 1),
                r: self.brush_radius.clamp(0, 10),
                spin,
            });
            self.push_seed_layout_to_worker();
            self.refresh_challenge_workspace();
            self.stepped_pixels = None;
            if let Some(c) = ctx {
                c.request_repaint();
            }
            return;
        }
        self.apply_free_brush_cells(stroke.cells, effective_mode, ctx);
    }

    fn apply_free_brush_cells(
        &mut self,
        cells: Vec<(usize, usize)>,
        effective_mode: IsingBrushMode,
        ctx: Option<&egui::Context>,
    ) {
        if self.pin_cells.len() != DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE {
            self.pin_cells
                .resize(DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE, 0);
        }
        let pin_sym = match effective_mode {
            IsingBrushMode::PinUp => Some(1u8),
            IsingBrushMode::PinDown => Some(2u8),
            IsingBrushMode::Unpin => Some(0u8),
            IsingBrushMode::SpinUp | IsingBrushMode::SpinDown => None,
        };
        if let Some(sym) = pin_sym {
            for &(x, y) in &cells {
                if x < DEFAULT_ISING_SIZE && y < DEFAULT_ISING_SIZE {
                    self.pin_cells[flatten_2d_index(x, y, DEFAULT_ISING_SIZE)] = sym;
                }
            }
            self.sync_pinned_rle();
        }
        let _ = self.worker.send_command(SimCommand::ApplyBrushCells {
            cells,
            mode: effective_mode.to_code(),
        });
        if !self.worker.is_running() {
            self.pending_ode_resync = true;
        }
        self.stepped_pixels = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    fn sync_pinned_rle(&mut self) {
        let next_rle = if self.pin_cells.iter().any(|&c| c != 0) {
            encode_u8_grid(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, &self.pin_cells, 2).ok()
        } else {
            None
        };
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle = next_rle;
    }

    /// Clears all pinned boundary sites across UI and worker threads.
    pub fn clear_pins(&mut self, ctx: Option<&egui::Context>) {
        self.pin_cells.fill(0);
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_pins".to_string()));
        self.stepped_pixels = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn handle_plot_brush(&mut self, ctx: &InteractionContext) {
        let Some(pos) = ctx.pointer_pos else {
            return;
        };
        let rect = ctx.response.rect;
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return;
        }
        let (w, h) = self
            .worker
            .latest_snapshot()
            .map_or((DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE), |s| {
                (s.width.max(1), s.height.max(1))
            });
        let (max_x, max_y) = ((w as i32) - 1, (h as i32) - 1);
        let rel_x = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
        let rel_y = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
        let cx = ((rel_x * (max_x as f32)).round() as i32).clamp(0, max_x);
        let cy = ((rel_y * (max_y as f32)).round() as i32).clamp(0, max_y);
        let max_r = if self.active_challenge.is_some() {
            10
        } else {
            max_x.max(max_y)
        };
        let r = self.brush_radius.clamp(0, max_r);
        let invert = ctx.modifiers.shift
            || ctx.response.secondary_clicked()
            || ctx.response.dragged_by(egui::PointerButton::Secondary);
        let is_obstacle = if invert {
            !self.brush_cooperator
        } else {
            self.brush_cooperator
        };
        self.record_droplet_seed(DropletSeed {
            cx,
            cy,
            r,
            spin: if is_obstacle { 1 } else { -1 },
        });
        self.sync_shared_params();
        if self.active_challenge.is_some() {
            self.push_seed_layout_to_worker();
            self.refresh_challenge_workspace();
            ctx.response.ctx.request_repaint();
            return;
        }
        if self
            .worker
            .send_command(SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            })
            .is_ok()
        {
            if !self.worker.is_running() {
                self.pending_ode_resync = true;
            }
            ctx.response.ctx.request_repaint();
        }
    }

    pub(crate) fn step_ising_simulation(&mut self) {
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, Some(42));
        let (theme, pinned_rle) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.theme, p.pinned_rle.clone())
        };
        apply_pinned_rle_to_lattice(&mut lattice, pinned_rle.as_deref());
        let steps = self.steps_per_frame.clamp(100, 5000);
        match self.mode {
            IsingInteractionMode::Ferromagnetic => {
                let t_phys = self.temperature.max(1e-3) / KB;
                lattice.evolve(steps, t_phys, self.j_coupling, self.h_field);
            }
            IsingInteractionMode::FermiGame => {
                let mat = DMatrix::from_row_slice(2, 2, &self.game_payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    lattice.evolve_fermi_game_with_mutation(
                        steps,
                        self.temperature,
                        &payoff,
                        self.mutation_rate,
                    );
                }
            }
        }
        let palette = theme.palette();
        let pixels: Vec<egui::Color32> = lattice
            .spins()
            .iter()
            .zip(lattice.pinned().iter())
            .map(|(&s, &pin)| ising_spin_color(if pin { s.saturating_mul(2) } else { s }, &palette))
            .collect();
        self.stepped_pixels = Some((lattice.width(), lattice.height(), pixels));
        self.dirty = true;
    }
}

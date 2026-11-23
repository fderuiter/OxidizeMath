#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Box-constraint barriers (`[w_low, w_high] x [b_low, b_high]`) for [`super::LossLandscapeTool`]:
//! zero-allocation projection/reflection bounds, side-panel UI controls, and 2D/3D barrier overlays.

use super::arena::MAX_TRAIL_POINTS;
use super::surfaces::{SurfaceEvaluator, SurfaceGrid};
use super::{LOSS_MIN_SPAN, LOSS_PARAM_MAX_BOUND, LOSS_PARAM_MIN_BOUND};
use crate::app::SemanticPalette;
use crate::camera::Camera3D;
use eframe::egui;
use egui_plot::{Line, LineStyle, PlotUi};
use math_explorer::pure_math::analysis::momentum_dynamics::PhasePoint2D;
use math_explorer::pure_math::analysis::optimization::AdamState2D;
use nalgebra::Vector2;

/// Fixed-capacity circular trail ring buffer for zero-allocation particle history.
pub struct TrailRing {
    buf: Box<[[f64; 2]; MAX_TRAIL_POINTS]>,
    start: usize,
    len: usize,
}

impl Default for TrailRing {
    fn default() -> Self {
        Self {
            buf: Box::new([[0.0; 2]; MAX_TRAIL_POINTS]),
            start: 0,
            len: 0,
        }
    }
}

impl TrailRing {
    #[inline]
    pub fn clear(&mut self) {
        (self.start, self.len) = (0, 0);
    }

    #[inline]
    pub fn push(&mut self, pt: [f64; 2]) {
        if self.len < MAX_TRAIL_POINTS {
            self.buf[(self.start + self.len) % MAX_TRAIL_POINTS] = pt;
            self.len += 1;
        } else {
            self.buf[self.start] = pt;
            self.start = (self.start + 1) % MAX_TRAIL_POINTS;
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = [f64; 2]> + '_ {
        (0..self.len).map(move |i| self.buf[(self.start + i) % MAX_TRAIL_POINTS])
    }
}

/// Default interior box barrier `[w_low, w_high, b_low, b_high]` for the linear MSE preset.
pub const DEFAULT_BARRIER_BOUNDS: [f64; 4] = [0.5, 3.5, -0.5, 2.5];

/// Returns [`DEFAULT_BARRIER_BOUNDS`] for serde default deserialization.
#[inline]
#[must_use]
pub const fn default_barrier_bounds() -> [f64; 4] {
    DEFAULT_BARRIER_BOUNDS
}

/// Predicate for `skip_serializing_if` so default barrier bounds add zero bytes to legacy JSON.
#[inline]
#[must_use]
pub fn is_default_barrier_bounds(b: &[f64; 4]) -> bool {
    (b[0] - DEFAULT_BARRIER_BOUNDS[0]).abs() < 1e-12
        && (b[1] - DEFAULT_BARRIER_BOUNDS[1]).abs() < 1e-12
        && (b[2] - DEFAULT_BARRIER_BOUNDS[2]).abs() < 1e-12
        && (b[3] - DEFAULT_BARRIER_BOUNDS[3]).abs() < 1e-12
}

/// Validates `[w_low, w_high, b_low, b_high]` finiteness, global range, and strict ordering.
#[inline]
#[must_use]
pub fn are_barrier_bounds_valid(b: &[f64; 4]) -> bool {
    b.iter()
        .all(|v| v.is_finite() && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(v))
        && b[0] + 1e-6 <= b[1]
        && b[2] + 1e-6 <= b[3]
}

/// Clamps `[w_low, w_high, b_low, b_high]` into ordered finite intervals inside `[-100, 100]`.
pub fn clamp_barrier_bounds(b: &mut [f64; 4]) {
    for (idx, val) in b.iter_mut().enumerate() {
        if !val.is_finite() {
            *val = DEFAULT_BARRIER_BOUNDS[idx];
        } else {
            *val = val.clamp(LOSS_PARAM_MIN_BOUND, LOSS_PARAM_MAX_BOUND);
        }
    }
    if b[1] - b[0] < LOSS_MIN_SPAN {
        let mid = (0.5 * (b[0] + b[1])).clamp(
            LOSS_PARAM_MIN_BOUND + LOSS_MIN_SPAN,
            LOSS_PARAM_MAX_BOUND - LOSS_MIN_SPAN,
        );
        b[0] = mid - 0.5 * LOSS_MIN_SPAN;
        b[1] = mid + 0.5 * LOSS_MIN_SPAN;
    }
    if b[3] - b[2] < LOSS_MIN_SPAN {
        let mid = (0.5 * (b[2] + b[3])).clamp(
            LOSS_PARAM_MIN_BOUND + LOSS_MIN_SPAN,
            LOSS_PARAM_MAX_BOUND - LOSS_MIN_SPAN,
        );
        b[2] = mid - 0.5 * LOSS_MIN_SPAN;
        b[3] = mid + 0.5 * LOSS_MIN_SPAN;
    }
}

/// Intersects outer domain `(w_min, w_max, b_min, b_max)` with active box barriers when enabled.
#[inline]
#[must_use]
pub fn effective_arena_bounds(
    domain: (f64, f64, f64, f64),
    enabled: bool,
    barrier: [f64; 4],
) -> (f64, f64, f64, f64) {
    if !enabled {
        return domain;
    }
    let w_lo = barrier[0].clamp(domain.0, domain.1);
    let w_hi = barrier[1].clamp(w_lo, domain.1);
    let b_lo = barrier[2].clamp(domain.2, domain.3);
    let b_hi = barrier[3].clamp(b_lo, domain.3);
    (w_lo, w_hi, b_lo, b_hi)
}

/// Projects a first-order iterate (`SGD` / `Adam`) onto `[q_min, q_max]` when box barriers are active.
#[inline]
pub fn project_first_order_point(
    q: &mut Vector2<f64>,
    momentum: Option<&mut Vector2<f64>>,
    q_min: Vector2<f64>,
    q_max: Vector2<f64>,
) {
    let clamped_x = q.x.clamp(q_min.x, q_max.x);
    let clamped_y = q.y.clamp(q_min.y, q_max.y);
    if let Some(m) = momentum {
        if (clamped_x - q.x).abs() > 1e-12 {
            m.x = 0.0;
        }
        if (clamped_y - q.y).abs() > 1e-12 {
            m.y = 0.0;
        }
    }
    q.x = clamped_x;
    q.y = clamped_y;
}

/// Computes sensible default interior barrier bounds (`65%` central box) when switching surfaces.
#[inline]
#[must_use]
pub fn default_barrier_for_domain(domain: (f64, f64, f64, f64)) -> [f64; 4] {
    let dw = 0.175 * (domain.1 - domain.0);
    let db = 0.175 * (domain.3 - domain.2);
    [domain.0 + dw, domain.1 - dw, domain.2 + db, domain.3 - db]
}

/// Renders the optional box-barrier checkbox and slider controls inside the side panel.
pub fn draw_box_barrier_controls(
    ui: &mut egui::Ui,
    domain: (f64, f64, f64, f64),
    enabled: &mut bool,
    bounds: &mut [f64; 4],
) -> bool {
    let mut reset_needed = ui
        .checkbox(enabled, "Box Barriers [w, b] (Projected / Reflective)")
        .changed();
    if !*enabled {
        return reset_needed;
    }
    let span = LOSS_MIN_SPAN.min(0.25 * (domain.1 - domain.0).max(0.02));
    let b_span = LOSS_MIN_SPAN.min(0.25 * (domain.3 - domain.2).max(0.02));
    let w_hi_cap = (bounds[1] - span).max(domain.0);
    reset_needed |= ui
        .add(egui::Slider::new(&mut bounds[0], domain.0..=w_hi_cap).text("Barrier w_min"))
        .changed();
    let w_lo_cap = (bounds[0] + span).min(domain.1);
    reset_needed |= ui
        .add(egui::Slider::new(&mut bounds[1], w_lo_cap..=domain.1).text("Barrier w_max"))
        .changed();
    let b_hi_cap = (bounds[3] - b_span).max(domain.2);
    reset_needed |= ui
        .add(egui::Slider::new(&mut bounds[2], domain.2..=b_hi_cap).text("Barrier b_min"))
        .changed();
    let b_lo_cap = (bounds[2] + b_span).min(domain.3);
    reset_needed |= ui
        .add(egui::Slider::new(&mut bounds[3], b_lo_cap..=domain.3).text("Barrier b_max"))
        .changed();
    reset_needed
}

/// Draws the feasible box-barrier boundary rectangle on the 2D contour plot.
pub fn draw_2d_box_barrier(
    plot_ui: &mut PlotUi<'_>,
    domain: (f64, f64, f64, f64),
    enabled: bool,
    barrier: [f64; 4],
    palette: &SemanticPalette,
) {
    if !enabled {
        return;
    }
    let (w0, w1, b0, b1) = effective_arena_bounds(domain, true, barrier);
    let rect_loop = vec![[w0, b0], [w1, b0], [w1, b1], [w0, b1], [w0, b0]];
    plot_ui.line(
        Line::new("Box Barrier Boundary", rect_loop)
            .color(palette.error)
            .style(LineStyle::dashed_dense())
            .width(2.2_f32),
    );
}

/// Draws the feasible box-barrier loop draped onto the 3D loss surface wireframe.
pub fn draw_3d_box_barrier(
    plot_ui: &mut PlotUi<'_>,
    grid: &SurfaceGrid,
    eval: SurfaceEvaluator<'_>,
    cam: &Camera3D,
    domain: (f64, f64, f64, f64),
    cfg: &super::arena::ArenaConfig,
    palette: &SemanticPalette,
) {
    if !cfg.box_barriers_enabled {
        return;
    }
    let (w0, w1, b0, b1) = effective_arena_bounds(domain, true, cfg.barrier_bounds);
    let corners = [[w0, b0], [w1, b0], [w1, b1], [w0, b1], [w0, b0]];
    let mut pts = Vec::with_capacity(65);
    for edge in corners.windows(2) {
        let (a, c) = (edge[0], edge[1]);
        for s in 0..16 {
            let t = s as f64 / 16.0;
            let w = a[0] + t * (c[0] - a[0]);
            let b = a[1] + t * (c[1] - a[1]);
            let loss = eval.evaluate(w, b);
            pts.push(cam.project(&grid.world_point(w, b, loss, 0.04)));
        }
    }
    let last_loss = eval.evaluate(w0, b0);
    pts.push(cam.project(&grid.world_point(w0, b0, last_loss, 0.04)));
    plot_ui.line(
        Line::new("Box Barrier Boundary", pts)
            .color(palette.error)
            .width(2.2_f32),
    );
}

/// Applies either projected box-barrier clamping or outer-domain reflection to an `SGD` iterate.
#[inline]
pub fn enforce_sgd_bounds(
    phase: &mut PhasePoint2D,
    q_min: Vector2<f64>,
    q_max: Vector2<f64>,
    restitution: f64,
    project: bool,
) {
    if project {
        project_first_order_point(&mut phase.q, Some(&mut phase.p), q_min, q_max);
    } else {
        phase.reflect_bounds(q_min, q_max, restitution);
    }
}

/// Applies either projected box-barrier clamping or outer-domain reflection to an `Adam` iterate.
#[inline]
pub fn enforce_adam_bounds(
    phase: &mut PhasePoint2D,
    adam_state: &mut AdamState2D<f64>,
    q_min: Vector2<f64>,
    q_max: Vector2<f64>,
    restitution: f64,
    project: bool,
) {
    if project {
        project_first_order_point(&mut phase.q, Some(&mut adam_state.m), q_min, q_max);
    } else {
        let raw_q = phase.q;
        phase.reflect_bounds(q_min, q_max, restitution);
        if (phase.q.x - raw_q.x).abs() > 1e-12 {
            adam_state.m.x *= -restitution;
        }
        if (phase.q.y - raw_q.y).abs() > 1e-12 {
            adam_state.m.y *= -restitution;
        }
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::super::LossLandscapeTool;
    use crate::framework::InteractiveTool;

    #[test]
    fn test_box_barriers_confine_all_racers_and_roundtrip_state() {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut tool = LossLandscapeTool::default();
        // Global minimum is at (2.0, 1.0); place box barrier [0.2, 1.4] x [0.0, 2.2] excluding w=2.0.
        tool.arena_config.box_barriers_enabled = true;
        tool.arena_config.barrier_bounds = [0.2, 1.4, 0.0, 2.2];
        tool.arena_config.learning_rate = 0.08;
        tool.reset_arena();

        let (_res, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..80 {
                tool.step_simulation();
            }
        });
        assert_eq!(metrics.memory_loads, 0);

        for racer in &tool.arena.racers {
            if !racer.kind.is_available_in_build() {
                continue;
            }
            assert!(racer.phase.q.x >= 0.2 - 1e-9 && racer.phase.q.x <= 1.4 + 1e-9);
            assert!(racer.phase.q.y >= 0.0 - 1e-9 && racer.phase.q.y <= 2.2 + 1e-9);
        }
        // SGD and Adam should press cleanly against the w_max = 1.4 barrier wall.
        assert!((tool.arena.racers[0].phase.q.x - 1.4).abs() < 1e-3);

        let saved = tool.save_state().expect("serialize barrier state");
        let mut restored = LossLandscapeTool::default();
        restored
            .try_load_state(&saved)
            .expect("restore barrier state");
        assert!(restored.arena_config.box_barriers_enabled);
        assert_eq!(restored.arena_config.barrier_bounds, [0.2, 1.4, 0.0, 2.2]);
    }
}

// [cite:lorahub]

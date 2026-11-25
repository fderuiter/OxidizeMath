//! Shared maze sculpting, Bellman overlay, RLE persistence, and policy-arrow helpers
//! for `GridWorldTool`, `QTableInspectorTool`, and `RewardPlotsTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::async_sim::SculptStatePayload;
use crate::framework::{
    decode_rle_mask, encode_rle_mask, rasterize_disk_stroke, BrushMode, BrushStroke,
    SpatialBrushKit,
};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::{
    BellmanSolution, DenseRlWorkspace, GridCellRole, GridState, MarkovDecisionProcess, Move,
    SculptedGridMdp, TabularGridPolicy, MAX_RL_GRID_DIM, MIN_RL_GRID_DIM,
};

/// Active reinforcement learning update rule across AI tab tools.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum RlSolverMode {
    /// Off-policy Bellman temporal-difference Q-learning.
    #[default]
    TabularQ,
    /// On-policy episodic softmax REINFORCE with state-value baseline.
    PolicyGradient,
}

impl RlSolverMode {
    /// Human-readable solver label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::TabularQ => "Tabular Q-Learning",
            Self::PolicyGradient => "REINFORCE Policy Gradient",
        }
    }

    /// Encodes this mode into a scalar payload tag.
    #[must_use]
    pub const fn to_code(self) -> f64 {
        match self {
            Self::TabularQ => 0.0,
            Self::PolicyGradient => 1.0,
        }
    }

    /// Decodes a scalar payload tag back into [`Self`].
    #[must_use]
    pub fn from_code(code: f64) -> Self {
        if (code - 1.0).abs() < 0.5 {
            Self::PolicyGradient
        } else {
            Self::TabularQ
        }
    }

    /// Serde helper omitting default solver mode for compact URL payloads.
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::TabularQ)
    }
}

/// Scalar field and greedy policy overlay mode for `GridWorldTool`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum RlOverlayMode {
    /// Displays empirical greedy arrows alongside sculpted terrain roles.
    #[default]
    EmpiricalPolicy,
    /// Shades cells by exact Bellman optimal state value $V^*(s)$ and optimal arrows $\pi^*(s)$.
    BellmanVStar,
    /// Shades cells by exact Bellman policy evaluation $V_\pi(s)$ of the agent's current policy.
    PolicyValueVPi,
    /// Shades cells by Bellman optimality residual $|\hat{V}(s) - V^*(s)|$.
    BellmanResidual,
}

impl RlOverlayMode {
    /// Human-readable overlay label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::EmpiricalPolicy => "Empirical Policy",
            Self::BellmanVStar => "Bellman Optimal V*(s)",
            Self::PolicyValueVPi => "Policy Value V_π(s)",
            Self::BellmanResidual => "Bellman Residual |V - V*|",
        }
    }

    /// Serde helper omitting default overlay mode.
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::EmpiricalPolicy)
    }
}

/// Curated sculptable maze topologies selectable inside all three RL tools.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum RlMazePreset {
    /// Classic baseline layout with two central trap cells and zero wind.
    #[default]
    BaselineTraps,
    /// Southern trap cliff with upward crosswind columns (`p_wind = 0.35`).
    WindyCliff,
    /// Solid bounce-back wall corridor forcing multi-turn detours.
    ObstacleCorridor,
    /// Alternating crosswind currents surrounding a wall chicane (`p_wind = 0.30`).
    CrosswindSpiral,
    /// Obstacle-free arena for clean baseline value propagation.
    OpenArena,
}

impl RlMazePreset {
    /// All selectable maze presets.
    pub const ALL: [Self; 5] = [
        Self::BaselineTraps,
        Self::WindyCliff,
        Self::ObstacleCorridor,
        Self::CrosswindSpiral,
        Self::OpenArena,
    ];

    /// Human-readable preset title.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BaselineTraps => "Baseline Traps",
            Self::WindyCliff => "Windy Cliff",
            Self::ObstacleCorridor => "Obstacle Corridor",
            Self::CrosswindSpiral => "Crosswind Chicane",
            Self::OpenArena => "Open Arena",
        }
    }

    /// Constructs a [`SculptedGridMdp`] for `(width, height, gamma)` matching this preset.
    #[must_use]
    pub fn build_mdp(self, width: i32, height: i32, gamma: f64) -> SculptedGridMdp {
        let w = width.clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM);
        let h = height.clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM);
        match self {
            Self::BaselineTraps => SculptedGridMdp::default_with_traps(w, h, gamma),
            Self::OpenArena => SculptedGridMdp::new(w, h, gamma, 0.0),
            Self::WindyCliff => build_windy_cliff(w, h, gamma),
            Self::ObstacleCorridor => build_obstacle_corridor(w, h, gamma),
            Self::CrosswindSpiral => build_crosswind_chicane(w, h, gamma),
        }
    }
}

fn build_windy_cliff(w: i32, h: i32, gamma: f64) -> SculptedGridMdp {
    let mut mdp = SculptedGridMdp::new(w, h, gamma, 0.35);
    for x in 1..(w - 1) {
        mdp.set_cell_role(GridState { x, y: h - 1 }, GridCellRole::Trap);
        for y in 0..(h - 1) {
            mdp.set_cell_role(GridState { x, y }, GridCellRole::WindDown);
        }
    }
    mdp
}

fn build_obstacle_corridor(w: i32, h: i32, gamma: f64) -> SculptedGridMdp {
    let mut mdp = SculptedGridMdp::new(w, h, gamma, 0.0);
    let mid_x = (w / 2).clamp(1, w - 1);
    for y in 0..(h - 1) {
        mdp.set_cell_role(GridState { x: mid_x, y }, GridCellRole::SolidWall);
    }
    if w >= 4 && h >= 4 {
        mdp.set_cell_role(
            GridState {
                x: (mid_x + 1).min(w - 1),
                y: 1,
            },
            GridCellRole::Trap,
        );
    }
    mdp
}

fn build_crosswind_chicane(w: i32, h: i32, gamma: f64) -> SculptedGridMdp {
    let mut mdp = SculptedGridMdp::new(w, h, gamma, 0.30);
    for y in 1..(h - 1) {
        for x in 0..w {
            let role = if y % 2 == 1 {
                GridCellRole::WindRight
            } else {
                GridCellRole::WindUp
            };
            mdp.set_cell_role(GridState { x, y }, role);
        }
    }
    if w >= 4 && h >= 4 {
        mdp.set_cell_role(GridState { x: 1, y: 1 }, GridCellRole::SolidWall);
        mdp.set_cell_role(GridState { x: w - 2, y: h - 2 }, GridCellRole::Trap);
    }
    mdp
}

/// Selectable brush modes exposed on RL grid canvases.
pub const RL_BRUSH_MODES: [(BrushMode, &str); 8] = [
    (BrushMode::Solid, "■ Wall"),
    (BrushMode::RlTrap, "✕ Trap"),
    (BrushMode::RlGoal, "★ Goal"),
    (BrushMode::WindUp, "↑ Wind N"),
    (BrushMode::WindDown, "↓ Wind S"),
    (BrushMode::WindLeft, "← Wind W"),
    (BrushMode::WindRight, "→ Wind E"),
    (BrushMode::Erase, "⌫ Erase"),
];

/// Creates a grid-calibrated [`SpatialBrushKit`] (`1` cell step, `0.5..=3.5` cell radius).
#[must_use]
pub fn new_rl_brush_kit() -> SpatialBrushKit {
    SpatialBrushKit::new(BrushMode::Solid, 0.5)
        .with_radius_bounds(0.5, 3.5, 1.0)
        .with_cursor_step(1.0, 1.0)
}

/// Maps a [`BrushMode`] onto its corresponding [`GridCellRole`].
#[must_use]
pub const fn brush_mode_to_cell_role(mode: BrushMode) -> GridCellRole {
    match mode {
        BrushMode::Solid | BrushMode::ZeroFluxWall => GridCellRole::SolidWall,
        BrushMode::RlTrap => GridCellRole::Trap,
        BrushMode::RlGoal => GridCellRole::Goal,
        BrushMode::WindUp => GridCellRole::WindUp,
        BrushMode::WindDown => GridCellRole::WindDown,
        BrushMode::WindLeft => GridCellRole::WindLeft,
        BrushMode::WindRight => GridCellRole::WindRight,
        _ => GridCellRole::Free,
    }
}

/// Applies an interpolated [`BrushStroke`] to `mdp`, returning `true` if any cell role changed.
pub fn apply_rl_brush_stroke(mdp: &mut SculptedGridMdp, stroke: &BrushStroke) -> bool {
    let w = mdp.width as usize;
    let h = mdp.height as usize;
    let start = SpatialBrushKit::plot_to_grid_y_down(stroke.start, w, h);
    let end = SpatialBrushKit::plot_to_grid_y_down(stroke.end, w, h);
    let cell_radius = (stroke.radius - 0.51).floor().max(0.0) as i32;
    let target_role = brush_mode_to_cell_role(stroke.mode);
    let mut changed = false;

    rasterize_disk_stroke(start, end, cell_radius, (w, h), |gx, gy| {
        let st = GridState {
            x: gx as i32,
            y: gy as i32,
        };
        let before = mdp.cell_role(st);
        let before_goal = mdp.goal;
        mdp.set_cell_role(st, target_role);
        if mdp.cell_role(st) != before || mdp.goal != before_goal {
            changed = true;
        }
    });
    changed
}

/// Serde predicate returning `true` when `val` is approximately `0.0`.
#[must_use]
pub fn is_zero_f64(val: &f64) -> bool {
    val.abs() <= 1e-12
}

/// Encodes `mdp` cell roles as RLE pairs only when they differ from `default_with_traps`.
#[must_use]
pub fn encode_custom_cells_rle(mdp: &SculptedGridMdp) -> Vec<[u32; 2]> {
    let baseline = SculptedGridMdp::default_with_traps(mdp.width, mdp.height, mdp.gamma);
    if mdp.cells_to_u8() == baseline.cells_to_u8() {
        Vec::new()
    } else {
        encode_rle_mask(&mdp.cells_to_u8())
    }
}

/// Restores `rle_mask` onto `mdp` (empty `rle_mask` restores `default_with_traps` cells).
pub fn apply_rle_to_mdp(mdp: &mut SculptedGridMdp, rle_mask: &[[u32; 2]]) -> Result<(), String> {
    if rle_mask.is_empty() {
        let wind = mdp.wind_slip_prob;
        let gamma = mdp.gamma;
        *mdp = SculptedGridMdp::default_with_traps(mdp.width, mdp.height, gamma);
        mdp.wind_slip_prob = wind;
        return Ok(());
    }
    let expected = (mdp.width as usize) * (mdp.height as usize);
    let raw = decode_rle_mask(rle_mask, expected)?;
    mdp.apply_cells_u8(&raw).map_err(|err| err.to_string())
}

/// Packs `mdp` and `solver` into a worker-synchronizable [`SculptStatePayload`].
#[must_use]
pub fn build_rl_sculpt_payload(mdp: &SculptedGridMdp, solver: RlSolverMode) -> SculptStatePayload {
    SculptStatePayload {
        rle_mask: encode_custom_cells_rle(mdp),
        sparse_potential: vec![
            (0, f64::from(mdp.width)),
            (1, f64::from(mdp.height)),
            (2, mdp.wind_slip_prob),
            (3, mdp.gamma),
            (4, solver.to_code()),
        ],
    }
}

/// Unpacks a [`SculptStatePayload`] onto `mdp` and `solver`.
pub fn apply_rl_sculpt_payload(
    mdp: &mut SculptedGridMdp,
    solver: &mut RlSolverMode,
    payload: &SculptStatePayload,
) -> Result<(), String> {
    let mut w = mdp.width;
    let mut h = mdp.height;
    let mut wind = mdp.wind_slip_prob;
    let mut gamma = mdp.gamma;
    for &(key, val) in &payload.sparse_potential {
        if !val.is_finite() {
            return Err("Non-finite RL sculpt parameter".into());
        }
        match key {
            0 => w = (val.round() as i32).clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM),
            1 => h = (val.round() as i32).clamp(MIN_RL_GRID_DIM, MAX_RL_GRID_DIM),
            2 => wind = val.clamp(0.0, 1.0),
            3 => gamma = val.clamp(0.0, 1.0),
            4 => *solver = RlSolverMode::from_code(val),
            _ => {}
        }
    }
    *mdp = SculptedGridMdp::default_with_traps(w, h, gamma);
    mdp.wind_slip_prob = wind;
    apply_rle_to_mdp(mdp, &payload.rle_mask)
}

/// Solves exact Bellman optimality $V^*(s), Q^*(s,a)$ and evaluates $V_\pi(s)$ for `ws`.
#[must_use]
pub fn solve_bellman_with_policy(
    mdp: &SculptedGridMdp,
    ws: &DenseRlWorkspace,
    solver: RlSolverMode,
) -> BellmanSolution {
    let mut sol = BellmanSolution::solve(mdp, 1e-7, 160);
    let (w, h) = (mdp.width as usize, mdp.height as usize);
    let policy = match solver {
        RlSolverMode::TabularQ => TabularGridPolicy::from_q_table(w, h, &ws.q_table),
        RlSolverMode::PolicyGradient => {
            TabularGridPolicy::from_softmax_logits(w, h, &ws.theta_logits)
        }
    };
    sol.evaluate_policy(mdp, &policy, 1e-7, 120);
    sol
}

/// Computes mean $|\hat{V}(s) - V^*(s)|$ across passable non-terminal states.
#[must_use]
pub fn mean_bellman_residual(
    mdp: &SculptedGridMdp,
    ws: &DenseRlWorkspace,
    solver: RlSolverMode,
    bellman: &BellmanSolution,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 0..mdp.height {
        for x in 0..mdp.width {
            let st = GridState { x, y };
            if mdp.is_terminal(&st) || mdp.cell_role(st) == GridCellRole::SolidWall {
                continue;
            }
            let empirical = match solver {
                RlSolverMode::TabularQ => ws.max_q_at(st),
                RlSolverMode::PolicyGradient => ws.state_value_at(st),
            };
            sum += (empirical - bellman.v_star_at(st)).abs();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / (count as f64)
    }
}

/// Renders the shared maze preset selector, brush mode buttons, radius control, and wind slider.
/// Returns `true` if the MDP topology or wind slip probability changed.
pub fn render_rl_sculpt_toolbar(
    ui: &mut egui::Ui,
    id_salt: &str,
    mdp: &mut SculptedGridMdp,
    brush_kit: &mut SpatialBrushKit,
    preset: &mut RlMazePreset,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Maze:");
        egui::ComboBox::from_id_salt((id_salt, "rl_maze_preset"))
            .selected_text(preset.label())
            .show_ui(ui, |ui| {
                for c in RlMazePreset::ALL {
                    if ui.selectable_value(preset, c, c.label()).clicked() {
                        *mdp = c.build_mdp(mdp.width, mdp.height, mdp.gamma);
                        changed = true;
                    }
                }
            });
    });
    ui.horizontal_wrapped(|ui| {
        for (mode, text) in RL_BRUSH_MODES {
            let r = ui.selectable_label(brush_kit.mode == mode, text);
            if r.accessible_hover_text(mode.label()).clicked() {
                brush_kit.mode = mode;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("Brush r:");
        ui.add(egui::Slider::new(&mut brush_kit.radius, 0.5..=3.5).step_by(1.0));
    });
    let slip_slider = egui::Slider::new(&mut mdp.wind_slip_prob, 0.0..=0.8).text("Wind Slip p");
    let slip_tip = "Probability that wind cells push the agent 1 extra cell";
    if ui
        .add(slip_slider)
        .accessible_hover_text(slip_tip)
        .changed()
    {
        changed = true;
    }
    changed
}

/// Non-color badge glyph rendered on the MDP start cell across all overlay modes.
pub const START_BADGE_GLYPH: &str = "S";

/// Short non-color glyph indicating non-free [`GridCellRole`]s (WCAG 1.4.1 non-color cue).
#[must_use]
pub const fn role_glyph(role: GridCellRole) -> Option<&'static str> {
    match role {
        GridCellRole::Free => None,
        GridCellRole::SolidWall => Some("■"),
        GridCellRole::Goal => Some("★"),
        GridCellRole::Trap => Some("✕"),
        GridCellRole::WindUp => Some("↑"),
        GridCellRole::WindDown => Some("↓"),
        GridCellRole::WindLeft => Some("←"),
        GridCellRole::WindRight => Some("→"),
    }
}

/// Resolves the non-color glyph for `role`, falling back to [`START_BADGE_GLYPH`] on the start cell.
#[must_use]
pub const fn cell_role_or_start_glyph(role: GridCellRole, is_start: bool) -> Option<&'static str> {
    match (role_glyph(role), is_start) {
        (Some(g), _) => Some(g),
        (None, true) => Some(START_BADGE_GLYPH),
        (None, false) => None,
    }
}

/// Draws a directional policy arrow centered inside `center` using `stroke`.
pub fn paint_policy_arrow(
    painter: &egui::Painter,
    center: egui::Pos2,
    action: Move,
    half_len: f32,
    stroke: egui::Stroke,
) {
    let dir = match action {
        Move::Up => egui::vec2(0.0, -1.0),
        Move::Down => egui::vec2(0.0, 1.0),
        Move::Left => egui::vec2(-1.0, 0.0),
        Move::Right => egui::vec2(1.0, 0.0),
    };
    let perp = egui::vec2(-dir.y, dir.x);
    let tail = center - dir * (half_len * 0.65);
    let tip = center + dir * half_len;
    painter.line_segment([tail, tip], stroke);
    let head = half_len * 0.45;
    painter.line_segment([tip, tip - dir * head + perp * (head * 0.65)], stroke);
    painter.line_segment([tip, tip - dir * head - perp * (head * 0.65)], stroke);
}

/// Returns the semantic background fill for a sculpted cell when no continuous scalar is active.
#[must_use]
pub fn role_base_fill(role: GridCellRole, palette: &SemanticPalette) -> egui::Color32 {
    match role {
        GridCellRole::Goal => palette.success,
        GridCellRole::Trap => palette.error,
        GridCellRole::SolidWall => palette.graph_edge,
        _ => palette.surface_fill,
    }
}

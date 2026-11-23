// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]

use super::{chord_tangent_canvas, chord_tangent_scene, chord_tangent_ui};
use crate::framework::{InteractiveTool, PaneSidePanel, ToolStateBounds, scoped_panel_id};
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    CurvePoint, MAX_LATTICE_PRIME, PrimeWeierstrassCurve, WeierstrassCurve,
};
use math_explorer::pure_math::number_theory::primes::is_prime;
use serde::{Deserialize, Serialize};

const PLAYGROUND_STATE_BOUNDS: ToolStateBounds = ToolStateBounds::new(16 * 1024, 8, 64);

/// Coordinate field domain for [`WeierstrassPlaygroundTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveFieldMode {
    /// Continuous real curve $E(\mathbb{R})$.
    Real,
    /// Finite prime-field torus $E(\mathbb{F}_p)$ with $3 < p \le 997$.
    FiniteField,
}

/// Active point updated by unmodified canvas clicks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClickTarget {
    /// Primary point $P$ (and cyclic orbit generator).
    PointP,
    /// Secondary secant point $Q$.
    PointQ,
}

/// Bounded serializable preset configuration for [`WeierstrassPlaygroundTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeierstrassPlaygroundState {
    /// Real curve linear coefficient $a \in \[-20, 20\]$.
    pub real_a: f64,
    /// Real curve constant coefficient $b \in \[-20, 20\]$.
    pub real_b: f64,
    /// Desired x-coordinate for real point $P$.
    pub real_px: f64,
    /// Desired x-coordinate for real point $Q$.
    pub real_qx: f64,
    /// Sign branch for $y_P$ (`true` selects $+\sqrt{x^3+ax+b}$).
    pub real_p_upper: bool,
    /// Sign branch for $y_Q$ (`true` selects $+\sqrt{x^3+ax+b}$).
    pub real_q_upper: bool,
    /// Selected field geometry ($\mathbb{R}$ vs $\mathbb{F}_p \times \mathbb{F}_p$).
    pub field_mode: CurveFieldMode,
    /// Prime field modulus $p \in \[5, 997\]$.
    pub prime_p: i64,
    /// Modular linear coefficient $a$.
    pub mod_a: i64,
    /// Modular constant coefficient $b$.
    pub mod_b: i64,
    /// Selected index into affine lattice points for $P$.
    pub selected_p_idx: usize,
    /// Selected index into affine lattice points for $Q$.
    pub selected_q_idx: usize,
    /// Scalar multiplier $k \in \[1, 1024\]$ for $\[k\]P$ orbit.
    pub scalar_k: i64,
    /// Whether to draw the secant/tangent line and reflection $-R \to P + Q$.
    pub show_secant: bool,
    /// Whether to draw the cyclic subgroup trajectory $\{\[1\]P, \dots, \[k\]P\}$.
    pub show_orbit: bool,
}

impl Default for WeierstrassPlaygroundState {
    fn default() -> Self {
        Self {
            real_a: -1.5,
            real_b: 1.0,
            real_px: -1.0,
            real_qx: 0.5,
            real_p_upper: true,
            real_q_upper: true,
            field_mode: CurveFieldMode::FiniteField,
            prime_p: 17,
            mod_a: 2,
            mod_b: 2,
            selected_p_idx: 0,
            selected_q_idx: 3,
            scalar_k: 4,
            show_secant: true,
            show_orbit: true,
        }
    }
}

/// Interactive chord-and-tangent and toroidal lattice orbit explorer.
pub struct WeierstrassPlaygroundTool {
    pub(crate) state: WeierstrassPlaygroundState,
    pub(crate) click_target: ClickTarget,
    pub(crate) lattice_points: Vec<CurvePoint<i64>>,
    pub(crate) is_playing: bool,
    pub(crate) focused_canvas: Option<egui::Id>,
    last_tick_secs: f64,
    revision: u64,
}

impl Default for WeierstrassPlaygroundTool {
    fn default() -> Self {
        let mut tool = Self {
            state: WeierstrassPlaygroundState::default(),
            click_target: ClickTarget::PointP,
            lattice_points: Vec::new(),
            is_playing: false,
            focused_canvas: None,
            last_tick_secs: 0.0,
            revision: 1,
        };
        tool.rebuild_lattice();
        tool
    }
}

pub(crate) fn snap_prime_i64(candidate: i64) -> i64 {
    let clamped = candidate.clamp(5, MAX_LATTICE_PRIME);
    if is_prime(clamped as u64) {
        return clamped;
    }
    for delta in 1..=30_i64 {
        let down = clamped - delta;
        if down >= 5 && is_prime(down as u64) {
            return down;
        }
        let up = clamped + delta;
        if up <= MAX_LATTICE_PRIME && is_prime(up as u64) {
            return up;
        }
    }
    17
}

pub(crate) fn snap_real_point(
    curve: &WeierstrassCurve,
    target_x: f64,
    upper: bool,
) -> CurvePoint<f64> {
    let clamped = target_x.clamp(-4.5, 4.5);
    let rhs0 = curve.rhs(clamped);
    if rhs0 >= 0.0 {
        let y = rhs0.sqrt();
        return CurvePoint::Affine {
            x: clamped,
            y: if upper { y } else { -y },
        };
    }
    for step in 1..=400 {
        let offset = (step as f64) * 0.025;
        for cand in [clamped + offset, clamped - offset] {
            if (-5.0..=5.0).contains(&cand) {
                let rhs = curve.rhs(cand);
                if rhs >= 0.0 {
                    let y = rhs.sqrt();
                    return CurvePoint::Affine {
                        x: cand,
                        y: if upper { y } else { -y },
                    };
                }
            }
        }
    }
    CurvePoint::Infinity
}

fn validate_playground_state(cfg: &WeierstrassPlaygroundState) -> Result<(), String> {
    for val in [cfg.real_a, cfg.real_b, cfg.real_px, cfg.real_qx] {
        if !val.is_finite() || !(-20.0..=20.0).contains(&val) {
            return Err("Real curve parameters must be finite and within [-20, 20]".into());
        }
    }
    WeierstrassCurve::new(cfg.real_a, cfg.real_b).map_err(|e| e.to_string())?;
    if !(5..=MAX_LATTICE_PRIME).contains(&cfg.prime_p) || !is_prime(cfg.prime_p as u64) {
        return Err(format!(
            "Prime modulus p must be a prime in [5, {MAX_LATTICE_PRIME}]"
        ));
    }
    if !(-997..=997).contains(&cfg.mod_a) || !(-997..=997).contains(&cfg.mod_b) {
        return Err("Modular curve coefficients must lie in [-997, 997]".into());
    }
    PrimeWeierstrassCurve::new(cfg.mod_a, cfg.mod_b, cfg.prime_p).map_err(|e| e.to_string())?;
    if !(1..=1024).contains(&cfg.scalar_k) {
        return Err("Scalar multiplier k must lie in [1, 1024]".into());
    }
    Ok(())
}

impl WeierstrassPlaygroundTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn rebuild_lattice(&mut self) {
        self.lattice_points.clear();
        if let Ok(curve) =
            PrimeWeierstrassCurve::new(self.state.mod_a, self.state.mod_b, self.state.prime_p)
        {
            let cap = (self.state.prime_p as usize)
                .saturating_mul(2)
                .saturating_add(4);
            let mut scratch = vec![CurvePoint::Infinity; cap];
            if let Ok(written) = curve.write_affine_points(&mut scratch) {
                scratch.truncate(written);
                self.lattice_points = scratch;
            }
        }
        if !self.lattice_points.is_empty() {
            let len = self.lattice_points.len();
            self.state.selected_p_idx %= len;
            self.state.selected_q_idx %= len;
        }
        self.bump_revision();
    }

    pub(crate) fn current_fp_points(&self) -> (CurvePoint<i64>, CurvePoint<i64>) {
        if self.lattice_points.is_empty() {
            return (CurvePoint::Infinity, CurvePoint::Infinity);
        }
        let len = self.lattice_points.len();
        (
            self.lattice_points[self.state.selected_p_idx % len],
            self.lattice_points[self.state.selected_q_idx % len],
        )
    }
}

impl InteractiveTool for WeierstrassPlaygroundTool {
    fn name(&self) -> &'static str {
        "Weierstrass Curve Playground"
    }

    fn domain(&self) -> &'static str {
        "elliptic_curves"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.real_a.is_finite()
            && self.state.real_b.is_finite()
            && self.state.real_px.is_finite()
            && self.state.real_qx.is_finite()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        PLAYGROUND_STATE_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: WeierstrassPlaygroundState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Weierstrass playground state: {e}"))?;
        validate_playground_state(&parsed)?;
        self.state = parsed;
        self.rebuild_lattice();
        Ok(())
    }

    fn step_simulation(&mut self) {
        let max_cycle = match self.state.field_mode {
            CurveFieldMode::Real => 24_i64,
            CurveFieldMode::FiniteField => PrimeWeierstrassCurve::new(
                self.state.mod_a,
                self.state.mod_b,
                self.state.prime_p,
            )
            .ok()
            .and_then(|c| c.point_count().ok())
            .map_or(24_i64, |n| (n as i64).clamp(2, 128)),
        };
        self.state.scalar_k = (self.state.scalar_k % max_cycle) + 1;
        self.bump_revision();
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        chord_tangent_scene::build_export_datasets(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if self.is_playing {
            let now = ui.input(|i| i.time);
            if now - self.last_tick_secs >= 0.25 {
                self.last_tick_secs = now;
                self.step_simulation();
            }
            ui.ctx().request_repaint();
        }
        let panel_id = scoped_panel_id(ui, "weierstrass_playground_controls");
        PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading(self.name());
            ui.separator();
            chord_tangent_ui::render_side_panel(self, ui);
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            chord_tangent_canvas::render_interactive_canvas(self, ui);
        });
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for WeierstrassPlaygroundTool {
    fn theory_description(&self) -> String {
        "Short-Weierstrass elliptic curve group law y^2 = x^3 + ax + b over R and prime fields F_p: chord-and-tangent addition P + Q, toroidal secant wrap-around on F_p x F_p, and cyclic subgroup orbits [k]P.".into()
    }

    fn phonetic_description(&self) -> String {
        "Short Weierstrass elliptic curve chord and tangent group law over real numbers and prime finite fields".into()
    }

    fn theory_citation(&self) -> String {
        "Silverman, The Arithmetic of Elliptic Curves (2009); Washington, Elliptic Curves: Number Theory and Cryptography (2008)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Chord-and-Tangent".into(),
            "Collinear curve intersections P, Q, and -R sum to the identity O, reflecting -R vertically to P + Q".into(),
        );
        map.insert(
            "Toroidal Lattice".into(),
            "Modular secants wrap periodically across the F_p x F_p torus to reveal cyclic cryptographic subgroups".into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WeierstrassPlaygroundTool",
        display_name: "Weierstrass Curve Playground",
        domain: "elliptic_curves",
        tags: &["elliptic-curves", "weierstrass", "finite-fields", "cryptography", "torus"],
        build: || Box::new(WeierstrassPlaygroundTool::default()),
    }
}

const PLAYGROUND_STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Toroidal Elliptic Subgroup",
    "domain": "elliptic_curves",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "real_a": -1.5,
    "real_b": 1.0,
    "real_px": -1.0,
    "real_qx": 0.5,
    "real_p_upper": true,
    "real_q_upper": true,
    "field_mode": "FiniteField",
    "prime_p": 23,
    "mod_a": 1,
    "mod_b": 1,
    "selected_p_idx": 0,
    "selected_q_idx": 2,
    "scalar_k": 6,
    "show_secant": true,
    "show_orbit": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "weierstrass_torus_starter",
        title: "Toroidal Elliptic Subgroup",
        description: "Click lattice points on y² ≡ x³ + x + 1 (mod 23) to watch toroidal secants wrap around F_p × F_p and trace [k]P orbits.",
        category: "Number Theory & Algebra",
        icon: "🍩",
        domain: "elliptic_curves",
        tool_name: Some("WeierstrassPlaygroundTool"),
        preset_generator: || PLAYGROUND_STARTER_PRESET_JSON,
    }
}

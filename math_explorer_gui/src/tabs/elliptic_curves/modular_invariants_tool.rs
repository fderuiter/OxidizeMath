// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]

use super::modular_invariants_ui;
use crate::framework::{scoped_panel_id, InteractiveTool, PaneSidePanel, ToolStateBounds};
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    analyze_frobenius_prime, eisenstein_e4_q_series, hasse_weil_bound, hurwitz_trace_sum,
    j_invariant_q_scaled, j_invariant_real, modular_valuation_summary, psi,
    ramanujan_delta_q_series, theorem_1_1_bounds, theorem_1_2_bounds, FrobeniusClassSummary,
    ModularValuationSummary, PrimeWeierstrassCurve, WeierstrassCurve, MAX_LATTICE_PRIME,
    MAX_SCALED_J_PRECISION,
};
use math_explorer::pure_math::number_theory::primes::{is_prime, primes_up_to};
use serde::{Deserialize, Serialize};

const MODULAR_STATE_BOUNDS: ToolStateBounds = ToolStateBounds::new(16 * 1024, 8, 64);

/// Selectable analytical plot lens for [`ModularInvariantsTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModularPlotMode {
    /// Frobenius traces $a_p$ vs Hasse-Weil $\pm 2\sqrt{p}$ band and class numbers $h(a_p^2 - 4p)$.
    HasseWeilTraces,
    /// Hurwitz-Kronecker trace sum $\sum_{|t| < \sqrt{p}} H(t^2 - p)$ vs $(p - 2)/3$.
    HurwitzKroneckerSums,
    /// Breuer Theorems 1.1 & 1.2 $p$-adic valuation lower bounds across degree $i + j < \psi(N)$.
    BreuerValuationBounds,
    /// Truncated `QSeries<i64>` coefficient bars ($\Delta(q)$, $E_4(q)$, or $q \cdot j(q)$).
    ModularQSeriesBars,
}

impl ModularPlotMode {
    /// Human-readable display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HasseWeilTraces => "Hasse-Weil Traces & Class Numbers",
            Self::HurwitzKroneckerSums => "Hurwitz-Kronecker Trace Identity",
            Self::BreuerValuationBounds => "Breuer Modular Polynomial Valuations",
            Self::ModularQSeriesBars => "Modular Form Q-Series Expansion",
        }
    }
}

/// Truncated modular $q$-series selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModularQSeriesKind {
    /// Ramanujan cusp form $\Delta(q) = \sum_{n \ge 1} \tau(n) q^n$.
    RamanujanDelta,
    /// Weight-4 Eisenstein series $E_4(q) = 1 + 240 \sum_{n \ge 1} \sigma_3(n) q^n$.
    EisensteinE4,
    /// Scaled $j$-invariant $q \cdot j(q) = E_4(q)^3 / (\Delta(q)/q)$.
    ScaledJInvariant,
}

impl ModularQSeriesKind {
    /// Human-readable display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::RamanujanDelta => "Ramanujan Discriminant Δ(q)",
            Self::EisensteinE4 => "Eisenstein Series E₄(q)",
            Self::ScaledJInvariant => "Scaled j-Invariant q·j(q)",
        }
    }
}

/// Bounded serializable preset configuration for [`ModularInvariantsTool`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModularInvariantsState {
    /// Integer Weierstrass coefficient $a$.
    pub coeff_a: i64,
    /// Integer Weierstrass coefficient $b$.
    pub coeff_b: i64,
    /// Upper prime bound for Hasse-Weil and Hurwitz sweeps ($5 \le p \le 997$).
    pub max_prime: u64,
    /// Prime $p$ inspected for Frobenius BQF / class-number summary ($5 \le p \le 997$).
    pub inspect_prime: u64,
    /// Classical modular polynomial level $N \in \[2, 60\]$.
    pub level_n: u64,
    /// Monomial degree $i$ in $\Phi_N(X, Y)$.
    pub deg_i: u64,
    /// Monomial degree $j$ in $\Phi_N(X, Y)$ with $i + j < \psi(N)$.
    pub deg_j: u64,
    /// Truncation order for `QSeries<i64>` expansions ($4 \le \text{precision} \le 32$).
    pub q_precision: usize,
    /// Selected modular $q$-series family.
    pub q_series_kind: ModularQSeriesKind,
    /// Active plot lens.
    pub view_mode: ModularPlotMode,
}

impl Default for ModularInvariantsState {
    fn default() -> Self {
        Self {
            coeff_a: -1,
            coeff_b: 1,
            max_prime: 149,
            inspect_prime: 23,
            level_n: 5,
            deg_i: 1,
            deg_j: 1,
            q_precision: 10,
            q_series_kind: ModularQSeriesKind::RamanujanDelta,
            view_mode: ModularPlotMode::HasseWeilTraces,
        }
    }
}

/// Analytical visualizer bridging Frobenius traces, class numbers, Breuer valuations, and $q$-series.
pub struct ModularInvariantsTool {
    pub(crate) state: ModularInvariantsState,
    revision: u64,
    pub(crate) trace_pts: Vec<[f64; 2]>,
    pub(crate) hasse_upper: Vec<[f64; 2]>,
    pub(crate) hasse_lower: Vec<[f64; 2]>,
    pub(crate) class_num_pts: Vec<[f64; 2]>,
    pub(crate) hurwitz_actual: Vec<[f64; 2]>,
    pub(crate) hurwitz_expected: Vec<[f64; 2]>,
    pub(crate) breuer_bars: [Vec<[f64; 2]>; 4],
    pub(crate) q_coeffs: Vec<i64>,
    pub(crate) frobenius_summary: Option<FrobeniusClassSummary>,
    pub(crate) valuation_summary: Option<ModularValuationSummary>,
    pub(crate) real_j: Option<f64>,
}

impl Default for ModularInvariantsTool {
    fn default() -> Self {
        let mut tool = Self {
            state: ModularInvariantsState::default(),
            revision: 1,
            trace_pts: Vec::new(),
            hasse_upper: Vec::new(),
            hasse_lower: Vec::new(),
            class_num_pts: Vec::new(),
            hurwitz_actual: Vec::new(),
            hurwitz_expected: Vec::new(),
            breuer_bars: std::array::from_fn(|_| Vec::new()),
            q_coeffs: Vec::new(),
            frobenius_summary: None,
            valuation_summary: None,
            real_j: None,
        };
        tool.recompute();
        tool
    }
}

pub(crate) fn snap_prime_u64(candidate: u64) -> u64 {
    let clamped = candidate.clamp(5, MAX_LATTICE_PRIME as u64);
    if is_prime(clamped) {
        return clamped;
    }
    for delta in 1..=30_u64 {
        let down = clamped.saturating_sub(delta);
        if down >= 5 && is_prime(down) {
            return down;
        }
        let up = clamped.saturating_add(delta);
        if up <= MAX_LATTICE_PRIME as u64 && is_prime(up) {
            return up;
        }
    }
    23
}

fn validate_modular_state(cfg: &ModularInvariantsState) -> Result<(), String> {
    if !(-50..=50).contains(&cfg.coeff_a) || !(-50..=50).contains(&cfg.coeff_b) {
        return Err("Curve coefficients a and b must lie within [-50, 50]".into());
    }
    if 4 * cfg.coeff_a.pow(3) + 27 * cfg.coeff_b.pow(2) == 0 {
        return Err("Curve coefficients define a singular curve (4a^3 + 27b^2 = 0)".into());
    }
    if !(5..=MAX_LATTICE_PRIME as u64).contains(&cfg.max_prime) {
        return Err(format!("max_prime must lie in [5, {MAX_LATTICE_PRIME}]"));
    }
    if !(5..=MAX_LATTICE_PRIME as u64).contains(&cfg.inspect_prime) || !is_prime(cfg.inspect_prime)
    {
        return Err(format!(
            "inspect_prime must be a prime in [5, {MAX_LATTICE_PRIME}]"
        ));
    }
    if !(2..=60).contains(&cfg.level_n) {
        return Err("Modular level N must lie in [2, 60]".into());
    }
    if cfg.deg_i.saturating_add(cfg.deg_j) >= psi(cfg.level_n) {
        return Err("Degrees i + j must be strictly less than psi(N)".into());
    }
    if !(4..=32).contains(&cfg.q_precision) {
        return Err("Q-series precision must lie in [4, 32]".into());
    }
    Ok(())
}

impl ModularInvariantsTool {
    pub(crate) fn recompute(&mut self) {
        let max_deg = psi(self.state.level_n).saturating_sub(1);
        if self.state.deg_i + self.state.deg_j > max_deg {
            self.state.deg_i = 0;
            self.state.deg_j = 0;
        }
        self.real_j = WeierstrassCurve::new(self.state.coeff_a as f64, self.state.coeff_b as f64)
            .ok()
            .map(|c| j_invariant_real(&c));
        self.recompute_prime_sweeps();
        self.recompute_breuer_and_q_series();
        self.revision = self.revision.wrapping_add(1);
    }

    fn recompute_prime_sweeps(&mut self) {
        self.trace_pts.clear();
        self.hasse_upper.clear();
        self.hasse_lower.clear();
        self.class_num_pts.clear();
        self.hurwitz_actual.clear();
        self.hurwitz_expected.clear();

        for p in primes_up_to(self.state.max_prime) {
            if p <= 3 {
                continue;
            }
            let x = p as f64;
            let bound = hasse_weil_bound(p);
            self.hasse_upper.push([x, bound]);
            self.hasse_lower.push([x, -bound]);
            let (h_sum, h_exp, _) = hurwitz_trace_sum(p);
            self.hurwitz_actual.push([x, h_sum]);
            self.hurwitz_expected.push([x, h_exp]);

            if let Ok(curve) =
                PrimeWeierstrassCurve::new(self.state.coeff_a, self.state.coeff_b, p as i64)
            {
                if let Ok(summary) = analyze_frobenius_prime(&curve) {
                    self.trace_pts.push([x, summary.frobenius_trace as f64]);
                    self.class_num_pts
                        .push([x, summary.imaginary_class_number as f64]);
                }
            }
        }

        let inspect_curve = PrimeWeierstrassCurve::new(
            self.state.coeff_a,
            self.state.coeff_b,
            self.state.inspect_prime as i64,
        )
        .ok();
        self.frobenius_summary = inspect_curve
            .as_ref()
            .and_then(|c| analyze_frobenius_prime(c).ok());
        self.valuation_summary = inspect_curve.as_ref().and_then(|c| {
            modular_valuation_summary(c, self.state.level_n, self.state.deg_i, self.state.deg_j)
                .ok()
        });
    }

    fn recompute_breuer_and_q_series(&mut self) {
        for series in &mut self.breuer_bars {
            series.clear();
        }
        let psi_n = psi(self.state.level_n);
        for total_deg in 0..psi_n.min(40) {
            let x = total_deg as f64;
            if let Ok(b11) = theorem_1_1_bounds(self.state.level_n, total_deg, 0) {
                self.breuer_bars[0].push([x, b11.v2_bound.unwrap_or(0) as f64]);
                self.breuer_bars[1].push([x, b11.v3_bound.unwrap_or(0) as f64]);
                self.breuer_bars[2].push([x, b11.v5_bound.unwrap_or(0) as f64]);
            }
            if let Ok(b12) = theorem_1_2_bounds(self.state.level_n, total_deg, 0) {
                self.breuer_bars[3].push([x, b12.v7_bound.unwrap_or(0) as f64]);
            }
        }

        let prec = self.state.q_precision;
        self.q_coeffs = match self.state.q_series_kind {
            ModularQSeriesKind::RamanujanDelta => ramanujan_delta_q_series(prec).coeffs,
            ModularQSeriesKind::EisensteinE4 => eisenstein_e4_q_series(prec).coeffs,
            ModularQSeriesKind::ScaledJInvariant => {
                j_invariant_q_scaled(prec.min(MAX_SCALED_J_PRECISION)).coeffs
            }
        };
    }
}

impl InteractiveTool for ModularInvariantsTool {
    fn name(&self) -> &'static str {
        "Modular Invariants & Traces"
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

    fn state_bounds(&self) -> ToolStateBounds {
        MODULAR_STATE_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: ModularInvariantsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid modular invariants state: {e}"))?;
        validate_modular_state(&parsed)?;
        self.state = parsed;
        self.recompute();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        modular_invariants_ui::build_export_datasets(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(ui, "modular_invariants_controls");
        PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading(self.name());
            ui.separator();
            modular_invariants_ui::render_controls(self, ui);
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            modular_invariants_ui::render_plot(self, ui);
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

impl scientific_metadata::theory::TheoryDescribable for ModularInvariantsTool {
    fn theory_description(&self) -> String {
        format!(
            "Hasse-Weil Frobenius trace bounds |a_p| <= 2*sqrt(p), imaginary quadratic class numbers h(a_p^2 - 4p), Hurwitz-Kronecker trace sums sum H(t^2 - p) = (p - 2)/3, Breuer modular polynomial p-adic valuations at level N = {}, and modular form q-series expansions.",
            self.state.level_n
        )
    }

    fn phonetic_description(&self) -> String {
        "Hasse Weil Frobenius traces, Hurwitz Kronecker class numbers, Breuer modular polynomial bounds, and modular q series".into()
    }

    fn theory_citation(&self) -> String {
        "Breuer, Divisibility of Coefficients of Classical Modular Polynomials; Kronecker-Hurwitz Class Number Trace Formula; Silverman, The Arithmetic of Elliptic Curves (2009)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Hasse-Weil".into(),
            "Frobenius trace a_p = p + 1 - #E(F_p) bounded within [-2*sqrt(p), 2*sqrt(p)]".into(),
        );
        map.insert(
            "Hurwitz-Kronecker".into(),
            "Weighted imaginary quadratic class number trace formula over prime fields".into(),
        );
        map.insert(
            "Breuer Valuations".into(),
            "Lower bounds on 2-, 3-, 5-, and 7-adic valuations of classical modular polynomial coefficients".into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ModularInvariantsTool",
        display_name: "Modular Invariants & Traces",
        domain: "elliptic_curves",
        tags: &["hasse-weil", "frobenius", "hurwitz-kronecker", "modular-forms", "q-series"],
        build: || Box::new(ModularInvariantsTool::default()),
    }
}

const MODULAR_STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hasse-Weil Band & Modular Valuations",
    "domain": "elliptic_curves",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "coeff_a": 1,
    "coeff_b": 2,
    "max_prime": 251,
    "inspect_prime": 29,
    "level_n": 7,
    "deg_i": 2,
    "deg_j": 1,
    "q_precision": 12,
    "q_series_kind": "RamanujanDelta",
    "view_mode": "HasseWeilTraces"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "modular_invariants_starter",
        title: "Hasse-Weil Band & Modular Valuations",
        description: "Sweep Frobenius traces across primes inside the ±2√p Hasse-Weil envelope alongside Hurwitz-Kronecker class numbers and Breuer bounds.",
        category: "Number Theory & Algebra",
        icon: "📈",
        domain: "elliptic_curves",
        tool_name: Some("ModularInvariantsTool"),
        preset_generator: || MODULAR_STARTER_PRESET_JSON,
    }
}

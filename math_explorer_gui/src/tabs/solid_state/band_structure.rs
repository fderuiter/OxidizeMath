#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:solid_state]

#[path = "band_structure_ui.rs"]
mod band_structure_ui;

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::physics::solid_state::quasiperiodic::{
    QuasiperiodicChainModel, QuasiperiodicWorkspace,
};
use std::f64::consts::TAU;

/// Dispersion regime in [`BandStructureTool`]: periodic Bloch vs quasiperiodic chains.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BandRegime {
    /// Periodic 1D tight-binding $E(k) = E_0 - 2t\cos(ka)$ and optional free-electron parabola.
    #[default]
    PeriodicBloch,
    /// Off-diagonal 1D Fibonacci hopping chain opening a self-similar Cantor gap spectrum.
    FibonacciChain,
    /// Incommensurate on-site Aubry-André cosine potential with localization transition at $\lambda = 2t$.
    AubryAndre,
}

impl BandRegime {
    pub(crate) const fn is_periodic(&self) -> bool {
        matches!(self, Self::PeriodicBloch)
    }

    /// Human-readable selector label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PeriodicBloch => "Periodic 1D Bloch",
            Self::FibonacciChain => "Fibonacci Cantor Chain",
            Self::AubryAndre => "Aubry-André Localization",
        }
    }
}

/// Primary visualization lens when inspecting finite quasiperiodic chains.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum QuasiperiodicPlotView {
    /// Sorted eigenenergies $E_n$ vs normalized mode index $(n + 1)/N$ compared against periodic Bloch.
    #[default]
    SpectrumAndGaps,
    /// Devil's-staircase Integrated Density of States $\text{IDOS}(E)$.
    IdosStaircase,
    /// Per-eigenstate Inverse Participation Ratio $\text{IPR}(E_n) = \sum_i |\psi_n(i)|^4$.
    EigenstateIpr,
}

impl QuasiperiodicPlotView {
    /// Human-readable view label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SpectrumAndGaps => "Spectrum & Cantor Gaps",
            Self::IdosStaircase => "IDOS Devil's Staircase",
            Self::EigenstateIpr => "Eigenstate IPR (Localization)",
        }
    }
}

/// Interactive 1D periodic Bloch and quasiperiodic (Fibonacci / Aubry-André) band explorer.
pub struct BandStructureTool {
    pub(crate) e0: f64,
    pub(crate) t: f64,
    pub(crate) a: f64,
    pub(crate) hbar: f64,
    pub(crate) m: f64,
    pub(crate) show_free_electron: bool,
    pub(crate) regime: BandRegime,
    pub(crate) plot_view: QuasiperiodicPlotView,
    pub(crate) chain_sites: usize,
    pub(crate) modulation: f64,
    pub(crate) aubry_lambda: f64,
    pub(crate) phason_phase: f64,
    pub(crate) workspace: QuasiperiodicWorkspace,
    pub(crate) last_error: Option<String>,
}

impl Default for BandStructureTool {
    fn default() -> Self {
        Self {
            e0: 0.0,
            t: 1.0,
            a: 1.0,
            hbar: 1.0,
            m: 1.0,
            show_free_electron: false,
            regime: BandRegime::PeriodicBloch,
            plot_view: QuasiperiodicPlotView::SpectrumAndGaps,
            chain_sites: 55,
            modulation: 0.45,
            aubry_lambda: 2.0,
            phason_phase: 0.0,
            workspace: QuasiperiodicWorkspace::new(233),
            last_error: None,
        }
    }
}

impl BandStructureTool {
    pub(crate) fn refresh_quasiperiodic_spectrum(&mut self) {
        if self.regime.is_periodic() {
            self.last_error = None;
            return;
        }
        let model = match self.regime {
            BandRegime::PeriodicBloch => return,
            BandRegime::FibonacciChain => QuasiperiodicChainModel::Fibonacci {
                sites: self.chain_sites,
                t_avg: self.t,
                modulation: self.modulation,
                e0: self.e0,
            },
            BandRegime::AubryAndre => QuasiperiodicChainModel::AubryAndre {
                sites: self.chain_sites,
                t: self.t,
                lambda: self.aubry_lambda,
                phase: self.phason_phase,
                e0: self.e0,
            },
        };
        self.last_error = self.workspace.solve_into(model).err().map(str::to_string);
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BandStructureState {
    e0: f64,
    t: f64,
    a: f64,
    hbar: f64,
    m: f64,
    show_free_electron: bool,
    #[serde(default, skip_serializing_if = "BandRegime::is_periodic")]
    regime: BandRegime,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plot_view: Option<QuasiperiodicPlotView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chain_sites: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    modulation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    aubry_lambda: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    phason_phase: Option<f64>,
}

impl BandStructureState {
    fn validate(&self) -> Result<(), String> {
        if !self.e0.is_finite() || !(-5.0..=5.0).contains(&self.e0) {
            return Err(format!("e0 out of [-5.0, 5.0]: {}", self.e0));
        }
        if !self.t.is_finite() || !(0.0..=5.0).contains(&self.t) {
            return Err(format!("t out of [0.0, 5.0]: {}", self.t));
        }
        if !self.a.is_finite()
            || !(0.1..=5.0).contains(&self.a)
            || !self.hbar.is_finite()
            || !(0.1..=5.0).contains(&self.hbar)
            || !self.m.is_finite()
            || !(0.1..=5.0).contains(&self.m)
        {
            return Err("a, hbar, and m must be finite in [0.1, 5.0]".to_string());
        }
        if let Some(sites) = self.chain_sites {
            if !(8..=233).contains(&sites) {
                return Err(format!("chain_sites out of [8, 233]: {sites}"));
            }
        }
        if let Some(delta) = self.modulation {
            if !delta.is_finite() || !(0.0..=0.95).contains(&delta) {
                return Err(format!("modulation out of [0.0, 0.95]: {delta}"));
            }
        }
        if let Some(lam) = self.aubry_lambda {
            if !lam.is_finite() || !(0.0..=5.0).contains(&lam) {
                return Err(format!("aubry_lambda out of [0.0, 5.0]: {lam}"));
            }
        }
        if let Some(phi) = self.phason_phase {
            if !phi.is_finite() || !(-TAU..=TAU).contains(&phi) {
                return Err(format!("phason_phase out of [-2π, 2π]: {phi}"));
            }
        }
        Ok(())
    }
}

impl InteractiveTool for BandStructureTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Band Structure"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 4_096,
            max_depth: 4,
            max_array_len: 16,
        }
    }

    fn save_state(&self) -> Option<String> {
        let qp = !self.regime.is_periodic();
        serde_json::to_string(&BandStructureState {
            e0: self.e0,
            t: self.t,
            a: self.a,
            hbar: self.hbar,
            m: self.m,
            show_free_electron: self.show_free_electron,
            regime: self.regime,
            plot_view: qp.then_some(self.plot_view),
            chain_sites: qp.then_some(self.chain_sites),
            modulation: qp.then_some(self.modulation),
            aubry_lambda: qp.then_some(self.aubry_lambda),
            phason_phase: qp.then_some(self.phason_phase),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: BandStructureState =
            serde_json::from_str(state).map_err(|e| format!("Invalid BandStructure state: {e}"))?;
        s.validate()?;
        self.e0 = s.e0;
        self.t = s.t;
        self.a = s.a;
        self.hbar = s.hbar;
        self.m = s.m;
        self.show_free_electron = s.show_free_electron;
        self.regime = s.regime;
        if let Some(view) = s.plot_view {
            self.plot_view = view;
        }
        if let Some(sites) = s.chain_sites {
            self.chain_sites = sites;
        }
        if let Some(delta) = s.modulation {
            self.modulation = delta;
        }
        if let Some(lam) = s.aubry_lambda {
            self.aubry_lambda = lam;
        }
        if let Some(phi) = s.phason_phase {
            self.phason_phase = phi;
        }
        self.refresh_quasiperiodic_spectrum();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.refresh_quasiperiodic_spectrum();
        self.draw_controls_panel(parent_ui);
        self.draw_central_plot(parent_ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BandStructureTool",
        display_name: "Band Structure",
        domain: "solid_state",
        tags: &[
            "tight-binding",
            "brillouin zone",
            "dispersion",
            "bloch",
            "effective mass",
            "quasiperiodic",
            "fibonacci",
            "aubry-andre",
            "cantor",
        ],
        build: || Box::new(BandStructureTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BandStructureTool {
    fn theory_description(&self) -> String {
        match self.regime {
            BandRegime::PeriodicBloch => {
                let detail = if self.show_free_electron {
                    format!("parabolic free-electron dispersion with effective mass $m = {:.2}$", self.m)
                } else {
                    format!(
                        "1D tight-binding dispersion $E(k) = E_0 - 2t\\cos(ka)$ ($E_0 = {:.2}$, $t = {:.2}$, $a = {:.2}$)",
                        self.e0, self.t, self.a
                    )
                };
                format!("Periodic Bloch band dispersion in the first Brillouin zone $[-\\pi/a, \\pi/a]$: {detail}.")
            }
            BandRegime::FibonacciChain => format!(
                "Quasiperiodic 1D Fibonacci chain ($N = {}$, modulation $\\delta = {:.2}$) exhibiting a singular-continuous Cantor spectrum and devil's-staircase IDOS plateaus at $\\tau^{{-m}}$ (mod 1).",
                self.chain_sites, self.modulation
            ),
            BandRegime::AubryAndre => format!(
                "Incommensurate Aubry-André chain ($N = {}$, $\\lambda = {:.2}$, $2t = {:.2}$) with self-dual metal-insulator localization transition at $\\lambda = 2t$.",
                self.chain_sites,
                self.aubry_lambda,
                2.0 * self.t
            ),
        }
    }
    fn phonetic_description(&self) -> String {
        "Electronic band structure, Fibonacci Cantor spectrum, and Aubry-Andre localization theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Localization Problem in One Dimension: Mapping and Escape (Kohmoto, Kadanoff & Tang, 1983); Analyticity breaking and Anderson localization in incommensurate lattices (Aubry & Andre, 1980); Introduction to Solid State Physics (Kittel, 2004); Solid State Physics (Ashcroft & Mermin, 1976)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "TightBinding".into(),
            "E(k) = E_0 - 2t cos(ka) periodic Bloch orbital overlap dispersion".into(),
        );
        map.insert(
            "FreeElectron".into(),
            "E(k) = hbar^2 k^2 / (2m) parabolic free Fermi gas dispersion".into(),
        );
        map.insert(
            "FibonacciCantor".into(),
            "Off-diagonal golden-mean hopping chain with singular-continuous Cantor spectrum"
                .into(),
        );
        map.insert(
            "AubryAndre".into(),
            "Self-dual incommensurate cosine potential localizing all states for lambda > 2t"
                .into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Electronic Band Structure",
    "domain": "solid_state",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "e0": 0.0,
    "t": 1.5,
    "a": 1.2,
    "hbar": 1.0,
    "m": 1.0,
    "show_free_electron": true
  }
}"#;

const FIBONACCI_STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fibonacci Cantor Band Spectrum",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "e0": 0.0,
    "t": 1.0,
    "a": 1.0,
    "hbar": 1.0,
    "m": 1.0,
    "show_free_electron": false,
    "regime": "FibonacciChain",
    "plot_view": "IdosStaircase",
    "chain_sites": 55,
    "modulation": 0.45,
    "aubry_lambda": 2.0,
    "phason_phase": 0.0
  }
}"#;

const AUBRY_ANDRE_STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Aubry-André Localization Transition",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "e0": 0.0,
    "t": 1.0,
    "a": 1.0,
    "hbar": 1.0,
    "m": 1.0,
    "show_free_electron": false,
    "regime": "AubryAndre",
    "plot_view": "EigenstateIpr",
    "chain_sites": 89,
    "modulation": 0.45,
    "aubry_lambda": 2.6,
    "phason_phase": 0.1
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "band_structure_starter",
        title: "Electronic Band Structure",
        description: "1D tight-binding E(k) = E₀ - 2t·cos(ka) and free-electron Brillouin zone dispersion.",
        category: "Condensed Matter",
        icon: "📉",
        domain: "solid_state",
        tool_name: Some("BandStructureTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "band_fibonacci_cantor_starter",
        title: "Fibonacci Cantor Band Spectrum",
        description: "Self-similar Cantor gaps and devil's-staircase IDOS on a 55-site golden-ratio Fibonacci chain.",
        category: "Condensed Matter",
        icon: "🪜",
        domain: "solid_state",
        tool_name: Some("BandStructureTool"),
        preset_generator: || FIBONACCI_STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "band_aubry_andre_starter",
        title: "Aubry-André Localization Transition",
        description: "Incommensurate cosine chain across the self-dual metal-insulator critical point λ = 2t.",
        category: "Condensed Matter",
        icon: "📍",
        domain: "solid_state",
        tool_name: Some("BandStructureTool"),
        preset_generator: || AUBRY_ANDRE_STARTER_PRESET_JSON,
    }
}

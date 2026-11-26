//! `InteractiveTool`, `TheoryDescribable`, and `inventory` starter gallery registration
//! for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MarkovSequenceWorkbenchTool;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::presets::{
    MarkovWorkbenchPresetConfig, MAX_BAUM_WELCH_ITERATIONS, MAX_OBSERVATION_SEQUENCE_LEN,
};
use eframe::egui;

impl InteractiveTool for MarkovSequenceWorkbenchTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Markov Sequence & Reliability"
    }

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.hmm.bw_tolerance.is_finite()
            && self.config.hmm.bw_dirichlet_alpha.is_finite()
            && self.config.ctmc.horizon_time.is_finite()
            && self.config.ctmc.eval_time.is_finite()
            && self.config.hmm.initial.iter().all(|v| v.is_finite())
            && self
                .config
                .dtmc
                .absorbing_rewards
                .iter()
                .all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        let mut changed = false;
        let tol = if self.config.hmm.bw_tolerance.is_finite() {
            self.config.hmm.bw_tolerance.clamp(0.0, 1.0)
        } else {
            1e-6
        };
        if (tol - self.config.hmm.bw_tolerance).abs() > f64::EPSILON {
            self.config.hmm.bw_tolerance = tol;
            changed = true;
        }
        let alpha = if self.config.hmm.bw_dirichlet_alpha.is_finite() {
            self.config.hmm.bw_dirichlet_alpha.clamp(0.0, 10.0)
        } else {
            1e-4
        };
        if (alpha - self.config.hmm.bw_dirichlet_alpha).abs() > f64::EPSILON {
            self.config.hmm.bw_dirichlet_alpha = alpha;
            changed = true;
        }
        let horizon = if self.config.ctmc.horizon_time.is_finite() {
            self.config.ctmc.horizon_time.clamp(0.05, 1000.0)
        } else {
            10.0
        };
        if (horizon - self.config.ctmc.horizon_time).abs() > f64::EPSILON {
            self.config.ctmc.horizon_time = horizon;
            changed = true;
        }
        let eval_t = if self.config.ctmc.eval_time.is_finite() {
            self.config.ctmc.eval_time.clamp(0.0, 1000.0)
        } else {
            1.0
        };
        if (eval_t - self.config.ctmc.eval_time).abs() > f64::EPSILON {
            self.config.ctmc.eval_time = eval_t;
            changed = true;
        }
        self.config.hmm.bw_max_iterations = self
            .config
            .hmm
            .bw_max_iterations
            .clamp(1, MAX_BAUM_WELCH_ITERATIONS);
        self.config.hmm.sim_length = self
            .config
            .hmm
            .sim_length
            .clamp(1, MAX_OBSERVATION_SEQUENCE_LEN);
        if changed {
            let _ = self.recompute_all();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        changed
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: MarkovWorkbenchPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid MarkovSequenceWorkbenchTool state: {e}"))?;
        self.apply_config(cfg)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        self.poll_io(&ctx);
        let side_id = crate::framework::scoped_panel_id(parent_ui, "markov_workbench_side_panel");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_side_controls(ui);
        });
        self.show_bottom_dock(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_central_plot(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for MarkovSequenceWorkbenchTool {
    fn theory_description(&self) -> String {
        "Unified discrete and continuous Markov workbench: decodes hidden regimes via exact log-space Viterbi dynamic programming and scaled Forward-Backward posteriors $\\gamma_t(i)$, fits HMM parameters via zero-allocation Dirichlet-smoothed Baum-Welch EM, evaluates continuous-time reliability $P(t) = \\exp(G t)$ and MTTF, and solves absorbing chain fundamental matrices $N = (I - Q)^{-1}$ and expected possession value.".into()
    }

    fn phonetic_description(&self) -> String {
        "Markov sequence and reliability workbench with log Viterbi decoding, Baum Welch expectation maximization, matrix exponential generator reliability, and absorbing fundamental matrix valuation".into()
    }

    fn theory_citation(&self) -> String {
        "A tutorial on hidden Markov models and selected applications in speech recognition (Rabiner, 1989); Error bounds for convolutional codes and an asymptotically optimum decoding algorithm (Viterbi, 1967); Markov Chains (Norris, 1997)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            (
                "bw_dirichlet_alpha".to_string(),
                "Symmetric Dirichlet prior pseudocount preventing zero-probability collapse during Baum-Welch M-steps.".to_string(),
            ),
            (
                "eval_time".to_string(),
                "Continuous evaluation time t for transition matrix P(t) = exp(G t).".to_string(),
            ),
        ])
    }
}

pub(crate) const PRESET_MARKOV_HOT_HAND_HMM: &str = r#"{"metadata":{"schema_version":"1.0","title":"Markov Sequence & Reliability - Hot-Hand Regime Inference","domain":"game_theory","timestamp":"2026-11-23T00:00:00Z"},"config":{"mode":"Hmm","rng_seed":101,"hmm":{"state_names":["Cold","Hot"],"symbol_names":["Miss","Make"],"initial":[0.6,0.4],"transitions":[[0.75,0.25],[0.3,0.7]],"emissions":[[0.65,0.35],[0.25,0.75]],"observations":[1,1,1,0,1,1,0,0,0,1,1,1,1,0,1,1],"bw_max_iterations":40,"bw_tolerance":0.000001,"bw_dirichlet_alpha":0.0001,"sim_length":80},"ctmc":{"state_names":["Dual Healthy","Degraded (1 Up)","System Failed"],"generator":[[-0.4,0.4,0.0],[1.2,-1.5,0.3],[0.0,0.0,0.0]],"transient_states":[0,1],"source_state":0,"horizon_time":12.0,"eval_time":3.0},"dtmc":{"state_names":["Perimeter","Paint Advantage","Score (+2)","Turnover (0)"],"transition_matrix":[[0.45,0.3,0.1,0.15],[0.15,0.35,0.38,0.12],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],"state_types":["Transient","Transient","Absorbing","Absorbing"],"absorbing_rewards":[2.0,0.0]}}}"#;

pub(crate) const PRESET_MARKOV_RELIABILITY_CTMC: &str = r#"{"metadata":{"schema_version":"1.0","title":"Markov Sequence & Reliability - Redundant Server MTTF","domain":"game_theory","timestamp":"2026-11-23T00:00:00Z"},"config":{"mode":"Ctmc","rng_seed":202,"hmm":{"state_names":["Bull","Bear","Sideways"],"symbol_names":["Large Up","Small Up","Flat","Small Down","Large Down"],"initial":[0.33,0.33,0.34],"transitions":[[0.7,0.2,0.1],[0.2,0.7,0.1],[0.25,0.25,0.5]],"emissions":[[0.3,0.3,0.2,0.1,0.1],[0.1,0.1,0.2,0.3,0.3],[0.1,0.2,0.4,0.2,0.1]],"observations":[1,0,2,3,4],"bw_max_iterations":40,"bw_tolerance":0.000001,"bw_dirichlet_alpha":0.0001,"sim_length":80},"ctmc":{"state_names":["Dual Healthy","Degraded (1 Up)","System Failed"],"generator":[[-0.4,0.4,0.0],[1.2,-1.5,0.3],[0.0,0.0,0.0]],"transient_states":[0,1],"source_state":0,"horizon_time":15.0,"eval_time":4.0},"dtmc":{"state_names":["Perimeter","Paint Advantage","Score (+2)","Turnover (0)"],"transition_matrix":[[0.45,0.3,0.1,0.15],[0.15,0.35,0.38,0.12],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],"state_types":["Transient","Transient","Absorbing","Absorbing"],"absorbing_rewards":[2.0,0.0]}}}"#;

pub(crate) const PRESET_MARKOV_BASKETBALL_EPV: &str = r#"{"metadata":{"schema_version":"1.0","title":"Markov Sequence & Reliability - Basketball Possession EPV","domain":"game_theory","timestamp":"2026-11-23T00:00:00Z"},"config":{"mode":"AbsorbingDtmc","rng_seed":303,"hmm":{"state_names":["Bull","Bear","Sideways"],"symbol_names":["Large Up","Small Up","Flat","Small Down","Large Down"],"initial":[0.33,0.33,0.34],"transitions":[[0.7,0.2,0.1],[0.2,0.7,0.1],[0.25,0.25,0.5]],"emissions":[[0.3,0.3,0.2,0.1,0.1],[0.1,0.1,0.2,0.3,0.3],[0.1,0.2,0.4,0.2,0.1]],"observations":[1,0,2,3,4],"bw_max_iterations":40,"bw_tolerance":0.000001,"bw_dirichlet_alpha":0.0001,"sim_length":80},"ctmc":{"state_names":["Dual Healthy","Degraded (1 Up)","System Failed"],"generator":[[-0.4,0.4,0.0],[1.2,-1.5,0.3],[0.0,0.0,0.0]],"transient_states":[0,1],"source_state":0,"horizon_time":12.0,"eval_time":3.0},"dtmc":{"state_names":["Perimeter","Paint Advantage","Score (+2)","Turnover (0)"],"transition_matrix":[[0.45,0.3,0.1,0.15],[0.15,0.35,0.38,0.12],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],"state_types":["Transient","Transient","Absorbing","Absorbing"],"absorbing_rewards":[2.0,0.0]}}}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MarkovSequenceWorkbenchTool",
        display_name: "Markov Sequence & Reliability",
        domain: "game_theory",
        tags: &["markov", "hmm", "viterbi", "baum-welch", "ctmc", "reliability", "epv"],
        build: || Box::new(MarkovSequenceWorkbenchTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "markov_hot_hand_hmm",
        title: "Hot-Hand HMM & Baum-Welch Regimes",
        description: "γ_t(i) & argmax ln δ_t: Exact log-space Viterbi decoding and unsupervised Baum-Welch EM on shooting sequences.",
        category: "Game Theory",
        icon: "🏀",
        domain: "game_theory",
        tool_name: Some("MarkovSequenceWorkbenchTool"),
        preset_generator: || PRESET_MARKOV_HOT_HAND_HMM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "markov_reliability_ctmc",
        title: "CTMC Redundant Server Reliability & MTTF",
        description: "P(t) = exp(Gt): Padé matrix-exponential reliability curves and mean time to system failure.",
        category: "Game Theory",
        icon: "⚙️",
        domain: "game_theory",
        tool_name: Some("MarkovSequenceWorkbenchTool"),
        preset_generator: || PRESET_MARKOV_RELIABILITY_CTMC,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "markov_basketball_epv_dtmc",
        title: "Absorbing DTMC Possession Valuation (EPV)",
        description: "N = (I - Q)⁻¹ & EPV = B r: Fundamental visit matrix, absorption odds, and expected possession value.",
        category: "Game Theory",
        icon: "📈",
        domain: "game_theory",
        tool_name: Some("MarkovSequenceWorkbenchTool"),
        preset_generator: || PRESET_MARKOV_BASKETBALL_EPV,
    }
}

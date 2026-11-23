//! Interactive Shannon Entropy, Divergence & Channel Capacity Workbench (`ShannonChannelTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::algorithmic_information::shannon::{
    awgn_bandwidth_capacity, awgn_capacity, bec_capacity, blahut_arimoto_capacity, bsc_capacity,
    categorical_entropy, jensen_shannon_divergence, joint_mutual_information, kl_divergence,
    renyi_entropy, BlahutArimotoWorkspace, JointMiWorkspace,
};
use serde::{Deserialize, Serialize};

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

pub const MAX_UI_ALPHABET: usize = 6;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShannonChannelKind {
    Bsc,
    Bec,
    ZChannel,
    Awgn,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShannonToolConfig {
    /// Crossover / erasure probability in `(0, 1)` (first scalar float for perturbation tests).
    pub noise_param: f64,
    pub snr_linear: f64,
    pub bandwidth_hz: f64,
    pub renyi_alpha: f64,
    pub channel_kind: ShannonChannelKind,
    pub weights_p: Vec<f64>,
    pub weights_q: Vec<f64>,
}

impl ShannonToolConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.noise_param.is_finite() || !(0.0..=0.999).contains(&self.noise_param) {
            return Err("noise_param must be finite and in [0.0, 0.999]".into());
        }
        if !self.snr_linear.is_finite() || !(0.01..=1e4).contains(&self.snr_linear) {
            return Err("snr_linear must be finite and in [0.01, 10000]".into());
        }
        if !self.bandwidth_hz.is_finite() || !(1.0..=1e7).contains(&self.bandwidth_hz) {
            return Err("bandwidth_hz must be finite and in [1, 1e7]".into());
        }
        if !self.renyi_alpha.is_finite() || !(0.0..=20.0).contains(&self.renyi_alpha) {
            return Err("renyi_alpha must be finite and in [0, 20]".into());
        }
        if !(2..=MAX_UI_ALPHABET).contains(&self.weights_p.len())
            || self.weights_p.len() != self.weights_q.len()
        {
            return Err("weights_p and weights_q must have matching length in 2..=6".into());
        }
        if self
            .weights_p
            .iter()
            .chain(self.weights_q.iter())
            .any(|w| !w.is_finite() || *w <= 0.0 || *w > 1e4)
        {
            return Err("weights_p and weights_q entries must be finite in (0, 10000]".into());
        }
        Ok(())
    }
}

pub struct ShannonChannelTool {
    pub(crate) cfg: ShannonToolConfig,
    pub(crate) pmf_p: Vec<f64>,
    pub(crate) pmf_q: Vec<f64>,
    pub(crate) entropy_h: f64,
    pub(crate) renyi_h: f64,
    pub(crate) kl_bits: f64,
    pub(crate) jsd_bits: f64,
    pub(crate) closed_capacity: f64,
    pub(crate) ba_capacity: f64,
    pub(crate) induced_mi: f64,
    pub(crate) bandwidth_bps: f64,
    pub(crate) opt_input: Vec<f64>,
    pub(crate) capacity_curve: Vec<[f64; 2]>,
    ba_ws: BlahutArimotoWorkspace,
    mi_ws: JointMiWorkspace,
    state_rev: u64,
}

impl Default for ShannonChannelTool {
    fn default() -> Self {
        let mut tool = Self {
            cfg: presets::default_config(),
            pmf_p: vec![0.5, 0.5],
            pmf_q: vec![0.5, 0.5],
            entropy_h: 1.0,
            renyi_h: 1.0,
            kl_bits: 0.0,
            jsd_bits: 0.0,
            closed_capacity: 0.0,
            ba_capacity: 0.0,
            induced_mi: 0.0,
            bandwidth_bps: 0.0,
            opt_input: vec![0.5, 0.5],
            capacity_curve: Vec::new(),
            ba_ws: BlahutArimotoWorkspace::new(4, 4),
            mi_ws: JointMiWorkspace::new(4, 4),
            state_rev: 1,
        };
        tool.recompute();
        tool
    }
}

fn normalize_weights(weights: &[f64], out: &mut Vec<f64>) {
    out.resize(weights.len(), 0.0);
    let sum: f64 = weights.iter().copied().sum();
    if sum > 0.0 {
        for (dst, &w) in out.iter_mut().zip(weights.iter()) {
            *dst = w / sum;
        }
    }
}

impl ShannonChannelTool {
    pub(crate) fn recompute(&mut self) {
        normalize_weights(&self.cfg.weights_p, &mut self.pmf_p);
        normalize_weights(&self.cfg.weights_q, &mut self.pmf_q);

        self.entropy_h = categorical_entropy(&self.pmf_p).unwrap_or(0.0);
        self.renyi_h = renyi_entropy(&self.pmf_p, self.cfg.renyi_alpha).unwrap_or(0.0);
        self.kl_bits = kl_divergence(&self.pmf_p, &self.pmf_q).unwrap_or(0.0);
        self.jsd_bits = jensen_shannon_divergence(&self.pmf_p, &self.pmf_q).unwrap_or(0.0);

        self.recompute_channel();
        self.recompute_curve();
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    fn recompute_channel(&mut self) {
        let p = self.cfg.noise_param.clamp(0.0, 0.999);
        match self.cfg.channel_kind {
            ShannonChannelKind::Bsc => {
                let w = [1.0 - p, p, p, 1.0 - p];
                self.closed_capacity = bsc_capacity(p).unwrap_or(0.0);
                self.solve_dmc(&w, 2, 2);
                self.bandwidth_bps = self.cfg.bandwidth_hz * self.closed_capacity;
            }
            ShannonChannelKind::Bec => {
                let w = [1.0 - p, p, 0.0, 0.0, p, 1.0 - p];
                self.closed_capacity = bec_capacity(p).unwrap_or(0.0);
                self.solve_dmc(&w, 2, 3);
                self.bandwidth_bps = self.cfg.bandwidth_hz * self.closed_capacity;
            }
            ShannonChannelKind::ZChannel => {
                let w = [1.0, 0.0, p, 1.0 - p];
                self.solve_dmc(&w, 2, 2);
                self.closed_capacity = self.ba_capacity;
                self.bandwidth_bps = self.cfg.bandwidth_hz * self.ba_capacity;
            }
            ShannonChannelKind::Awgn => {
                self.closed_capacity = awgn_capacity(self.cfg.snr_linear).unwrap_or(0.0);
                self.ba_capacity = self.closed_capacity;
                self.induced_mi = self.closed_capacity;
                self.opt_input = vec![0.5, 0.5];
                self.bandwidth_bps =
                    awgn_bandwidth_capacity(self.cfg.bandwidth_hz, self.cfg.snr_linear)
                        .unwrap_or(0.0);
            }
        }
    }

    fn solve_dmc(&mut self, w: &[f64], n_in: usize, n_out: usize) {
        self.opt_input.resize(n_in, 0.0);
        if let Ok(res) = blahut_arimoto_capacity(
            w,
            n_in,
            n_out,
            1e-9,
            250,
            &mut self.ba_ws,
            &mut self.opt_input,
        ) {
            self.ba_capacity = res.capacity_bits;
            let mut joint = vec![0.0; n_in * n_out];
            for (i, &ri) in self.opt_input.iter().enumerate().take(n_in) {
                for j in 0..n_out {
                    let idx = flatten_2d_index(j, i, n_out);
                    joint[idx] = ri * w[idx];
                }
            }
            self.induced_mi =
                joint_mutual_information(&joint, n_in, n_out, &mut self.mi_ws).unwrap_or(0.0);
        }
    }

    fn recompute_curve(&mut self) {
        self.capacity_curve.clear();
        let samples = 60usize;
        match self.cfg.channel_kind {
            ShannonChannelKind::Awgn => {
                let max_snr = self.cfg.snr_linear.max(20.0);
                for k in 0..=samples {
                    let snr = (k as f64) * max_snr / (samples as f64);
                    let cap = awgn_capacity(snr).unwrap_or(0.0);
                    self.capacity_curve.push([snr, cap]);
                }
            }
            kind => {
                let mut local_ws = BlahutArimotoWorkspace::new(2, 2);
                let mut local_opt = [0.0; 2];
                for k in 0..=samples {
                    let q = (k as f64) * 0.99 / (samples as f64);
                    let cap = match kind {
                        ShannonChannelKind::Bsc => bsc_capacity(q).unwrap_or(0.0),
                        ShannonChannelKind::Bec => bec_capacity(q).unwrap_or(0.0),
                        ShannonChannelKind::ZChannel => {
                            let w = [1.0, 0.0, q, 1.0 - q];
                            blahut_arimoto_capacity(
                                &w,
                                2,
                                2,
                                1e-7,
                                150,
                                &mut local_ws,
                                &mut local_opt,
                            )
                            .map_or(0.0, |r| r.capacity_bits)
                        }
                        ShannonChannelKind::Awgn => 0.0,
                    };
                    self.capacity_curve.push([q, cap]);
                }
            }
        }
    }

    pub fn resize_alphabet(&mut self, new_len: usize) {
        let n = new_len.clamp(2, MAX_UI_ALPHABET);
        if n == self.cfg.weights_p.len() {
            return;
        }
        self.cfg.weights_p.resize(n, 1.0);
        self.cfg.weights_q.resize(n, 1.0);
        self.recompute();
    }
}

impl InteractiveTool for ShannonChannelTool {
    fn name(&self) -> &'static str {
        "Shannon Entropy & Channels"
    }

    fn domain(&self) -> &'static str {
        "algorithmic_information"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(2048, 3, 96)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("entropy_bits", self.entropy_h),
            ("renyi_bits", self.renyi_h),
            ("kl_divergence_bits", self.kl_bits),
            ("jsd_bits", self.jsd_bits),
            ("channel_capacity_bits", self.ba_capacity),
            ("mutual_information_bits", self.induced_mi),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: ShannonToolConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid ShannonChannelTool state: {e}"))?;
        cfg.validate()?;
        self.cfg = cfg;
        self.recompute();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Channel Capacity Curve".to_string(),
                self.capacity_curve.clone(),
            ),
            (
                "Source PMF P(x)".to_string(),
                self.pmf_p
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| [i as f64, p])
                    .collect(),
            ),
            (
                "Reference PMF Q(x)".to_string(),
                self.pmf_q
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| [i as f64, q])
                    .collect(),
            ),
        ]
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "shannon_channel_side");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_view(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for ShannonChannelTool {
    fn theory_description(&self) -> String {
        "Evaluates Shannon entropy $H(P) = -\\sum_i p_i \\log_2 p_i$, Rényi entropy $H_\\alpha(P)$, Kullback-Leibler and Jensen-Shannon divergences, joint mutual information $I(X;Y)$, closed-form BSC, BEC, and AWGN Shannon-Hartley capacities, and iterative Blahut-Arimoto capacity $C = \\max_{p(x)} I(X;Y)$.".into()
    }

    fn phonetic_description(&self) -> String {
        "Shannon entropy Renyi divergence mutual information and Blahut Arimoto channel capacity workbench".into()
    }

    fn theory_citation(&self) -> String {
        "A Mathematical Theory of Communication (Shannon, Bell System Technical Journal, 1948); Computation of Channel Capacity and Rate-Distortion Functions (Blahut, IEEE Trans. Info. Theory, 1972)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ShannonChannelTool",
        display_name: "Shannon Entropy & Channels",
        domain: "algorithmic_information",
        tags: &["shannon", "entropy", "kl-divergence", "channel-capacity", "blahut-arimoto", "awgn"],
        build: || Box::new(ShannonChannelTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "shannon_asymmetric_z_channel",
        title: "Asymmetric Z-Channel Blahut-Arimoto",
        description: "W(0|1) = 0.25, W(0|0) = 1: Non-uniform optimal input p*(1) < 0.5 maximizes I(X;Y).",
        category: "Information Theory",
        icon: "📡",
        domain: "algorithmic_information",
        tool_name: Some("ShannonChannelTool"),
        preset_generator: || presets::PRESET_Z_CHANNEL_ASYMMETRIC,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "shannon_awgn_hartley_regime",
        title: "Shannon-Hartley AWGN High-SNR Link",
        description: "C = B log₂(1 + S/N): 20 kHz Gaussian channel at SNR = 31 yields 100 kbps capacity.",
        category: "Information Theory",
        icon: "📶",
        domain: "algorithmic_information",
        tool_name: Some("ShannonChannelTool"),
        preset_generator: || presets::PRESET_AWGN_HARTLEY,
    }
}

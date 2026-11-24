//! Takeaway `.csv` and `.json` artifact formatters across HMM, CTMC, and Absorbing DTMC modes
//! for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MarkovSequenceWorkbenchTool;

fn csv_escape_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

impl MarkovSequenceWorkbenchTool {
    /// Formats the decoded Viterbi state sequence and per-step Forward-Backward posteriors as CSV.
    #[must_use]
    pub fn format_hmm_timeline_csv(&self) -> String {
        let mut header = String::from(
            "step,observation_idx,observation_label,viterbi_state_idx,viterbi_state_label",
        );
        for name in &self.config.hmm.state_names {
            header.push_str(&format!(",posterior_{}", csv_escape_field(name)));
        }
        header.push('\n');
        let mut out = header;
        for (t, &obs_idx) in self.config.hmm.observations.iter().enumerate() {
            let vit_idx = self.hmm_out.viterbi_path.get(t).copied().unwrap_or(0);
            let obs_lbl = self
                .config
                .hmm
                .symbol_names
                .get(obs_idx)
                .map_or("?", String::as_str);
            let st_lbl = self
                .config
                .hmm
                .state_names
                .get(vit_idx)
                .map_or("?", String::as_str);
            out.push_str(&format!(
                "{t},{obs_idx},{},{vit_idx},{}",
                csv_escape_field(obs_lbl),
                csv_escape_field(st_lbl)
            ));
            for s in 0..self.config.hmm.state_names.len() {
                let p = self
                    .hmm_out
                    .posteriors
                    .get(s)
                    .and_then(|row| row.get(t))
                    .copied()
                    .unwrap_or(0.0);
                out.push_str(&format!(",{p:.6}"));
            }
            out.push('\n');
        }
        out
    }

    /// Formats the fitted `HiddenMarkovModel<f64>` specification as pretty JSON.
    pub fn format_hmm_model_json(&self) -> Result<String, String> {
        let hmm = self.build_hmm()?;
        serde_json::to_string_pretty(&hmm).map_err(|e| e.to_string())
    }

    /// Formats CTMC `P(eval_time)`, steady-state probabilities, and MTTF as CSV.
    #[must_use]
    pub fn format_ctmc_report_csv(&self) -> String {
        let n = self.config.ctmc.state_names.len();
        let mut out = String::from("state_idx,state_name,steady_state,mttf");
        for j in 0..n {
            out.push_str(&format!(",p_eval_to_s{j}"));
        }
        out.push('\n');
        for (i, name) in self.config.ctmc.state_names.iter().enumerate() {
            let pi_str = self
                .ctmc_out
                .steady_state
                .as_ref()
                .and_then(|v| v.get(i))
                .map_or_else(String::new, |v| format!("{v:.6}"));
            let mttf_str = self
                .config
                .ctmc
                .transient_states
                .iter()
                .position(|&t| t == i)
                .and_then(|pos| self.ctmc_out.mttf.as_ref().and_then(|m| m.get(pos)))
                .map_or_else(String::new, |v| format!("{v:.6}"));
            out.push_str(&format!(
                "{i},{},{pi_str},{mttf_str}",
                csv_escape_field(name)
            ));
            for j in 0..n {
                let p = self
                    .ctmc_out
                    .eval_matrix
                    .get(i)
                    .and_then(|r| r.get(j))
                    .copied()
                    .unwrap_or(0.0);
                out.push_str(&format!(",{p:.6}"));
            }
            out.push('\n');
        }
        out
    }

    /// Formats CTMC generator, `P(eval_time)`, steady-state, and MTTF report as pretty JSON.
    pub fn format_ctmc_report_json(&self) -> Result<String, String> {
        let payload = serde_json::json!({
            "state_names": self.config.ctmc.state_names,
            "generator": self.config.ctmc.generator,
            "eval_time": self.config.ctmc.eval_time,
            "transition_probabilities_at_eval": self.ctmc_out.eval_matrix,
            "steady_state": self.ctmc_out.steady_state,
            "transient_states": self.config.ctmc.transient_states,
            "expected_absorption_times_mttf": self.ctmc_out.mttf,
        });
        serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())
    }

    /// Formats Absorbing DTMC EPV, expected absorption times, `B = NR`, and `N = (I - Q)^{-1}` as CSV.
    #[must_use]
    pub fn format_dtmc_report_csv(&self) -> String {
        let num_t = self.dtmc_out.transient_indices.len();
        let num_a = self.dtmc_out.absorbing_indices.len();
        let mut out = String::from("transient_idx,state_name,epv,expected_steps_to_absorption");
        for k in 0..num_a {
            let abs_i = self.dtmc_out.absorbing_indices[k];
            out.push_str(&format!(
                ",absorb_prob_{}",
                csv_escape_field(&self.config.dtmc.state_names[abs_i])
            ));
        }
        for j in 0..num_t {
            let tr_j = self.dtmc_out.transient_indices[j];
            out.push_str(&format!(
                ",fundamental_visits_{}",
                csv_escape_field(&self.config.dtmc.state_names[tr_j])
            ));
        }
        out.push('\n');
        for (row_idx, &state_i) in self.dtmc_out.transient_indices.iter().enumerate() {
            let epv = self.dtmc_out.epv.get(row_idx).copied().unwrap_or(0.0);
            let steps = self
                .dtmc_out
                .expected_steps
                .get(row_idx)
                .copied()
                .unwrap_or(0.0);
            out.push_str(&format!(
                "{state_i},{},{epv:.6},{steps:.6}",
                csv_escape_field(&self.config.dtmc.state_names[state_i])
            ));
            for &b in self.dtmc_out.absorption_probs[row_idx].iter().take(num_a) {
                out.push_str(&format!(",{b:.6}"));
            }
            for &v in self.dtmc_out.fundamental[row_idx].iter().take(num_t) {
                out.push_str(&format!(",{v:.6}"));
            }
            out.push('\n');
        }
        out
    }

    /// Formats Absorbing DTMC fundamental matrix `N`, absorption `B`, steps, and EPV as pretty JSON.
    pub fn format_dtmc_report_json(&self) -> Result<String, String> {
        let payload = serde_json::json!({
            "state_names": self.config.dtmc.state_names,
            "transition_matrix": self.config.dtmc.transition_matrix,
            "state_types": self.config.dtmc.state_types,
            "absorbing_rewards": self.config.dtmc.absorbing_rewards,
            "transient_indices": self.dtmc_out.transient_indices,
            "absorbing_indices": self.dtmc_out.absorbing_indices,
            "fundamental_matrix": self.dtmc_out.fundamental,
            "absorption_probabilities": self.dtmc_out.absorption_probs,
            "expected_absorption_times": self.dtmc_out.expected_steps,
            "expected_possession_value": self.dtmc_out.epv,
        });
        serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())
    }
}

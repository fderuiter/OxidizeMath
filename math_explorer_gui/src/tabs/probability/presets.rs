#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bayesian_tool::ConjugateBayesianTool;
use super::distributions_tool::DistributionsExplorerTool;
use super::factor_bp_tool::FactorBeliefPropagationTool;
use crate::framework::{StarterPresetMetadata, ToolMetadata};

pub const PRESET_BETA_SKEWED: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Distributions Explorer",
    "domain": "probability",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "family": "Gamma",
    "param_a": 4.0,
    "param_b": 1.2,
    "quantile_prob": 0.95,
    "sample_count": 1024,
    "seed": 108,
    "show_cdf": true
  }
}"#;

pub const PRESET_CONJUGATE_CLINICAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conjugate Bayesian Updater",
    "domain": "probability",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "family": "BetaBinomial",
    "prior_a": 3.0,
    "prior_b": 3.0,
    "sample_size": 40.0,
    "observed_stat": 0.80,
    "credible_level": 0.95
  }
}"#;

pub const PRESET_LOOPY_POTTS_BP: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Factor Graph & Belief Propagation",
    "domain": "probability",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "topology": "LoopyCycle4",
    "num_states": 2,
    "coupling": -1.25,
    "root_bias": 1.1,
    "pinned_node": 2,
    "damping": 0.55,
    "max_iters": 80
  }
}"#;

impl scientific_metadata::theory::TheoryDescribable for DistributionsExplorerTool {
    fn theory_description(&self) -> String {
        "Parametric continuous and discrete distributions built atop `#![no_std]` Lanczos `ln_gamma`, Lentz continued-fraction regularized incomplete gamma/beta, Acklam + Halley quantiles (`< 1e-10` relative error), and deterministic `OxidizeRng` sampling.".into()
    }
    fn phonetic_description(&self) -> String {
        "No standard library probability distributions quantiles and Monte Carlo sampling".into()
    }
    fn theory_citation(&self) -> String {
        "Lanczos (1964), A precision approximation of the gamma function, SIAM J. Numer. Anal. 1, 86-96; DiDonato & Morris (1992), Algorithm 708, ACM TOMS 18(3), 360-373".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl scientific_metadata::theory::TheoryDescribable for ConjugateBayesianTool {
    fn theory_description(&self) -> String {
        "Closed-form zero-allocation exponential-family conjugate updating (Beta-Binomial, Gamma-Poisson, Normal-Normal, Normal-Inverse-Gamma) yielding exact posterior densities, equal-tailed credible intervals, marginal likelihoods, and analytical Bayes factors.".into()
    }
    fn phonetic_description(&self) -> String {
        "Conjugate Bayesian posterior credible intervals and Bayes factors".into()
    }
    fn theory_citation(&self) -> String {
        "Gelman et al. (2013), Bayesian Data Analysis (3rd ed.), Chapman & Hall/CRC; Kass & Raftery (1995), Bayes Factors, JASA 90(430), 773-795".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl scientific_metadata::theory::TheoryDescribable for FactorBeliefPropagationTool {
    fn theory_description(&self) -> String {
        "Zero-allocation log-domain sum-product and max-product belief propagation on factor graphs: computes exact marginal beliefs and Bethe log-partition $\\ln Z = -F_{\\text{Bethe}}$ on trees, damped fixed points on loopy Potts cycles, and deterministic MAP configurations.".into()
    }
    fn phonetic_description(&self) -> String {
        "Factor graph sum product belief propagation Bethe free energy and max product MAP decoding"
            .into()
    }
    fn theory_citation(&self) -> String {
        "Kschischang, Frey & Loeliger (2001), Factor graphs and the sum-product algorithm, IEEE Trans. Inf. Theory 47(2), 498-519; Yedidia, Freeman & Weiss (2005), Constructing free-energy approximations and generalized belief propagation algorithms, IEEE Trans. Inf. Theory 51(7), 2282-2312".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "DistributionsExplorerTool",
        display_name: "Distributions Explorer",
        domain: "probability",
        tags: &["probability", "distributions", "quantile", "gamma", "beta"],
        build: || Box::new(DistributionsExplorerTool::default()),
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "prob_beta_skewed_starter",
        title: "Gamma Waiting-Time & 95% Quantile",
        description: "Explore exact Lentz incomplete-gamma CDF, Halley 95% quantile, and deterministic Marsaglia-Tsang sampling.",
        category: "Probability Workbench",
        icon: "📊",
        domain: "probability",
        tool_name: Some("DistributionsExplorerTool"),
        preset_generator: || PRESET_BETA_SKEWED,
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ConjugateBayesianTool",
        display_name: "Conjugate Bayesian Updater",
        domain: "probability",
        tags: &["bayesian", "conjugate", "posterior", "credible_interval", "bayes_factor"],
        build: || Box::new(ConjugateBayesianTool::default()),
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "prob_conjugate_beta_binomial_starter",
        title: "Clinical Responder Beta-Binomial Posterior",
        description: "Update a weakly informative Beta(3, 3) prior with 32/40 clinical responders and inspect exact 95% credible intervals and BF10.",
        category: "Probability Workbench",
        icon: "🧬",
        domain: "probability",
        tool_name: Some("ConjugateBayesianTool"),
        preset_generator: || PRESET_CONJUGATE_CLINICAL,
    }
}

inventory::submit! {
    ToolMetadata {
        id: "FactorBeliefPropagationTool",
        display_name: "Factor Graph & Belief Propagation",
        domain: "probability",
        tags: &["belief_propagation", "factor_graph", "sum_product", "max_product", "bethe"],
        build: || Box::new(FactorBeliefPropagationTool::default()),
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "prob_factor_bp_tree_starter",
        title: "Frustrated 2x2 Antiferromagnetic Cycle BP",
        description: "Run damped log-domain sum-product and max-product belief propagation on a loopy 4-variable cycle with repulsive coupling.",
        category: "Probability Workbench",
        icon: "🕸",
        domain: "probability",
        tool_name: Some("FactorBeliefPropagationTool"),
        preset_generator: || PRESET_LOOPY_POTTS_BP,
    }
}

// [cite:probability]

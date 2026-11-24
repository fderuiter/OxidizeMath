#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Federated `ToolMetadata` and `StarterPresetMetadata` registrations for Probability & Statistics.

use super::bayesian_tool::BayesianInferenceTool;
use super::copula_tool::CopulaDependenceTool;
use super::distribution_tool::DistributionExplorerTool;

/// Starter preset: Heavy-tailed Gamma survival distribution with digamma MLE vs MoM comparison.
pub const PRESET_DIST_GAMMA_SURVIVAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Distribution Explorer - Gamma Survival & Digamma MLE",
    "domain": "probability_statistics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "quantile_p": 0.75,
    "param_a": 3.5,
    "param_b": 1.4,
    "sample_size": 240,
    "seed": 77,
    "family": "Gamma"
  }
}"#;

/// Starter preset: Overdispersed Zero-Inflated Poisson count distribution ("Median Attack").
pub const PRESET_DIST_ZIP_OVERDISPERSED: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Distribution Explorer - Zero-Inflated Poisson Counts",
    "domain": "probability_statistics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "quantile_p": 0.50,
    "param_a": 0.62,
    "param_b": 3.8,
    "sample_size": 220,
    "seed": 19,
    "family": "ZeroInflatedPoisson"
  }
}"#;

/// Starter preset: Skewed Beta-Binomial clinical responder posterior with HPD vs Equal-Tailed intervals.
pub const PRESET_BAYES_CLINICAL_RESPONDER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bayesian Inference Lab - Clinical Trial Beta-Binomial HPD",
    "domain": "probability_statistics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "proposal_scale": 0.11,
    "credible_level": 0.90,
    "prior_a": 1.0,
    "prior_b": 1.0,
    "obs_stat": 19.0,
    "obs_count": 24,
    "steps_per_frame": 32,
    "warmup_draws": 480,
    "seed": 314,
    "family": "BetaBinomial"
  }
}"#;

/// Starter preset: Unknown-variance Normal-Inverse-Gamma Student-t marginal posterior + MCMC.
pub const PRESET_BAYES_NIG_STUDENT_T: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bayesian Inference Lab - Normal-Inverse-Gamma Marginal t",
    "domain": "probability_statistics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "proposal_scale": 0.36,
    "credible_level": 0.95,
    "prior_a": 0.0,
    "prior_b": 2.5,
    "obs_stat": 1.85,
    "obs_count": 18,
    "steps_per_frame": 28,
    "warmup_draws": 400,
    "seed": 271,
    "family": "NormalInverseGamma"
  }
}"#;

/// Starter preset: 4D Positive-Correlated Gaussian Copula via Genz QMC.
pub const PRESET_COPULA_4D_QMC: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Multivariate Copula Lab - 4D Genz QMC Positive Tail",
    "domain": "probability_statistics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "correlation_rho": 0.68,
    "marginal_u1": 0.80,
    "marginal_u2": 0.75,
    "zip_zero_prob": 0.42,
    "dimension": 4,
    "qmc_samples": 512,
    "seed": 55
  }
}"#;

/// Starter preset: Bivariate Negative-Correlation "Hero Ball" Same-Game Parlay Pricing.
pub const PRESET_COPULA_HERO_BALL_SGP: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Multivariate Copula Lab - Negative Correlation SGP",
    "domain": "probability_statistics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "correlation_rho": -0.25,
    "marginal_u1": 0.90,
    "marginal_u2": 0.60,
    "zip_zero_prob": 0.55,
    "dimension": 2,
    "qmc_samples": 256,
    "seed": 88
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "DistributionExplorerTool",
        display_name: "Distribution Explorer",
        domain: "probability_statistics",
        tags: &["distributions", "mle", "mom", "quantiles", "weibull", "gamma", "poisson", "zip"],
        build: || Box::new(DistributionExplorerTool::default()),
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BayesianInferenceTool",
        display_name: "Bayesian Inference Lab",
        domain: "probability_statistics",
        tags: &["bayesian", "conjugate", "mcmc", "metropolis-hastings", "hpd", "bayes-factor"],
        build: || Box::new(BayesianInferenceTool::default()),
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CopulaDependenceTool",
        display_name: "Multivariate Copula Lab",
        domain: "probability_statistics",
        tags: &["copula", "qmc", "cholesky", "sgp", "frechet-hoeffding", "zip-regression"],
        build: || Box::new(CopulaDependenceTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prob_stat_gamma_mle",
        title: "Gamma Survival & Digamma MLE",
        description: "Exact Gamma PDF/CDF quantiles with seeded Newton-digamma MLE vs Method-of-Moments recovery.",
        category: "Probability & Statistics",
        icon: "📈",
        domain: "probability_statistics",
        tool_name: Some("DistributionExplorerTool"),
        preset_generator: || PRESET_DIST_GAMMA_SURVIVAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prob_stat_zip_counts",
        title: "Zero-Inflated Poisson Counts",
        description: "High structural zero-inflation (rho = 0.62) illustrating median-vs-mean skew and unclamped fitting.",
        category: "Probability & Statistics",
        icon: "🎯",
        domain: "probability_statistics",
        tool_name: Some("DistributionExplorerTool"),
        preset_generator: || PRESET_DIST_ZIP_OVERDISPERSED,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prob_stat_bayes_responder",
        title: "Clinical Trial Beta-Binomial HPD",
        description: "Exact Beta(20, 6) responder posterior comparing shortest HPD against equal-tailed intervals and MCMC.",
        category: "Probability & Statistics",
        icon: "🧬",
        domain: "probability_statistics",
        tool_name: Some("BayesianInferenceTool"),
        preset_generator: || PRESET_BAYES_CLINICAL_RESPONDER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prob_stat_bayes_nig",
        title: "Normal-Inverse-Gamma Student-t",
        description: "Joint unknown-(mu, sigma^2) Bayesian update with exact marginal Student-t posterior and Bayes factor.",
        category: "Probability & Statistics",
        icon: "⚖",
        domain: "probability_statistics",
        tool_name: Some("BayesianInferenceTool"),
        preset_generator: || PRESET_BAYES_NIG_STUDENT_T,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prob_stat_copula_4d_qmc",
        title: "4D Gaussian Copula Genz QMC",
        description: "Four-dimensional positive-definite copula joint tail probability bounded by Frechet-Hoeffding envelopes.",
        category: "Probability & Statistics",
        icon: "🔗",
        domain: "probability_statistics",
        tool_name: Some("CopulaDependenceTool"),
        preset_generator: || PRESET_COPULA_4D_QMC,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prob_stat_copula_sgp",
        title: "Negative-Correlation SGP & ZIP EM",
        description: "Bivariate Plackett quadrature under negative correlation alongside EM covariate ZIP count regression.",
        category: "Probability & Statistics",
        icon: "🎲",
        domain: "probability_statistics",
        tool_name: Some("CopulaDependenceTool"),
        preset_generator: || PRESET_COPULA_HERO_BALL_SGP,
    }
}

// [cite:clinical_trials_statistics]

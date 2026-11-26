//! Multi-step guided story tours (`PresetStoryMetadata`) and live `math_commons::ast`
//! (`CompiledExpr`) playable challenges (`GoalChallengeMetadata`) for all four paradox labs.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(any(feature = "pure_math", feature = "applied"))]
use super::{GoalChallengeMetadata, PresetStoryMetadata, StoryStep};
#[cfg(any(feature = "pure_math", feature = "applied"))]
use crate::framework::CalloutSpec;

#[cfg(any(feature = "pure_math", feature = "applied"))]
const fn pin(
    tag: &'static str,
    target_point: [f64; 2],
    label_offset: [f32; 2],
    text: &'static str,
) -> CalloutSpec {
    CalloutSpec {
        tag,
        target_point,
        label_offset,
        text,
    }
}

#[cfg(feature = "pure_math")]
mod parrondo_kelly_packs {
    use super::*;

    const KELLY_DEEP_LOSS: &str = r#"{"metadata":{"schema_version":"1.0","title":"Deep Bias Collapse","domain":"financial_math","timestamp":"2026-11-23T00:00:00Z"},"config":{"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.038,"modulus":3,"schedule":"ABB","num_walkers":512,"num_steps":240,"seed":42,"view_mode":"SplitOverview"}}"#;
    const KELLY_WEDGE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Harmer-Abbott Paradox Wedge","domain":"financial_math","timestamp":"2026-11-23T00:00:00Z"},"config":{"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.005,"modulus":3,"schedule":"ABB","num_walkers":512,"num_steps":240,"seed":42,"view_mode":"SplitOverview"}}"#;
    const KELLY_AABB: &str = r#"{"metadata":{"schema_version":"1.0","title":"AABB Ratchet Schedule","domain":"financial_math","timestamp":"2026-11-23T00:00:00Z"},"config":{"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.004,"modulus":3,"schedule":"AABB","num_walkers":512,"num_steps":240,"seed":42,"view_mode":"SplitOverview"}}"#;

    const KELLY_RATCHET_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Excessive Bias Crushes Every Schedule",
            narrative: "At bias ε = 0.038, standalone Game A (E[ΔW_A] = -0.076) and Game B (E[ΔW_B] < -0.06) drag down even the ABB ratchet: every Kelly badge reports NoEdge.",
            preset_generator: || KELLY_DEEP_LOSS,
            callouts: &[pin(
                "parrondo_kelly_step",
                [180.0, -9.0],
                [-24.0, 22.0],
                "ε = 0.038: ABB ratchet overwhelmed",
            )],
            focus_observables: &["gain_a", "gain_b", "gain_mix", "gain_schedule"],
        },
        StoryStep {
            title: "Entering the Harmer-Abbott Paradox Wedge",
            narrative: "Lowering bias to ε = 0.005 keeps Game A (-0.010) and Game B (-0.0087) individually losing, yet Game A agitates walkers out of the W mod 3 = 0 trap so ABB compounds at +0.0176/step!",
            preset_generator: || KELLY_WEDGE,
            callouts: &[pin(
                "parrondo_kelly_step",
                [180.0, 3.2],
                [-28.0, -24.0],
                "ABB synthesizes Kelly f* > 0 from two NoEdge games",
            )],
            focus_observables: &["gain_a", "gain_b", "gain_mix", "gain_schedule"],
        },
        StoryStep {
            title: "Comparing Periodic Schedules (AABB vs ABB)",
            narrative: "Switching to AABB at ε = 0.004 reshapes stationary modulo-3 occupancy π_sched, demonstrating how non-commuting Markov operators P_A and P_B rectify asymmetric traps.",
            preset_generator: || KELLY_AABB,
            callouts: &[pin(
                "parrondo_kelly_step",
                [180.0, 3.6],
                [-28.0, -24.0],
                "Broken detailed balance rectifies Z/3Z trap",
            )],
            focus_observables: &["gain_a", "gain_b", "gain_mix", "gain_schedule"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "parrondo_kelly_ratchet_tour",
            title: "Parrondo's Kelly Ratchet Tour",
            description: "Discover how alternating two Kelly-NoEdge games breaks detailed balance on Z/3Z to synthesize positive stationary drift.",
            category: "Guided Story",
            icon: "📖",
            domain: "financial_math",
            tool_name: "ParrondoParadoxTool",
            steps: KELLY_RATCHET_STEPS,
        }
    }

    inventory::submit! {
        GoalChallengeMetadata {
            id: "parrondo_kelly_wedge_hunt",
            title: "Unlock the Parrondo Kelly Wedge",
            description: "Keep standalone Game A and Game B strictly losing while tuning into the modulo-M paradox wedge where both random mixing and periodic scheduling win.",
            category: "Challenge",
            icon: "🎯",
            domain: "financial_math",
            tool_name: "ParrondoParadoxTool",
            initial_preset_generator: || KELLY_DEEP_LOSS,
            goal_summary: "Keep Game A and Game B losing (gain_a < -0.002, gain_b < -0.002) while achieving gain_mix > 0.008 and gain_schedule > 0.012.",
            predicate_expr: "gain_a < -0.002 && gain_b < -0.002 && gain_mix > 0.008 && gain_schedule > 0.012",
            hint: "Open the 2D Regime Atlas on (p_a_base, epsilon) and teleport into the 'Full Paradox Wedge' near p_A = 0.50, ε ≈ 0.005.",
            win_message: "Challenge solved! Standalone Game A and Game B remain losing (Kelly NoEdge), yet broken detailed balance yields positive ratchet growth.",
            callouts: &[pin(
                "parrondo_kelly_challenge",
                [160.0, 0.0],
                [16.0, -22.0],
                "Break-even threshold E[ΔW] = 0",
            )],
            focus_observables: &["gain_a", "gain_b", "gain_mix", "gain_schedule"],
        }
    }
}

#[cfg(feature = "applied")]
mod parrondo_jump_packs {
    use super::*;

    const JUMP_UNSOLVED: &str = r#"{"metadata":{"schema_version":"1.0","title":"Over-Biased Jump Chain","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"bias_epsilon":0.038,"mix_gamma":0.05,"jump_switch_rate":1.2,"variant":"CapitalDependent","focus_schedule":"PeriodicABB","steps":120,"walkers":256,"seed":42}}"#;
    const JUMP_CAPITAL_WEDGE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Capital-Dependent Ratchet","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"bias_epsilon":0.005,"mix_gamma":0.5,"jump_switch_rate":1.2,"variant":"CapitalDependent","focus_schedule":"PeriodicABB","steps":120,"walkers":256,"seed":42}}"#;
    const JUMP_HISTORY_WEDGE: &str = r#"{"metadata":{"schema_version":"1.0","title":"History-Dependent Parrondo Chain","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"bias_epsilon":0.004,"mix_gamma":0.5,"jump_switch_rate":2.0,"variant":"HistoryDependent","focus_schedule":"RandomMix","steps":120,"walkers":256,"seed":42}}"#;

    const JUMP_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Starved Mixture & Heavy Bias",
            narrative: "With ε = 0.038 and mixture weight γ = 0.05, walkers rarely invoke Game A to escape bad modulo-3 states: discrete chains and Gillespie jumps drift downward.",
            preset_generator: || JUMP_UNSOLVED,
            callouts: &[pin(
                "parrondo_jump_step",
                [90.0, -4.5],
                [-24.0, 20.0],
                "ε = 0.038, γ = 0.05: all schedules lose",
            )],
            focus_observables: &["drift_a", "drift_b", "drift_abb", "drift_mix"],
        },
        StoryStep {
            title: "Capital-Dependent ABB & Gillespie SSA",
            narrative: "At ε = 0.005 and γ = 0.50, Pure A (-0.010) and Pure B (-0.0087) still lose, while Periodic ABB (+0.0176) and continuous-time Poisson switching climb steadily.",
            preset_generator: || JUMP_CAPITAL_WEDGE,
            callouts: &[pin(
                "parrondo_jump_step",
                [90.0, 1.6],
                [-26.0, -24.0],
                "ABB & SSA rectify modulo-3 trap",
            )],
            focus_observables: &["drift_a", "drift_b", "drift_abb", "drift_mix"],
        },
        StoryStep {
            title: "History-Dependent 4-State Parrondo Chain",
            narrative: "Switching to HistoryDependent replaces modulo-3 capital traps with 2-step memory states {LL, LW, WL, WW}, proving memory coupling alone suffices for Parrondo's paradox.",
            preset_generator: || JUMP_HISTORY_WEDGE,
            callouts: &[pin(
                "parrondo_jump_step",
                [90.0, 1.4],
                [-26.0, -24.0],
                "2-step memory {LL,LW,WL,WW} ratchet",
            )],
            focus_observables: &["drift_a", "drift_b", "drift_abb", "drift_mix"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "parrondo_jump_ratchet_tour",
            title: "Parrondo's Jump & Memory Ratchet Tour",
            description: "Compare capital-dependent, history-dependent, and continuous-time Gillespie SSA switching ratchets.",
            category: "Guided Story",
            icon: "📖",
            domain: "paradox_lab",
            tool_name: "ParrondoJumpRatchetTool",
            steps: JUMP_STEPS,
        }
    }

    inventory::submit! {
        GoalChallengeMetadata {
            id: "parrondo_jump_ratchet_challenge",
            title: "Tune the Stochastic Jump Ratchet",
            description: "Keep Pure Game A and Pure Game B strictly losing while tuning bias ε and mixing weight γ so both Periodic ABB and Random Mix drift upward.",
            category: "Challenge",
            icon: "🎯",
            domain: "paradox_lab",
            tool_name: "ParrondoJumpRatchetTool",
            initial_preset_generator: || JUMP_UNSOLVED,
            goal_summary: "Keep Pure A and Pure B losing (drift_a < -0.002, drift_b < -0.002) while raising drift_abb > 0.012 and drift_mix > 0.010.",
            predicate_expr: "drift_a < -0.002 && drift_b < -0.002 && drift_abb > 0.012 && drift_mix > 0.010",
            hint: "Open the 2D Regime Atlas on (bias_epsilon, mix_gamma) or set ε ≈ 0.005 and γ ≈ 0.50.",
            win_message: "Challenge solved! Both discrete periodic ABB and stochastic γ-mixing synthesize positive drift out of two losing Markov chains.",
            callouts: &[pin(
                "parrondo_jump_challenge",
                [80.0, 0.0],
                [16.0, -20.0],
                "Zero-drift break-even line",
            )],
            focus_observables: &["drift_a", "drift_b", "drift_abb", "drift_mix"],
        }
    }
}

#[cfg(feature = "applied")]
mod braess_and_simpson_packs {
    use super::*;

    const BRAESS_CLOSED: &str = r#"{"metadata":{"schema_version":"1.0","title":"Parallel Highways (Closed Shortcut)","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"demand":1.0,"congestion_slope":1.0,"bypass_latency":1.0,"shortcut_cost":0.0,"shortcut_toll":0.0,"shortcut_open":false,"continuum_mfg_mode":false,"mfg_viscosity":0.08,"mfg_grid_points":33}}"#;
    const BRAESS_GRIDLOCK: &str = r#"{"metadata":{"schema_version":"1.0","title":"Selfish Shortcut Gridlock","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"demand":1.0,"congestion_slope":1.0,"bypass_latency":1.0,"shortcut_cost":0.0,"shortcut_toll":0.0,"shortcut_open":true,"continuum_mfg_mode":false,"mfg_viscosity":0.08,"mfg_grid_points":33}}"#;
    const BRAESS_PIGOU: &str = r#"{"metadata":{"schema_version":"1.0","title":"Pigouvian Externality Pricing","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"demand":0.8,"congestion_slope":1.0,"bypass_latency":1.0,"shortcut_cost":0.0,"shortcut_toll":0.5,"shortcut_open":true,"continuum_mfg_mode":false,"mfg_viscosity":0.08,"mfg_grid_points":33}}"#;
    const BRAESS_CHALLENGE_INIT: &str = r#"{"metadata":{"schema_version":"1.0","title":"Untolled Wheatstone Slowdown","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"demand":0.8,"congestion_slope":1.0,"bypass_latency":1.0,"shortcut_cost":0.0,"shortcut_toll":0.0,"shortcut_open":true,"continuum_mfg_mode":false,"mfg_viscosity":0.08,"mfg_grid_points":33}}"#;

    const BRAESS_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Symmetric Parallel Split (Shortcut Closed)",
            narrative: "When central link A→B is closed at demand D = 1.0, commuters split 50/50 across upper and lower routes for an average latency of 1.50 (PoA = 1.00).",
            preset_generator: || BRAESS_CLOSED,
            callouts: &[pin(
                "braess_step",
                [1.0, 1.5],
                [18.0, -24.0],
                "Closed A→B: L = 1.50 (PoA = 1.00)",
            )],
            focus_observables: &["wardrop_latency", "closed_latency", "pigou_latency", "price_of_anarchy"],
        },
        StoryStep {
            title: "Opening Zero-Cost Shortcut Triggers Braess Slowdown",
            narrative: "Opening zero-cost link A→B makes zig-zag S→A→B→T a dominant strategy: 100% of drivers pile onto congested links S→A and B→T, raising everyone's travel time from 1.50 to 2.00 (PoA = 4/3)!",
            preset_generator: || BRAESS_GRIDLOCK,
            callouts: &[pin(
                "braess_step",
                [1.0, 2.0],
                [-36.0, -24.0],
                "Braess Paradox: L_Wardrop = 2.00 > 1.50!",
            )],
            focus_observables: &["wardrop_latency", "closed_latency", "pigou_latency", "price_of_anarchy"],
        },
        StoryStep {
            title: "Pigouvian Marginal-Externality Toll Restores Optimum",
            narrative: "At D = 0.8, charging Pigouvian toll τ* = 0.50 internalizes congestion externalities so Wardrop equilibrium matches the Pigou social optimum (L = 1.375 < L_closed = 1.40, PoA = 1.00).",
            preset_generator: || BRAESS_PIGOU,
            callouts: &[pin(
                "braess_step",
                [0.8, 1.375],
                [18.0, -24.0],
                "Pigou toll τ* = 0.50 achieves L = 1.375 < L_closed",
            )],
            focus_observables: &["wardrop_latency", "closed_latency", "pigou_latency", "price_of_anarchy"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "braess_wheatstone_paradox_tour",
            title: "Braess's Wheatstone Congestion Tour",
            description: "Why adding a zero-cost road slows down every commuter—and how Pigouvian congestion pricing eliminates the Price of Anarchy.",
            category: "Guided Story",
            icon: "📖",
            domain: "paradox_lab",
            tool_name: "BraessNetworkParadoxTool",
            steps: BRAESS_STEPS,
        }
    }

    inventory::submit! {
        GoalChallengeMetadata {
            id: "braess_pigou_pricing_challenge",
            title: "Eliminate the Price of Anarchy",
            description: "At demand D = 0.8, untolled selfish routing suffers L_Wardrop = 1.60 > L_closed = 1.40. Price the shortcut to match the Pigou social optimum (1.375)!",
            category: "Challenge",
            icon: "🎯",
            domain: "paradox_lab",
            tool_name: "BraessNetworkParadoxTool",
            initial_preset_generator: || BRAESS_CHALLENGE_INIT,
            goal_summary: "Beat closed-road latency (wardrop_latency < closed_latency - 0.015) and drive Price of Anarchy to social optimality (price_of_anarchy <= 1.008).",
            predicate_expr: "wardrop_latency < closed_latency - 0.015 && wardrop_latency <= pigou_latency + 0.01 && price_of_anarchy <= 1.008",
            hint: "Merely closing the shortcut only reaches L = 1.40 (PoA = 1.018). Keep A→B open and tune Shortcut Toll τ (or use the (demand, shortcut_toll) 2D Atlas) until selfish routing internalizes marginal congestion!",
            win_message: "Challenge solved! Your Pigouvian toll internalized congestion externalities, beating closed-road travel time and driving PoA to 1.00!",
            callouts: &[pin(
                "braess_challenge",
                [0.8, 1.375],
                [18.0, -24.0],
                "Target Pigou optimum L* = 1.375 at D = 0.8",
            )],
            focus_observables: &["wardrop_latency", "closed_latency", "pigou_latency", "price_of_anarchy"],
        }
    }

    const SIMPSON_CONFOUNDED: &str = r#"{"metadata":{"schema_version":"1.0","title":"Confounded Observational Trial","domain":"clinical_trials","timestamp":"2026-11-23T00:00:00Z"},"config":{"randomization_type":"Stratified","n_subjects":24,"block_size":4}}"#;
    const SIMPSON_WASHOUT: &str = r#"{"metadata":{"schema_version":"1.0","title":"Confounded Null Washout","domain":"clinical_trials","timestamp":"2026-11-23T00:00:00Z"},"config":{"randomization_type":"Stratified","n_subjects":24,"block_size":4,"simpson_config":{"mild_treatment_share":0.33,"severe_treatment_share":0.67,"stratum_size":400,"mild_counts":{"treat_event":13,"treat_no_event":119,"ctrl_event":40,"ctrl_no_event":228},"severe_counts":{"treat_event":80,"treat_no_event":188,"ctrl_event":53,"ctrl_no_event":79}}}}"#;
    const SIMPSON_BALANCED: &str = r#"{"metadata":{"schema_version":"1.0","title":"Balanced Stratified Trial","domain":"clinical_trials","timestamp":"2026-11-23T00:00:00Z"},"config":{"randomization_type":"Stratified","n_subjects":24,"block_size":4,"simpson_config":{"mild_treatment_share":0.5,"severe_treatment_share":0.5,"stratum_size":400,"mild_counts":{"treat_event":20,"treat_no_event":180,"ctrl_event":30,"ctrl_no_event":170},"severe_counts":{"treat_event":60,"treat_no_event":140,"ctrl_event":80,"ctrl_no_event":120}}}}"#;

    const SIMPSON_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Severity Confounding Flips Pooled Relative Risk",
            narrative: "Treatment lowers adverse events in Mild (10% vs 15%, RR₁ = 0.67) AND Severe (30% vs 40%, RR₂ = 0.75) patients—yet routing 80% of Severe cases to Treatment inflates Pooled RR = 1.22!",
            preset_generator: || SIMPSON_CONFOUNDED,
            callouts: &[pin(
                "simpson_step",
                [1.22, 1.0],
                [18.0, 18.0],
                "Confounded 20%/80% split: Pooled RR = 1.22 > 1",
            )],
            focus_observables: &["mild_rr", "severe_rr", "pooled_rr", "simpson_reversal"],
        },
        StoryStep {
            title: "Moderate Imbalance Masks Protective Signal",
            narrative: "At a 33% Mild / 67% Severe allocation split, severity confounding cancels out the true drug benefit, producing a false null Pooled RR ≈ 0.99.",
            preset_generator: || SIMPSON_WASHOUT,
            callouts: &[pin(
                "simpson_step",
                [0.99, 1.0],
                [18.0, 18.0],
                "33%/67% split washes out efficacy (RR ≈ 1.0)",
            )],
            focus_observables: &["mild_rr", "severe_rr", "pooled_rr", "simpson_reversal"],
        },
        StoryStep {
            title: "Equalized Stratum Shares Restore True Causal Effect",
            narrative: "Balancing Mild and Severe treatment shares at 50%/50% orthogonalizes treatment assignment from baseline severity, aligning Pooled RR = 0.73 with within-stratum efficacy.",
            preset_generator: || SIMPSON_BALANCED,
            callouts: &[pin(
                "simpson_step",
                [0.73, 1.0],
                [18.0, 18.0],
                "Balanced 50%/50% arms: Pooled RR = 0.73 < 1",
            )],
            focus_observables: &["mild_rr", "severe_rr", "pooled_rr", "simpson_reversal"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "simpson_confounding_reversal_tour",
            title: "Simpson's Paradox Clinical Confounding Tour",
            description: "See how unequal severity allocation flips a protective drug into an apparent hazard—and how stratified balancing restores causal validity.",
            category: "Guided Story",
            icon: "📖",
            domain: "clinical_trials",
            tool_name: "RandomizationTool",
            steps: SIMPSON_STEPS,
        }
    }

    inventory::submit! {
        GoalChallengeMetadata {
            id: "simpson_deconfounding_challenge",
            title: "Resolve Simpson's Clinical Reversal",
            description: "Eliminate severity confounding by tuning Mild and Severe treatment shares until Pooled RR aligns with the protective within-stratum effect.",
            category: "Challenge",
            icon: "🎯",
            domain: "clinical_trials",
            tool_name: "RandomizationTool",
            initial_preset_generator: || SIMPSON_CONFOUNDED,
            goal_summary: "Preserve protective stratum risks (mild_rr < 0.85, severe_rr < 0.85) while lowering pooled_rr < 0.80 and clearing simpson_reversal < 0.5.",
            predicate_expr: "mild_rr < 0.85 && severe_rr < 0.85 && pooled_rr < 0.80 && simpson_reversal < 0.5",
            hint: "Adjust the Mild and Severe Treatment Share sliders toward equal allocation (or click the 'Balanced Protective' diagonal in the 2D Regime Atlas).",
            win_message: "Challenge solved! Balancing treatment shares across prognostic strata eliminated Simpson's Paradox and revealed Pooled RR < 0.80.",
            callouts: &[pin(
                "simpson_challenge",
                [1.0, 1.0],
                [16.0, 18.0],
                "Null threshold RR = 1.0",
            )],
            focus_observables: &["mild_rr", "severe_rr", "pooled_rr", "simpson_reversal"],
        }
    }
}

// [cite:paradox_lab]

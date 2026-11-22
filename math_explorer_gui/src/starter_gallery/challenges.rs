//! Declarative metadata and federated `inventory` registries for multi-step guided stories
//! and playable AST target challenges across `SimulationFramework` tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{StarterCard, StarterPresetMetadata};
use crate::framework::CalloutSpec;

/// Filter category tabs displayed inside the Starter Template Gallery modal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StarterGalleryTab {
    /// Single-step pre-configured parameter snapshots.
    #[default]
    Snapshots,
    /// Multi-step narrative tours with synchronized plot callouts.
    GuidedStories,
    /// Playable live observable challenges evaluated via `math_commons::ast`.
    Challenges,
}

/// Single step in a guided mathematical paradox tour.
#[derive(Clone, Copy, Debug)]
pub struct StoryStep {
    /// Short headline for this tour step.
    pub title: &'static str,
    /// Narrative explanation shown in the HUD banner.
    pub narrative: &'static str,
    /// Generator returning a valid `PresetWrapper` JSON string.
    pub preset_generator: fn() -> &'static str,
    /// Plot callout pins synchronized when entering this step.
    pub callouts: &'static [CalloutSpec],
    /// Observable names highlighted in the HUD readout strip.
    pub focus_observables: &'static [&'static str],
}

/// Federated guided story registration metadata submitted across flagship tools.
#[derive(Clone, Copy, Debug)]
pub struct PresetStoryMetadata {
    /// Unique URL-safe story identifier (`&preset=<id>`).
    pub id: &'static str,
    /// User-facing story card title.
    pub title: &'static str,
    /// Concise summary shown in `StarterGallery`.
    pub description: &'static str,
    /// Thematic category badge.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain matching `InteractiveTool::domain`.
    pub domain: &'static str,
    /// Canonical `ToolMetadata::id` for the target tool.
    pub tool_name: &'static str,
    /// Ordered sequence of narrative parameter steps.
    pub steps: &'static [StoryStep],
}

inventory::collect!(PresetStoryMetadata);

/// Federated playable AST challenge metadata submitted across flagship tools.
#[derive(Clone, Copy, Debug)]
pub struct ChallengeMetadata {
    /// Unique URL-safe challenge identifier (`&challenge=<id>`).
    pub id: &'static str,
    /// User-facing challenge card title.
    pub title: &'static str,
    /// Concise summary shown in `StarterGallery`.
    pub description: &'static str,
    /// Thematic category badge.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain matching `InteractiveTool::domain`.
    pub domain: &'static str,
    /// Canonical `ToolMetadata::id` for the target tool.
    pub tool_name: &'static str,
    /// Generator returning the unsolved initial `PresetWrapper` JSON.
    pub initial_preset_generator: fn() -> &'static str,
    /// Plain-language target prompt displayed in the Challenge HUD.
    pub goal_summary: &'static str,
    /// Conjunction of relational expressions compiled with `math_commons::ast::parse`.
    pub predicate_expr: &'static str,
    /// Optional hint revealed on request.
    pub hint: &'static str,
    /// Screen-reader and HUD celebration message announced on completion.
    pub win_message: &'static str,
    /// Initial plot callout pins synchronized when starting the challenge.
    pub callouts: &'static [CalloutSpec],
    /// Observable names highlighted in the HUD readout strip.
    pub focus_observables: &'static [&'static str],
}

inventory::collect!(ChallengeMetadata);

#[cfg(any(feature = "pure_math", feature = "applied", feature = "physics"))]
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

/// Returns `true` when `entry_domain` matches `requested_domain` (accounting for `biology`/`morphogenesis`).
#[must_use]
pub fn domain_matches_gallery(entry_domain: &str, requested_domain: &str) -> bool {
    entry_domain == requested_domain
        || (requested_domain == "biology" && entry_domain == "morphogenesis")
        || (requested_domain == "morphogenesis" && entry_domain == "biology")
}

/// Returns all registered guided stories for `domain`, sorted deterministically.
#[must_use]
pub fn get_stories_for_domain(domain: &str) -> Vec<&'static PresetStoryMetadata> {
    let mut out: Vec<&'static PresetStoryMetadata> = inventory::iter::<PresetStoryMetadata>
        .into_iter()
        .filter(|s| domain_matches_gallery(s.domain, domain) && !s.steps.is_empty())
        .collect();
    out.sort_by_key(|s| (s.title, s.id));
    out
}

/// Returns all registered playable challenges for `domain`, sorted deterministically.
#[must_use]
pub fn get_challenges_for_domain(domain: &str) -> Vec<&'static ChallengeMetadata> {
    let mut out: Vec<&'static ChallengeMetadata> = inventory::iter::<ChallengeMetadata>
        .into_iter()
        .filter(|c| domain_matches_gallery(c.domain, domain))
        .collect();
    out.sort_by_key(|c| (c.title, c.id));
    out
}

/// Converts registered guided stories for `domain` into `StarterCard` items for gallery rendering.
#[must_use]
pub fn get_story_cards_for_domain(domain: &str) -> Vec<StarterCard> {
    get_stories_for_domain(domain)
        .into_iter()
        .map(|s| StarterCard {
            id: s.id,
            title: s.title,
            description: s.description,
            category: s.category,
            icon: s.icon,
            domain: s.domain,
            tool_name: Some(s.tool_name),
            preset_json: s.steps.first().map(|step| (step.preset_generator)()),
        })
        .collect()
}

/// Converts registered playable challenges for `domain` into `StarterCard` items for gallery rendering.
#[must_use]
pub fn get_challenge_cards_for_domain(domain: &str) -> Vec<StarterCard> {
    get_challenges_for_domain(domain)
        .into_iter()
        .map(|c| StarterCard {
            id: c.id,
            title: c.title,
            description: c.description,
            category: c.category,
            icon: c.icon,
            domain: c.domain,
            tool_name: Some(c.tool_name),
            preset_json: Some((c.initial_preset_generator)()),
        })
        .collect()
}

/// Looks up a registered `PresetStoryMetadata` by ID (optionally scoped to `domain`).
#[must_use]
pub fn find_story_by_id(id: &str, domain: Option<&str>) -> Option<&'static PresetStoryMetadata> {
    let trimmed = id.trim();
    inventory::iter::<PresetStoryMetadata>
        .into_iter()
        .find(|s| {
            s.id.eq_ignore_ascii_case(trimmed)
                && !s.steps.is_empty()
                && domain.is_none_or(|d| domain_matches_gallery(s.domain, d))
        })
}

/// Looks up a registered `ChallengeMetadata` by ID (optionally scoped to `domain`).
#[must_use]
pub fn find_challenge_by_id(id: &str, domain: Option<&str>) -> Option<&'static ChallengeMetadata> {
    let trimmed = id.trim();
    inventory::iter::<ChallengeMetadata>.into_iter().find(|c| {
        c.id.eq_ignore_ascii_case(trimmed)
            && domain.is_none_or(|d| domain_matches_gallery(c.domain, d))
    })
}

/// Looks up a registered static `StarterPresetMetadata` snapshot by ID within `domain`.
#[must_use]
pub fn find_starter_preset_by_id(
    id: &str,
    domain: Option<&str>,
) -> Option<&'static StarterPresetMetadata> {
    let trimmed = id.trim();
    inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .find(|p| {
            p.id.eq_ignore_ascii_case(trimmed)
                && domain.is_none_or(|d| domain_matches_gallery(p.domain, d))
        })
}

#[cfg(feature = "pure_math")]
mod kelly_packs {
    use super::*;

    const STEP_HALF: &str = r#"{"metadata":{"schema_version":"1.0","title":"Half Kelly Stability","domain":"financial_math","timestamp":"2026-11-21T00:00:00Z"},"config":{"initial_bankroll":1000.0,"probability":0.6,"odds":2.0,"num_bets":200,"stake_multiplier":0.5}}"#;
    const STEP_FULL: &str = r#"{"metadata":{"schema_version":"1.0","title":"Full Kelly Optimum","domain":"financial_math","timestamp":"2026-11-21T00:00:00Z"},"config":{"initial_bankroll":1000.0,"probability":0.6,"odds":2.0,"num_bets":200}}"#;
    const STEP_RUIN: &str = r#"{"metadata":{"schema_version":"1.0","title":"Overbetting Ruin","domain":"financial_math","timestamp":"2026-11-21T00:00:00Z"},"config":{"initial_bankroll":1000.0,"probability":0.6,"odds":2.0,"num_bets":200,"stake_multiplier":2.5}}"#;

    const KELLY_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Half-Kelly Cushion",
            narrative: "A 60% coin at even-money (decimal odds 2.0) has f* = 20%. Betting 0.5f* (10%) compounds steadily with low volatility.",
            preset_generator: || STEP_HALF,
            callouts: &[pin("kelly_step", [140.0, 2200.0], [-20.0, -28.0], "0.5f* retains 75% log-growth")],
            focus_observables: &["kelly_fraction", "stake_multiplier", "expected_growth_rate"],
        },
        StoryStep {
            title: "Optimal Full Kelly Peak",
            narrative: "At 1.0f* (20% stake), expected logarithmic growth g = p ln(1+bf) + q ln(1-f) reaches its global maximum (+0.0201/bet).",
            preset_generator: || STEP_FULL,
            callouts: &[pin("kelly_step", [140.0, 4500.0], [-24.0, -28.0], "Optimal f* = 20% maximizes g > 0")],
            focus_observables: &["kelly_fraction", "stake_multiplier", "expected_growth_rate"],
        },
        StoryStep {
            title: "Overbetting Past 2f* Triggers Ruin",
            narrative: "Above 2f* (40% stake), volatility drag overwhelms positive expected value: g turns negative (-0.0340 at 2.5f*) and wealth decays toward zero.",
            preset_generator: || STEP_RUIN,
            callouts: &[pin("kelly_step", [140.0, 120.0], [18.0, -26.0], "2.5f* overbet: EV > 0 yet g < 0")],
            focus_observables: &["expected_value", "stake_multiplier", "expected_growth_rate"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "kelly_paradox_tour",
            title: "Kelly Overbetting Ruin Tour",
            description: "Why wagering past 2f* on a favorable coin destroys wealth despite positive expected value.",
            category: "Guided Story",
            icon: "📖",
            domain: "financial_math",
            tool_name: "BankrollGrowthTool",
            steps: KELLY_STEPS,
        }
    }

    inventory::submit! {
        ChallengeMetadata {
            id: "kelly_ruin_trap",
            title: "Kelly Ruin Trap",
            description: "Keep positive EV while pushing stake size past 2f* until expected log-growth turns negative.",
            category: "Challenge",
            icon: "🎯",
            domain: "financial_math",
            tool_name: "BankrollGrowthTool",
            initial_preset_generator: || STEP_FULL,
            goal_summary: "Keep a positive-edge bet (expected_value > 0.05) while raising Stake Multiplier above 2.0f* so expected_growth_rate drops below -0.01.",
            predicate_expr: "expected_value > 0.05 && stake_multiplier > 2.0 && expected_growth_rate < -0.01",
            hint: "Drag the 'Stake Multiplier (× f*)' slider above 2.2 while keeping Win Probability at 0.60 and Decimal Odds at 2.0.",
            win_message: "Challenge solved: Kelly Ruin Trap! Even with positive expected value, betting above 2f* yields negative logarithmic growth.",
            callouts: &[pin("kelly_challenge_baseline", [0.0, 1000.0], [36.0, -24.0], "Initial bankroll W₀ = 1000")],
            focus_observables: &["expected_value", "stake_multiplier", "expected_growth_rate"],
        }
    }
}

#[cfg(feature = "applied")]
mod replicator_packs {
    use super::*;

    const PD_COOP: &str = r#"{"metadata":{"schema_version":"1.0","title":"Prisoner's Dilemma Collapse","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[3.0,0.0],[5.0,1.0]],"initial_population":[0.9,0.1],"time_horizon":10.0,"dt":0.05,"strategy_names":["Cooperate","Defect"]}}"#;
    const STAG_LOW: &str = r#"{"metadata":{"schema_version":"1.0","title":"Stag Hunt Below Threshold","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[4.0,0.0],[3.0,2.0]],"initial_population":[0.55,0.45],"time_horizon":12.0,"dt":0.05,"strategy_names":["Stag","Hare"]}}"#;
    const STAG_HIGH: &str = r#"{"metadata":{"schema_version":"1.0","title":"Stag Hunt Coordination","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[4.0,0.0],[3.0,2.0]],"initial_population":[0.75,0.25],"time_horizon":12.0,"dt":0.05,"strategy_names":["Stag","Hare"]}}"#;
    const PD_INIT: &str = r#"{"metadata":{"schema_version":"1.0","title":"Mild Harmony Baseline","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[3.0,2.0],[2.5,1.5]],"initial_population":[0.9,0.1],"time_horizon":10.0,"dt":0.05,"strategy_names":["Cooperate","Defect"]}}"#;
    const RPS_NEUTRAL: &str = r#"{"metadata":{"schema_version":"1.0","title":"Neutral RPS Cycle","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[0.0,-1.0,1.0],[1.0,0.0,-1.0],[-1.0,1.0,0.0]],"initial_population":[0.5,0.3,0.2],"time_horizon":40.0,"dt":0.05,"strategy_names":["Rock","Paper","Scissors"]}}"#;
    const RPS_INWARD: &str = r#"{"metadata":{"schema_version":"1.0","title":"Damped Spiral RPS","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[0.0,-0.7,1.0],[1.0,0.0,-0.7],[-0.7,1.0,0.0]],"initial_population":[0.55,0.25,0.2],"time_horizon":40.0,"dt":0.05,"strategy_names":["Rock","Paper","Scissors"]}}"#;
    const RPS_OUTWARD: &str = r#"{"metadata":{"schema_version":"1.0","title":"Heteroclinic Spiral RPS","domain":"game_theory","timestamp":"2026-11-21T00:00:00Z"},"config":{"payoff_matrix":[[0.0,-1.3,1.0],[1.0,0.0,-1.3],[-1.3,1.0,0.0]],"initial_population":[0.4,0.32,0.28],"time_horizon":45.0,"dt":0.05,"strategy_names":["Rock","Paper","Scissors"]}}"#;

    const DILEMMA_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Prisoner's Dilemma Fitness Collapse",
            narrative: "Starting with 90% Cooperators gives high mean fitness xᵀAx = 2.89, yet individual selection favors Defectors until xᵀAx plunges to 1.00.",
            preset_generator: || PD_COOP,
            callouts: &[
                pin("pd_start", [0.3, 2.89], [36.0, -18.0], "Initial xᵀAx = 2.89"),
                pin("pd_end", [7.5, 1.0], [-24.0, -26.0], "Defection lowers xᵀAx to 1.00"),
            ],
            focus_observables: &["initial_mean_fitness", "final_mean_fitness", "fitness_change"],
        },
        StoryStep {
            title: "Stag Hunt Trust Breakdown",
            narrative: "In Stag Hunt, when initial Stag trust (55%) sits below the unstable 2/3 invasion threshold, risk-dominant Hare wins (xᵀAx = 2.0).",
            preset_generator: || STAG_LOW,
            callouts: &[pin("stag_low", [6.0, 2.0], [10.0, -24.0], "Below x* = 2/3: Hare basin (xᵀAx = 2.0)")],
            focus_observables: &["x0_initial", "x0_final", "final_mean_fitness"],
        },
        StoryStep {
            title: "Stag Hunt Cooperative Escape",
            narrative: "Crossing x₀ = 0.75 > 2/3 flips the replicator vector field into the payoff-dominant Stag basin, doubling mean fitness to 4.0.",
            preset_generator: || STAG_HIGH,
            callouts: &[pin("stag_high", [6.0, 4.0], [10.0, -24.0], "Above x* = 2/3: Stag optimum (xᵀAx = 4.0)")],
            focus_observables: &["x0_initial", "x0_final", "final_mean_fitness"],
        },
    ];

    const SPIRAL_RPS_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "Zero-Sum Neutral Cycle",
            narrative: "Symmetric zero-sum Rock-Paper-Scissors conserves a Hamiltonian invariant, locking strategies into closed periodic orbits with xᵀAx = 0.",
            preset_generator: || RPS_NEUTRAL,
            callouts: &[pin("rps_step", [20.0, 0.33], [12.0, -24.0], "Closed neutral oscillations around 1/3")],
            focus_observables: &["final_mean_fitness", "oscillation_span"],
        },
        StoryStep {
            title: "Positive-Sum Inward Spiral",
            narrative: "Softening loss penalties to -0.7 makes coexistence (1/3, 1/3, 1/3) a stable spiral attractor with positive mean fitness (+0.10).",
            preset_generator: || RPS_INWARD,
            callouts: &[pin("rps_step", [30.0, 0.33], [12.0, -24.0], "Damped spiral converges to coexistence")],
            focus_observables: &["final_mean_fitness", "oscillation_span"],
        },
        StoryStep {
            title: "Negative-Sum Heteroclinic Spiral",
            narrative: "Harsher loss penalties (-1.3) destabilize coexistence, flinging the population outward toward boundary dominance eras.",
            preset_generator: || RPS_OUTWARD,
            callouts: &[pin("rps_step", [35.0, 0.90], [-20.0, -24.0], "Expanding heteroclinic boundary waves")],
            focus_observables: &["final_mean_fitness", "oscillation_span"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "prisoners_dilemma_tragedy",
            title: "Social Dilemma Fitness Paradox",
            description: "Watch natural selection lower population fitness xᵀAx in Prisoner's Dilemma and cross the Stag Hunt tipping point.",
            category: "Guided Story",
            icon: "🤝",
            domain: "game_theory",
            tool_name: "ReplicatorDynamicsTool",
            steps: DILEMMA_STEPS,
        }
    }

    inventory::submit! {
        PresetStoryMetadata {
            id: "spiral_rps_tour",
            title: "Rock-Paper-Scissors Spiral Regimes",
            description: "Compare neutral Hamiltonian cycles, damped coexistence spirals, and outward heteroclinic waves.",
            category: "Guided Story",
            icon: "🌀",
            domain: "game_theory",
            tool_name: "ReplicatorDynamicsTool",
            steps: SPIRAL_RPS_STEPS,
        }
    }

    inventory::submit! {
        ChallengeMetadata {
            id: "prisoners_dilemma_fitness_drop",
            title: "Trigger Fitness Decline",
            description: "Construct a Prisoner's Dilemma payoff matrix where Defectors win yet population mean fitness xᵀAx drops by over 1.0.",
            category: "Challenge",
            icon: "🎯",
            domain: "game_theory",
            tool_name: "ReplicatorDynamicsTool",
            initial_preset_generator: || PD_INIT,
            goal_summary: "Click 'Prisoner's Dilemma' or edit payoffs (and click 'Run Simulation') so Defectors fixate (x1_final > 0.95) and fitness_change < -1.0.",
            predicate_expr: "x1_final > 0.95 && fitness_change < -1.0",
            hint: "Set Defect-vs-Cooperate > Cooperate-vs-Cooperate (5 > 3) and Defect-vs-Defect > Cooperate-vs-Defect (1 > 0), or click 'Prisoner's Dilemma'.",
            win_message: "Challenge solved: Frequency-dependent selection drove Defectors to fixation while slashing mean population fitness!",
            callouts: &[pin("pd_goal", [5.0, 1.5], [12.0, -22.0], "Drive Mean Fitness (xᵀAx) down while Defect -> 1.0")],
            focus_observables: &["initial_mean_fitness", "final_mean_fitness", "fitness_change", "x1_final"],
        }
    }

    inventory::submit! {
        ChallengeMetadata {
            id: "stag_hunt_tipping_point",
            title: "Escape the Hare Basin",
            description: "Cross the unstable Stag Hunt invasion threshold to reach full cooperative Stag coordination.",
            category: "Challenge",
            icon: "🦌",
            domain: "game_theory",
            tool_name: "ReplicatorDynamicsTool",
            initial_preset_generator: || STAG_LOW,
            goal_summary: "Raise initial Stag share above the 2/3 threshold and click 'Run Simulation' so x0_final > 0.95 and final_mean_fitness > 3.8.",
            predicate_expr: "x0_final > 0.95 && final_mean_fitness > 3.8",
            hint: "Raise Stag's initial slider above 0.70 (with Hare below 0.30) and click '▶ Run Simulation'.",
            win_message: "Challenge solved: Stag trust cleared the 2/3 separatrix and converged to the payoff-dominant equilibrium!",
            callouts: &[pin("stag_threshold", [0.2, 0.67], [36.0, -18.0], "Unstable Stag tipping threshold x* = 2/3")],
            focus_observables: &["x0_initial", "x0_final", "final_mean_fitness"],
        }
    }
}

#[cfg(feature = "physics")]
mod potential_flow_packs {
    use super::*;

    const CYL_SYM: &str = r#"{"metadata":{"schema_version":"1.0","title":"d'Alembert Symmetric Cylinder","domain":"fluid_dynamics","timestamp":"2026-11-21T00:00:00Z"},"config":{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0},{"element_type":"Doublet","strength":251.32741228718345,"x":0.0,"y":0.0,"angle":0.0},{"element_type":"Vortex","strength":0.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":22,"view_range":8.0}}"#;
    const CYL_SUB: &str = r#"{"metadata":{"schema_version":"1.0","title":"Subcritical Kutta-Joukowski Lift","domain":"fluid_dynamics","timestamp":"2026-11-21T00:00:00Z"},"config":{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0},{"element_type":"Doublet","strength":251.32741228718345,"x":0.0,"y":0.0,"angle":0.0},{"element_type":"Vortex","strength":-125.66370614359172,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":22,"view_range":8.0}}"#;
    const CYL_SUP: &str = r#"{"metadata":{"schema_version":"1.0","title":"Supercritical Detached Stagnation","domain":"fluid_dynamics","timestamp":"2026-11-21T00:00:00Z"},"config":{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0},{"element_type":"Doublet","strength":251.32741228718345,"x":0.0,"y":0.0,"angle":0.0},{"element_type":"Vortex","strength":-300.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":22,"view_range":8.0}}"#;

    const LIFT_STEPS: &[StoryStep] = &[
        StoryStep {
            title: "d'Alembert's Zero-Drag Cylinder",
            narrative: "Fore-and-aft streamline symmetry around an inviscid cylinder (R = 2) places stagnation points at (±2, 0), yielding zero lift and zero drag.",
            preset_generator: || CYL_SYM,
            callouts: &[
                pin("stag_front", [-2.0, 0.0], [-42.0, -26.0], "Front stagnation (-2, 0)"),
                pin("stag_rear", [2.0, 0.0], [42.0, -26.0], "Rear stagnation (+2, 0): D' = 0"),
            ],
            focus_observables: &["circulation", "kutta_joukowski_lift", "pressure_drag"],
        },
        StoryStep {
            title: "Circulation Breaks Top-Bottom Symmetry",
            narrative: "Adding clockwise circulation (Γ = -40π) accelerates streamlines over the top and shifts stagnation points downward, creating upward lift L' = -ρUΓ.",
            preset_generator: || CYL_SUB,
            callouts: &[pin("lift_peak", [0.0, 2.1], [0.0, -28.0], "Packed upper streamlines -> upward lift")],
            focus_observables: &["circulation", "kutta_joukowski_lift", "pressure_drag"],
        },
        StoryStep {
            title: "Supercritical Spin Detaches Stagnation",
            narrative: "Exceeding |Γ| = 4πRU merges surface stagnation points and detaches a saddle stagnation point into the fluid at (0, -3.69).",
            preset_generator: || CYL_SUP,
            callouts: &[pin("detached_saddle", [0.0, -3.69], [44.0, 22.0], "Detached saddle point (0, -3.69)")],
            focus_observables: &["circulation", "kutta_joukowski_lift", "stagnation_count"],
        },
    ];

    inventory::submit! {
        PresetStoryMetadata {
            id: "dalembert_to_kutta_lift",
            title: "d'Alembert's Paradox & Magnus Lift",
            description: "Tour symmetric zero-drag cylinder streamlines, Kutta-Joukowski circulation lift, and detached stagnation.",
            category: "Guided Story",
            icon: "✈️",
            domain: "fluid_dynamics",
            tool_name: "PotentialFlowTool",
            steps: LIFT_STEPS,
        }
    }

    inventory::submit! {
        ChallengeMetadata {
            id: "kutta_joukowski_lift",
            title: "Zero-Drag Circulation Lift",
            description: "Tune vortex circulation around a cylinder to generate upward Kutta-Joukowski lift above 1000 with zero pressure drag.",
            category: "Challenge",
            icon: "🌪️",
            domain: "fluid_dynamics",
            tool_name: "PotentialFlowTool",
            initial_preset_generator: || CYL_SYM,
            goal_summary: "Set negative (clockwise) Vortex circulation Γ so upward kutta_joukowski_lift exceeds 1000 while pressure_drag stays 0.",
            predicate_expr: "kutta_joukowski_lift > 1000.0 && pressure_drag <= 0.001",
            hint: "Drag the Vortex strength slider below -105 (clockwise spin accelerates flow over the top of the cylinder).",
            win_message: "Challenge solved: Kutta-Joukowski circulation lift L' = -ρUΓ > 1000 achieved with zero inviscid pressure drag!",
            callouts: &[pin("cylinder_top", [0.0, 2.0], [0.0, -26.0], "Accelerate upper flow (Γ < -100) for lift")],
            focus_observables: &["freestream_speed", "circulation", "kutta_joukowski_lift", "pressure_drag"],
        }
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;

impl scientific_metadata::theory::TheoryDescribable for GridWorldTool {
    fn theory_description(&self) -> String {
        format!(
            "Tabular Q-Learning updates state-action values via Bellman temporal difference $Q(s,a) \\leftarrow Q(s,a) + \\alpha [R + \\gamma \\max_{{a'}} Q(s',a') - Q(s,a)]$ with $\\alpha = {:.2}$, $\\gamma = {:.2}$, and $\\epsilon = {:.2}$.",
            self.learning_rate, self.discount_factor, self.exploration_rate
        )
    }

    fn phonetic_description(&self) -> String {
        "Reinforcement Learning involves an agent that interacts with its environment to learn optimal actions, aiming to maximize a cumulative reward based on the Bellman equation.".into()
    }

    fn theory_citation(&self) -> String {
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT press."
            .into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "learning_rate".into(),
            "The step size used to update the Q-values, controlling how much new information overrides old information.".into(),
        );
        map.insert(
            "discount_factor".into(),
            "Determines the importance of future rewards. A factor of 0 makes the agent opportunistic, while 1 makes it strive for long-term high reward.".into(),
        );
        map.insert(
            "exploration_rate".into(),
            "The probability that the agent will choose a random action instead of the best known action, facilitating exploration of the state space.".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Grid World RL","domain":"ai","timestamp":"2026-11-16T00:00:00Z"},"config":{"width":5,"height":5,"gamma":0.95,"playback_speed":8.0,"is_playing":false}}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "grid_world_starter",
        title: "Grid World RL",
        description: "Markov Decision Process navigation with obstacle avoidance and discount reward propagation.",
        category: "Reinforcement Learning",
        icon: "🗺️",
        domain: "ai",
        tool_name: Some("GridWorldTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

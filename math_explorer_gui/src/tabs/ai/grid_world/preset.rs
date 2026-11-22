#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::framework::{decode_u8_grid, encode_u8_grid};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::ai::reinforcement_learning::grid_world::GridState;

fn default_rl_rate() -> f64 {
    0.1
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GridWorldToolState {
    pub(super) width: i32,
    pub(super) height: i32,
    #[serde(alias = "discount_factor")]
    pub(super) gamma: f64,
    #[serde(default = "default_rl_rate")]
    pub(super) learning_rate: f64,
    #[serde(default = "default_rl_rate")]
    pub(super) exploration_rate: f64,
    pub(super) playback_speed: f32,
    pub(super) is_playing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) start: Option<[i32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) goal: Option<[i32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) layout_rle: Option<String>,
}

pub(super) fn resolve_endpoints(
    parsed: &GridWorldToolState,
) -> Result<(GridState, GridState), String> {
    let start = match parsed.start {
        Some([sx, sy]) if (0..parsed.width).contains(&sx) && (0..parsed.height).contains(&sy) => {
            GridState { x: sx, y: sy }
        }
        Some(_) => return Err("GridWorld start coordinates out of bounds".into()),
        None => GridState { x: 0, y: 0 },
    };
    let goal = match parsed.goal {
        Some([gx, gy]) if (0..parsed.width).contains(&gx) && (0..parsed.height).contains(&gy) => {
            GridState { x: gx, y: gy }
        }
        Some(_) => return Err("GridWorld goal coordinates out of bounds".into()),
        None => GridState {
            x: parsed.width - 1,
            y: parsed.height - 1,
        },
    };
    if start == goal {
        return Err("GridWorld start and goal cannot occupy the same cell".into());
    }
    Ok((start, goal))
}

pub(super) fn decode_walls_and_traps(
    parsed: &GridWorldToolState,
    start: GridState,
    goal: GridState,
) -> Result<(Vec<GridState>, Vec<GridState>), String> {
    if let Some(ref rle) = parsed.layout_rle {
        let w = parsed.width as usize;
        let h = parsed.height as usize;
        let cells = decode_u8_grid(rle, w, h, 2)?;
        let mut walls = Vec::new();
        let mut traps = Vec::new();
        for (idx, sym) in cells.into_iter().enumerate() {
            let st = GridState {
                x: (idx % w) as i32,
                y: (idx / w) as i32,
            };
            if st == start || st == goal {
                continue;
            }
            match sym {
                1 => walls.push(st),
                2 => traps.push(st),
                _ => {}
            }
        }
        Ok((walls, traps))
    } else {
        let traps = vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }]
            .into_iter()
            .filter(|t| t.x < parsed.width && t.y < parsed.height && *t != start && *t != goal)
            .collect();
        Ok((Vec::new(), traps))
    }
}

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
        map.insert("learning_rate".into(), "The step size used to update the Q-values, controlling how much new information overrides old information.".into());
        map.insert("discount_factor".into(), "Determines the importance of future rewards. A factor of 0 makes the agent opportunistic, while 1 makes it strive for long-term high reward.".into());
        map.insert("exploration_rate".into(), "The probability that the agent will choose a random action instead of the best known action, facilitating exploration of the state space.".into());
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Grid World RL","domain":"ai","timestamp":"2026-11-16T00:00:00Z"},"config":{"width":5,"height":5,"gamma":0.95,"playback_speed":8.0,"is_playing":false}}"#;

fn generate_grid_world_labyrinth_preset() -> &'static str {
    static LABYRINTH_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LABYRINTH_PRESET
        .get_or_init(|| {
            let w = 8usize;
            let h = 8usize;
            let mut grid = vec![0u8; w * h];
            for &(wx, wy) in &[
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (5, 3),
                (5, 4),
                (5, 5),
                (5, 6),
                (5, 7),
            ] {
                grid[flatten_2d_index(wx, wy, w)] = 1;
            }
            grid[flatten_2d_index(5, 2, w)] = 2;
            grid[flatten_2d_index(2, 5, w)] = 2;
            let rle = encode_u8_grid(w, h, &grid, 2).expect("valid GridWorld labyrinth RLE");
            serde_json::json!({
                "metadata": {
                    "schema_version": "1.0",
                    "title": "Grid World Labyrinth Maze",
                    "domain": "ai",
                    "timestamp": "2026-11-21T00:00:00Z"
                },
                "config": {
                    "width": 8,
                    "height": 8,
                    "gamma": 0.95,
                    "learning_rate": 0.25,
                    "exploration_rate": 0.15,
                    "playback_speed": 12.0,
                    "is_playing": false,
                    "start": [0, 0],
                    "goal": [7, 7],
                    "layout_rle": rle
                }
            })
            .to_string()
        })
        .as_str()
}

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

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "grid_world_labyrinth_maze",
        title: "Grid World Labyrinth Maze",
        description: "Sculpted 8x8 switchback wall labyrinth with hazard traps demonstrating Bellman temporal-difference rerouting.",
        category: "Reinforcement Learning",
        icon: "🧱",
        domain: "ai",
        tool_name: Some("GridWorldTool"),
        preset_generator: generate_grid_world_labyrinth_preset,
    }
}

//! Declarative Challenge HUD, guided story stepper, and `math_commons::ast` predicate engine
//! for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CalloutSpec, SimulationFramework};
use crate::starter_gallery::{ChallengeMetadata, PresetStoryMetadata};
use eframe::egui;
use math_commons::ast::{parse as parse_ast, Expr};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[path = "challenge_hud/banner.rs"]
mod banner;

/// Maximum relational clauses permitted in a single compiled goal predicate.
pub const MAX_GOAL_CLAUSES: usize = 16;

/// Relational comparison operator evaluated between two `math_commons::ast::Expr` trees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonOp {
    /// Strict greater-than (`>`).
    Gt,
    /// Greater-than or equal (`>=`).
    Ge,
    /// Strict less-than (`<`).
    Lt,
    /// Less-than or equal (`<=`).
    Le,
    /// Approximate equality (`==` / `=`).
    ApproxEq,
}

impl ComparisonOp {
    fn evaluate(self, lhs: f64, rhs: f64) -> bool {
        lhs.is_finite()
            && rhs.is_finite()
            && match self {
                Self::Gt => lhs > rhs,
                Self::Ge => lhs >= rhs - 1e-9,
                Self::Lt => lhs < rhs,
                Self::Le => lhs <= rhs + 1e-9,
                Self::ApproxEq => (lhs - rhs).abs() <= 1e-3,
            }
    }
}

#[derive(Clone, Debug)]
struct CompiledGoalClause {
    lhs: Expr,
    op: ComparisonOp,
    rhs: Expr,
}

/// Conjunction of compiled AST comparison clauses evaluated against finite tool observables.
#[derive(Clone, Debug)]
pub struct CompiledGoalPredicate {
    raw: String,
    clauses: Vec<CompiledGoalClause>,
}

fn split_comparison_operator(clause: &str) -> (ComparisonOp, &str, &str) {
    for (token, op) in [
        (">=", ComparisonOp::Ge),
        ("<=", ComparisonOp::Le),
        ("==", ComparisonOp::ApproxEq),
        (">", ComparisonOp::Gt),
        ("<", ComparisonOp::Lt),
        ("=", ComparisonOp::ApproxEq),
    ] {
        if let Some((lhs, rhs)) = clause.split_once(token) {
            return (op, lhs.trim(), rhs.trim());
        }
    }
    (ComparisonOp::Gt, clause.trim(), "0.0")
}

impl CompiledGoalPredicate {
    /// Compiles a conjunction (`&&` or `;`) of relational AST formulas using `math_commons::ast::parse`.
    pub fn compile(formula: &str) -> Result<Self, String> {
        let trimmed = formula.trim();
        let parts: Vec<&str> = trimmed
            .split("&&")
            .flat_map(|chunk| chunk.split(';'))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err("Predicate formula cannot be empty".to_string());
        }
        if parts.len() > MAX_GOAL_CLAUSES {
            return Err(format!(
                "Predicate exceeds max clauses ({MAX_GOAL_CLAUSES})"
            ));
        }
        let mut clauses = Vec::with_capacity(parts.len());
        for part in parts {
            let (op, lhs_str, rhs_str) = split_comparison_operator(part);
            if lhs_str.is_empty() || rhs_str.is_empty() {
                return Err(format!("Incomplete comparison in '{part}'"));
            }
            let lhs = parse_ast(lhs_str).map_err(|e| e.user_message_for_input(lhs_str))?;
            let rhs = parse_ast(rhs_str).map_err(|e| e.user_message_for_input(rhs_str))?;
            clauses.push(CompiledGoalClause { lhs, op, rhs });
        }
        Ok(Self {
            raw: trimmed.to_string(),
            clauses,
        })
    }

    /// Returns the raw predicate string.
    #[must_use]
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Evaluates all clauses via `Expr::eval` (bounded by `MAX_EVAL_DEPTH = 64`) over finite observables.
    pub fn eval_observables(
        &self,
        observables: &[(&'static str, f64)],
        scratch: &mut HashMap<String, f64>,
    ) -> Result<bool, String> {
        if observables.is_empty() || observables.iter().any(|(_, v)| !v.is_finite()) {
            return Ok(false);
        }
        scratch.clear();
        for &(name, val) in observables {
            scratch.insert(name.to_string(), val);
        }
        for clause in &self.clauses {
            let lhs_val = clause.lhs.eval(scratch).map_err(|e| e.to_string())?;
            let rhs_val = clause.rhs.eval(scratch).map_err(|e| e.to_string())?;
            if !clause.op.evaluate(lhs_val, rhs_val) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Interactive AST sandbox state allowing custom user predicates over live tool observables.
#[derive(Clone, Debug)]
pub struct AstSandboxState {
    /// Whether the sandbox input row is expanded.
    pub is_open: bool,
    /// User-edited predicate text.
    pub formula: String,
    compiled: Result<CompiledGoalPredicate, String>,
    /// Latest evaluation result (`Ok(true/false)` or `Err(reason)`).
    pub last_eval: Result<bool, String>,
    announced_satisfied: bool,
}

impl Default for AstSandboxState {
    fn default() -> Self {
        Self {
            is_open: false,
            formula: String::new(),
            compiled: Err("Enter a target inequality".to_string()),
            last_eval: Ok(false),
            announced_satisfied: false,
        }
    }
}

impl AstSandboxState {
    /// Updates the custom sandbox formula and recompiles its AST clauses.
    pub fn set_formula(&mut self, formula: &str) {
        self.formula = formula.to_string();
        self.compiled = CompiledGoalPredicate::compile(formula);
        self.last_eval = Ok(false);
        self.announced_satisfied = false;
    }
}

/// Active mode of the `SimulationFramework` Challenge & Story HUD.
#[derive(Clone, Debug, Default)]
pub enum ChallengeHudMode {
    /// No active guided story or challenge.
    #[default]
    Inactive,
    /// Multi-step guided story tour.
    Story {
        /// Static metadata for the active story.
        story: &'static PresetStoryMetadata,
        /// Current 0-based step index.
        step_index: usize,
        /// Compact serialized tool state for this step's baseline.
        step_state: Option<String>,
    },
    /// Playable AST target challenge.
    Challenge {
        /// Static metadata for the active challenge.
        challenge: &'static ChallengeMetadata,
        /// Compiled AST goal predicate.
        compiled: Result<CompiledGoalPredicate, String>,
        /// Compact serialized tool state for the initial challenge baseline.
        initial_state: Option<String>,
        /// Whether the predicate currently evaluates to `true`.
        currently_satisfied: bool,
        /// Latched completion status once the target predicate holds.
        solved: bool,
        /// Whether the optional hint callout text is expanded.
        show_hint: bool,
    },
}

/// Runtime state for guided stories, AST target challenges, and live ARIA announcements.
#[derive(Default)]
pub struct ChallengeHudState {
    /// Active story or challenge mode.
    pub mode: ChallengeHudMode,
    /// Interactive AST sandbox state.
    pub sandbox: AstSandboxState,
    vars_scratch: HashMap<String, f64>,
    last_announced_story: Option<(&'static str, usize)>,
    announced_challenge_win: Option<&'static str>,
}

impl SimulationFramework {
    pub(super) fn hash_hud_for_url(&self, hasher: &mut impl Hasher) {
        match &self.challenge_hud.mode {
            ChallengeHudMode::Inactive => 0_u8.hash(hasher),
            ChallengeHudMode::Story {
                story, step_index, ..
            } => {
                (1_u8, story.id, step_index).hash(hasher);
            }
            ChallengeHudMode::Challenge { challenge, .. } => (2_u8, challenge.id).hash(hasher),
        }
    }

    /// Clears any active guided story or challenge banner and automated plot callouts.
    pub fn clear_challenge_hud(&mut self) {
        self.challenge_hud.mode = ChallengeHudMode::Inactive;
        self.challenge_hud.sandbox.is_open = false;
        self.challenge_hud.last_announced_story = None;
        self.challenge_hud.announced_challenge_win = None;
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(&[]);
        }
    }

    /// Opens the interactive AST Sandbox banner initialized with `formula`.
    pub fn open_ast_sandbox(&mut self, formula: &str) {
        self.challenge_hud.sandbox.is_open = true;
        self.challenge_hud.sandbox.set_formula(formula);
    }

    /// Resets solved challenge latch and re-applies HUD callout pins after parameter reset.
    pub(crate) fn on_hud_parameters_reset(&mut self) {
        if let ChallengeHudMode::Challenge {
            ref mut solved,
            ref mut currently_satisfied,
            ..
        } = self.challenge_hud.mode
        {
            *solved = false;
            *currently_satisfied = false;
        }
        self.challenge_hud.announced_challenge_win = None;
        self.resync_active_hud_callouts();
    }

    /// Re-applies active story step or challenge callout pins to the active tool.
    pub fn resync_active_hud_callouts(&mut self) {
        let specs: &[CalloutSpec] = match &self.challenge_hud.mode {
            ChallengeHudMode::Story {
                story, step_index, ..
            } => story.steps.get(*step_index).map_or(&[], |s| s.callouts),
            ChallengeHudMode::Challenge { challenge, .. } => challenge.callouts,
            ChallengeHudMode::Inactive => &[],
        };
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(specs);
        }
    }

    /// Launches a multi-step guided story at `step_index`, syncing tool state, callouts, and baseline.
    pub fn launch_story(
        &mut self,
        story: &'static PresetStoryMetadata,
        step_index: usize,
    ) -> Result<(), String> {
        let step = story
            .steps
            .get(step_index)
            .ok_or_else(|| format!("Story '{}' step {step_index} out of bounds", story.id))?;
        self.apply_starter_card(
            story.title,
            Some(story.tool_name),
            Some((step.preset_generator)()),
        )?;
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(step.callouts);
        }
        let step_state = self.active_tool.as_ref().and_then(|t| t.save_state());
        if self.challenge_hud.sandbox.formula.is_empty() {
            if let Some((obs, _)) = self
                .active_tool
                .as_ref()
                .and_then(|t| t.observables().first().copied())
            {
                self.challenge_hud
                    .sandbox
                    .set_formula(&format!("{obs} > 0.0"));
            }
        }
        self.challenge_hud.announced_challenge_win = None;
        self.challenge_hud.mode = ChallengeHudMode::Story {
            story,
            step_index,
            step_state,
        };
        self.sync_active_baseline();
        Ok(())
    }

    /// Advances or rewinds the active guided story to `step_index`.
    pub fn step_story(&mut self, step_index: usize) -> Result<(), String> {
        let ChallengeHudMode::Story { story, .. } = self.challenge_hud.mode else {
            return Err("No active guided story to step".to_string());
        };
        self.launch_story(story, step_index)
    }

    /// Launches a playable AST target challenge, compiling its goal formula and syncing baseline state.
    pub fn launch_challenge(
        &mut self,
        challenge: &'static ChallengeMetadata,
    ) -> Result<(), String> {
        let compiled = CompiledGoalPredicate::compile(challenge.predicate_expr);
        self.apply_starter_card(
            challenge.title,
            Some(challenge.tool_name),
            Some((challenge.initial_preset_generator)()),
        )?;
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(challenge.callouts);
        }
        let initial_state = self.active_tool.as_ref().and_then(|t| t.save_state());
        self.challenge_hud
            .sandbox
            .set_formula(challenge.predicate_expr);
        self.challenge_hud.last_announced_story = None;
        self.challenge_hud.announced_challenge_win = None;
        self.challenge_hud.mode = ChallengeHudMode::Challenge {
            challenge,
            compiled,
            initial_state,
            currently_satisfied: false,
            solved: false,
            show_hint: false,
        };
        self.sync_active_baseline();
        let _ = self.evaluate_challenge_predicates(None);
        Ok(())
    }

    /// Resets the active challenge back to its initial unsolved parameter configuration.
    pub fn reset_active_challenge(&mut self) -> Result<(), String> {
        let ChallengeHudMode::Challenge { challenge, .. } = self.challenge_hud.mode else {
            return Err("No active challenge to reset".to_string());
        };
        self.launch_challenge(challenge)
    }

    /// Evaluates active challenge and AST sandbox predicates when `!self.is_diverged`.
    pub fn evaluate_challenge_predicates(&mut self, ctx: Option<&egui::Context>) -> Option<bool> {
        if self.is_diverged {
            return None;
        }
        let observables = self
            .active_tool
            .as_ref()
            .map(|t| t.observables())
            .unwrap_or_default();
        if self.challenge_hud.sandbox.is_open {
            self.eval_sandbox_with_observables(&observables, ctx);
        }
        let ChallengeHudMode::Challenge {
            challenge,
            ref compiled,
            ref mut currently_satisfied,
            ref mut solved,
            ..
        } = self.challenge_hud.mode
        else {
            return None;
        };
        let satisfied = compiled
            .as_ref()
            .ok()
            .and_then(|p| {
                p.eval_observables(&observables, &mut self.challenge_hud.vars_scratch)
                    .ok()
            })
            .unwrap_or(false);
        *currently_satisfied = satisfied;
        if satisfied {
            *solved = true;
            if let Some(egui_ctx) = ctx {
                if self.challenge_hud.announced_challenge_win != Some(challenge.id) {
                    crate::accessibility::announce_live_status(egui_ctx, challenge.win_message);
                    self.challenge_hud.announced_challenge_win = Some(challenge.id);
                }
            }
        }
        Some(satisfied)
    }

    pub(in crate::framework) fn eval_sandbox_with_observables(
        &mut self,
        observables: &[(&'static str, f64)],
        ctx: Option<&egui::Context>,
    ) {
        let eval_res = match &self.challenge_hud.sandbox.compiled {
            Ok(pred) => pred.eval_observables(observables, &mut self.challenge_hud.vars_scratch),
            Err(err) => Err(err.clone()),
        };
        let now_true = matches!(eval_res, Ok(true));
        if now_true && !self.challenge_hud.sandbox.announced_satisfied {
            if let Some(egui_ctx) = ctx {
                let msg = format!(
                    "Sandbox AST goal satisfied: {}",
                    self.challenge_hud.sandbox.formula
                );
                crate::accessibility::announce_live_status(egui_ctx, msg);
                self.challenge_hud.sandbox.announced_satisfied = true;
            }
        } else if !now_true {
            self.challenge_hud.sandbox.announced_satisfied = false;
        }
        self.challenge_hud.sandbox.last_eval = eval_res;
    }

    /// Dispatches polite ARIA live announcements for newly entered story steps or solved challenges.
    pub fn flush_hud_announcements(&mut self, ctx: &egui::Context) {
        if let ChallengeHudMode::Story {
            story, step_index, ..
        } = self.challenge_hud.mode
        {
            if self.challenge_hud.last_announced_story != Some((story.id, step_index)) {
                if let Some(step) = story.steps.get(step_index) {
                    let msg = format!(
                        "Guided story {}, step {} of {}: {}. {}",
                        story.title,
                        step_index + 1,
                        story.steps.len(),
                        step.title,
                        step.narrative
                    );
                    crate::accessibility::announce_live_status(ctx, msg);
                    self.challenge_hud.last_announced_story = Some((story.id, step_index));
                }
            }
        }
        let _ = self.evaluate_challenge_predicates(Some(ctx));
    }
}

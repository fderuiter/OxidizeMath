//! Pre-allocated 2D bracketed turtle geometry interpreter for botanical L-systems.

use super::grammar::{LSystemEngine, MAX_TURTLE_STACK_DEPTH};
use crate::error::BotanicalError;

/// Configuration governing 2D vascular turtle interpretation of an L-system tape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurtleConfig {
    /// Base forward step length $L_0 > 0$ for unit-parameter drawing symbols (`F`, `G`, `A`, `B`).
    pub step_length: f64,
    /// Base yaw turn angle $\delta$ in radians for `+` (CCW) and `-` (CW).
    pub turn_angle_rad: f64,
    /// Initial heading angle in radians (`PI / 2` points vertically upward along $+y$).
    pub initial_heading_rad: f64,
    /// Per-bracket depth length decay factor $d \in (0, 1]$ applied on `[` pushes.
    pub depth_scale: f64,
}

impl Default for TurtleConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            step_length: 1.0,
            turn_angle_rad: 25.0_f64.to_radians(),
            initial_heading_rad: std::f64::consts::FRAC_PI_2,
            depth_scale: 1.0,
        }
    }
}

impl TurtleConfig {
    /// Validates that all geometric parameters are finite and strictly positive where required.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), BotanicalError> {
        if !self.step_length.is_finite() || self.step_length <= 0.0 {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "step_length",
                value: self.step_length,
            });
        }
        if !self.turn_angle_rad.is_finite() {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "turn_angle_rad",
                value: self.turn_angle_rad,
            });
        }
        if !self.initial_heading_rad.is_finite() {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "initial_heading_rad",
                value: self.initial_heading_rad,
            });
        }
        if !self.depth_scale.is_finite() || self.depth_scale <= 0.0 || self.depth_scale > 2.0 {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "depth_scale",
                value: self.depth_scale,
            });
        }
        Ok(())
    }
}

/// Instantaneous 2D turtle state pushed onto the pre-allocated meristem stack at `[` branches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurtleState {
    /// Cartesian $x$ position.
    pub x: f64,
    /// Cartesian $y$ position.
    pub y: f64,
    /// Heading angle in radians.
    pub heading_rad: f64,
    /// Current bracket nesting depth (`0` = main trunk).
    pub depth: u16,
    /// Cumulative depth scaling factor.
    pub scale: f64,
}

/// Single 2D internode segment emitted into [`LSystemEngine`]'s pre-allocated buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurtleSegment {
    /// Segment start $x_0$.
    pub x0: f64,
    /// Segment start $y_0$.
    pub y0: f64,
    /// Segment end $x_1$.
    pub x1: f64,
    /// Segment end $y_1$.
    pub y1: f64,
    /// Branch hierarchy depth (`0` = primary stem).
    pub depth: u16,
}

/// Axis-aligned bounding box of all interpreted 2D [`TurtleSegment`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurtleBounds {
    /// Minimum $x$ coordinate.
    pub min_x: f64,
    /// Maximum $x$ coordinate.
    pub max_x: f64,
    /// Minimum $y$ coordinate.
    pub min_y: f64,
    /// Maximum $y$ coordinate.
    pub max_y: f64,
    /// Maximum bracket depth observed across segments.
    pub max_depth: u16,
}

impl Default for TurtleBounds {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            min_x: 0.0,
            max_x: 0.0,
            min_y: 0.0,
            max_y: 0.0,
            max_depth: 0,
        }
    }
}

impl TurtleBounds {
    /// Horizontal span `max_x - min_x`.
    #[inline]
    #[verified_engine::verified]
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Vertical span `max_y - min_y`.
    #[inline]
    #[verified_engine::verified]
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    #[inline]
    #[verified_engine::verified]
    fn include_point(&mut self, x: f64, y: f64) {
        if x < self.min_x {
            self.min_x = x;
        }
        if x > self.max_x {
            self.max_x = x;
        }
        if y < self.min_y {
            self.min_y = y;
        }
        if y > self.max_y {
            self.max_y = y;
        }
    }
}

impl LSystemEngine {
    /// Interprets the current double-buffered symbol and parameter tapes into pre-allocated
    /// [`TurtleSegment`]s with zero heap allocations.
    ///
    /// Supported turtle symbols:
    /// - `F`, `G`, `A`, `B`: Advance by `step_length * param * state.scale` and emit a segment.
    /// - `f`: Advance without emitting a visible segment.
    /// - `+`: Turn counter-clockwise by `turn_angle_rad * param`.
    /// - `-`: Turn clockwise by `turn_angle_rad * param`.
    /// - `|`: Reverse heading by $\pi$ radians.
    /// - `[`: Push current [`TurtleState`] onto the pre-allocated stack and scale by `depth_scale`.
    /// - `]`: Pop [`TurtleState`] from the stack.
    #[verified_engine::verified]
    pub fn interpret_turtle(&mut self) -> Result<(), BotanicalError> {
        let cfg = self.turtle_config();
        self.segments.clear();
        self.stack.clear();

        let mut state = TurtleState {
            x: 0.0,
            y: 0.0,
            heading_rad: cfg.initial_heading_rad,
            depth: 0,
            scale: 1.0,
        };
        let mut bounds = TurtleBounds::default();

        for i in 0..self.current_symbols.len() {
            let sym = self.current_symbols[i];
            let param = self.current_params[i];
            match sym {
                b'F' | b'G' | b'A' | b'B' => {
                    let len = cfg.step_length * param * state.scale;
                    let (sin_h, cos_h) = state.heading_rad.sin_cos();
                    let nx = state.x + len * cos_h;
                    let ny = state.y + len * sin_h;
                    self.segments.push(TurtleSegment {
                        x0: state.x,
                        y0: state.y,
                        x1: nx,
                        y1: ny,
                        depth: state.depth,
                    });
                    bounds.include_point(nx, ny);
                    if state.depth > bounds.max_depth {
                        bounds.max_depth = state.depth;
                    }
                    state.x = nx;
                    state.y = ny;
                }
                b'f' => {
                    let len = cfg.step_length * param * state.scale;
                    let (sin_h, cos_h) = state.heading_rad.sin_cos();
                    state.x += len * cos_h;
                    state.y += len * sin_h;
                    bounds.include_point(state.x, state.y);
                }
                b'+' => {
                    state.heading_rad += cfg.turn_angle_rad * param;
                }
                b'-' => {
                    state.heading_rad -= cfg.turn_angle_rad * param;
                }
                b'|' => {
                    state.heading_rad += std::f64::consts::PI;
                }
                b'[' => {
                    if self.stack.len() >= MAX_TURTLE_STACK_DEPTH {
                        return Err(BotanicalError::TurtleStackOverflow {
                            max_depth: MAX_TURTLE_STACK_DEPTH,
                        });
                    }
                    self.stack.push(state);
                    state.depth = state.depth.saturating_add(1);
                    state.scale *= cfg.depth_scale;
                }
                b']' => {
                    state = self
                        .stack
                        .pop()
                        .ok_or(BotanicalError::UnmatchedPopBracket)?;
                }
                _ => {}
            }
        }

        self.bounds = bounds;
        Ok(())
    }
}

//! Bounded iterative Lindenmayer systems (L-systems) and nutrient-modulated 2D turtle geometry.

use super::vogel::NutrientCoupling;
use std::f64::consts::FRAC_PI_2;

/// Canonical botanical L-system production rule presets from Prusinkiewicz & Lindenmayer (1990).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LSystemPreset {
    /// Axial monopodial shoot (`X -> F+[[X]-X]-F[-FX]+X`, `F -> FF`, $\delta = 25^\circ$).
    #[default]
    BotanicalShoot,
    /// Sympodial bush/tree (`F -> FF+[+F-F-F]-[-F+F+F]`, $\delta = 22.5^\circ$).
    SympodialTree,
    /// Bilateral fern frond (`X -> F[-X][+X]FX`, `F -> FF`, $\delta = 25.7^\circ$).
    MonopodialFern,
}

impl LSystemPreset {
    /// Axiom byte sequence for the botanical preset.
    #[verified_engine::verified]
    pub fn axiom(self) -> &'static [u8] {
        match self {
            Self::BotanicalShoot | Self::MonopodialFern => b"X",
            Self::SympodialTree => b"F",
        }
    }

    /// Default branching angle in degrees.
    #[verified_engine::verified]
    pub fn default_angle_deg(self) -> f64 {
        match self {
            Self::BotanicalShoot => 25.0,
            Self::SympodialTree => 22.5,
            Self::MonopodialFern => 25.7,
        }
    }

    #[verified_engine::verified]
    fn expand_symbol(self, sym: u8) -> &'static [u8] {
        match (self, sym) {
            (Self::BotanicalShoot, b'X') => b"F+[[X]-X]-F[-FX]+X",
            (Self::BotanicalShoot, b'F') | (Self::MonopodialFern, b'F') => b"FF",
            (Self::SympodialTree, b'F') => b"FF+[+F-F-F]-[-F+F+F]",
            (Self::MonopodialFern, b'X') => b"F[-X][+X]FX",
            _ => &[],
        }
    }
}

/// Configuration for iterative L-system derivation and 2D turtle interpretation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LSystemConfig {
    /// Production rule preset.
    pub preset: LSystemPreset,
    /// Number of parallel rewriting iterations (bounded to `0..=7`).
    pub iterations: usize,
    /// Base internode length $L_0 > 0$.
    pub step_length: f64,
    /// Base branching angle $\delta_0$ in degrees.
    pub branch_angle_deg: f64,
}

impl Default for LSystemConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            preset: LSystemPreset::BotanicalShoot,
            iterations: 4,
            step_length: 0.35,
            branch_angle_deg: 25.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TurtlePose {
    x: f64,
    y: f64,
    heading: f64,
}

/// Preallocated, zero-allocation L-system derivation and 2D turtle segment generator.
#[derive(Debug, Clone)]
pub struct LSystemEngine {
    max_symbols: usize,
    max_segments: usize,
    max_stack_depth: usize,
    symbols_a: Vec<u8>,
    symbols_b: Vec<u8>,
    active_symbols_len: usize,
    turtle_stack: Vec<TurtlePose>,
    segments: Vec<[f64; 4]>,
    active_segments_len: usize,
}

impl LSystemEngine {
    /// Allocates fixed symbol, stack, and segment buffers so subsequent updates never allocate.
    #[verified_engine::verified]
    pub fn new(max_symbols: usize, max_segments: usize, max_stack_depth: usize) -> Self {
        let sym_cap = max_symbols.max(64);
        let seg_cap = max_segments.max(32);
        let stack_cap = max_stack_depth.max(8);
        Self {
            max_symbols: sym_cap,
            max_segments: seg_cap,
            max_stack_depth: stack_cap,
            symbols_a: vec![0u8; sym_cap],
            symbols_b: vec![0u8; sym_cap],
            active_symbols_len: 0,
            turtle_stack: vec![TurtlePose::default(); stack_cap],
            segments: vec![[0.0; 4]; seg_cap],
            active_segments_len: 0,
        }
    }

    /// Rewrites the L-system and interprets 2D turtle segments scaled by `nutrient` with **zero heap allocations**.
    #[verified_engine::verified]
    pub fn generate(&mut self, config: LSystemConfig, nutrient: &NutrientCoupling) {
        self.rewrite_in_place(config.preset, config.iterations.min(7));
        let rate = nutrient.normalized_rate();
        let step_scale = (0.35 + 0.95 * rate) / (1.8_f64.powi(config.iterations.min(7) as i32));
        let step_len = config.step_length.max(1e-4) * step_scale * 8.0;
        let angle_rad = (config.branch_angle_deg * (0.75 + 0.45 * rate)).to_radians();
        self.interpret_turtle_in_place(step_len, angle_rad);
    }

    #[verified_engine::verified]
    fn rewrite_in_place(&mut self, preset: LSystemPreset, iterations: usize) {
        let axiom = preset.axiom();
        let init_len = axiom.len().min(self.max_symbols);
        self.symbols_a[..init_len].copy_from_slice(&axiom[..init_len]);
        let mut cur_len = init_len;

        for _ in 0..iterations {
            let mut next_len = 0usize;
            for idx in 0..cur_len {
                let sym = self.symbols_a[idx];
                let repl = preset.expand_symbol(sym);
                let bytes = if repl.is_empty() {
                    std::slice::from_ref(&sym)
                } else {
                    repl
                };
                if next_len + bytes.len() > self.max_symbols {
                    break;
                }
                self.symbols_b[next_len..next_len + bytes.len()].copy_from_slice(bytes);
                next_len += bytes.len();
            }
            std::mem::swap(&mut self.symbols_a, &mut self.symbols_b);
            cur_len = next_len;
        }
        self.active_symbols_len = cur_len;
    }

    #[verified_engine::verified]
    fn interpret_turtle_in_place(&mut self, step_len: f64, angle_rad: f64) {
        let mut pose = TurtlePose {
            x: 0.0,
            y: 0.0,
            heading: FRAC_PI_2,
        };
        let mut stack_top = 0usize;
        let mut seg_count = 0usize;

        for idx in 0..self.active_symbols_len {
            match self.symbols_a[idx] {
                b'F' => {
                    let (sin_h, cos_h) = pose.heading.sin_cos();
                    let nx = pose.x + step_len * cos_h;
                    let ny = pose.y + step_len * sin_h;
                    if seg_count < self.max_segments {
                        self.segments[seg_count] = [pose.x, pose.y, nx, ny];
                        seg_count += 1;
                    }
                    pose.x = nx;
                    pose.y = ny;
                }
                b'+' => pose.heading += angle_rad,
                b'-' => pose.heading -= angle_rad,
                b'[' if stack_top < self.max_stack_depth => {
                    self.turtle_stack[stack_top] = pose;
                    stack_top += 1;
                }
                b']' if stack_top > 0 => {
                    stack_top -= 1;
                    pose = self.turtle_stack[stack_top];
                }
                _ => {}
            }
        }
        self.active_segments_len = seg_count;
    }

    /// Generated 2D line segments `[x0, y0, x1, y1]`.
    #[verified_engine::verified]
    pub fn segments(&self) -> &[[f64; 4]] {
        &self.segments[..self.active_segments_len]
    }

    /// Rewritten byte string produced by the bounded parallel grammar.
    #[verified_engine::verified]
    pub fn symbols(&self) -> &[u8] {
        &self.symbols_a[..self.active_symbols_len]
    }
}

impl Default for LSystemEngine {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(16_384, 16_384, 128)
    }
}

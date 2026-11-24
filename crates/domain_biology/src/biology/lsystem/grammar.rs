//! Fixed-capacity D0L and parametric L-system grammar and double-buffered derivation engine.

use super::turtle::{TurtleBounds, TurtleConfig, TurtleSegment, TurtleState};
use crate::error::BotanicalError;

/// Hard upper bound on symbol count in the double-buffered byte tape ($2^{16} = 65{,}536$).
pub const MAX_LSYSTEM_SYMBOLS: usize = 65_536;

/// Maximum number of production rules in a stack-allocated [`LSystemGrammar`].
pub const MAX_LSYSTEM_RULES: usize = 16;

/// Maximum number of successor tokens emitted by a single production rule or axiom.
pub const MAX_RULE_EXPANSION: usize = 32;

/// Maximum bracket nesting depth `[...]` for the pre-allocated 2D turtle stack.
pub const MAX_TURTLE_STACK_DEPTH: usize = 512;

/// Single module token in a production successor or axiom: ASCII symbol byte plus affine
/// parameter map $p' = p \cdot \text{param\_scale} + \text{param\_offset}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpansionToken {
    /// ASCII byte representing the L-system module (`b'F'`, `b'G'`, `b'X'`, `b'+'`, `b'-'`, `b'['`, `b']'`).
    pub symbol: u8,
    /// Multiplicative scale applied to the parent module's real parameter.
    pub param_scale: f64,
    /// Additive offset applied after scaling the parent module's real parameter.
    pub param_offset: f64,
}

impl ExpansionToken {
    /// Creates a D0L unit token preserving parameter scale `1.0`.
    #[verified_engine::verified]
    pub fn d0l(symbol: u8) -> Self {
        Self {
            symbol,
            param_scale: 1.0,
            param_offset: 0.0,
        }
    }

    /// Creates a parametric token mapping parent parameter $p \mapsto p \cdot \text{scale} + \text{offset}$.
    #[verified_engine::verified]
    pub fn parametric(symbol: u8, param_scale: f64, param_offset: f64) -> Self {
        Self {
            symbol,
            param_scale,
            param_offset,
        }
    }
}

/// Stack-allocated deterministic or parametric production rule `predecessor(p : p >= min_param) -> tokens`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductionRule {
    /// Predecessor ASCII byte matched during parallel rewriting.
    pub predecessor: u8,
    /// Minimum parameter threshold required for this production to fire (`0.0` for unconditional D0L).
    pub min_param: f64,
    /// Fixed-capacity array of successor tokens.
    pub tokens: [ExpansionToken; MAX_RULE_EXPANSION],
    /// Active number of successor tokens in `tokens`.
    pub token_count: usize,
}

impl ProductionRule {
    /// Builds a D0L production rule from an ASCII predecessor and ASCII expansion byte slice.
    #[verified_engine::verified]
    pub fn from_ascii(predecessor: u8, expansion: &[u8]) -> Result<Self, BotanicalError> {
        if expansion.len() > MAX_RULE_EXPANSION {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "rule_expansion_len",
                value: expansion.len() as f64,
            });
        }
        let mut tokens = [ExpansionToken::d0l(b' '); MAX_RULE_EXPANSION];
        for (idx, &b) in expansion.iter().enumerate() {
            tokens[idx] = ExpansionToken::d0l(b);
        }
        Ok(Self {
            predecessor,
            min_param: 0.0,
            tokens,
            token_count: expansion.len(),
        })
    }

    /// Builds a parametric production rule with an explicit minimum parameter guard and token slice.
    #[verified_engine::verified]
    pub fn from_tokens(
        predecessor: u8,
        min_param: f64,
        expansion: &[ExpansionToken],
    ) -> Result<Self, BotanicalError> {
        if !min_param.is_finite() || min_param < 0.0 {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "min_param",
                value: min_param,
            });
        }
        if expansion.len() > MAX_RULE_EXPANSION {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "rule_expansion_len",
                value: expansion.len() as f64,
            });
        }
        let mut tokens = [ExpansionToken::d0l(b' '); MAX_RULE_EXPANSION];
        for (idx, &tok) in expansion.iter().enumerate() {
            if !tok.param_scale.is_finite() || !tok.param_offset.is_finite() {
                return Err(BotanicalError::InvalidGrammarParameter {
                    name: "token_param",
                    value: tok.param_scale + tok.param_offset,
                });
            }
            tokens[idx] = tok;
        }
        Ok(Self {
            predecessor,
            min_param,
            tokens,
            token_count: expansion.len(),
        })
    }
}

/// Stack-allocated deterministic/parametric L-system grammar specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LSystemGrammar {
    /// Axiom symbol bytes.
    pub axiom_symbols: [u8; MAX_RULE_EXPANSION],
    /// Initial parameter values paired with `axiom_symbols`.
    pub axiom_params: [f64; MAX_RULE_EXPANSION],
    /// Active length of the axiom (`1..=MAX_RULE_EXPANSION`).
    pub axiom_len: usize,
    /// Fixed-capacity table of production rules evaluated in order.
    pub rules: [ProductionRule; MAX_LSYSTEM_RULES],
    /// Number of active rules in `rules`.
    pub rule_count: usize,
}

impl LSystemGrammar {
    /// Creates a grammar initialized with an ASCII axiom and unit (`1.0`) parameters.
    #[verified_engine::verified]
    pub fn new_ascii(axiom: &[u8]) -> Result<Self, BotanicalError> {
        if axiom.is_empty() {
            return Err(BotanicalError::EmptyAxiom);
        }
        if axiom.len() > MAX_RULE_EXPANSION {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "axiom_len",
                value: axiom.len() as f64,
            });
        }
        let mut axiom_symbols = [b' '; MAX_RULE_EXPANSION];
        let axiom_params = [1.0_f64; MAX_RULE_EXPANSION];
        for (idx, &b) in axiom.iter().enumerate() {
            axiom_symbols[idx] = b;
        }
        let empty_rule = ProductionRule {
            predecessor: 0,
            min_param: 0.0,
            tokens: [ExpansionToken::d0l(b' '); MAX_RULE_EXPANSION],
            token_count: 0,
        };
        Ok(Self {
            axiom_symbols,
            axiom_params,
            axiom_len: axiom.len(),
            rules: [empty_rule; MAX_LSYSTEM_RULES],
            rule_count: 0,
        })
    }

    /// Appends a [`ProductionRule`] to the grammar.
    #[verified_engine::verified]
    pub fn add_rule(&mut self, rule: ProductionRule) -> Result<(), BotanicalError> {
        if self.rule_count >= MAX_LSYSTEM_RULES {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "rule_count",
                value: self.rule_count as f64,
            });
        }
        self.rules[self.rule_count] = rule;
        self.rule_count += 1;
        Ok(())
    }

    /// Finds the first matching production rule for `(symbol, param)`.
    #[inline]
    #[verified_engine::verified]
    pub fn match_rule(&self, symbol: u8, param: f64) -> Option<&ProductionRule> {
        self.rules[..self.rule_count]
            .iter()
            .find(|r| r.predecessor == symbol && param >= r.min_param)
    }
}

/// Zero-allocation double-buffered L-system derivation and 2D turtle engine.
#[derive(Debug, Clone)]
pub struct LSystemEngine {
    grammar: LSystemGrammar,
    turtle_config: TurtleConfig,
    tape_capacity: usize,
    generation: usize,
    pub(super) current_symbols: Vec<u8>,
    next_symbols: Vec<u8>,
    pub(super) current_params: Vec<f64>,
    next_params: Vec<f64>,
    pub(super) stack: Vec<TurtleState>,
    pub(super) segments: Vec<TurtleSegment>,
    pub(super) bounds: TurtleBounds,
}

impl LSystemEngine {
    /// Pre-allocates double-buffered byte/parameter tapes, turtle state stack, and 2D segment
    /// buffers up to `tape_capacity` (clamped at `1..=MAX_LSYSTEM_SYMBOLS`).
    #[verified_engine::verified(
        alloc_ok = "constructor pre-allocates double-buffered symbol tapes"
    )]
    pub fn new(
        grammar: LSystemGrammar,
        turtle_config: TurtleConfig,
        tape_capacity: usize,
    ) -> Result<Self, BotanicalError> {
        turtle_config.validate()?;
        if tape_capacity == 0 || tape_capacity > MAX_LSYSTEM_SYMBOLS {
            return Err(BotanicalError::SymbolTapeOverflow {
                requested: tape_capacity,
                max: MAX_LSYSTEM_SYMBOLS,
            });
        }
        if grammar.axiom_len == 0 {
            return Err(BotanicalError::EmptyAxiom);
        }
        if grammar.axiom_len > tape_capacity {
            return Err(BotanicalError::SymbolTapeOverflow {
                requested: grammar.axiom_len,
                max: tape_capacity,
            });
        }

        let mut engine = Self {
            grammar,
            turtle_config,
            tape_capacity,
            generation: 0,
            current_symbols: Vec::with_capacity(tape_capacity),
            next_symbols: Vec::with_capacity(tape_capacity),
            current_params: Vec::with_capacity(tape_capacity),
            next_params: Vec::with_capacity(tape_capacity),
            stack: Vec::with_capacity(MAX_TURTLE_STACK_DEPTH),
            segments: Vec::with_capacity(tape_capacity),
            bounds: TurtleBounds::default(),
        };
        engine.reset_to_axiom()?;
        Ok(engine)
    }

    /// Resets the double-buffered symbol tape back to generation `0` (axiom) and re-interprets
    /// turtle segments with zero heap allocations.
    #[verified_engine::verified]
    pub fn reset_to_axiom(&mut self) -> Result<(), BotanicalError> {
        self.current_symbols.clear();
        self.current_params.clear();
        for i in 0..self.grammar.axiom_len {
            self.current_symbols.push(self.grammar.axiom_symbols[i]);
            self.current_params.push(self.grammar.axiom_params[i]);
        }
        self.generation = 0;
        self.interpret_turtle()
    }

    /// Updates grammar and turtle configuration in place and resets to generation `0` without reallocating.
    #[verified_engine::verified]
    pub fn reconfigure(
        &mut self,
        grammar: LSystemGrammar,
        turtle_config: TurtleConfig,
    ) -> Result<(), BotanicalError> {
        turtle_config.validate()?;
        if grammar.axiom_len == 0 {
            return Err(BotanicalError::EmptyAxiom);
        }
        if grammar.axiom_len > self.tape_capacity {
            return Err(BotanicalError::SymbolTapeOverflow {
                requested: grammar.axiom_len,
                max: self.tape_capacity,
            });
        }
        self.grammar = grammar;
        self.turtle_config = turtle_config;
        self.reset_to_axiom()
    }

    /// Advances the L-system by one parallel derivation step on the double-buffered tape and
    /// refreshes the 2D turtle segment buffer. Returns `Err(SymbolTapeOverflow)` without
    /// mutating the active generation if expansion would exceed `tape_capacity`.
    #[verified_engine::verified]
    pub fn step(&mut self) -> Result<usize, BotanicalError> {
        self.next_symbols.clear();
        self.next_params.clear();

        for idx in 0..self.current_symbols.len() {
            let sym = self.current_symbols[idx];
            let param = self.current_params[idx];
            if let Some(rule) = self.grammar.match_rule(sym, param) {
                let needed = self.next_symbols.len() + rule.token_count;
                if needed > self.tape_capacity {
                    return Err(BotanicalError::SymbolTapeOverflow {
                        requested: needed,
                        max: self.tape_capacity,
                    });
                }
                for tok in &rule.tokens[..rule.token_count] {
                    self.next_symbols.push(tok.symbol);
                    self.next_params
                        .push(param * tok.param_scale + tok.param_offset);
                }
            } else {
                let needed = self.next_symbols.len() + 1;
                if needed > self.tape_capacity {
                    return Err(BotanicalError::SymbolTapeOverflow {
                        requested: needed,
                        max: self.tape_capacity,
                    });
                }
                self.next_symbols.push(sym);
                self.next_params.push(param);
            }
        }

        std::mem::swap(&mut self.current_symbols, &mut self.next_symbols);
        std::mem::swap(&mut self.current_params, &mut self.next_params);
        self.generation += 1;
        self.interpret_turtle()?;
        Ok(self.current_symbols.len())
    }

    /// Resets to the axiom and derives up to `target_generations`, halting cleanly at the last
    /// complete generation that fits inside `tape_capacity`. Returns the achieved generation count.
    #[verified_engine::verified]
    pub fn derive_bounded(&mut self, target_generations: usize) -> Result<usize, BotanicalError> {
        self.reset_to_axiom()?;
        for _ in 0..target_generations {
            if self.step().is_err() {
                break;
            }
        }
        Ok(self.generation)
    }

    /// Active symbol byte slice in the current derivation tape.
    #[inline]
    #[verified_engine::verified]
    pub fn symbols(&self) -> &[u8] {
        &self.current_symbols
    }

    /// Active module parameter slice paired 1:1 with [`Self::symbols`].
    #[inline]
    #[verified_engine::verified]
    pub fn params(&self) -> &[f64] {
        &self.current_params
    }

    /// Pre-allocated 2D turtle line segments interpreted from the current tape.
    #[inline]
    #[verified_engine::verified]
    pub fn segments(&self) -> &[TurtleSegment] {
        &self.segments
    }

    /// Axis-aligned 2D bounding box enclosing [`Self::segments`].
    #[inline]
    #[verified_engine::verified]
    pub fn bounds(&self) -> TurtleBounds {
        self.bounds
    }

    /// Current completed derivation generation index.
    #[inline]
    #[verified_engine::verified]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Active [`TurtleConfig`] governing step length, branching angle, and contraction.
    #[inline]
    #[verified_engine::verified]
    pub fn turtle_config(&self) -> TurtleConfig {
        self.turtle_config
    }

    /// Active [`LSystemGrammar`].
    #[inline]
    #[verified_engine::verified]
    pub fn grammar(&self) -> &LSystemGrammar {
        &self.grammar
    }
}

//! Canonical botanical D0L and parametric L-system presets (Prusinkiewicz & Lindenmayer 1990).

use super::grammar::{
    ExpansionToken, LSystemEngine, LSystemGrammar, MAX_LSYSTEM_SYMBOLS, ProductionRule,
};
use super::turtle::TurtleConfig;
use crate::error::BotanicalError;

/// Canonical botanical and vascular L-system architectures.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BotanicalLSystemPreset {
    /// Deterministic D0L herbaceous shoot (`X -> F+[[X]-X]-F[-FX]+X`, `F -> FF`, $\delta = 25^\circ$).
    #[default]
    HerbaceousShoot,
    /// Parametric dichotomous tree (`A(l) -> F(l)[+A(r_1 l)][-A(r_2 l)]`) with asymmetric contraction.
    ParametricDichotomousTree,
    /// Parametric monopodial conifer/fern axis (`A(l) -> F(l)[+B(r_2 l)][-B(r_2 l)]A(r_1 l)`).
    MonopodialShoot,
    /// Sympodial bush architecture (`F -> FF+[+F-F-F]-[-F+F+F]`, $\delta = 22.5^\circ$).
    SympodialBush,
    /// Space-filling vascular leaf venation dragon curve (`X -> X+YF+`, `Y -> -FX-Y`, $\delta = 90^\circ$).
    DragonVenation,
}

impl BotanicalLSystemPreset {
    /// All canonical botanical L-system presets.
    pub const ALL: [Self; 5] = [
        Self::HerbaceousShoot,
        Self::ParametricDichotomousTree,
        Self::MonopodialShoot,
        Self::SympodialBush,
        Self::DragonVenation,
    ];

    /// Human-readable display label.
    #[verified_engine::verified]
    pub fn label(self) -> &'static str {
        match self {
            Self::HerbaceousShoot => "Herbaceous Shoot (D0L)",
            Self::ParametricDichotomousTree => "Parametric Dichotomous Canopy",
            Self::MonopodialShoot => "Monopodial Apical Axis",
            Self::SympodialBush => "Sympodial Woody Bush",
            Self::DragonVenation => "Dragon Leaf Venation",
        }
    }

    /// Recommended default derivation depth balancing visual richness and fast interactivity.
    #[verified_engine::verified]
    pub fn default_generations(self) -> usize {
        match self {
            Self::HerbaceousShoot => 5,
            Self::ParametricDichotomousTree => 8,
            Self::MonopodialShoot => 7,
            Self::SympodialBush => 4,
            Self::DragonVenation => 10,
        }
    }

    /// Default branching turn angle in degrees.
    #[verified_engine::verified]
    pub fn default_angle_deg(self) -> f64 {
        match self {
            Self::HerbaceousShoot => 25.0,
            Self::ParametricDichotomousTree => 30.0,
            Self::MonopodialShoot => 32.0,
            Self::SympodialBush => 22.5,
            Self::DragonVenation => 90.0,
        }
    }

    /// Default primary contraction ratio $r_1 \in (0, 1]$ for parametric presets.
    #[verified_engine::verified]
    pub fn default_contraction_primary(self) -> f64 {
        match self {
            Self::ParametricDichotomousTree => 0.75,
            Self::MonopodialShoot => 0.84,
            _ => 0.82,
        }
    }

    /// Default lateral contraction ratio $r_2 \in (0, 1]$ for parametric presets.
    #[verified_engine::verified]
    pub fn default_contraction_lateral(self) -> f64 {
        match self {
            Self::ParametricDichotomousTree => 0.68,
            Self::MonopodialShoot => 0.56,
            _ => 0.72,
        }
    }

    /// Constructs the [`LSystemGrammar`] and [`TurtleConfig`] parameterized by branching angle
    /// (in degrees) and primary/lateral contraction ratios `(r1, r2)`.
    #[verified_engine::verified]
    pub fn build_grammar(
        self,
        angle_deg: f64,
        contraction_r1: f64,
        contraction_r2: f64,
    ) -> Result<(LSystemGrammar, TurtleConfig), BotanicalError> {
        if !angle_deg.is_finite()
            || !contraction_r1.is_finite()
            || contraction_r1 <= 0.0
            || !contraction_r2.is_finite()
            || contraction_r2 <= 0.0
        {
            return Err(BotanicalError::InvalidGrammarParameter {
                name: "preset_contraction_or_angle",
                value: angle_deg + contraction_r1 + contraction_r2,
            });
        }
        let cfg = TurtleConfig {
            step_length: 1.0,
            turn_angle_rad: angle_deg.to_radians(),
            initial_heading_rad: std::f64::consts::FRAC_PI_2,
            depth_scale: 1.0,
        };

        match self {
            Self::HerbaceousShoot => {
                let mut g = LSystemGrammar::new_ascii(b"X")?;
                g.add_rule(ProductionRule::from_ascii(b'X', b"F+[[X]-X]-F[-FX]+X")?)?;
                g.add_rule(ProductionRule::from_ascii(b'F', b"FF")?)?;
                Ok((g, cfg))
            }
            Self::ParametricDichotomousTree => {
                let mut g = LSystemGrammar::new_ascii(b"A")?;
                let tokens = [
                    ExpansionToken::parametric(b'F', 1.0, 0.0),
                    ExpansionToken::d0l(b'['),
                    ExpansionToken::d0l(b'+'),
                    ExpansionToken::parametric(b'A', contraction_r1, 0.0),
                    ExpansionToken::d0l(b']'),
                    ExpansionToken::d0l(b'['),
                    ExpansionToken::d0l(b'-'),
                    ExpansionToken::parametric(b'A', contraction_r2, 0.0),
                    ExpansionToken::d0l(b']'),
                ];
                g.add_rule(ProductionRule::from_tokens(b'A', 0.02, &tokens)?)?;
                Ok((g, cfg))
            }
            Self::MonopodialShoot => {
                let mut g = LSystemGrammar::new_ascii(b"A")?;
                let apical = [
                    ExpansionToken::parametric(b'F', 1.0, 0.0),
                    ExpansionToken::d0l(b'['),
                    ExpansionToken::d0l(b'+'),
                    ExpansionToken::parametric(b'B', contraction_r2, 0.0),
                    ExpansionToken::d0l(b']'),
                    ExpansionToken::d0l(b'['),
                    ExpansionToken::d0l(b'-'),
                    ExpansionToken::parametric(b'B', contraction_r2, 0.0),
                    ExpansionToken::d0l(b']'),
                    ExpansionToken::parametric(b'A', contraction_r1, 0.0),
                ];
                let lateral = [
                    ExpansionToken::parametric(b'F', 1.0, 0.0),
                    ExpansionToken::d0l(b'['),
                    ExpansionToken::d0l(b'+'),
                    ExpansionToken::parametric(b'F', contraction_r2, 0.0),
                    ExpansionToken::d0l(b']'),
                    ExpansionToken::parametric(b'B', contraction_r1, 0.0),
                ];
                g.add_rule(ProductionRule::from_tokens(b'A', 0.02, &apical)?)?;
                g.add_rule(ProductionRule::from_tokens(b'B', 0.04, &lateral)?)?;
                Ok((g, cfg))
            }
            Self::SympodialBush => {
                let mut g = LSystemGrammar::new_ascii(b"F")?;
                g.add_rule(ProductionRule::from_ascii(b'F', b"FF+[+F-F-F]-[-F+F+F]")?)?;
                Ok((g, cfg))
            }
            Self::DragonVenation => {
                let mut g = LSystemGrammar::new_ascii(b"FX")?;
                g.add_rule(ProductionRule::from_ascii(b'X', b"X+YF+")?)?;
                g.add_rule(ProductionRule::from_ascii(b'Y', b"-FX-Y")?)?;
                Ok((g, cfg))
            }
        }
    }

    /// Instantiates a pre-allocated [`LSystemEngine`] derived to [`Self::default_generations`].
    #[verified_engine::verified]
    pub fn build_engine(self) -> Result<LSystemEngine, BotanicalError> {
        let (grammar, cfg) = self.build_grammar(
            self.default_angle_deg(),
            self.default_contraction_primary(),
            self.default_contraction_lateral(),
        )?;
        let mut engine = LSystemEngine::new(grammar, cfg, MAX_LSYSTEM_SYMBOLS)?;
        engine.derive_bounded(self.default_generations())?;
        Ok(engine)
    }
}

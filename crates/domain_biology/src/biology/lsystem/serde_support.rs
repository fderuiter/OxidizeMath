// [cite:biology]

use super::grammar::{
    ExpansionToken, LSystemGrammar, MAX_LSYSTEM_RULES, MAX_RULE_EXPANSION, ProductionRule,
};
use super::turtle::TurtleConfig;
use crate::error::BotanicalError;

impl<'de> serde::Deserialize<'de> for ExpansionToken {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawExpansionToken {
            symbol: u8,
            param_scale: f64,
            param_offset: f64,
        }
        let raw = RawExpansionToken::deserialize(deserializer)?;
        if !raw.param_scale.is_finite() || !raw.param_offset.is_finite() {
            return Err(serde::de::Error::custom(
                BotanicalError::InvalidGrammarParameter {
                    name: "token_param",
                    value: raw.param_scale + raw.param_offset,
                },
            ));
        }
        Ok(Self::parametric(
            raw.symbol,
            raw.param_scale,
            raw.param_offset,
        ))
    }
}

impl<'de> serde::Deserialize<'de> for ProductionRule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawProductionRule {
            predecessor: u8,
            min_param: f64,
            tokens: [ExpansionToken; MAX_RULE_EXPANSION],
            token_count: usize,
        }
        let raw = RawProductionRule::deserialize(deserializer)?;
        if raw.token_count > MAX_RULE_EXPANSION {
            return Err(serde::de::Error::custom(
                BotanicalError::InvalidGrammarParameter {
                    name: "rule_expansion_len",
                    value: raw.token_count as f64,
                },
            ));
        }
        Self::from_tokens(
            raw.predecessor,
            raw.min_param,
            &raw.tokens[..raw.token_count],
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<'de> serde::Deserialize<'de> for LSystemGrammar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawLSystemGrammar {
            axiom_symbols: [u8; MAX_RULE_EXPANSION],
            axiom_params: [f64; MAX_RULE_EXPANSION],
            axiom_len: usize,
            rules: [ProductionRule; MAX_LSYSTEM_RULES],
            rule_count: usize,
        }
        let raw = RawLSystemGrammar::deserialize(deserializer)?;
        if raw.axiom_len == 0 {
            return Err(serde::de::Error::custom(BotanicalError::EmptyAxiom));
        }
        if raw.axiom_len > MAX_RULE_EXPANSION {
            return Err(serde::de::Error::custom(
                BotanicalError::InvalidGrammarParameter {
                    name: "axiom_len",
                    value: raw.axiom_len as f64,
                },
            ));
        }
        if let Some(&bad) = raw.axiom_params.iter().find(|p| !p.is_finite()) {
            return Err(serde::de::Error::custom(
                BotanicalError::InvalidGrammarParameter {
                    name: "axiom_params",
                    value: bad,
                },
            ));
        }
        if raw.rule_count > MAX_LSYSTEM_RULES {
            return Err(serde::de::Error::custom(
                BotanicalError::InvalidGrammarParameter {
                    name: "rule_count",
                    value: raw.rule_count as f64,
                },
            ));
        }
        Ok(Self {
            axiom_symbols: raw.axiom_symbols,
            axiom_params: raw.axiom_params,
            axiom_len: raw.axiom_len,
            rules: raw.rules,
            rule_count: raw.rule_count,
        })
    }
}

impl<'de> serde::Deserialize<'de> for TurtleConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawTurtleConfig {
            step_length: f64,
            turn_angle_rad: f64,
            initial_heading_rad: f64,
            depth_scale: f64,
        }
        let raw = RawTurtleConfig::deserialize(deserializer)?;
        let cfg = Self {
            step_length: raw.step_length,
            turn_angle_rad: raw.turn_angle_rad,
            initial_heading_rad: raw.initial_heading_rad,
            depth_scale: raw.depth_scale,
        };
        cfg.validate().map_err(serde::de::Error::custom)?;
        Ok(cfg)
    }
}

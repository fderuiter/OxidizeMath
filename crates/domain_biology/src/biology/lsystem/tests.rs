use super::*;
use crate::error::BotanicalError;
use scientific_metadata::theory::TheoryDescribable;

#[test]
#[verified_engine::verified]
fn test_d0l_algae_fibonacci_string_lengths() {
    let mut grammar = LSystemGrammar::new_ascii(b"A").unwrap();
    grammar
        .add_rule(ProductionRule::from_ascii(b'A', b"AB").unwrap())
        .unwrap();
    grammar
        .add_rule(ProductionRule::from_ascii(b'B', b"A").unwrap())
        .unwrap();

    let mut engine = LSystemEngine::new(grammar, TurtleConfig::default(), 1024).unwrap();
    let expected = [1_usize, 2, 3, 5, 8, 13, 21, 34, 55];
    assert_eq!(engine.symbols().len(), expected[0]);
    for &exp_len in &expected[1..] {
        let len = engine.step().unwrap();
        assert_eq!(len, exp_len);
    }
}

#[test]
#[verified_engine::verified]
fn test_parametric_dichotomous_tree_contraction_and_segments() {
    let preset = BotanicalLSystemPreset::ParametricDichotomousTree;
    let (grammar, cfg) = preset.build_grammar(30.0, 0.75, 0.5).unwrap();
    let mut engine = LSystemEngine::new(grammar, cfg, MAX_LSYSTEM_SYMBOLS).unwrap();
    assert_eq!(engine.segments().len(), 1);

    engine.step().unwrap();
    // Generation 1: F(1.0)[+A(0.75)][-A(0.5)] -> 3 visible segments (F, A, A)
    assert_eq!(engine.segments().len(), 3);
    let s0 = engine.segments()[0];
    let s1 = engine.segments()[1];
    let s2 = engine.segments()[2];
    let len0 = ((s0.x1 - s0.x0).hypot(s0.y1 - s0.y0) - 1.0).abs();
    let len1 = ((s1.x1 - s1.x0).hypot(s1.y1 - s1.y0) - 0.75).abs();
    let len2 = ((s2.x1 - s2.x0).hypot(s2.y1 - s2.y0) - 0.50).abs();
    assert!(len0 < 1e-12 && len1 < 1e-12 && len2 < 1e-12);
    assert!(engine.bounds().height() > 1.3);
    assert!(engine.bounds().width() > 0.5);
}

#[test]
#[verified_engine::verified]
fn test_symbol_tape_capped_at_65536_and_overflow_guarded() {
    assert!(
        LSystemEngine::new(
            LSystemGrammar::new_ascii(b"F").unwrap(),
            TurtleConfig::default(),
            MAX_LSYSTEM_SYMBOLS + 1
        )
        .is_err()
    );

    let mut grammar = LSystemGrammar::new_ascii(b"F").unwrap();
    grammar
        .add_rule(ProductionRule::from_ascii(b'F', b"FFFF").unwrap())
        .unwrap();
    let mut engine = LSystemEngine::new(grammar, TurtleConfig::default(), MAX_LSYSTEM_SYMBOLS)
        .expect("65536 capacity must succeed");

    // 4^8 = 65536 fits exactly; 4^9 = 262144 must return SymbolTapeOverflow without corrupting generation 8.
    for step_idx in 1..=8 {
        let count = engine.step().unwrap();
        assert_eq!(count, 4_usize.pow(step_idx));
    }
    assert_eq!(engine.symbols().len(), MAX_LSYSTEM_SYMBOLS);
    let err = engine.step().unwrap_err();
    assert!(matches!(
        err,
        BotanicalError::SymbolTapeOverflow {
            max: MAX_LSYSTEM_SYMBOLS,
            ..
        }
    ));
    assert_eq!(engine.generation(), 8);
    assert_eq!(engine.symbols().len(), MAX_LSYSTEM_SYMBOLS);

    // derive_bounded halts cleanly at generation 8
    let achieved = engine.derive_bounded(12).unwrap();
    assert_eq!(achieved, 8);
}

#[test]
#[verified_engine::verified]
fn test_all_botanical_presets_and_theory_metadata() {
    for preset in BotanicalLSystemPreset::ALL {
        let engine = preset.build_engine().unwrap();
        assert!(!engine.segments().is_empty());
        assert!(engine.bounds().width() > 0.0 || engine.bounds().height() > 0.0);
        assert!(!engine.theory_description().is_empty());
        assert!(!engine.phonetic_description().is_empty());
        assert!(!engine.theory_citation().contains("[cite:"));
    }
}

#[cfg(feature = "serde")]
#[test]
#[verified_engine::verified]
fn test_lsystem_serde_roundtrip_and_invariant_validation() {
    for preset in BotanicalLSystemPreset::ALL {
        let preset_json = serde_json::to_string(&preset).unwrap();
        let restored_preset: BotanicalLSystemPreset = serde_json::from_str(&preset_json).unwrap();
        assert_eq!(preset, restored_preset);

        let (grammar, cfg) = preset
            .build_grammar(
                preset.default_angle_deg(),
                preset.default_contraction_primary(),
                preset.default_contraction_lateral(),
            )
            .unwrap();
        let grammar_json = serde_json::to_string(&grammar).unwrap();
        let restored_grammar: LSystemGrammar = serde_json::from_str(&grammar_json).unwrap();
        assert_eq!(grammar, restored_grammar);

        let cfg_json = serde_json::to_string(&cfg).unwrap();
        let restored_cfg: TurtleConfig = serde_json::from_str(&cfg_json).unwrap();
        assert_eq!(cfg, restored_cfg);
    }

    assert!(serde_json::from_str::<TurtleConfig>(
        r#"{"step_length":0.0,"turn_angle_rad":0.4,"initial_heading_rad":1.5,"depth_scale":1.0}"#
    )
    .is_err());
    assert!(serde_json::from_str::<TurtleConfig>(
        r#"{"step_length":1.0,"turn_angle_rad":0.4,"initial_heading_rad":1.5,"depth_scale":2.5}"#
    )
    .is_err());

    let mut bad_grammar = serde_json::to_value(LSystemGrammar::new_ascii(b"F").unwrap()).unwrap();
    bad_grammar["axiom_len"] = serde_json::json!(0);
    assert!(serde_json::from_value::<LSystemGrammar>(bad_grammar).is_err());

    let mut overflow_rule =
        serde_json::to_value(ProductionRule::from_ascii(b'F', b"FF").unwrap()).unwrap();
    overflow_rule["token_count"] = serde_json::json!(MAX_RULE_EXPANSION + 1);
    assert!(serde_json::from_value::<ProductionRule>(overflow_rule).is_err());
}

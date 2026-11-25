//! Integration tests for optional `serde` support on the `math_explorer` facade.

#[test]
fn facade_serde_json_dev_dep_linked() {
    let value: serde_json::Value = serde_json::from_str(r#"{"ok":true}"#).expect("valid json");
    assert_eq!(value["ok"], serde_json::Value::Bool(true));
}

#[cfg(feature = "serde")]
#[test]
fn facade_primitives_and_grid_types_roundtrip_json() {
    use math_explorer::math_kernel::types::{Dimension, GridIndex, StepSize};
    use math_explorer::prelude::{
        Interval, NonNegativeFloat, NonNegativeInterval, PositiveFloat, PositiveInterval,
        UnitInterval, UnitIntervalEnclosure,
    };

    let iv = Interval::new(-0.5, 1.25).expect("interval");
    assert_eq!(
        serde_json::from_str::<Interval>(&serde_json::to_string(&iv).unwrap()).unwrap(),
        iv
    );
    assert!(serde_json::from_str::<Interval>("[2.0, 1.0]").is_err());

    let u_iv = UnitIntervalEnclosure::new(0.2, 0.8).expect("unit enclosure");
    assert_eq!(
        serde_json::from_str::<UnitIntervalEnclosure>(&serde_json::to_string(&u_iv).unwrap())
            .unwrap(),
        u_iv
    );
    assert!(serde_json::from_str::<UnitIntervalEnclosure>("[-0.1, 0.8]").is_err());

    let p_iv = PositiveInterval::new(0.5, 2.0).expect("positive interval");
    assert_eq!(
        serde_json::from_str::<PositiveInterval>(&serde_json::to_string(&p_iv).unwrap()).unwrap(),
        p_iv
    );
    assert!(serde_json::from_str::<PositiveInterval>("[0.0, 2.0]").is_err());

    let nn_iv = NonNegativeInterval::new(0.0, 4.0).expect("non-negative interval");
    assert_eq!(
        serde_json::from_str::<NonNegativeInterval>(&serde_json::to_string(&nn_iv).unwrap())
            .unwrap(),
        nn_iv
    );
    assert!(serde_json::from_str::<NonNegativeInterval>("[-0.1, 4.0]").is_err());

    let u = UnitInterval::new(0.75).expect("unit interval");
    let u_json = serde_json::to_string(&u).expect("serialize UnitInterval");
    assert_eq!(
        serde_json::from_str::<UnitInterval>(&u_json).expect("deserialize UnitInterval"),
        u
    );
    assert!(serde_json::from_str::<UnitInterval>("1.25").is_err());

    let p = PositiveFloat::new(3.5).expect("positive float");
    let p_json = serde_json::to_string(&p).expect("serialize PositiveFloat");
    assert_eq!(serde_json::from_str::<PositiveFloat>(&p_json).unwrap(), p);
    assert!(serde_json::from_str::<PositiveFloat>("0.0").is_err());

    let nn = NonNegativeFloat::new(0.0).expect("non-negative float");
    let nn_json = serde_json::to_string(&nn).expect("serialize NonNegativeFloat");
    assert_eq!(
        serde_json::from_str::<NonNegativeFloat>(&nn_json).unwrap(),
        nn
    );

    assert_eq!(
        serde_json::from_str::<StepSize>(&serde_json::to_string(&StepSize(0.1)).unwrap()).unwrap(),
        StepSize(0.1)
    );
    assert_eq!(
        serde_json::from_str::<GridIndex>(&serde_json::to_string(&GridIndex(9)).unwrap()).unwrap(),
        GridIndex(9)
    );
    assert_eq!(
        serde_json::from_str::<Dimension>(&serde_json::to_string(&Dimension(64)).unwrap()).unwrap(),
        Dimension(64)
    );
}

#[cfg(feature = "serde")]
#[test]
fn facade_fit_poi_ast_and_errors_roundtrip_json() {
    use math_explorer::ast::{CompiledExpr, parse};
    use math_explorer::error::MathError;
    use math_explorer::fit::{CurveFit, FitModel, fit_model};
    use math_explorer::poi::LocalExtremum;

    let pts = [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]];
    let curve = fit_model(&pts, FitModel::Linear, false).expect("linear fit");
    let curve_json = serde_json::to_string(&curve).unwrap();
    assert_eq!(
        serde_json::from_str::<CurveFit>(&curve_json).unwrap(),
        curve
    );

    let expr = parse("cos(x) + k").expect("parse expr");
    let compiled = CompiledExpr::compile_with_vars(&expr, &["x"]).expect("compile expr");
    let compiled_json = serde_json::to_string(&compiled).unwrap();
    let decoded_compiled: CompiledExpr = serde_json::from_str(&compiled_json).unwrap();
    assert_eq!(decoded_compiled, compiled);

    let ext = LocalExtremum {
        point: [0.5, 2.0],
        is_max: true,
    };
    assert_eq!(
        serde_json::from_str::<LocalExtremum>(&serde_json::to_string(&ext).unwrap()).unwrap(),
        ext
    );

    let err = MathError::OutOfBounds {
        value: 2.0,
        min: 0.0,
        max: 1.0,
    };
    assert_eq!(
        serde_json::from_str::<MathError>(&serde_json::to_string(&err).unwrap()).unwrap(),
        err
    );
}

#[cfg(all(feature = "serde", feature = "pure_math"))]
#[test]
fn facade_pure_math_and_external_reexports_roundtrip_json() {
    use math_explorer::math_types::{Float, Integer, Rational};
    use math_explorer::nalgebra::Vector3;
    use math_explorer::num_complex::Complex64;
    use math_explorer::petgraph::graph::Graph;
    use math_explorer::rand::distributions::Uniform;

    let vec3 = Vector3::new(1.0_f64, -2.0, 3.5);
    assert_eq!(
        serde_json::from_str::<Vector3<f64>>(&serde_json::to_string(&vec3).unwrap()).unwrap(),
        vec3
    );

    let z = Complex64::new(1.25, -0.75);
    assert_eq!(
        serde_json::from_str::<Complex64>(&serde_json::to_string(&z).unwrap()).unwrap(),
        z
    );

    let mut graph = Graph::<u32, f64>::new();
    let n0 = graph.add_node(10);
    let n1 = graph.add_node(20);
    graph.add_edge(n0, n1, 0.5);
    let g_decoded: Graph<u32, f64> =
        serde_json::from_str(&serde_json::to_string(&graph).unwrap()).unwrap();
    assert_eq!(g_decoded.node_count(), 2);
    assert_eq!(g_decoded.edge_count(), 1);

    let big_int = Integer::from(123_456_789_i64);
    assert_eq!(
        serde_json::from_str::<Integer>(&serde_json::to_string(&big_int).unwrap()).unwrap(),
        big_int
    );

    let big_float = Float::with_val(64, 2.5_f64);
    assert_eq!(
        serde_json::from_str::<Float>(&serde_json::to_string(&big_float).unwrap()).unwrap(),
        big_float
    );

    let third_256 = Float::with_val(256, 1.0) / Float::with_val(256, 3.0);
    let third_decoded: Float =
        serde_json::from_str(&serde_json::to_string(&third_256).unwrap()).unwrap();
    assert_eq!(third_decoded.precision(), 256);
    assert_eq!(third_decoded, third_256);

    let rat = Rational::from((22, 7));
    assert_eq!(
        serde_json::from_str::<Rational>(&serde_json::to_string(&rat).unwrap()).unwrap(),
        rat
    );

    let dist = Uniform::new(-1.0_f64, 1.0_f64);
    assert_eq!(
        serde_json::from_str::<Uniform<f64>>(&serde_json::to_string(&dist).unwrap()).unwrap(),
        dist
    );
}

#[cfg(all(feature = "serde", feature = "pure_math"))]
#[test]
fn facade_rng_roundtrips_json() {
    use math_explorer::rng::{OxidizeRng, QmcError, RngCore};

    let mut rng = OxidizeRng::new(777).with_stream(3);
    let _ = rng.next_u64();
    let json = serde_json::to_string(&rng).expect("serialize OxidizeRng");
    let mut decoded: OxidizeRng = serde_json::from_str(&json).expect("deserialize OxidizeRng");
    assert_eq!(rng, decoded);
    assert_eq!(rng.next_u64(), decoded.next_u64());

    let qmc_err = QmcError::InvalidBase { base: 1 };
    assert_eq!(
        serde_json::from_str::<QmcError>(&serde_json::to_string(&qmc_err).unwrap()).unwrap(),
        qmc_err
    );
}

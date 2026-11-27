// [cite:engine_diffeq]
#[test]
fn diffeq_serde_json_dev_dep_linked() {
    let v: usize = serde_json::from_str("4").expect("json scalar");
    assert_eq!(v, 4);
}

#[cfg(feature = "serde")]
#[test]
fn diffeq_sde_steppers_roundtrip_no_std() {
    use crate::{
        CorrelatedWienerIncrement, EulerMaruyamaSolver, MilsteinSolver, Sriw1Solver,
        StochasticCalculus,
    };

    for calc in [StochasticCalculus::Ito, StochasticCalculus::Stratonovich] {
        let enc = serde_json::to_string(&calc).unwrap();
        assert_eq!(
            serde_json::from_str::<StochasticCalculus>(&enc).unwrap(),
            calc
        );

        let em = EulerMaruyamaSolver::new(calc);
        let em_json = serde_json::to_string(&em).unwrap();
        assert_eq!(
            serde_json::from_str::<EulerMaruyamaSolver>(&em_json).unwrap(),
            em
        );

        let mil = MilsteinSolver::new(calc).with_levy_truncation(6);
        let mil_json = serde_json::to_string(&mil).unwrap();
        assert_eq!(
            serde_json::from_str::<MilsteinSolver>(&mil_json).unwrap(),
            mil
        );

        let sriw = Sriw1Solver::new(calc);
        let sriw_json = serde_json::to_string(&sriw).unwrap();
        assert_eq!(
            serde_json::from_str::<Sriw1Solver>(&sriw_json).unwrap(),
            sriw
        );
    }

    let inc = CorrelatedWienerIncrement::<f64> {
        dw: 0.03125,
        dz: -0.00042,
    };
    let inc_json = serde_json::to_string(&inc).unwrap();
    assert_eq!(
        serde_json::from_str::<CorrelatedWienerIncrement<f64>>(&inc_json).unwrap(),
        inc
    );
}

#[cfg(feature = "serde")]
#[test]
fn diffeq_baoab_roundtrips_and_validates_params_no_std() {
    use crate::{BaoabLangevinSolver, BaoabParams, LangevinSystem};
    use math_commons::math_kernel::types::SolverWorkspace;
    use math_rng::OxidizeRng;

    struct Harmonic1D;
    impl LangevinSystem<f64> for Harmonic1D {
        fn dim(&self) -> usize {
            1
        }
        fn grad_potential(&self, q: &[f64], out_grad: &mut [f64]) {
            out_grad[0] = 2.0 * q[0];
        }
    }

    let params = BaoabParams {
        mass: 1.5_f64,
        friction: 0.8,
        temperature: 0.5,
        dt: 0.01,
    };
    let params_json = serde_json::to_string(&params).unwrap();
    assert_eq!(
        serde_json::from_str::<BaoabParams<f64>>(&params_json).unwrap(),
        params
    );

    let solver = BaoabLangevinSolver::new(params);
    let solver_json = serde_json::to_string(&solver).unwrap();
    let decoded_solver: BaoabLangevinSolver<f64> = serde_json::from_str(&solver_json).unwrap();
    assert_eq!(decoded_solver, solver);

    let (mut q1, mut p1) = ([1.0_f64], [-0.25_f64]);
    let (mut q2, mut p2) = (q1, p1);
    let (mut buf1, mut buf2) = ([0.0_f64; 4], [0.0_f64; 4]);
    let (mut rng1, mut rng2) = (OxidizeRng::new(99), OxidizeRng::new(99));
    solver.step(
        &Harmonic1D,
        &mut q1,
        &mut p1,
        &mut SolverWorkspace::new(&mut buf1),
        &mut rng1,
    );
    decoded_solver.step(
        &Harmonic1D,
        &mut q2,
        &mut p2,
        &mut SolverWorkspace::new(&mut buf2),
        &mut rng2,
    );
    assert_eq!((q1, p1), (q2, p2));

    for bad_json in [
        r#"{"mass":0.0,"friction":0.5,"temperature":1.0,"dt":0.01}"#,
        r#"{"mass":1.0,"friction":-0.1,"temperature":1.0,"dt":0.01}"#,
        r#"{"mass":1.0,"friction":0.5,"temperature":-0.1,"dt":0.01}"#,
        r#"{"mass":1.0,"friction":0.5,"temperature":1.0,"dt":0.0}"#,
    ] {
        assert!(serde_json::from_str::<BaoabLangevinSolver<f64>>(bad_json).is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn diffeq_error_roundtrips_no_std() {
    use crate::DiffeqError;
    use math_commons::error::MathError;

    let samples = [
        DiffeqError::InvalidParameter { parameter: "mass" },
        DiffeqError::InvalidStepSize,
        DiffeqError::BufferTooSmall {
            buffer: "state",
            required: 4,
            actual: 2,
        },
        DiffeqError::WorkspaceAllocationFailed { capacity: 128 },
        DiffeqError::Math(MathError::WorkspaceTooSmall {
            required: 8,
            available: 4,
        }),
    ];
    for sample in samples {
        let json = serde_json::to_string(&sample).expect("serialize DiffeqError");
        let decoded: DiffeqError = serde_json::from_str(&json).expect("deserialize DiffeqError");
        assert_eq!(decoded, sample);
    }

    let kalman_samples = [
        crate::KalmanError::MatrixInversionError,
        crate::KalmanError::InitializationError("initial_state"),
        crate::KalmanError::Math(MathError::WorkspaceTooSmall {
            required: 4,
            available: 1,
        }),
    ];
    for sample in kalman_samples {
        let json = serde_json::to_string(&sample).expect("serialize KalmanError");
        let decoded: crate::KalmanError =
            serde_json::from_str(&json).expect("deserialize KalmanError");
        assert_eq!(decoded, sample);
    }
}

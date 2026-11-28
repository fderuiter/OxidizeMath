// [cite:engine_diffeq]
use super::*;
use math_commons::error::MathError;
use math_diagnostics::Diagnostic;

struct Harmonic1D;

impl LangevinSystem<f64> for Harmonic1D {
    fn dim(&self) -> usize {
        2
    }

    fn grad_potential(&self, q: &[f64], out_grad: &mut [f64]) {
        out_grad[0] = q[0];
        out_grad[1] = q[1];
    }
}

struct ToySde;

impl SdeSystem<f64> for ToySde {
    fn state_dim(&self) -> usize {
        2
    }

    fn wiener_dim(&self) -> usize {
        2
    }

    fn drift(&self, _t: f64, x: &[f64], out_drift: &mut [f64]) {
        out_drift[0] = -x[0];
        out_drift[1] = -0.5 * x[1];
    }

    fn diffusion(&self, _t: f64, x: &[f64], out_diffusion: &mut [f64]) {
        out_diffusion[0] = 0.2 * x[0];
        out_diffusion[1] = 0.0;
        out_diffusion[2] = 0.0;
        out_diffusion[3] = 0.1 * x[1];
    }
}

fn assert_stack_error_traits<
    E: Copy + Clone + PartialEq + Eq + core::fmt::Display + core::error::Error + Diagnostic,
>() {
}

#[test]
#[verified_engine::verified]
fn test_diffeq_error_copy_and_fallible_companions() {
    assert_stack_error_traits::<DiffeqError>();
    assert_eq!(
        BaoabLangevinSolver::try_new(BaoabParams {
            mass: 0.0,
            friction: 1.0,
            temperature: 1.0,
            dt: 0.01,
        }),
        Err(MathError::StrictlyOutOfBounds {
            value: 0.0,
            min: 0.0,
            max: f64::INFINITY,
        })
    );
    let solver = BaoabLangevinSolver::try_new(BaoabParams {
        mass: 1.0,
        friction: 0.5,
        temperature: 1.0,
        dt: 0.01,
    })
    .expect("valid BAOAB params");

    let mut q = [1.0, -0.5];
    let mut p = [0.0, 0.2];
    let mut tiny_buf = [0.0; 2];
    let mut rng = OxidizeRng::new(7);
    let err = solver
        .try_step(
            &Harmonic1D,
            &mut q,
            &mut p,
            &mut ScalarWorkspace::new(&mut tiny_buf),
            &mut rng,
        )
        .expect_err("workspace of len 2 < 4 should fail");
    let err_copy = err;
    assert_eq!(err, err_copy);
    assert_eq!(err.code(), "DIFFEQ-005");
    assert_eq!(
        err,
        DiffeqError::Math(MathError::WorkspaceTooSmall {
            required: 2,
            available: 0,
        })
    );

    let mut ok_buf = [0.0; 4];
    solver
        .try_step(
            &Harmonic1D,
            &mut q,
            &mut p,
            &mut ScalarWorkspace::new(&mut ok_buf),
            &mut rng,
        )
        .expect("sufficient workspace succeeds");

    let em = EulerMaruyamaSolver::new(StochasticCalculus::Stratonovich);
    let mil = MilsteinSolver::new(StochasticCalculus::Ito).with_levy_truncation(2);
    let sriw = Sriw1Solver::new(StochasticCalculus::Stratonovich);
    let mut sde_state = [1.0, 2.0];
    let mut short_buf = [0.0; 3];

    assert_eq!(
        em.try_step(
            &ToySde,
            0.0,
            -0.1,
            &mut sde_state,
            &mut ScalarWorkspace::new(&mut short_buf),
            &mut rng
        ),
        Err(DiffeqError::InvalidStepSize)
    );
    assert!(
        mil.try_step(
            &ToySde,
            0.0,
            0.01,
            &mut sde_state,
            &mut ScalarWorkspace::new(&mut short_buf),
            &mut rng
        )
        .is_err()
    );
    assert!(
        sriw.try_step(
            &ToySde,
            0.0,
            0.01,
            &mut sde_state,
            &mut ScalarWorkspace::new(&mut short_buf),
            &mut rng
        )
        .is_err()
    );
    assert!(
        try_eval_ito_effective_drift(
            &ToySde,
            StochasticCalculus::Stratonovich,
            0.0,
            &sde_state,
            &mut [0.0; 1],
            &mut ScalarWorkspace::new(&mut short_buf),
        )
        .is_err()
    );
}

#[cfg(feature = "std")]
#[test]
#[verified_engine::verified]
fn test_diffeq_error_preserves_root_math_error_code_and_type() {
    let wrapped = DiffeqError::from(MathError::WorkspaceTooSmall {
        required: 8,
        available: 2,
    });
    let meta = wrapped.metadata();
    assert_eq!(
        meta.get("code").map(std::string::String::as_str),
        Some("MATH-005")
    );
    assert_eq!(
        meta.get("wrapper_code").map(std::string::String::as_str),
        Some("DIFFEQ-005")
    );
    assert_eq!(
        meta.get("root_error_type").map(std::string::String::as_str),
        Some("MathError")
    );
    assert_eq!(
        meta.get("error_type").map(std::string::String::as_str),
        Some("DiffeqError")
    );
}

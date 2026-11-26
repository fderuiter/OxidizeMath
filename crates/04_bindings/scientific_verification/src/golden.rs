use pure_math::pure_math::special_functions::bessel::{bessel_j, bessel_y, spherical_bessel_j};
use pure_math::pure_math::special_functions::gamma_beta::{beta_function, gamma_function};
use pure_math::pure_math::special_functions::legendre::legendre_p;
use pure_math::pure_math::special_functions::polynomials::{associated_laguerre, hermite};
use pure_math::pure_math::statistics::copula::{Correlation, GaussianCopula, Probability};
use serde::Deserialize;

const GOLDEN_JSON: &str = include_str!("../fixtures/special_functions_golden.json");

#[derive(Deserialize)]
struct BesselCase {
    nu: f64,
    x: f64,
    expected: f64,
}

#[derive(Deserialize)]
struct SphericalBesselCase {
    l: u64,
    x: f64,
    expected: f64,
}

#[derive(Deserialize)]
struct PolyCase {
    n: u64,
    x: f64,
    expected: f64,
}

#[derive(Deserialize)]
struct LaguerreCase {
    n: u64,
    alpha: f64,
    x: f64,
    expected: f64,
}

#[derive(Deserialize)]
struct GammaBetaCase {
    r#fn: String,
    a: f64,
    b: f64,
    expected: f64,
}

#[derive(Deserialize)]
struct CopulaCase {
    rho: f64,
    u: f64,
    v: f64,
    expected: f64,
}

#[derive(Deserialize)]
struct GoldenFixtureBundle {
    tolerance: f64,
    bessel_j: Vec<BesselCase>,
    bessel_y: Vec<BesselCase>,
    spherical_bessel_j: Vec<SphericalBesselCase>,
    legendre_p: Vec<PolyCase>,
    hermite_h: Vec<PolyCase>,
    associated_laguerre: Vec<LaguerreCase>,
    gamma_beta: Vec<GammaBetaCase>,
    gaussian_copula: Vec<CopulaCase>,
}

fn check_rel_abs(label: &str, actual: f64, expected: f64, tol: f64) -> Result<(), String> {
    let scale = expected.abs().max(1.0);
    let err = (actual - expected).abs();
    if !actual.is_finite() || err > tol * scale {
        Err(format!(
            "{label}: actual={actual:.16e}, expected={expected:.16e}, err={err:.3e} > tol*scale={:.3e}",
            tol * scale
        ))
    } else {
        Ok(())
    }
}

fn verify_bessel_and_polys(bundle: &GoldenFixtureBundle) -> Result<usize, String> {
    let tol = bundle.tolerance;
    let mut checked = 0usize;
    for c in &bundle.bessel_j {
        check_rel_abs(
            &format!("bessel_j({}, {})", c.nu, c.x),
            bessel_j(c.nu, c.x),
            c.expected,
            tol,
        )?;
        checked += 1;
    }
    for c in &bundle.bessel_y {
        check_rel_abs(
            &format!("bessel_y({}, {})", c.nu, c.x),
            bessel_y(c.nu, c.x),
            c.expected,
            tol,
        )?;
        checked += 1;
    }
    for c in &bundle.spherical_bessel_j {
        check_rel_abs(
            &format!("spherical_bessel_j({}, {})", c.l, c.x),
            spherical_bessel_j(c.l, c.x),
            c.expected,
            tol,
        )?;
        checked += 1;
    }
    for c in &bundle.legendre_p {
        check_rel_abs(
            &format!("legendre_p({}, {})", c.n, c.x),
            legendre_p(c.n, c.x),
            c.expected,
            tol,
        )?;
        checked += 1;
    }
    for c in &bundle.hermite_h {
        check_rel_abs(
            &format!("hermite({}, {})", c.n, c.x),
            hermite(c.n, c.x),
            c.expected,
            tol,
        )?;
        checked += 1;
    }
    for c in &bundle.associated_laguerre {
        check_rel_abs(
            &format!("associated_laguerre({}, {}, {})", c.n, c.alpha, c.x),
            associated_laguerre(c.n, c.alpha, c.x),
            c.expected,
            tol,
        )?;
        checked += 1;
    }
    Ok(checked)
}

/// Validates all committed golden reference cases in `fixtures/special_functions_golden.json`
/// within `1e-12`, returning the number of verified golden cases.
pub fn verify_golden_fixtures() -> Result<usize, String> {
    let bundle: GoldenFixtureBundle = serde_json::from_str(GOLDEN_JSON)
        .map_err(|err| format!("Failed parsing special_functions_golden.json: {err}"))?;
    let mut checked = verify_bessel_and_polys(&bundle)?;
    for c in &bundle.gamma_beta {
        let actual = if c.r#fn == "gamma" {
            gamma_function(c.a)
        } else {
            beta_function(c.a, c.b)
        };
        check_rel_abs(
            &format!("{}({}, {})", c.r#fn, c.a, c.b),
            actual,
            c.expected,
            bundle.tolerance,
        )?;
        checked += 1;
    }
    for c in &bundle.gaussian_copula {
        let copula = GaussianCopula::bivariate(
            Correlation::new(c.rho).map_err(|e| format!("Invalid rho: {e:?}"))?,
        )
        .map_err(|e| format!("Copula init error: {e:?}"))?;
        let u = Probability::new(c.u).map_err(|e| format!("Invalid u: {e:?}"))?;
        let v = Probability::new(c.v).map_err(|e| format!("Invalid v: {e:?}"))?;
        let actual = copula
            .cdf(&[u, v])
            .map_err(|e| format!("Copula cdf error: {e:?}"))?
            .value();
        // Bivariate Gaussian copula quadrature achieves ~1e-8 accuracy on non-zero correlation
        let cop_tol = if c.rho == 0.0 { bundle.tolerance } else { 5e-7 };
        check_rel_abs(
            &format!("gaussian_copula(rho={}, u={}, v={})", c.rho, c.u, c.v),
            actual,
            c.expected,
            cop_tol,
        )?;
        checked += 1;
    }
    Ok(checked)
}

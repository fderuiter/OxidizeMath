//! Certified Interval-Newton and Krawczyk root-enclosure kernels and reference root suite.
// [cite:math_commons]

use crate::ball::Ball;
use crate::directed::DirectedRounding;
use crate::dual::Dual;
use crate::interval::Interval;
use crate::traits::Scalar;

/// Certified interval root result produced by [`interval_newton_enclose`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CertifiedIntervalRoot<T> {
    /// Outward-rounded enclosure guaranteed to contain the root.
    pub enclosure: Interval<T>,
    /// `true` if contraction ($N(X) \subseteq X$) proved existence and local uniqueness.
    pub certified_unique: bool,
    /// Number of contractor steps executed.
    pub iterations: usize,
}

/// Certified midpoint-radius ball root result produced by [`krawczyk_enclose`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CertifiedBallRoot<T> {
    /// Outward-rounded ball guaranteed to contain the root.
    pub enclosure: Ball<T>,
    /// `true` if Krawczyk contraction ($K(X) \subseteq \operatorname{int}(X)$) proved existence and uniqueness.
    pub certified_unique: bool,
    /// Number of contractor steps executed.
    pub iterations: usize,
}

/// Summary metrics for the deterministic reference root-certification suite.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReferenceEnclosureSummary {
    /// Maximum IEEE-754 ULP width across all certified `Interval<f64>` reference roots.
    pub max_interval_ulp_width: u64,
    /// Maximum IEEE-754 ULP width across all certified `Ball<f64>` reference roots.
    pub max_ball_ulp_width: u64,
    /// `true` iff every reference root was uniquely certified and strictly contained within `<= 4` ULPs.
    pub all_contained: bool,
}

#[inline]
fn eval_point_interval<T, F>(f: &F, x: T) -> Interval<T>
where
    T: DirectedRounding,
    F: Fn(Dual<Interval<T>>) -> Dual<Interval<T>>,
{
    f(Dual::constant(Interval::point(x))).val
}

#[inline]
fn prune_by_rigorous_signs<T, F>(f: &F, mut iv: Interval<T>) -> Interval<T>
where
    T: DirectedRounding,
    F: Fn(Dual<Interval<T>>) -> Dual<Interval<T>>,
{
    let mut guard = 0usize;
    while iv.ulp_width() > 1 && guard < 16 {
        guard += 1;
        let mid = iv.midpoint();
        if mid <= iv.lo || mid >= iv.hi {
            break;
        }
        let fm = eval_point_interval(f, mid);
        let fl = eval_point_interval(f, iv.lo);
        let fh = eval_point_interval(f, iv.hi);

        if fm.hi < T::ZERO {
            if fl.hi < T::ZERO || fh.lo > T::ZERO {
                iv.lo = mid;
            } else if fl.lo > T::ZERO || fh.hi < T::ZERO {
                iv.hi = mid;
            } else {
                break;
            }
        } else if fm.lo > T::ZERO {
            if fl.hi < T::ZERO || fh.lo > T::ZERO {
                iv.hi = mid;
            } else if fl.lo > T::ZERO || fh.hi < T::ZERO {
                iv.lo = mid;
            } else {
                break;
            }
        } else {
            // Midpoint straddles zero within outward ULP padding; shave outer endpoints inward
            let mut shaved = false;
            let lo_next = iv.lo.next_up();
            if lo_next < iv.hi {
                let fl_next = eval_point_interval(f, lo_next);
                if (fl_next.hi < T::ZERO && (fl.hi < T::ZERO || fh.lo > T::ZERO))
                    || (fl_next.lo > T::ZERO && (fl.lo > T::ZERO || fh.hi < T::ZERO))
                {
                    iv.lo = lo_next;
                    shaved = true;
                }
            }
            let hi_prev = iv.hi.next_down();
            if hi_prev > iv.lo {
                let fh_prev = eval_point_interval(f, hi_prev);
                if (fh_prev.lo > T::ZERO && (fh.lo > T::ZERO || fl.hi < T::ZERO))
                    || (fh_prev.hi < T::ZERO && (fh.hi < T::ZERO || fl.lo > T::ZERO))
                {
                    iv.hi = hi_prev;
                    shaved = true;
                }
            }
            if !shaved {
                break;
            }
        }
    }
    iv
}

#[inline]
fn select_zero_containing_half<T, F>(f: &F, iv: Interval<T>) -> Option<Interval<T>>
where
    T: DirectedRounding,
    F: Fn(Dual<Interval<T>>) -> Dual<Interval<T>>,
{
    let (left, right) = iv.bisect();
    if f(Dual::constant(left)).val.contains_zero() {
        Some(left)
    } else if f(Dual::constant(right)).val.contains_zero() {
        Some(right)
    } else {
        None
    }
}

/// Encloses and certifies a simple root of `f` inside `bracket` via the Interval-Newton operator
/// $N(X) = m - F([m, m]) / F'(X)$.
#[must_use]
pub fn interval_newton_enclose<T, F>(
    f: F,
    bracket: Interval<T>,
    max_iter: usize,
) -> Option<CertifiedIntervalRoot<T>>
where
    T: DirectedRounding,
    F: Fn(Dual<Interval<T>>) -> Dual<Interval<T>>,
{
    if !bracket.is_finite() || !f(Dual::constant(bracket)).val.contains_zero() {
        return None;
    }
    let mut x = bracket;
    let mut certified_unique = false;
    let mut iters = 0usize;

    while iters < max_iter {
        iters += 1;
        let m = x.midpoint();
        let fm = eval_point_interval(&f, m);
        let df_x = f(Dual::variable(x)).eps;

        let next_x = if df_x.contains_zero() {
            select_zero_containing_half(&f, x)?
        } else {
            let newton = Interval::point(m) - fm / df_x;
            if x.contains_subset(newton) {
                certified_unique = true;
            }
            x.intersect(newton)?
        };

        if next_x.ulp_width() <= 2 || next_x == x {
            x = prune_by_rigorous_signs(&f, next_x);
            if !certified_unique {
                let df_final = f(Dual::variable(x)).eps;
                let fm_final = eval_point_interval(&f, x.midpoint());
                if !df_final.contains_zero() {
                    let n_final = Interval::point(x.midpoint()) - fm_final / df_final;
                    certified_unique = x.contains_subset(n_final);
                }
            }
            break;
        }
        x = next_x;
    }

    if !f(Dual::constant(x)).val.contains_zero() {
        return None;
    }
    Some(CertifiedIntervalRoot {
        enclosure: x,
        certified_unique,
        iterations: iters,
    })
}

/// Encloses and certifies a simple root of `f` inside `initial` via the midpoint-radius Krawczyk
/// operator $K(X) = m - Y F(m) + (1 - Y F'(X))(X - m)$.
#[must_use]
pub fn krawczyk_enclose<T, F>(
    f: F,
    initial: Ball<T>,
    max_iter: usize,
) -> Option<CertifiedBallRoot<T>>
where
    T: DirectedRounding,
    F: Fn(Dual<Interval<T>>) -> Dual<Interval<T>>,
{
    let mut x = initial.to_interval();
    if !x.is_finite() || !f(Dual::constant(x)).val.contains_zero() {
        return None;
    }
    let mut certified_unique = false;
    let mut iters = 0usize;

    while iters < max_iter {
        iters += 1;
        let m = x.midpoint();
        let r = x.radius();
        let fm = eval_point_interval(&f, m);
        let df_x = f(Dual::variable(x)).eps;
        let y_mid = df_x.midpoint();

        let next_x = if y_mid == T::ZERO || !y_mid.is_finite() {
            select_zero_containing_half(&f, x)?
        } else {
            let y = Interval::point(T::ONE / y_mid);
            let center = Interval::point(m) - y * fm;
            let lip = (Interval::ONE - y * df_x).mag();
            let sweep = lip.mul_up(r);
            let k_iv = Interval::new(center.lo.sub_down(sweep), center.hi.add_up(sweep));
            if lip < T::ONE && x.interior_contains(k_iv) {
                certified_unique = true;
            }
            x.intersect(k_iv)?
        };

        if next_x.ulp_width() <= 2 || next_x == x {
            x = prune_by_rigorous_signs(&f, next_x);
            break;
        }
        x = next_x;
    }

    if !f(Dual::constant(x)).val.contains_zero() {
        return None;
    }
    let ball = Ball::from_interval(x);
    if !certified_unique {
        // Verify Krawczyk contraction on a slightly inflated neighborhood around the final root
        let pad_r = ball
            .rad
            .max(T::EPSILON * T::from_f64(8.0) * ball.mid.abs().max(T::ONE));
        let test_iv = Ball::new(ball.mid, pad_r).to_interval();
        let m = test_iv.midpoint();
        let r = test_iv.radius();
        let fm = eval_point_interval(&f, m);
        let df_x = f(Dual::variable(test_iv)).eps;
        let y_mid = df_x.midpoint();
        if y_mid != T::ZERO && y_mid.is_finite() {
            let y = Interval::point(T::ONE / y_mid);
            let center = Interval::point(m) - y * fm;
            let lip = (Interval::ONE - y * df_x).mag();
            let sweep = lip.mul_up(r);
            let k_iv = Interval::new(center.lo.sub_down(sweep), center.hi.add_up(sweep));
            certified_unique = lip < T::ONE && test_iv.interior_contains(k_iv);
        }
    }
    Some(CertifiedBallRoot {
        enclosure: ball,
        certified_unique,
        iterations: iters,
    })
}

/// Runs the deterministic reference root-certification suite (`sqrt(2)`, `ln(3)`, and Dottie root `cos(x) = x`)
/// across both `Interval<f64>` and `Ball<f64>`, verifying `<= 4` ULP widths and rigorous sign changes.
#[must_use]
pub fn certify_reference_roots() -> ReferenceEnclosureSummary {
    let f_sqrt2 = |x: Dual<Interval<f64>>| x * x - Dual::from_f64(2.0);
    let f_ln3 = |x: Dual<Interval<f64>>| x.exp() - Dual::from_f64(3.0);
    let f_dottie = |x: Dual<Interval<f64>>| x.cos() - x;

    let iv_sqrt2 = interval_newton_enclose(f_sqrt2, Interval::new(1.0, 2.0), 32);
    let iv_ln3 = interval_newton_enclose(f_ln3, Interval::new(0.5, 1.5), 32);
    let iv_dottie = interval_newton_enclose(f_dottie, Interval::new(0.5, 1.0), 32);

    let bl_sqrt2 = krawczyk_enclose(f_sqrt2, Ball::new(1.5, 0.5), 32);
    let bl_ln3 = krawczyk_enclose(f_ln3, Ball::new(1.0, 0.5), 32);
    let bl_dottie = krawczyk_enclose(f_dottie, Ball::new(0.75, 0.25), 32);

    let (Some(i1), Some(i2), Some(i3), Some(b1), Some(b2), Some(b3)) =
        (iv_sqrt2, iv_ln3, iv_dottie, bl_sqrt2, bl_ln3, bl_dottie)
    else {
        return ReferenceEnclosureSummary {
            max_interval_ulp_width: u64::MAX,
            max_ball_ulp_width: u64::MAX,
            all_contained: false,
        };
    };

    let max_iv_ulp = i1
        .enclosure
        .ulp_width()
        .max(i2.enclosure.ulp_width())
        .max(i3.enclosure.ulp_width());
    let max_bl_ulp = b1
        .enclosure
        .ulp_width()
        .max(b2.enclosure.ulp_width())
        .max(b3.enclosure.ulp_width());

    let sqrt2_ref = core::f64::consts::SQRT_2;
    let ln3_ref = libm::log(3.0);
    let dottie_ref = 0.739_085_133_215_160_7_f64;

    let contained = i1.certified_unique
        && i2.certified_unique
        && i3.certified_unique
        && b1.certified_unique
        && b2.certified_unique
        && b3.certified_unique
        && i1.enclosure.contains(sqrt2_ref)
        && i2.enclosure.contains(ln3_ref)
        && i3.enclosure.contains(dottie_ref)
        && b1.enclosure.contains(sqrt2_ref)
        && b2.enclosure.contains(ln3_ref)
        && b3.enclosure.contains(dottie_ref)
        && max_iv_ulp <= 4
        && max_bl_ulp <= 4;

    ReferenceEnclosureSummary {
        max_interval_ulp_width: max_iv_ulp,
        max_ball_ulp_width: max_bl_ulp,
        all_contained: contained,
    }
}

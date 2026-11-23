// [cite:math_commons]
/// Clebsch-Gordan coefficient calculator for angular momentum coupling.
///
/// # Example
/// ```
/// use math_commons::wigner::ClebschGordan;
///
/// let cg = ClebschGordan {
///     tj1: 3,
///     tm1: -1,
///     tj2: 2,
///     tm2: 2,
///     tj12: 5,
///     tm12: 1,
/// };
/// let val = cg.value();
/// let expected = (3.0_f64 / 10.0).sqrt();
/// assert!((val - expected).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClebschGordan {
    #[allow(missing_docs)]
    pub tj1: i32,
    #[allow(missing_docs)]
    pub tm1: i32,
    #[allow(missing_docs)]
    pub tj2: i32,
    #[allow(missing_docs)]
    pub tm2: i32,
    #[allow(missing_docs)]
    pub tj12: i32,
    #[allow(missing_docs)]
    pub tm12: i32,
}

impl ClebschGordan {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        let (tj1, tm1, tj2, tm2, tj, tm) =
            (self.tj1, self.tm1, self.tj2, self.tm2, self.tj12, self.tm12);

        if tm1 + tm2 != tm {
            return 0.0;
        }
        if tj < (tj1 - tj2).abs() || tj > tj1 + tj2 {
            return 0.0;
        }
        if (tj1 + tj2 + tj) % 2 != 0 {
            return 0.0;
        }

        let w3j = wigner_3j(tj1, tj2, tj, tm1, tm2, -tm);
        let mut cg_sign = 1.0;
        if (tj1 - tj2 + tm) % 4 == 2 || (tj1 - tj2 + tm) % 4 == -2 {
            cg_sign = -1.0;
        }

        cg_sign * (tj as f64 + 1.0).sqrt() * w3j
    }
}

#[verified_engine::verified]
fn fact(n: i32) -> f64 {
    if n <= 0 {
        1.0
    } else {
        (1..=n).map(|x| x as f64).product()
    }
}

#[verified_engine::verified]
fn triangle_coeff(ta: i32, tb: i32, tc: i32) -> f64 {
    fact((ta + tb - tc) / 2) * fact((ta - tb + tc) / 2) * fact((-ta + tb + tc) / 2)
        / fact((ta + tb + tc) / 2 + 1)
}

#[verified_engine::verified]
fn wigner_3j(tj1: i32, tj2: i32, tj3: i32, tm1: i32, tm2: i32, tm3: i32) -> f64 {
    if tm1 + tm2 + tm3 != 0 {
        return 0.0;
    }
    if tj3 < (tj1 - tj2).abs() || tj3 > tj1 + tj2 {
        return 0.0;
    }

    let prefactor = triangle_coeff(tj1, tj2, tj3)
        * fact((tj1 + tm1) / 2)
        * fact((tj1 - tm1) / 2)
        * fact((tj2 + tm2) / 2)
        * fact((tj2 - tm2) / 2)
        * fact((tj3 + tm3) / 2)
        * fact((tj3 - tm3) / 2);

    let mut sum = 0.0;

    let mut t_min = 0;
    t_min = t_min.max(tj2 - tj3 - tm1);
    t_min = t_min.max(tj1 - tj3 + tm2);
    if t_min % 2 != 0 {
        t_min += 1;
    }

    let mut t_max = tj1 + tj2 - tj3;
    t_max = t_max.min(tj1 - tm1);
    t_max = t_max.min(tj2 + tm2);

    let mut t = t_min;
    while t <= t_max {
        let denom = fact(t / 2)
            * fact((tj3 - tj2 + tm1 + t) / 2)
            * fact((tj3 - tj1 - tm2 + t) / 2)
            * fact((tj1 + tj2 - tj3 - t) / 2)
            * fact((tj1 - tm1 - t) / 2)
            * fact((tj2 + tm2 - t) / 2);

        let term = if (t / 2) % 2 != 0 { -1.0 } else { 1.0 } / denom;
        sum += term;
        t += 2;
    }

    let sign = if (tj1 - tj2 - tm3) % 4 == 2 || (tj1 - tj2 - tm3) % 4 == -2 {
        -1.0
    } else {
        1.0
    };
    sign * prefactor.sqrt() * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg() {
        let cg = ClebschGordan {
            tj1: 3,
            tm1: -1,
            tj2: 2,
            tm2: 2,
            tj12: 5,
            tm12: 1,
        };
        let val = cg.value();
        let expected = (3.0_f64 / 10.0).sqrt();
        assert!((val - expected).abs() < 1e-6);
    }
}

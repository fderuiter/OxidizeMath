//! Integration tests for partitions.
use pure_math::pure_math::number_theory::partitions::*;

#[test]
#[verified_engine::verified]
fn test_qseries_add() {
    let s1 = QSeries::from_vec(vec![1, 2, 3]);
    let s2 = QSeries::from_vec(vec![4, 5, 6, 7]);
    let s3 = &s1 + &s2;
    assert_eq!(s3.coeffs, vec![5, 7, 9, 7]);
}

#[test]
#[verified_engine::verified]
fn test_qseries_mul() {
    let s1 = QSeries::from_vec(vec![1, 1]); // 1+q
    let s2 = QSeries::from_vec(vec![1, 1]); // 1+q
    let s3 = &s1 * &s2; // (1+q)^2 = 1+2q+q^2
    assert_eq!(s3.coeffs, vec![1, 2]); // Truncated to precision 2

    let s4 = QSeries::from_vec(vec![1, 1, 1]); // 1+q+q^2
    let s5 = QSeries::from_vec(vec![1, -1, 0]); // 1-q
    let s6 = &s4 * &s5; // (1-q^3) = 1
    assert_eq!(s6.coeffs, vec![1, 0, 0]);
}

#[test]
#[verified_engine::verified]
fn test_qseries_div() {
    // (1-q^4) / (1-q^2) = 1+q^2
    let s1 = f_k(1, 10);
    let s2 = f_k(2, 10);
    let s3 = &(&s1 * &s2) / &s1;
    assert_eq!(s3.coeffs, s2.coeffs);

    // 1 / (1-q) = 1+q+q^2+...
    let one = QSeries::from_vec(vec![1, 0, 0, 0, 0]);
    let one_minus_q = QSeries::from_vec(vec![1, -1]);
    let geom_series = &one / &one_minus_q;
    assert_eq!(geom_series.coeffs, vec![1, 1, 1, 1, 1]);
}

#[test]
#[verified_engine::verified]
fn test_f_k() {
    // f_1 = 1 - q - q^2 + q^5 + q^7 - ... (Euler's pentagonal number theorem)
    let f1 = f_k(1, 10);
    assert_eq!(f1.coeffs, vec![1, -1, -1, 0, 0, 1, 0, 1, 0, 0]);

    // f_2 = 1 - q^2 - q^4 + q^10 + ...
    let f2 = f_k(2, 12);
    assert_eq!(f2.coeffs, vec![1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0]);
}

#[test]
#[verified_engine::verified]
fn test_f1_pow2() {
    let f1 = f_k(1, 10);
    let f1_pow2 = f1.pow(2);
    assert_eq!(f1_pow2.coeffs, vec![1, -2, -1, 2, 1, 2, -2, 0, -2, -2]);
}

#[test]
#[verified_engine::verified]
fn test_gen_p_star() {
    let p_star = gen_p_star(10);
    assert_eq!(p_star.coeffs, vec![1, -4, 2, 8, -5, -8, 6, 0, -23, 20]);
}

#[test]
#[verified_engine::verified]
fn test_gen_m() {
    let m = gen_m(7);
    assert_eq!(m.coeffs, vec![1, 1, -3, -2, 0, -8, 1]);
}

#[test]
#[verified_engine::verified]
fn test_theorem_1() {
    let precision = 40;
    let p_star = gen_p_star(precision);

    // P*(2n+1) = 0 (mod 4)
    for n in 0.. {
        let index = 2 * n + 1;
        if index >= precision {
            break;
        }
        assert_eq!(p_star.get_coeff(index) % 4, 0, "P*(2*{}+1) failed mod 4", n);
    }

    // P*(4n+3) = 0 (mod 8)
    for n in 0.. {
        let index = 4 * n + 3;
        if index >= precision {
            break;
        }
        assert_eq!(p_star.get_coeff(index) % 8, 0, "P*(4*{}+3) failed mod 8", n);
    }

    // P*(16n+7) = 0
    for n in 0.. {
        let index = 16 * n + 7;
        if index >= precision {
            break;
        }
        assert_eq!(p_star.get_coeff(index), 0, "P*(16*{}+7) failed", n);
    }
}

#[test]
#[verified_engine::verified]
fn test_theorem_2() {
    let precision = 40;
    let p_star = gen_p_star(precision);

    for n in 0..2 {
        // Check for n=0 and n=1
        let index1 = 16 * n + 15;
        if index1 >= precision {
            break;
        }
        assert_eq!(
            p_star.get_coeff(index1),
            -64 * p_star.get_coeff(n),
            "P*(16*{}+15) failed",
            n
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_corollary_2_alpha_1() {
    let precision = 30;
    let m = gen_m(precision);
    for n in 0.. {
        let index = 5 * n + 4;
        if index >= precision {
            break;
        }
        assert_eq!(m.get_coeff(index) % 5, 0, "M(5*{}+4) failed mod 5", n);
    }
}

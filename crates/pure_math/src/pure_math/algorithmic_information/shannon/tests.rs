use super::*;
use approx::assert_relative_eq;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_categorical_and_renyi_entropy_identities() {
    let uniform = [0.125; 8];
    let h_shannon = categorical_entropy(&uniform).unwrap();
    assert_relative_eq!(h_shannon, 3.0, epsilon = TOLERANCE_HIGH);

    let skewed = [0.5, 0.25, 0.125, 0.125];
    assert_relative_eq!(
        categorical_entropy(&skewed).unwrap(),
        1.75,
        epsilon = TOLERANCE_HIGH
    );

    let h0 = renyi_entropy(&skewed, 0.0).unwrap();
    let h05 = renyi_entropy(&skewed, 0.5).unwrap();
    let h1 = renyi_entropy(&skewed, 1.0).unwrap();
    let h2 = renyi_entropy(&skewed, 2.0).unwrap();
    let h_inf = renyi_entropy(&skewed, f64::INFINITY).unwrap();

    assert_relative_eq!(h0, 2.0, epsilon = TOLERANCE_HIGH);
    assert_relative_eq!(h1, 1.75, epsilon = TOLERANCE_HIGH);
    assert_relative_eq!(h_inf, 1.0, epsilon = TOLERANCE_HIGH);
    assert!(h0 >= h05 && h05 >= h1 && h1 >= h2 && h2 >= h_inf);
}

#[test]
#[verified_engine::verified]
fn test_divergences_and_joint_mutual_information() {
    let p = [0.5, 0.5];
    let q = [0.25, 0.75];
    let kl_pq = kl_divergence(&p, &q).unwrap();
    let expected_kl = 0.5 * (2.0_f64).log2() + 0.5 * (2.0_f64 / 3.0).log2();
    assert_relative_eq!(kl_pq, expected_kl, epsilon = TOLERANCE_HIGH);
    assert_relative_eq!(
        kl_divergence(&p, &p).unwrap(),
        0.0,
        epsilon = TOLERANCE_HIGH
    );

    let disjoint_p = [1.0, 0.0];
    let disjoint_q = [0.0, 1.0];
    assert_eq!(
        kl_divergence(&disjoint_p, &disjoint_q),
        Err(ShannonError::SupportMismatch { index: 0 })
    );
    assert_relative_eq!(
        jensen_shannon_divergence(&disjoint_p, &disjoint_q).unwrap(),
        1.0,
        epsilon = TOLERANCE_HIGH
    );
    assert_relative_eq!(
        jensen_shannon_divergence(&p, &q).unwrap(),
        jensen_shannon_divergence(&q, &p).unwrap(),
        epsilon = TOLERANCE_HIGH
    );

    let mut ws = JointMiWorkspace::new(2, 2);
    let independent = [0.15, 0.35, 0.15, 0.35];
    assert_relative_eq!(
        joint_mutual_information(&independent, 2, 2, &mut ws).unwrap(),
        0.0,
        epsilon = TOLERANCE_HIGH
    );
    let correlated = [0.5, 0.0, 0.0, 0.5];
    assert_relative_eq!(
        joint_mutual_information(&correlated, 2, 2, &mut ws).unwrap(),
        1.0,
        epsilon = TOLERANCE_HIGH
    );
}

#[test]
#[verified_engine::verified]
fn test_closed_form_capacities_match_blahut_arimoto() {
    let p = 0.15;
    let c_bsc = bsc_capacity(p).unwrap();
    let bsc_matrix = [1.0 - p, p, p, 1.0 - p];
    let mut ws = BlahutArimotoWorkspace::new(3, 3);
    let mut opt_r = [0.0; 2];
    let res_bsc =
        blahut_arimoto_capacity(&bsc_matrix, 2, 2, 1e-11, 200, &mut ws, &mut opt_r).unwrap();
    assert!(res_bsc.converged);
    assert_relative_eq!(res_bsc.capacity_bits, c_bsc, epsilon = TOLERANCE_FAST);
    assert_relative_eq!(opt_r[0], 0.5, epsilon = TOLERANCE_FAST);

    let eps = 0.25;
    let c_bec = bec_capacity(eps).unwrap();
    let bec_matrix = [1.0 - eps, eps, 0.0, 0.0, eps, 1.0 - eps];
    let res_bec =
        blahut_arimoto_capacity(&bec_matrix, 2, 3, 1e-11, 200, &mut ws, &mut opt_r).unwrap();
    assert_relative_eq!(res_bec.capacity_bits, c_bec, epsilon = TOLERANCE_FAST);

    // Z-channel with crossover q = 0.5 has exact capacity log2(5/4) bits achieved at r(1) = 2/5.
    let z_matrix = [1.0, 0.0, 0.5, 0.5];
    let res_z = blahut_arimoto_capacity(&z_matrix, 2, 2, 1e-11, 500, &mut ws, &mut opt_r).unwrap();
    assert_relative_eq!(
        res_z.capacity_bits,
        (1.25_f64).log2(),
        epsilon = TOLERANCE_FAST
    );
    assert_relative_eq!(opt_r[1], 0.4, epsilon = 1e-4);

    assert_relative_eq!(awgn_capacity(3.0).unwrap(), 1.0, epsilon = TOLERANCE_HIGH);
    assert_relative_eq!(
        awgn_bandwidth_capacity(1000.0, 7.0).unwrap(),
        3000.0,
        epsilon = TOLERANCE_HIGH
    );
}

#[test]
#[verified_engine::verified]
fn test_shannon_solvers_zero_heap_allocations() {
    let p = [0.4, 0.35, 0.15, 0.10];
    let q = [0.25, 0.25, 0.25, 0.25];
    let joint = [0.3, 0.1, 0.05, 0.55];
    let z_channel = [1.0, 0.0, 0.25, 0.75];
    let mut mi_ws = JointMiWorkspace::new(2, 2);
    let mut ba_ws = BlahutArimotoWorkspace::new(2, 2);
    let mut opt_input = [0.0; 2];

    let ((h, h2, kl, jsd, mi, ba), metrics) = VerifiedEngine::run_verified(|| {
        let h = categorical_entropy(&p).unwrap();
        let h2 = renyi_entropy(&p, 2.0).unwrap();
        let kl = kl_divergence(&p, &q).unwrap();
        let jsd = jensen_shannon_divergence(&p, &q).unwrap();
        let mi = joint_mutual_information(&joint, 2, 2, &mut mi_ws).unwrap();
        let ba = blahut_arimoto_capacity(&z_channel, 2, 2, 1e-10, 300, &mut ba_ws, &mut opt_input)
            .unwrap();
        (h, h2, kl, jsd, mi, ba)
    });

    assert!(metrics.function_calls > 0);
    assert!(h > h2 && kl > 0.0 && jsd > 0.0 && mi > 0.0 && ba.converged);
}

#[test]
#[verified_engine::verified]
fn test_shannon_validation_errors() {
    assert_eq!(
        categorical_entropy(&[]),
        Err(ShannonError::EmptyDistribution)
    );
    assert!(matches!(
        categorical_entropy(&[0.5, 0.6]),
        Err(ShannonError::UnnormalizedDistribution { .. })
    ));
    assert!(matches!(
        categorical_entropy(&[-0.2, 1.2]),
        Err(ShannonError::InvalidProbability { .. })
    ));
    assert!(matches!(
        renyi_entropy(&[0.5, 0.5], -1.0),
        Err(ShannonError::InvalidRenyiOrder { .. })
    ));
    assert!(matches!(
        bsc_capacity(1.2),
        Err(ShannonError::InvalidChannelParameter { .. })
    ));
    assert!(matches!(
        awgn_capacity(-0.1),
        Err(ShannonError::InvalidChannelParameter { .. })
    ));
}

// [cite:algorithmic_information_rust]

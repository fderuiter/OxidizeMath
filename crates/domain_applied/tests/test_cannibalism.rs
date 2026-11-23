//! Integration tests for test cannibalism.
#[cfg(test)]
mod tests {
    use domain_applied::applied::cannibalism::two_dimensional_ode::CannibalismModel;
    use nalgebra::Vector2;
    use pure_math::pure_math::analysis::ode::OdeSystem;

    #[test]
    #[verified_engine::verified]
    fn test_struct_implementation() {
        let n = 100.0;
        let c = 10.0;
        let beta_n = 0.5;
        let beta_c = 0.2;
        let k_n = 0.1;
        let phi_n_c = 5.0;
        let mu_n = 0.3;
        let mu_c = 0.4;

        let model = CannibalismModel::new(beta_n, beta_c, k_n, phi_n_c, mu_n, mu_c);
        let state = Vector2::new(n, c);

        // Time t is unused in this autonomous system
        let derivative = model.derivative(0.0, &state);

        let expected_dn = beta_n * n + beta_c * c - k_n * n - phi_n_c - mu_n * n;
        let expected_dc = k_n * n - mu_c * c;

        assert!(
            (derivative[0] - expected_dn).abs() < math_commons::registry::TOLERANCE_HIGH,
            "Model derivative dN/dt should match expected"
        );
        assert!(
            (derivative[1] - expected_dc).abs() < math_commons::registry::TOLERANCE_HIGH,
            "Model derivative dC/dt should match expected"
        );
    }
}

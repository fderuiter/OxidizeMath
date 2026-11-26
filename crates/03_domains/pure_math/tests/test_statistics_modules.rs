//! Integration tests for new statistics modules

#[cfg(test)]
mod glicko2_tests {
    use pure_math::pure_math::statistics::glicko2::{
        GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility,
        update_rating,
    };

    #[test]
    #[verified_engine::verified]
    fn test_glicko2_basic_update() {
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent = GlickoPlayer::new(
            Rating::new(1400.0).unwrap(),
            RatingDeviation::new(30.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let results = vec![MatchResult::new(opponent, 1.0).unwrap()];
        let tau = SystemConstant::default();

        let updated = update_rating(&player, &results, &tau).unwrap();

        // Rating should increase after winning
        assert!(updated.rating.value() > player.rating.value());
        println!(
            "Glicko-2: Rating changed from {} to {}",
            player.rating.value(),
            updated.rating.value()
        );
    }
}

#[cfg(test)]
mod kelly_tests {
    use pure_math::pure_math::statistics::kelly::{
        Odds, UnitInterval, expected_value, kelly_fraction,
    };

    #[test]
    #[verified_engine::verified]
    fn test_kelly_positive_edge() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();

        let ev = expected_value(&p, &odds);
        assert!(ev > 0.0);

        let kelly = kelly_fraction(&p, &odds).unwrap();
        assert!(kelly.value() > 0.0);
        assert!(kelly.value() < 1.0);
        println!("Kelly: Optimal fraction = {:.3}", kelly.value());
    }
}

#[cfg(test)]
mod tda_tests {
    use pure_math::error::TdaError;
    use pure_math::pure_math::statistics::tda::{
        Point2D, PointCloud, Simplex, SimplicialComplex, TdaFiltrationWorkspace, betti_number_0,
        betti_number_1, betti_numbers, compute_persistence, compute_persistence_from_distances,
        compute_rips_persistence, vietoris_rips_complex, vietoris_rips_from_distances,
    };

    #[test]
    #[verified_engine::verified]
    fn test_tda_basic() {
        // Three points forming a triangle
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 0.866),
        ];
        let cloud = PointCloud::new(points).unwrap();

        // Build complex at radius where all vertices are connected
        let complex = vietoris_rips_complex(&cloud, 1.0).unwrap();
        let (beta0, beta1) = betti_numbers(&complex).unwrap();

        // Should be one connected component
        assert_eq!(beta0, 1);
        println!("TDA: β₀ = {}, β₁ = {}", beta0, beta1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_dimension_levels_and_distance_validation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.5, 0.0),
            Point2D::new(0.5, 0.8),
        ];
        let cloud = PointCloud::new(points).unwrap();
        let dist = cloud.distance_matrix();

        let c0 = vietoris_rips_from_distances(&dist, 0.0).unwrap();
        assert_eq!(c0.dimension(), 0);
        assert_eq!(c0.count_simplices(1), 0);
        assert_eq!(c0.count_simplices(2), 0);

        let c1 = vietoris_rips_from_distances(&dist, 0.96).unwrap();
        assert_eq!(c1.dimension(), 1);
        assert_eq!(c1.count_simplices(2), 0);

        let c2 = vietoris_rips_from_distances(&dist, 1.05).unwrap();
        assert_eq!(c2.dimension(), 2);
        assert_eq!(c2.count_simplices(2), 1);

        let jagged = vec![vec![0.0, 1.0], vec![1.0]];
        assert!(matches!(
            vietoris_rips_from_distances(&jagged, 1.0),
            Err(TdaError::MatrixError { .. })
        ));
        let neg_dist = vec![vec![0.0, -1.0], vec![-1.0, 0.0]];
        assert!(matches!(
            vietoris_rips_from_distances(&neg_dist, 1.0),
            Err(TdaError::MatrixError { .. })
        ));
        let nan_dist = vec![vec![0.0, f64::NAN], vec![f64::NAN, 0.0]];
        assert!(matches!(
            vietoris_rips_from_distances(&nan_dist, 1.0),
            Err(TdaError::MatrixError { .. })
        ));
    }

    #[test]
    #[verified_engine::verified]
    fn test_sparse_vertex_ids_and_isolated_vertices_betti_parity() {
        let mut complex = SimplicialComplex::new();
        for &v in &[100_usize, 500, 1000, 2000, 4000] {
            complex.add_simplex(Simplex::new(vec![v]).unwrap()).unwrap();
        }
        complex
            .add_simplex(Simplex::new(vec![100, 500]).unwrap())
            .unwrap();
        complex
            .add_simplex(Simplex::new(vec![500, 1000]).unwrap())
            .unwrap();
        complex
            .add_simplex(Simplex::new(vec![100, 1000]).unwrap())
            .unwrap();

        assert_eq!(betti_number_0(&complex).unwrap(), 3);
        assert_eq!(betti_number_1(&complex).unwrap(), 1);
        assert_eq!(betti_numbers(&complex).unwrap(), (3, 1));
    }

    #[test]
    #[verified_engine::verified]
    fn test_symmetric_tie_breaking_unsorted_radii_and_workspace_parity() {
        // Regular hexagon of side 1.0: 6 equal side edges at 1.0, 6 chord edges at sqrt(3), 3 diameters at 2.0
        let hex: Vec<Point2D> = (0..6)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::FRAC_PI_3;
                Point2D::new(angle.cos(), angle.sin())
            })
            .collect();
        let cloud = PointCloud::new(hex.clone()).unwrap();
        let dist = cloud.distance_matrix();

        let exact = compute_rips_persistence(&cloud).unwrap();
        let h1: Vec<_> = exact
            .intervals
            .iter()
            .filter(|iv| iv.dimension == 1)
            .collect();
        assert_eq!(h1.len(), 1);
        assert!((h1[0].birth - 1.0).abs() < 1e-9);
        assert!((h1[0].death - 3.0_f64.sqrt()).abs() < 1e-9);

        let sorted_radii = [0.0, 0.5, 1.05, 1.5, 1.8, 2.1];
        let unsorted_radii = [1.8, 0.0, 2.1, 1.05, 0.5, 1.5];
        let b_sorted = compute_persistence(&cloud, &sorted_radii).unwrap();
        let b_unsorted = compute_persistence_from_distances(&dist, &unsorted_radii).unwrap();
        assert_eq!(b_sorted.intervals, b_unsorted.intervals);

        assert!(matches!(
            compute_persistence(&cloud, &[]),
            Err(TdaError::InvalidRadius { .. })
        ));
        assert!(matches!(
            compute_persistence(&cloud, &[0.0, -0.1]),
            Err(TdaError::InvalidRadius { .. })
        ));
        assert!(matches!(
            compute_persistence(&cloud, &[0.0, f64::NAN]),
            Err(TdaError::InvalidRadius { .. })
        ));
        assert!(matches!(
            compute_persistence(&cloud, &[0.0, f64::INFINITY]),
            Err(TdaError::InvalidRadius { .. })
        ));

        let mut ws = TdaFiltrationWorkspace::with_capacity(6);
        ws.load_points_2d(&hex).unwrap();
        for &r in &[0.0, 0.5, 1.05, 1.8, 1.05, 0.5, 2.1] {
            let step = ws.evaluate_radius(6, r).unwrap();
            let c_ref = vietoris_rips_from_distances(&dist, r).unwrap();
            assert_eq!((step.beta0, step.beta1), betti_numbers(&c_ref).unwrap());
            assert_eq!(step.edges, c_ref.count_simplices(1));
            assert_eq!(step.triangles, c_ref.count_simplices(2));
            assert_eq!(ws.active_edges().len(), step.edges);
            assert_eq!(ws.active_triangles().len(), step.triangles);
        }
    }
}

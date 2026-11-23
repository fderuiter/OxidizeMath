#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_depth_guardrail_rejects_above_six() {
    for kind in [
        TilingKind::PenroseP2,
        TilingKind::PenroseP3,
        TilingKind::EinsteinHat,
        TilingKind::EinsteinSpectre,
    ] {
        assert!(generate_aperiodic_tiling(kind, MAX_TILING_DEPTH + 1).is_err());
    }
}

#[test]
fn test_penrose_p2_and_p3_deduplicate_interior_edges_and_grow_with_depth() {
    for kind in [TilingKind::PenroseP2, TilingKind::PenroseP3] {
        let mut prev_tiles = 0usize;
        for depth in 0..=MAX_TILING_DEPTH {
            let patch = generate_aperiodic_tiling(kind, depth).unwrap();
            assert!(
                patch.tiles.len() >= prev_tiles && !patch.tiles.is_empty(),
                "{kind:?} depth {depth} failed tile growth"
            );
            prev_tiles = patch.tiles.len();
            assert_eq!(patch.raw_edge_count, patch.tiles.len() * 4);
            assert!(
                patch.deduplicated_edges.len() < patch.raw_edge_count,
                "{kind:?} depth {depth} must deduplicate shared interior edges"
            );
            assert!(!patch.boundary_paths.is_empty());
            assert!(!patch.interior_paths.is_empty());
            for tile in &patch.tiles {
                assert_eq!(tile.vertices.len(), 4);
                for &pt in &tile.vertices {
                    assert!(patch.bounds.contains(pt));
                }
            }
        }
    }
}

#[test]
fn test_einstein_hat_contains_hats_and_anti_hats_with_zero_interior_gaps() {
    let mut prev_count = 0usize;
    for depth in 0..=MAX_TILING_DEPTH {
        let patch = generate_aperiodic_tiling(TilingKind::EinsteinHat, depth).unwrap();
        assert!(
            patch.tiles.len() > prev_count,
            "EinsteinHat depth {depth} did not grow ({} <= {prev_count})",
            patch.tiles.len()
        );
        prev_count = patch.tiles.len();

        let hat_count = patch
            .tiles
            .iter()
            .filter(|t| t.role == TileRole::Hat)
            .count();
        let anti_count = patch
            .tiles
            .iter()
            .filter(|t| t.role == TileRole::AntiHat)
            .count();
        assert!(hat_count >= 3 && anti_count >= 1);
        assert_eq!(patch.raw_edge_count, patch.tiles.len() * 14);
        assert!(patch.deduplicated_edges.len() < patch.raw_edge_count);
        assert_eq!(
            patch.boundary_paths.len(),
            1,
            "EinsteinHat patch at depth {depth} must form a single simply-connected outer perimeter"
        );
        assert!(patch.boundary_paths[0].closed);
        assert!(!patch.interior_paths.is_empty());
    }
}

#[test]
fn test_einstein_spectre_chiral_monotile_deduplicates_shared_edges() {
    let mut prev_count = 0usize;
    for depth in 0..=MAX_TILING_DEPTH {
        let patch = generate_aperiodic_tiling(TilingKind::EinsteinSpectre, depth).unwrap();
        assert!(
            patch.tiles.len() > prev_count,
            "EinsteinSpectre depth {depth} did not grow ({} <= {prev_count})",
            patch.tiles.len()
        );
        prev_count = patch.tiles.len();

        assert!(patch.tiles.iter().all(|t| t.role == TileRole::Spectre));
        assert_eq!(patch.raw_edge_count, patch.tiles.len() * 14);
        assert!(patch.deduplicated_edges.len() < patch.raw_edge_count);
        assert!(!patch.boundary_paths.is_empty());
        assert!(!patch.interior_paths.is_empty());
    }
}

// [cite:graph_parameters_rust]

//! Zero-allocation integration tests for hyperbolic billiards and Möbius drag updates.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use pure_math::pure_math::hyperbolic::{
    BilliardDomainKind, DiskPoint, HalfPlanePoint, HyperbolicTilingBuffer, MAX_BILLIARD_SEGMENTS,
    MAX_TILING_POLYGONS, Psl2R, Psu11, TiledPolygon, TilingKind, TwinBilliardSystem,
    sample_disk_geodesic_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_zero_alloc_across_100_billiard_reflections_and_mobius_drags() {
    let tiling = HyperbolicTilingBuffer::new(TilingKind::Regular { p: 7, q: 3 }, 3)
        .expect("pre-allocated {7,3} tiling");
    let mut transformed_tiles = vec![TiledPolygon::default(); MAX_TILING_POLYGONS];
    let mut arc_scratch = [[0.0_f64; 2]; 24];

    let mut billiards = TwinBilliardSystem::new(
        BilliardDomainKind::RegularPolygon { p: 8, q: 3 },
        1337,
        1.0e-6,
    )
    .expect("pre-allocated octagon billiard");
    let mut seg_scratch = vec![Default::default(); MAX_BILLIARD_SEGMENTS];
    let mut code_scratch = vec![0u8; MAX_BILLIARD_SEGMENTS];
    let mut sep_scratch = vec![[0.0_f64; 2]; MAX_BILLIARD_SEGMENTS];

    let mut disk_iso = Psu11::identity();
    let mut hp_iso = Psl2R::identity();

    VerifiedEngine::run_verified(|| {
        for step in 0..100usize {
            let phase = (step as f64) * 0.05;
            let from = DiskPoint::new_clamped(0.15 * phase.cos(), 0.15 * phase.sin());
            let to = DiskPoint::new_clamped(0.18 * (phase + 0.1).cos(), 0.18 * (phase + 0.1).sin());
            let delta = Psu11::drag_between(from, to);
            disk_iso = delta.compose(disk_iso);
            disk_iso.renormalize();

            let hp_from = HalfPlanePoint::new_clamped(0.2 * phase.sin(), 1.0 + 0.1 * phase.cos());
            let hp_to = HalfPlanePoint::new_clamped(0.25 * phase.cos(), 1.1);
            hp_iso = Psl2R::drag_between(hp_from, hp_to).compose(hp_iso);
            hp_iso.renormalize();

            let count = tiling.transform_into(disk_iso, &mut transformed_tiles);
            assert!(count > 0);
            let first = &transformed_tiles[0];
            let written =
                sample_disk_geodesic_into(first.vertices[0], first.vertices[1], &mut arc_scratch);
            assert_eq!(written, arc_scratch.len());
        }

        billiards.advance_reflections(100);
        assert!(billiards.total_bounces() >= 100);

        let n_seg = billiards.copy_primary_segments_into(&mut seg_scratch);
        let n_codes = billiards.copy_edge_hits_into(&mut code_scratch);
        let n_sep = billiards.copy_separation_series_into(&mut sep_scratch);
        assert!(n_seg >= 100);
        assert!(n_codes >= 100);
        assert!(n_sep >= 100);
    });
}

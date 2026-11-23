#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_attractor_watertight_tube_export_and_cold_path_stepping() {
    use crate::tabs::geometry_topology::export_utils::{
        count_open_and_nonmanifold_edges, signed_volume,
    };

    let mut plotter = AttractorPlotter {
        tube_radius: 0.25,
        ..Default::default()
    };
    for _ in 0..40 {
        plotter.step_simulation();
    }
    let mesh = plotter
        .export_mesh_3d()
        .expect("3D mesh hook enabled")
        .expect("watertight tube mesh");
    let (open, non_manifold) = count_open_and_nonmanifold_edges(&mesh);
    assert_eq!((open, non_manifold), (0, 0));
    assert!(signed_volume(&mesh) > 0.0);
    assert!(mesh.normals.iter().all(|n| n.is_finite()));

    let colored = plotter
        .export_colored_mesh_3d()
        .expect("colored 3D mesh hook enabled")
        .expect("colored tube mesh");
    assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
    assert_ne!(
        colored.colors.first().copied(),
        colored.colors.last().copied()
    );

    let saved = plotter.save_state().expect("serialize tube_radius");
    let mut restored = AttractorPlotter::default();
    restored
        .try_load_state(&saved)
        .expect("deserialize tube_radius");
    assert!((restored.tube_radius - 0.25).abs() < 1e-6);

    let cap_before = restored.history.capacity();
    restored.history.clear();
    restored.step_simulation();
    assert_eq!(restored.history.capacity(), cap_before);

    let custom_plotter = AttractorPlotter {
        mode: AttractorMode::Custom,
        tube_radius: 0.18,
        ..Default::default()
    };
    let custom_mesh = custom_plotter
        .export_mesh_3d()
        .expect("3D mesh hook enabled in custom mode")
        .expect("custom cold-path preview tube mesh");
    assert_eq!(count_open_and_nonmanifold_edges(&custom_mesh), (0, 0));
    assert!(signed_volume(&custom_mesh) > 0.0);
}

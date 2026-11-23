#![allow(missing_docs)]

use domain_applied::applied::isosurface::{
    IsosurfaceMode, MAX_SYNC_INTERIOR_DIM, Mesh, WatertightGridConfig, WatertightGridError,
    analyze_mesh_manifold, extract_isosurface, sample_watertight_grid,
};
use math_commons::ast::parse;
use oxidize_core::mesh::{export_mesh_to_obj_string, export_mesh_to_stl_binary};
use std::collections::HashMap;
use std::f64::consts::PI;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[verified_engine::verified]
fn assert_closed_manifold(mesh: &Mesh) {
    let stats = analyze_mesh_manifold(mesh);
    assert!(stats.triangle_count > 0, "Expected non-empty mesh");
    assert_eq!(mesh.vertices.len(), mesh.normals.len());
    assert_eq!(
        stats.open_boundary_edges, 0,
        "Expected 0 open boundary half-edges, got {stats:?}"
    );
    assert_eq!(
        stats.non_manifold_edges, 0,
        "Expected 0 non-manifold half-edges, got {stats:?}"
    );
    assert_eq!(
        stats.degenerate_triangles, 0,
        "Expected 0 degenerate triangles, got {stats:?}"
    );
    assert!(
        stats.is_watertight_two_manifold(),
        "Expected closed 2-manifold: {stats:?}"
    );
    assert_eq!(
        3 * stats.triangle_count,
        2 * stats.undirected_edges,
        "Closed 2-manifold must satisfy 3*F == 2*E: {stats:?}"
    );

    let mut signed_vol6 = 0.0_f64;
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        let (a, b, c) = (mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]);
        let (ax, ay, az) = (f64::from(a.x), f64::from(a.y), f64::from(a.z));
        let (bx, by, bz) = (f64::from(b.x), f64::from(b.y), f64::from(b.z));
        let (cx, cy, cz) = (f64::from(c.x), f64::from(c.y), f64::from(c.z));
        signed_vol6 +=
            ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    }
    assert!(
        signed_vol6 > 0.0,
        "Expected positive outward-wound signed volume, got {}",
        signed_vol6 / 6.0
    );
}

#[verified_engine::verified]
fn assert_welded_stl_watertight(stl_bytes: &[u8], expected_triangles: usize) {
    assert_eq!(stl_bytes.len(), 84 + 50 * expected_triangles);
    let tri_count = u32::from_le_bytes(stl_bytes[80..84].try_into().unwrap()) as usize;
    assert_eq!(tri_count, expected_triangles);

    let mut directed: HashMap<([u32; 3], [u32; 3]), usize> = HashMap::new();
    for idx in 0..tri_count {
        let base = 84 + idx * 50 + 12;
        let mut verts = [[0_u32; 3]; 3];
        for (vi, vert) in verts.iter_mut().enumerate() {
            for (ci, coord) in vert.iter_mut().enumerate() {
                let off = base + (vi * 3 + ci) * 4;
                let val = f32::from_le_bytes(stl_bytes[off..off + 4].try_into().unwrap());
                assert!(val.is_finite(), "STL vertex coordinate must be finite");
                *coord = val.to_bits();
            }
        }
        assert!(verts[0] != verts[1] && verts[1] != verts[2] && verts[2] != verts[0]);
        for (u, v) in [
            (verts[0], verts[1]),
            (verts[1], verts[2]),
            (verts[2], verts[0]),
        ] {
            *directed.entry((u, v)).or_insert(0) += 1;
        }
    }

    for (&(u, v), &count) in &directed {
        assert_eq!(count, 1, "Duplicate half-edge in welded STL");
        assert_eq!(
            directed.get(&(v, u)).copied(),
            Some(1),
            "Open boundary half-edge in welded STL"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_tpms_and_clipped_sphere_32_cubed_are_closed_manifolds() {
    let params = HashMap::new();
    let presets = [
        (
            "gyroid_sheet",
            "sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)",
            IsosurfaceMode::Sheet { thickness: 0.45 },
            PI,
        ),
        (
            "schwarz_p_sheet",
            "cos(x) + cos(y) + cos(z)",
            IsosurfaceMode::Sheet { thickness: 0.45 },
            PI,
        ),
        (
            "schwarz_d_sheet",
            "sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)",
            IsosurfaceMode::Sheet { thickness: 0.40 },
            PI,
        ),
        (
            "clipped_sphere_solid",
            "sqrt(x^2 + y^2 + z^2) - 3.6",
            IsosurfaceMode::Solid,
            2.5,
        ),
    ];

    for (label, formula, mode, bound) in presets {
        let expr = parse(formula).unwrap_or_else(|e| panic!("{label} parse failed: {e}"));
        let cfg = WatertightGridConfig::cubic(bound, 32, 0.0, mode);
        let grid = sample_watertight_grid(&cfg, |x, y, z| expr.eval_3d(x, y, z, &params))
            .unwrap_or_else(|e| panic!("{label} sample failed: {e}"));
        assert_eq!((grid.width(), grid.height(), grid.depth()), (34, 34, 34));

        let mesh = extract_isosurface(&grid, 0.0).expect("extract_isosurface");
        assert_closed_manifold(&mesh);

        let stl = export_mesh_to_stl_binary(&mesh).expect("stl export");
        assert_welded_stl_watertight(&stl, mesh.indices.len() / 3);

        let obj = export_mesh_to_obj_string(&mesh).expect("obj export");
        assert!(obj.starts_with("o oxidize_mesh\n"));
        assert_eq!(
            obj.lines().filter(|l| l.starts_with("f ")).count(),
            mesh.indices.len() / 3
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_watertight_grid_validation_and_empty_volume_zero_alloc() {
    let over =
        WatertightGridConfig::cubic(2.0, MAX_SYNC_INTERIOR_DIM + 1, 0.0, IsosurfaceMode::Solid);
    assert!(matches!(
        sample_watertight_grid(&over, |x, y, z| x + y + z),
        Err(WatertightGridError::InvalidResolution { .. })
    ));

    let bad_thick =
        WatertightGridConfig::cubic(2.0, 16, 0.0, IsosurfaceMode::Sheet { thickness: -0.1 });
    assert!(matches!(
        sample_watertight_grid(&bad_thick, |x, _, _| x),
        Err(WatertightGridError::InvalidThickness(_))
    ));

    let non_finite_cfg = WatertightGridConfig::cubic(2.0, 8, 0.0, IsosurfaceMode::Solid);
    assert!(matches!(
        sample_watertight_grid(&non_finite_cfg, |_, _, _| f64::NAN),
        Err(WatertightGridError::NonFiniteSample { .. })
    ));

    let empty_cfg = WatertightGridConfig::cubic(1.0, 16, 0.0, IsosurfaceMode::Solid);
    let empty_grid = sample_watertight_grid(&empty_cfg, |x, y, z| x * x + y * y + z * z + 5.0)
        .expect("positive empty field");
    let (empty_mesh, _) =
        VerifiedEngine::run_verified(|| extract_isosurface(&empty_grid, 0.0).unwrap());
    assert!(empty_mesh.indices.is_empty());
    assert!(!analyze_mesh_manifold(&empty_mesh).is_watertight_two_manifold());
}

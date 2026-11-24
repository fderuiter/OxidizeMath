#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

fn naive_point_to_surface_distance(point: &[f64; 3], surface: &Surface) -> f64 {
    surface
        .vertices
        .iter()
        .map(|v| {
            let (dx, dy, dz) = (point[0] - v[0], point[1] - v[1], point[2] - v[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn test_empty_surface_and_nan_queries() {
    let empty = Surface::from_vertices(vec![]);
    let point = [1.0, 2.0, 3.0];
    let (mut offsets, mut sorted) = ([], []);
    let empty_grid = FlatSpatialGrid::from_surface(&empty, &mut offsets, &mut sorted).unwrap();
    assert!(empty_grid.is_empty());
    assert_eq!(empty_grid.shortest_distance(&point), f64::INFINITY);
    assert_eq!(point_to_surface_distance(&point, &empty), f64::INFINITY);

    let non_empty = Surface::from_vertices(vec![[0.0, 0.0, 0.0]]);
    let (mut off_1, mut sort_1) = ([0_usize; 2], [[0.0_f64; 3]; 1]);
    let grid_1 = FlatSpatialGrid::from_surface(&non_empty, &mut off_1, &mut sort_1).unwrap();
    let nan_pt = [0.0, f64::NAN, 0.0];
    assert!(grid_1.shortest_distance(&nan_pt).is_nan());
    assert!(point_to_surface_distance(&nan_pt, &non_empty).is_nan());
}

#[test]
fn test_single_and_zero_extent_bounds() {
    let surface = Surface::from_vertices(vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let mut offsets = vec![0_usize; FlatSpatialGrid::required_offsets_len(2)];
    let mut sorted = vec![[0.0_f64; 3]; 2];
    let grid = FlatSpatialGrid::from_surface(&surface, &mut offsets, &mut sorted).unwrap();
    assert_eq!(grid.cell_size(), 1.0);
    assert_eq!(grid.shortest_distance(&[4.0, 6.0, 3.0]), 5.0);
    assert_eq!(point_to_surface_distance(&[4.0, 6.0, 3.0], &surface), 5.0);
}

#[test]
fn test_insufficient_capacity_returns_none_and_zero() {
    let white = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let pial = Surface::from_vertices(vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]);
    let (mut short_offsets, mut sorted) = ([0_usize; 4], [[0.0_f64; 3]; 2]);
    assert!(FlatSpatialGrid::from_surface(&white, &mut short_offsets, &mut sorted).is_none());

    let (mut small_ws, mut out) = (ThicknessWorkspace::with_max_vertices(2), [0.0_f64; 2]);
    assert_eq!(
        cortical_thickness_map_into(&white, &pial, &mut small_ws, &mut out),
        0
    );

    let (mut full_ws, mut short_out) = (ThicknessWorkspace::with_max_vertices(3), [0.0_f64; 1]);
    assert_eq!(
        cortical_thickness_map_into(&white, &pial, &mut full_ws, &mut short_out),
        0
    );
    assert_eq!(
        cortical_thickness_map_into(&white, &pial, &mut full_ws, &mut out),
        2
    );
    assert_eq!(out, [2.0, 2.0]);
}

#[test]
fn test_exact_distance_equivalence_random_and_sparse() {
    let vertices: Vec<[f64; 3]> = (0..100)
        .map(|i| {
            [
                ((i * 17) % 100) as f64 * 0.1,
                ((i * 31) % 100) as f64 * 0.1,
                ((i * 47) % 100) as f64 * 0.1,
            ]
        })
        .collect();
    let surface = Surface::from_vertices(vertices);
    let mut offsets = vec![0_usize; FlatSpatialGrid::required_offsets_len(100)];
    let mut sorted = vec![[0.0_f64; 3]; 100];
    let grid = FlatSpatialGrid::from_surface(&surface, &mut offsets, &mut sorted).unwrap();

    for j in 0..60 {
        let query = [
            ((j * 23) % 160) as f64 * 0.1 - 3.0,
            ((j * 41) % 160) as f64 * 0.1 - 3.0,
            ((j * 59) % 160) as f64 * 0.1 - 3.0,
        ];
        let expected = naive_point_to_surface_distance(&query, &surface);
        assert_eq!(expected, grid.shortest_distance(&query));
        assert_eq!(expected, point_to_surface_distance(&query, &surface));
    }

    let sparse = Surface::from_vertices(vec![[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]);
    let mut sp_off = vec![0_usize; FlatSpatialGrid::required_offsets_len(2)];
    let mut sp_sort = vec![[0.0_f64; 3]; 2];
    let sp_grid = FlatSpatialGrid::from_surface(&sparse, &mut sp_off, &mut sp_sort).unwrap();
    let owned_sparse = SpatialGrid::from_surface(&sparse);
    let owned_surface = SpatialGrid::from_surface(&surface);
    for query in [
        [0.0, 0.0, 0.0],
        [250.0, -300.0, 15.0],
        [f64::INFINITY, 0.0, 0.0],
    ] {
        let expected = naive_point_to_surface_distance(&query, &sparse);
        assert_eq!(expected, sp_grid.shortest_distance(&query));
        assert_eq!(expected, owned_sparse.shortest_distance(&query));
    }
    assert_eq!(
        owned_surface.shortest_distance(&[0.5, 0.5, 0.5]),
        grid.shortest_distance(&[0.5, 0.5, 0.5])
    );
    assert_eq!(
        cortical_thickness_map(&surface, &sparse),
        cortical_thickness_mesh(&surface, &sparse)
    );
}

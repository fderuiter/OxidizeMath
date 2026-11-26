/// Maximum vertex block size for zero-allocation stack-chunked surface operators.
pub(super) const SURFACE_CHUNK: usize = 512;

#[inline]
#[verified_engine::verified]
pub(super) fn sq_dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
}

#[verified_engine::verified]
pub(super) fn sequential_internal_energy(vertices: &[[f64; 3]], alpha: f64, beta: f64) -> f64 {
    let n = vertices.len();
    let mut tension = 0.0;
    for i in 0..n.saturating_sub(1) {
        tension += sq_dist3(&vertices[i], &vertices[i + 1]);
    }
    let mut bending = 0.0;
    for i in 1..n.saturating_sub(1) {
        let d2x = vertices[i + 1][0] - 2.0 * vertices[i][0] + vertices[i - 1][0];
        let d2y = vertices[i + 1][1] - 2.0 * vertices[i][1] + vertices[i - 1][1];
        let d2z = vertices[i + 1][2] - 2.0 * vertices[i][2] + vertices[i - 1][2];
        bending += d2x * d2x + d2y * d2y + d2z * d2z;
    }
    alpha * tension + beta * bending
}

#[verified_engine::verified]
pub(super) fn mesh_tension_energy(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> f64 {
    let n = vertices.len();
    let mut half_edge_sum = 0.0;
    for &[i0, i1, i2] in faces {
        if i0 < n && i1 < n && i2 < n {
            half_edge_sum += sq_dist3(&vertices[i0], &vertices[i1])
                + sq_dist3(&vertices[i1], &vertices[i2])
                + sq_dist3(&vertices[i2], &vertices[i0]);
        }
    }
    0.5 * half_edge_sum
}

#[verified_engine::verified]
pub(super) fn accumulate_chunk_topology(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    base: usize,
    chunk_len: usize,
    nbr_sum: &mut [[f64; 3]; SURFACE_CHUNK],
    nbr_deg: &mut [u32; SURFACE_CHUNK],
    normals: &mut [[f64; 3]; SURFACE_CHUNK],
) {
    let n = vertices.len();
    let end = base + chunk_len;
    for &[i0, i1, i2] in faces {
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        let (v0, v1, v2) = (vertices[i0], vertices[i1], vertices[i2]);
        let (ux, uy, uz) = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]);
        let (wx, wy, wz) = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]);
        let fn_vec = [uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx];

        for (vi, va, vb) in [(i0, v1, v2), (i1, v2, v0), (i2, v0, v1)] {
            if vi >= base && vi < end {
                let k = vi - base;
                nbr_sum[k][0] += va[0] + vb[0];
                nbr_sum[k][1] += va[1] + vb[1];
                nbr_sum[k][2] += va[2] + vb[2];
                nbr_deg[k] += 2;
                normals[k][0] += fn_vec[0];
                normals[k][1] += fn_vec[1];
                normals[k][2] += fn_vec[2];
            }
        }
    }
}

#[verified_engine::verified]
pub(super) fn mesh_bending_energy(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> f64 {
    let n = vertices.len();
    let mut bending = 0.0;
    let mut nbr_sum = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut nbr_deg = [0_u32; SURFACE_CHUNK];
    let mut normals = [[0.0_f64; 3]; SURFACE_CHUNK];

    let mut base = 0;
    while base < n {
        let chunk_len = (n - base).min(SURFACE_CHUNK);
        for k in 0..chunk_len {
            nbr_sum[k] = [0.0; 3];
            nbr_deg[k] = 0;
            normals[k] = [0.0; 3];
        }
        accumulate_chunk_topology(
            vertices,
            faces,
            base,
            chunk_len,
            &mut nbr_sum,
            &mut nbr_deg,
            &mut normals,
        );
        for k in 0..chunk_len {
            if nbr_deg[k] > 0 {
                let inv_d = 1.0 / f64::from(nbr_deg[k]);
                let v = vertices[base + k];
                let lx = nbr_sum[k][0] * inv_d - v[0];
                let ly = nbr_sum[k][1] * inv_d - v[1];
                let lz = nbr_sum[k][2] * inv_d - v[2];
                bending += lx * lx + ly * ly + lz * lz;
            }
        }
        base += SURFACE_CHUNK;
    }
    bending
}

#[verified_engine::verified]
pub(super) fn accumulate_biharmonic_terms(
    faces: &[[usize; 3]],
    base: usize,
    chunk_len: usize,
    lap: &[[f64; 3]; SURFACE_CHUNK],
    nbr_deg: &[u32; SURFACE_CHUNK],
    biharm: &mut [[f64; 3]; SURFACE_CHUNK],
) {
    let end = base + chunk_len;
    for &[i0, i1, i2] in faces {
        for (vi, ja, jb) in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            if vi >= base && vi < end {
                let k = vi - base;
                for j in [ja, jb] {
                    if j >= base && j < end && nbr_deg[j - base] > 0 {
                        let inv_dj = 1.0 / f64::from(nbr_deg[j - base]);
                        let lj = lap[j - base];
                        biharm[k][0] += lj[0] * inv_dj;
                        biharm[k][1] += lj[1] * inv_dj;
                        biharm[k][2] += lj[2] * inv_dj;
                    }
                }
            }
        }
    }
}

#[verified_engine::verified]
pub(super) fn evolve_sequential_polyline(vertices: &mut [[f64; 3]], step: f64) {
    let n = vertices.len();
    if n < 2 {
        return;
    }
    let mut base = 0;
    let mut disp = [[0.0_f64; 3]; SURFACE_CHUNK];
    while base < n {
        let chunk_len = (n - base).min(SURFACE_CHUNK);
        for (k, d) in disp.iter_mut().take(chunk_len).enumerate() {
            let i = base + k;
            let mut f = [0.0; 3];
            if i + 1 < n {
                f[0] += vertices[i + 1][0] - vertices[i][0];
                f[1] += vertices[i + 1][1] - vertices[i][1];
                f[2] += vertices[i + 1][2] - vertices[i][2];
            }
            if i > 0 {
                f[0] += vertices[i - 1][0] - vertices[i][0];
                f[1] += vertices[i - 1][1] - vertices[i][1];
                f[2] += vertices[i - 1][2] - vertices[i][2];
            }
            *d = [step * f[0], step * f[1], step * f[2]];
        }
        for (k, d) in disp.iter().take(chunk_len).enumerate() {
            vertices[base + k][0] += d[0];
            vertices[base + k][1] += d[1];
            vertices[base + k][2] += d[2];
        }
        base += SURFACE_CHUNK;
    }
}

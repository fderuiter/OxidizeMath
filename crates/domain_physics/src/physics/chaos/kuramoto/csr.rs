#[derive(Clone, Copy)]
pub(super) struct CsrAdjacencyRef<'a> {
    pub(super) row_offsets: &'a [usize],
    pub(super) col_indices: &'a [usize],
    pub(super) weights: &'a [f64],
}

#[verified_engine::verified]
pub(super) fn populate_natural_frequencies(omegas: &mut [f64], spread: f64) {
    let n = omegas.len();
    if n <= 1 {
        if let Some(first) = omegas.first_mut() {
            *first = 0.0;
        }
        return;
    }
    let denom = (n - 1) as f64;
    for (i, omega) in omegas.iter_mut().enumerate() {
        let normalized = (2.0 * (i as f64) / denom) - 1.0;
        *omega = spread * normalized;
    }
}

#[verified_engine::verified]
pub(super) fn stage_phases(n: usize, base: &[f64], deriv: &[f64], step: f64, out: &mut [f64]) {
    for i in 0..n {
        out[i] = base[i] + step * deriv[i];
    }
}

#[verified_engine::verified]
pub(super) fn compute_derivatives(
    n: usize,
    coupling_scale: f64,
    omegas: &[f64],
    csr: CsrAdjacencyRef<'_>,
    phases: &[f64],
    out: &mut [f64],
) {
    for i in 0..n {
        let theta_i = phases[i];
        let mut coupling_sum = 0.0;
        let start = csr.row_offsets[i];
        let end = csr.row_offsets[i + 1];
        for idx in start..end {
            let j = csr.col_indices[idx];
            coupling_sum += csr.weights[idx] * (phases[j] - theta_i).sin();
        }
        out[i] = omegas[i] + coupling_scale * coupling_sum;
    }
}

#[verified_engine::verified]
pub(super) fn rebuild_csr_adjacency(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
    row_offsets: &mut Vec<usize>,
    col_indices: &mut Vec<usize>,
    weights: &mut Vec<f64>,
) {
    row_offsets.clear();
    row_offsets.resize(num_nodes + 1, 0);
    col_indices.clear();
    weights.clear();
    if num_nodes == 0 {
        return;
    }

    let mut clean: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len());
    for &(u, v, w) in edges {
        if u < num_nodes && v < num_nodes && u != v && w.is_finite() {
            let (lo, hi) = if u < v { (u, v) } else { (v, u) };
            clean.push((lo, hi, w));
        }
    }
    clean.sort_unstable_by_key(|a| (a.0, a.1));
    clean.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    for &(u, v, _) in &clean {
        row_offsets[u + 1] += 1;
        row_offsets[v + 1] += 1;
    }
    for i in 0..num_nodes {
        row_offsets[i + 1] += row_offsets[i];
    }

    let total_directed = row_offsets[num_nodes];
    col_indices.resize(total_directed, 0);
    weights.resize(total_directed, 0.0);
    let mut write_heads = row_offsets[..num_nodes].to_vec();
    for &(u, v, w) in &clean {
        let pos_u = write_heads[u];
        col_indices[pos_u] = v;
        weights[pos_u] = w;
        write_heads[u] += 1;

        let pos_v = write_heads[v];
        col_indices[pos_v] = u;
        weights[pos_v] = w;
        write_heads[v] += 1;
    }
}

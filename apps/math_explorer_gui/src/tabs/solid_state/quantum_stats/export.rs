//! Deterministic dataset exports and Markdown/CSV thermodynamic datasheets for `QuantumStatsPhononTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{QuantumPhononView, QuantumStatsPhononTool};
use crate::framework::TraceSeries;
use std::fmt::Write as _;

/// Collects deterministic physical 2D curve datasets for CSV, JSON, PNG, and SVG exports.
#[must_use]
pub(crate) fn collect_export_datasets(
    tool: &QuantumStatsPhononTool,
) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut ds = Vec::with_capacity(10);
    if !tool.fd_points.is_empty() {
        ds.push((
            "Fermi-Dirac Occupancy f_FD(E)".to_string(),
            tool.fd_points.clone(),
        ));
    }
    if !tool.be_points.is_empty() {
        ds.push((
            "Bose-Einstein Occupancy f_BE(E)".to_string(),
            tool.be_points.clone(),
        ));
    }
    if !tool.mb_points.is_empty() {
        ds.push((
            "Maxwell-Boltzmann Occupancy f_MB(E)".to_string(),
            tool.mb_points.clone(),
        ));
    }
    if !tool.acoustic_points.is_empty() {
        ds.push((
            "Acoustic Phonon Dispersion (meV)".to_string(),
            tool.acoustic_points.clone(),
        ));
    }
    if !tool.optical_points.is_empty() {
        ds.push((
            "Optical Phonon Dispersion (meV)".to_string(),
            tool.optical_points.clone(),
        ));
    }
    if !tool.magnon_points.is_empty() {
        ds.push((
            "Heisenberg Magnon Dispersion (meV)".to_string(),
            tool.magnon_points.clone(),
        ));
    }
    if !tool.debye_cv_points.is_empty() {
        ds.push((
            "Debye Low-T Heat Capacity C_V/(N k_B)".to_string(),
            tool.debye_cv_points.clone(),
        ));
    }
    if !tool.dielectric_points.is_empty() {
        ds.push((
            "Thomas-Fermi Dielectric epsilon(q)".to_string(),
            tool.dielectric_points.clone(),
        ));
    }
    if !tool.yukawa_points.is_empty() {
        ds.push((
            "Yukawa Screened Potential V(r)".to_string(),
            tool.yukawa_points.clone(),
        ));
    }
    ds.extend(tool.snapshot_manager.export_datasets());
    ds
}

/// Builds active `TraceSeries` corresponding to the selected `QuantumPhononView`.
#[must_use]
pub(crate) fn build_active_trace_series(tool: &QuantumStatsPhononTool) -> Vec<TraceSeries> {
    match tool.plot_view {
        QuantumPhononView::QuantumOccupancy => {
            let mut v = vec![
                TraceSeries::new("Fermi-Dirac", tool.fd_points.clone()),
                TraceSeries::new("Maxwell-Boltzmann", tool.mb_points.clone()),
            ];
            if !tool.be_points.is_empty() {
                v.push(TraceSeries::new("Bose-Einstein", tool.be_points.clone()));
            }
            v
        }
        QuantumPhononView::PhononAndDebye => vec![
            TraceSeries::new("Acoustic Phonon", tool.acoustic_points.clone()),
            TraceSeries::new("Optical Phonon", tool.optical_points.clone()),
            TraceSeries::new("Magnon E(k)", tool.magnon_points.clone()),
        ],
        QuantumPhononView::ScreeningAndYukawa => vec![
            TraceSeries::new("Thomas-Fermi ε(q)", tool.dielectric_points.clone()),
            TraceSeries::new("Yukawa V(r)", tool.yukawa_points.clone()),
        ],
    }
}

/// Formats a standalone Markdown quantum-statistics and lattice datasheet.
#[must_use]
pub fn format_markdown_card(tool: &QuantumStatsPhononTool) -> String {
    let m = &tool.metrics;
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "# Quantum Statistics, Phonon & Screening Datasheet\n");
    let _ = writeln!(
        out,
        "- **Parameters**: Mode = {}, T = {:.2} K, μ = {:.4} eV, E_min = {:.4} eV, Θ_D = {:.1} K, k_TF = {:.3} Å⁻¹, J = {:.2} meV\n",
        tool.stats_mode.label(),
        tool.temperature_k,
        tool.chemical_potential_ev,
        tool.energy_min_ev,
        tool.debye_theta_k,
        tool.k_tf_inv_a,
        tool.exchange_j_mev
    );
    let _ = writeln!(out, "## Canonical Ensemble & Lattice Summary\n");
    let _ = writeln!(out, "| Observable | Value | Unit |\n| :--- | :--- | :--- |");
    for (name, val, unit) in [
        ("Thermal Energy k_B T", m.thermal_energy_mev, "meV"),
        (
            "Canonical Partition Function Z",
            m.partition_z,
            "dimensionless",
        ),
        ("Helmholtz Free Energy F", m.free_energy_ev, "eV"),
        ("Canonical Internal Energy U", m.internal_energy_ev, "eV"),
        ("Canonical Entropy S", m.entropy_mev_per_k, "meV/K"),
        (
            "Debye Heat Capacity C_V/(N k_B)",
            m.debye_cv_per_nk,
            "per atom",
        ),
        (
            "Thomas-Fermi Screening Length λ_TF",
            m.screening_length_a,
            "Å",
        ),
        (
            "Heisenberg Dimer Ground Energy",
            m.heisenberg_dimer_mev,
            "meV",
        ),
        ("Bosonic CCR Norm [b, b†]", m.boson_ccr_norm, "norm"),
    ] {
        let _ = writeln!(out, "| {name} | {val:.5} | {unit} |");
    }
    out
}

/// Formats a standalone CSV quantum-statistics and lattice datasheet.
#[must_use]
pub fn format_csv_card(tool: &QuantumStatsPhononTool) -> String {
    let m = &tool.metrics;
    let mut out = String::with_capacity(896);
    let _ = writeln!(out, "section,key,value,unit");
    let _ = writeln!(out, "parameter,temperature_k,{:.6},K", tool.temperature_k);
    let _ = writeln!(
        out,
        "parameter,chemical_potential_ev,{:.6},eV",
        tool.chemical_potential_ev
    );
    let _ = writeln!(out, "parameter,energy_min_ev,{:.6},eV", tool.energy_min_ev);
    let _ = writeln!(out, "parameter,debye_theta_k,{:.6},K", tool.debye_theta_k);
    let _ = writeln!(
        out,
        "parameter,k_tf_inv_a,{:.6},1/angstrom",
        tool.k_tf_inv_a
    );
    let _ = writeln!(
        out,
        "parameter,exchange_j_mev,{:.6},meV",
        tool.exchange_j_mev
    );
    let _ = writeln!(
        out,
        "metric,thermal_energy_mev,{:.6},meV",
        m.thermal_energy_mev
    );
    let _ = writeln!(out, "metric,partition_z,{:.6},dimensionless", m.partition_z);
    let _ = writeln!(out, "metric,free_energy_ev,{:.6},eV", m.free_energy_ev);
    let _ = writeln!(
        out,
        "metric,internal_energy_ev,{:.6},eV",
        m.internal_energy_ev
    );
    let _ = writeln!(
        out,
        "metric,entropy_mev_per_k,{:.6},meV/K",
        m.entropy_mev_per_k
    );
    let _ = writeln!(
        out,
        "metric,debye_cv_per_nk,{:.6},dimensionless",
        m.debye_cv_per_nk
    );
    let _ = writeln!(
        out,
        "metric,screening_length_a,{:.6},angstrom",
        m.screening_length_a
    );
    let _ = writeln!(
        out,
        "metric,heisenberg_dimer_mev,{:.6},meV",
        m.heisenberg_dimer_mev
    );
    let _ = writeln!(out, "metric,boson_ccr_norm,{:.6},norm", m.boson_ccr_norm);
    out
}

// [cite:mmwave_radiotherapy_setup]

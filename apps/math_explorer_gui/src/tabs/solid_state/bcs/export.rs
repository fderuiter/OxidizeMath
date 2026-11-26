//! Deterministic dataset exports and Markdown/CSV superconducting datasheets for `BcsSuperconductivityTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BcsPlotView, BcsSuperconductivityTool};
use crate::framework::TraceSeries;
use std::fmt::Write as _;

/// Collects deterministic physical 2D curve datasets for CSV, JSON, PNG, and SVG exports.
#[must_use]
pub(crate) fn collect_export_datasets(
    tool: &BcsSuperconductivityTool,
) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut ds = Vec::with_capacity(8);
    if !tool.gap_vs_temp_points.is_empty() {
        ds.push((
            "BCS Gap Delta(T) (meV)".to_string(),
            tool.gap_vs_temp_points.clone(),
        ));
    }
    if !tool.dispersion_points.is_empty() {
        ds.push((
            "Bogoliubov Quasiparticle E_k (meV)".to_string(),
            tool.dispersion_points.clone(),
        ));
    }
    if !tool.u_sq_points.is_empty() {
        ds.push((
            "Bogoliubov Coherence u_k^2".to_string(),
            tool.u_sq_points.clone(),
        ));
    }
    if !tool.v_sq_points.is_empty() {
        ds.push((
            "Bogoliubov Coherence v_k^2".to_string(),
            tool.v_sq_points.clone(),
        ));
    }
    if !tool.nis_didv_points.is_empty() {
        ds.push((
            "NIS Conductance dI/dV (G_NS/G_NN)".to_string(),
            tool.nis_didv_points.clone(),
        ));
    }
    if !tool.dynes_dos_points.is_empty() {
        ds.push((
            "Dynes DOS Ratio N_S(E)/N(0)".to_string(),
            tool.dynes_dos_points.clone(),
        ));
    }
    if !tool.frohlich_points.is_empty() {
        ds.push((
            "Frohlich Coupling |M_q|^2 (eV^2)".to_string(),
            tool.frohlich_points.clone(),
        ));
    }
    ds.extend(tool.snapshot_manager.export_datasets());
    ds
}

/// Builds active `TraceSeries` corresponding to the selected `BcsPlotView`.
#[must_use]
pub(crate) fn build_active_trace_series(tool: &BcsSuperconductivityTool) -> Vec<TraceSeries> {
    match tool.plot_view {
        BcsPlotView::GapVsTemperature => {
            vec![TraceSeries::new("Δ(T)", tool.gap_vs_temp_points.clone())]
        }
        BcsPlotView::QuasiparticleAndCoherence => vec![
            TraceSeries::new("E_k", tool.dispersion_points.clone()),
            TraceSeries::new("u_k²", tool.u_sq_points.clone()),
            TraceSeries::new("v_k²", tool.v_sq_points.clone()),
        ],
        BcsPlotView::NisTunnelingAndCoupling => vec![
            TraceSeries::new("NIS dI/dV", tool.nis_didv_points.clone()),
            TraceSeries::new("Dynes DOS", tool.dynes_dos_points.clone()),
        ],
    }
}

/// Formats a standalone Markdown BCS superconducting datasheet.
#[must_use]
pub fn format_markdown_card(tool: &BcsSuperconductivityTool) -> String {
    let m = &tool.metrics;
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "# BCS Superconductivity & NIS Tunneling Datasheet\n");
    let _ = writeln!(
        out,
        "- **Operating Conditions**: T = {:.3} K, ħω_D = {:.2} meV, N(0)V = {:.3}, N(0) = {:.2} 1/eV, Γ = {:.3} meV, α_F = {:.2}\n",
        tool.temperature_k,
        tool.debye_energy_mev,
        tool.coupling_nv,
        tool.dos_per_ev,
        tool.dynes_gamma_mev,
        tool.frohlich_alpha
    );
    let _ = writeln!(out, "## Derived Superconducting Observables\n");
    let _ = writeln!(out, "| Observable | Value | Unit |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    let _ = writeln!(out, "| Energy Gap Δ(T) | {:.4} | meV |", m.delta_t_mev);
    let _ = writeln!(
        out,
        "| Zero-Temperature Gap Δ(0) | {:.4} | meV |",
        m.delta_zero_mev
    );
    let _ = writeln!(out, "| Critical Temperature T_c | {:.4} | K |", m.tc_kelvin);
    let _ = writeln!(
        out,
        "| Universal Ratio 2Δ(0)/(k_B T_c) | {:.4} | dimensionless |",
        m.universal_ratio
    );
    let _ = writeln!(
        out,
        "| Zero-Bias NIS Conductance G_NS(0)/G_NN | {:.4} | G_NN |",
        m.zero_bias_conductance
    );
    let _ = writeln!(
        out,
        "| Fröhlich Vertex |M(q_D)|² | {:.4e} | eV² |",
        m.frohlich_mq_sq_ev2
    );
    let _ = writeln!(
        out,
        "| Fermionic CAR Norm {{c, c†}} | {:.4} | norm |",
        m.cooper_car_norm
    );
    out
}

/// Formats a standalone CSV BCS superconducting datasheet.
#[must_use]
pub fn format_csv_card(tool: &BcsSuperconductivityTool) -> String {
    let m = &tool.metrics;
    let mut out = String::with_capacity(768);
    let _ = writeln!(out, "section,key,value,unit");
    let _ = writeln!(out, "parameter,temperature_k,{:.6},K", tool.temperature_k);
    let _ = writeln!(
        out,
        "parameter,debye_energy_mev,{:.6},meV",
        tool.debye_energy_mev
    );
    let _ = writeln!(
        out,
        "parameter,coupling_nv,{:.6},dimensionless",
        tool.coupling_nv
    );
    let _ = writeln!(out, "parameter,dos_per_ev,{:.6},1/eV", tool.dos_per_ev);
    let _ = writeln!(
        out,
        "parameter,dynes_gamma_mev,{:.6},meV",
        tool.dynes_gamma_mev
    );
    let _ = writeln!(
        out,
        "parameter,frohlich_alpha,{:.6},dimensionless",
        tool.frohlich_alpha
    );
    let _ = writeln!(out, "metric,delta_t_mev,{:.6},meV", m.delta_t_mev);
    let _ = writeln!(out, "metric,delta_zero_mev,{:.6},meV", m.delta_zero_mev);
    let _ = writeln!(out, "metric,tc_kelvin,{:.6},K", m.tc_kelvin);
    let _ = writeln!(
        out,
        "metric,universal_ratio_2delta_kbtc,{:.6},dimensionless",
        m.universal_ratio
    );
    let _ = writeln!(
        out,
        "metric,zero_bias_nis_conductance,{:.6},G_NN",
        m.zero_bias_conductance
    );
    let _ = writeln!(
        out,
        "metric,frohlich_mq_sq,{:.6e},eV^2",
        m.frohlich_mq_sq_ev2
    );
    let _ = writeln!(out, "metric,cooper_car_norm,{:.6},norm", m.cooper_car_norm);
    out
}

// [cite:mmwave_radiotherapy_setup]

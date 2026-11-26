//! Theoretical equations, peer-reviewed citations, and curated starter gallery presets
//! for [`super::tool::MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// [`TheoryDescribable`] metadata for cross-domain 2D MRI k-space Fourier imaging.
#[derive(Default, Clone, Debug)]
pub struct MriKSpaceScannerTheory;

impl TheoryDescribable for MriKSpaceScannerTheory {
    fn theory_description(&self) -> String {
        "Magnetic resonance spatial encoding maps macroscopic spin-echo transverse magnetization \
         into 2D spatial-frequency space (k-space) via time-integrated magnetic field gradients \
         k(t) = (gamma / 2pi) integral G(tau) d tau. Low spatial frequencies near the DC origin \
         govern bulk T1/T2 tissue contrast, whereas outer high-k bins encode sharp cortical and \
         morphogenetic boundaries. Regular phase-encode undersampling violates the Nyquist field \
         of view criterion and produces coherent alias ghosts, while non-Cartesian and chaotic \
         Lorenz trajectories disperse undersampling energy into incoherent noise-like residuals."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional magnetic resonance k space Fourier encoding, spin echo relaxation \
         contrast, and chaotic Lorenz compressed sensing trajectories"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Lauterbur, P. C. (1973). Image Formation by Induced Local Interactions. Nature, 242, \
         190-191; Mansfield, P. (1977). Multi-planar image formation using NMR spin echoes. \
         J. Phys. C, 10(3), L55; Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: \
         The application of compressed sensing for rapid MR imaging. Magn. Reson. Med., 58(6), \
         1182-1195; Turing, A. M. (1952). The Chemical Basis of Morphogenesis. Phil. Trans. R. \
         Soc. B, 237(641), 37-72; Lorenz, E. N. (1963). Deterministic Nonperiodic Flow. J. \
         Atmos. Sci., 20(2), 130-141."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "k_space_encoding".to_string(),
            "Time-integrated magnetic field gradients map nuclear Larmor phase directly onto \
             2D spatial frequencies k_x and k_y [cycles/m]."
                .to_string(),
        );
        descs.insert(
            "spin_echo_contrast".to_string(),
            "Compartment intensity scales with Bloch longitudinal recovery (1 - exp(-TR/T1)) \
             and transverse spin-echo decay exp(-TE/T2)."
                .to_string(),
        );
        descs.insert(
            "keyhole_vs_highpass".to_string(),
            "Central DC k-space bins dominate image contrast and PSNR, whereas peripheral \
             high-frequency bins isolate sharp anatomical edges."
                .to_string(),
        );
        descs.insert(
            "chaotic_compressed_sensing".to_string(),
            "Projecting a deterministic 3D Lorenz strange attractor into variable-density \
             k-space replaces coherent Nyquist ghosts with incoherent aliasing."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"S(k_x, k_y) = \iint \rho(x,y)\left(1 - e^{-TR/T_1}\right)e^{-TE/T_2}\,e^{-i 2\pi (k_x x + k_y y)}\,dx\,dy"
                .to_string(),
        )
    }
}

/// JSON preset envelope for MRI 2D k-Space Keyhole Contrast Preservation.
pub const PRESET_MRI_KSPACE_KEYHOLE: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "MRI 2D k-Space - Keyhole Contrast Preservation", "domain": "mri", "timestamp": "2026-11-21T00:00:00Z" },
  "config": {
    "tr_s": 1.8, "te_s": 0.065, "t1_scale": 1.0, "t2_scale": 1.0,
    "phantom": "SheppLogan", "trajectory": "FullCartesian", "mask_filter": "Keyhole",
    "brush_mode": "Erase", "brush_radius": 3, "cartesian_stride": 2, "radial_spokes": 24,
    "spiral_turns": 14.0, "lorenz_steps": 1800, "lorenz_rho": 28.0,
    "cutoff_radius": 10, "random_keep_ratio": 0.35, "rng_seed": 42, "turing_steps": 120
  }
}"#;

/// JSON preset envelope for MRI 2D k-Space Cartesian 2x Nyquist Ghosting.
pub const PRESET_MRI_KSPACE_NYQUIST_GHOSTING: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "MRI 2D k-Space - Cartesian 2x Nyquist Ghosting", "domain": "mri", "timestamp": "2026-11-21T00:00:00Z" },
  "config": {
    "tr_s": 1.6, "te_s": 0.055, "t1_scale": 1.0, "t2_scale": 1.0,
    "phantom": "SheppLogan", "trajectory": "CartesianLines", "mask_filter": "TrajectoryOnly",
    "brush_mode": "Paint", "brush_radius": 2, "cartesian_stride": 2, "radial_spokes": 24,
    "spiral_turns": 14.0, "lorenz_steps": 1800, "lorenz_rho": 28.0,
    "cutoff_radius": 10, "random_keep_ratio": 0.35, "rng_seed": 42, "turing_steps": 120
  }
}"#;

/// JSON preset envelope for MRI 2D k-Space Lorenz Chaotic Compressed Sensing on Turing Spots.
pub const PRESET_MRI_KSPACE_LORENZ_CS: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "MRI 2D k-Space - Lorenz Chaotic Compressed Sensing", "domain": "mri", "timestamp": "2026-11-21T00:00:00Z" },
  "config": {
    "tr_s": 2.0, "te_s": 0.08, "t1_scale": 1.0, "t2_scale": 1.0,
    "phantom": "TuringMorphogenesis", "trajectory": "LorenzChaotic", "mask_filter": "TrajectoryOnly",
    "brush_mode": "Erase", "brush_radius": 3, "cartesian_stride": 2, "radial_spokes": 24,
    "spiral_turns": 14.0, "lorenz_steps": 2200, "lorenz_rho": 28.0,
    "cutoff_radius": 10, "random_keep_ratio": 0.35, "rng_seed": 77, "turing_steps": 180
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mri_kspace_keyhole",
        title: "Keyhole Contrast Preservation",
        description: "Circular low-pass k-space keyhole preserves bulk T1/T2 brain contrast from central DC frequencies while gently softening cortical edges.",
        category: "Medical Physics",
        icon: "🎯",
        domain: "mri",
        tool_name: Some("MriKSpaceScannerTool"),
        preset_generator: || PRESET_MRI_KSPACE_KEYHOLE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mri_kspace_nyquist_ghosting",
        title: "Cartesian 2x Nyquist Ghosting",
        description: "Skipping every second phase-encode line violates the Nyquist FOV criterion and folds coherent half-FOV ghosts across the slice.",
        category: "Medical Physics",
        icon: "👻",
        domain: "mri",
        tool_name: Some("MriKSpaceScannerTool"),
        preset_generator: || PRESET_MRI_KSPACE_NYQUIST_GHOSTING,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mri_kspace_lorenz_cs",
        title: "Lorenz Chaotic Compressed Sensing",
        description: "Chaotic 3D Lorenz attractor trajectory samples a 2D Turing morphogenesis phantom incoherently, suppressing structured alias ghosts.",
        category: "Medical Physics",
        icon: "🦋",
        domain: "mri",
        tool_name: Some("MriKSpaceScannerTool"),
        preset_generator: || PRESET_MRI_KSPACE_LORENZ_CS,
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// GUI [`TheoryDescribable`] wrapper for `domain_physics::physics::mri::bloch::BlochSimulator`.
#[derive(Default, Clone, Debug)]
pub struct MriBlochTheory;

impl TheoryDescribable for MriBlochTheory {
    fn theory_description(&self) -> String {
        "Phenomenological Bloch equations model macroscopic nuclear magnetization precession in an \
         applied magnetic field alongside longitudinal spin-lattice recovery (T1) toward thermal \
         equilibrium M0 and transverse spin-spin dephasing (T2)."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Nuclear magnetic resonance Bloch relaxation and Larmor precession".to_string()
    }

    fn theory_citation(&self) -> String {
        "Bloch, F. (1946). Nuclear Induction. Physical Review, 70(7-8), 460-474.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "t1_relaxation".to_string(),
            "Spin-lattice longitudinal recovery time constant toward thermal equilibrium M0 [s]."
                .to_string(),
        );
        descs.insert(
            "t2_relaxation".to_string(),
            "Spin-spin transverse dephasing time constant governing FID envelope decay [s]."
                .to_string(),
        );
        descs.insert(
            "larmor_precession".to_string(),
            "Torque-driven precession of magnetization vector around applied magnetic field B."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{d\vec{M}}{dt} = \gamma (\vec{M} \times \vec{B}) - \frac{M_x \hat{i} + M_y \hat{j}}{T_2} - \frac{M_z - M_0}{T_1}\hat{k}"
                .to_string(),
        )
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mri_bloch_relaxation",
        title: "Off-Resonance Bloch Precession",
        description: "dM/dt = γ(M × B) - M_xy/T2 - (Mz - M0)k/T1: Transverse FID decay coupled with longitudinal T1 recovery.",
        category: "Medical Physics",
        icon: "🧲",
        domain: "mri",
        tool_name: Some("MriTool"),
        preset_generator: || crate::starter_gallery::PRESET_MRI_BLOCH_RELAXATION,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "mri_spin_echo_refocusing",
        title: "MRI Spin-Echo Refocusing",
        description: "Counterfactual shadow twin forked mid-decay with a 180° Hahn refocusing pulse recovering ΔM_xy transverse echo.",
        category: "Counterfactual MRI",
        icon: "⑂",
        domain: "mri",
        tool_name: Some("MriTool"),
        preset_generator: || crate::starter_gallery::PRESET_MRI_SPIN_ECHO_REFOCUSING,
    }
}

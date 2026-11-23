//! Cold-path INRIA-standard 3DGS binary `.ply` scene exporter for [`super::GaussianSplattingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GaussianSplattingTool;
use oxidize_core::mesh::{export_splats_to_ply_binary, GaussianSplatPly};

impl GaussianSplattingTool {
    /// Converts the active 3D Gaussian scene at `self.scale` into an INRIA-standard binary `.ply`.
    pub fn serialize_splats_ply_binary(&self) -> Result<Vec<u8>, String> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err("Gaussian splat scale must be finite and positive.".to_string());
        }
        let scale_factor = self.scale.clamp(0.1, 10.0);
        let mut splats = Vec::with_capacity(self.scene.gaussians.len());
        for g in &self.scene.gaussians {
            let record: GaussianSplatPly = g.to_splat_ply(scale_factor);
            splats.push(record);
        }
        export_splats_to_ply_binary(&splats).map_err(|e| e.to_string())
    }
}

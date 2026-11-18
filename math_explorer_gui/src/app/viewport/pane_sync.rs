use eframe::egui;

pub(super) fn sync_pane_textures(
    ctx: &egui::Context,
    pane_idx: usize,
    tex_map: &mut std::collections::HashMap<egui::TextureId, egui::TextureId>,
    textures_delta: egui::TexturesDelta,
) {
    if textures_delta.is_empty() {
        return;
    }
    let main_tex = ctx.tex_manager();
    let mut writer = main_tex.write();
    for (sub_id, delta) in textures_delta.set {
        if sub_id == egui::TextureId::default() {
            if writer.meta(sub_id).is_some() {
                writer.set(sub_id, delta);
            }
        } else if let Some(&main_id) = tex_map.get(&sub_id) {
            if writer.meta(main_id).is_some() {
                writer.set(main_id, delta);
            }
        } else if delta.pos.is_none() {
            let main_id = writer.alloc(
                format!("pane_{pane_idx}_{sub_id:?}"),
                delta.image,
                delta.options,
            );
            tex_map.insert(sub_id, main_id);
        }
    }
    for sub_id in textures_delta.free {
        if sub_id != egui::TextureId::default() {
            if let Some(main_id) = tex_map.remove(&sub_id) {
                if writer.meta(main_id).is_some() {
                    writer.free(main_id);
                }
            }
        }
    }
}

pub(super) fn remap_shape_textures(
    shape: &mut egui::Shape,
    tex_map: &std::collections::HashMap<egui::TextureId, egui::TextureId>,
) {
    match shape {
        egui::Shape::Mesh(mesh) => {
            if let Some(&mapped) = tex_map.get(&mesh.texture_id) {
                std::sync::Arc::make_mut(mesh).texture_id = mapped;
            }
        }
        egui::Shape::Rect(rect_shape) => {
            if let Some(brush) = &mut rect_shape.brush {
                if let Some(&mapped) = tex_map.get(&brush.fill_texture_id) {
                    std::sync::Arc::make_mut(brush).fill_texture_id = mapped;
                }
            }
        }
        egui::Shape::Vec(shapes) => {
            for s in shapes {
                remap_shape_textures(s, tex_map);
            }
        }
        _ => {}
    }
}

pub(super) fn merge_pane_commands(ctx: &egui::Context, pane_ctx: &egui::Context) {
    let pane_reg = pane_ctx.data(|d| {
        d.get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
            .unwrap_or_default()
    });
    if pane_reg.commands.is_empty() {
        return;
    }
    ctx.data_mut(|d| {
        let mut root_reg = d
            .get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
            .unwrap_or_default();
        for cmd in pane_reg.commands {
            if !root_reg
                .commands
                .iter()
                .any(|c| c.name == cmd.name && c.context == cmd.context && c.trigger == cmd.trigger)
            {
                root_reg.commands.push(cmd);
            }
        }
        d.insert_temp(egui::Id::new("CMD_REGISTRY"), root_reg);
    });
}

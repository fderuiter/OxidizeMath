use eframe::egui;

#[allow(missing_docs)]
pub struct CoordinateMapper {
    #[allow(missing_docs)]
    pub screen_rect: egui::Rect,
    #[allow(missing_docs)]
    pub sim_rect: egui::Rect,
}

#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub struct Camera3D {
    #[allow(missing_docs)]
    pub pitch: f32,
    #[allow(missing_docs)]
    pub yaw: f32,
    #[allow(missing_docs)]
    pub zoom: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            pitch: 0.5,
            yaw: 0.5,
            zoom: 1.0,
        }
    }
}

impl Camera3D {
    #[allow(missing_docs)]
    pub fn new(pitch: f32, yaw: f32, zoom: f32) -> Self {
        Self { pitch, yaw, zoom }
    }

    #[allow(missing_docs)]
    pub fn handle_interaction(&mut self, response: &egui::Response, ui: &egui::Ui) {
        let multi_touch = ui.input(|i| i.multi_touch());

        if let Some(touch) = multi_touch {
            if touch.zoom_delta != 1.0 {
                let zoom_factor = 1.0 + (touch.zoom_delta - 1.0) * 0.5;
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.01, 100.0);
            }
            if touch.translation_delta != egui::Vec2::ZERO {
                self.yaw -= touch.translation_delta.x * 0.01;
                self.pitch -= touch.translation_delta.y * 0.01;
            }
        } else if response.dragged() {
            let delta = response.drag_delta();
            self.yaw -= delta.x * 0.01;
            self.pitch -= delta.y * 0.01;
        }

        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 && response.hovered() && multi_touch.is_none() {
            self.zoom *= 1.0 + (scroll * 0.001);
            self.zoom = self.zoom.clamp(0.01, 100.0);
        }

        if response.has_focus() {
            self.handle_keyboard_interaction(ui);
        }
    }

    fn handle_keyboard_interaction(&mut self, ui: &egui::Ui) {
        let mut yaw_delta = 0.0;
        let mut pitch_delta = 0.0;
        let mut zoom_factor = 1.0;
        if ui.input(|i| i.key_down(egui::Key::A)) {
            yaw_delta += 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::D)) {
            yaw_delta -= 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::W)) {
            pitch_delta += 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::S)) {
            pitch_delta -= 0.05;
        }
        if ui.input(|i| i.key_down(egui::Key::Q)) {
            zoom_factor *= 1.05;
        }
        if ui.input(|i| i.key_down(egui::Key::E)) {
            zoom_factor /= 1.05;
        }

        self.yaw -= yaw_delta;
        self.pitch -= pitch_delta;
        self.zoom *= zoom_factor;
        self.zoom = self.zoom.clamp(0.01, 100.0);

        self.register_camera_commands(ui);
    }

    fn register_camera_commands(&self, ui: &egui::Ui) {
        ui.ctx().data_mut(|d| {
            let mut registry = d
                .get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
                .unwrap_or_default();

            let commands = [
                (egui::Key::W, "Pitch Up", "Rotate camera pitch upwards"),
                (egui::Key::S, "Pitch Down", "Rotate camera pitch downwards"),
                (egui::Key::A, "Yaw Left", "Rotate camera yaw left"),
                (egui::Key::D, "Yaw Right", "Rotate camera yaw right"),
                (egui::Key::Q, "Zoom Out", "Zoom camera out"),
                (egui::Key::E, "Zoom In", "Zoom camera in"),
            ];

            for (key, name, desc) in commands {
                if !registry
                    .commands
                    .iter()
                    .any(|c| c.name == name && c.context == "Camera 3D")
                {
                    registry
                        .commands
                        .push(egui_plot::commands::CommandMetadata {
                            name: name.to_string(),
                            description: desc.to_string(),
                            trigger: egui_plot::commands::CommandTrigger::Key(key),
                            desktop_only: true,
                            context: "Camera 3D".to_string(),
                        });
                }
            }
            d.insert_temp(egui::Id::new("CMD_REGISTRY"), registry);
        });
    }

    #[allow(missing_docs)]
    pub fn project(&self, point: &[f64; 3]) -> [f64; 2] {
        let cy = (self.yaw as f64).cos();
        let sy = (self.yaw as f64).sin();
        let cp = (self.pitch as f64).cos();
        let sp = (self.pitch as f64).sin();

        // Apply yaw (rotate around Z)
        let x1 = point[0] * cy - point[1] * sy;
        let y1 = point[0] * sy + point[1] * cy;
        let z1 = point[2];

        // Apply pitch (rotate around X)
        let x2 = x1;
        let y2 = y1 * cp - z1 * sp;

        [x2 * (self.zoom as f64), y2 * (self.zoom as f64)]
    }

    #[allow(missing_docs)]
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Yaw");
        ui.drag_angle(&mut self.yaw);

        ui.label("Pitch");
        ui.drag_angle(&mut self.pitch);

        ui.add(egui::Slider::new(&mut self.zoom, 0.1..=5.0).text("Zoom"));
    }
}

impl CoordinateMapper {
    #[allow(missing_docs)]
    pub fn new(screen_rect: egui::Rect, sim_rect: egui::Rect) -> Self {
        Self {
            screen_rect,
            sim_rect,
        }
    }

    #[allow(missing_docs)]
    pub fn screen_to_sim(&self, pos: egui::Pos2) -> egui::Pos2 {
        let x_norm = (pos.x - self.screen_rect.min.x) / self.screen_rect.width();
        let y_norm = (pos.y - self.screen_rect.min.y) / self.screen_rect.height();

        egui::Pos2::new(
            self.sim_rect.min.x + x_norm * self.sim_rect.width(),
            self.sim_rect.min.y + y_norm * self.sim_rect.height(),
        )
    }

    #[allow(missing_docs)]
    pub fn sim_to_screen(&self, pos: egui::Pos2) -> egui::Pos2 {
        let x_norm = (pos.x - self.sim_rect.min.x) / self.sim_rect.width();
        let y_norm = (pos.y - self.sim_rect.min.y) / self.sim_rect.height();

        egui::Pos2::new(
            self.screen_rect.min.x + x_norm * self.screen_rect.width(),
            self.screen_rect.min.y + y_norm * self.screen_rect.height(),
        )
    }
}

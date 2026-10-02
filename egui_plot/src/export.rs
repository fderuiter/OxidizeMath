//! Cross-platform dataset and canvas snapshot export utilities.

/// Format dataset points to CSV string.
pub fn format_csv(datasets: &[(String, Vec<[f64; 2]>)]) -> String {
    let mut csv = String::from("Dataset,X,Y\n");
    for (name, pts) in datasets {
        for p in pts {
            csv.push_str(&format!("{name},{},{}\n", p[0], p[1]));
        }
    }
    csv
}

/// Format dataset points to JSON string.
pub fn format_json(datasets: &[(String, Vec<[f64; 2]>)]) -> String {
    let mut json = String::from("[\n");
    let mut first = true;
    for (name, pts) in datasets {
        for p in pts {
            if !first {
                json.push_str(",\n");
            }
            first = false;
            json.push_str(&format!(
                "  {{\"dataset\": \"{}\", \"x\": {}, \"y\": {}}}",
                name, p[0], p[1]
            ));
        }
    }
    json.push_str("\n]");
    json
}

/// Export dataset points to CSV string and initiate download or file save dialog.
pub fn export_csv(datasets: &[(String, Vec<[f64; 2]>)]) {
    let csv = format_csv(datasets);
    save_file_dialog("export.csv", csv.as_bytes(), "CSV File", &["csv"]);
}

/// Export dataset points to JSON string and initiate download or file save dialog.
pub fn export_json(datasets: &[(String, Vec<[f64; 2]>)]) {
    let json = format_json(datasets);
    save_file_dialog("export.json", json.as_bytes(), "JSON File", &["json"]);
}

/// Convert an `egui::ColorImage` into PNG encoded bytes.
pub fn color32_image_to_png(image: &egui::ColorImage) -> Result<Vec<u8>, String> {
    let width = image.width() as u32;
    let height = image.height() as u32;
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;

    let mut raw_rgba = Vec::with_capacity(image.pixels.len() * 4);
    for p in &image.pixels {
        raw_rgba.push(p.r());
        raw_rgba.push(p.g());
        raw_rgba.push(p.b());
        raw_rgba.push(p.a());
    }
    writer
        .write_image_data(&raw_rgba)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(bytes)
}

/// Trigger cross-platform file saving dialog or browser download action.
pub fn save_file_dialog(filename: &str, content: &[u8], _filter_name: &str, _filter_ext: &[&str]) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let builder = rfd::FileDialog::new()
            .set_file_name(filename)
            .add_filter(_filter_name, _filter_ext);
        if let Some(path) = builder.save_file() {
            let _ = std::fs::write(path, content);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast as _;
        use web_sys::{Blob, BlobPropertyBag, HtmlAnchorElement, Url};

        let mime_type = if filename.ends_with(".csv") {
            "text/csv"
        } else if filename.ends_with(".json") {
            "application/json"
        } else if filename.ends_with(".png") {
            "image/png"
        } else {
            "application/octet-stream"
        };

        if let Some(window) = web_sys::window() {
            if let Some(document) = window.document() {
                let uint8_array = js_sys::Uint8Array::from(content);
                let array = js_sys::Array::new();
                array.push(&uint8_array);
                let mut blob_props = BlobPropertyBag::new();
                blob_props.type_(mime_type);
                if let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&array, &blob_props)
                {
                    if let Ok(url) = Url::create_object_url_with_blob(&blob) {
                        if let Ok(element) = document.create_element("a") {
                            if let Ok(anchor) = element.dyn_into::<HtmlAnchorElement>() {
                                anchor.set_href(&url);
                                anchor.set_download(filename);
                                anchor.click();
                                let _ = Url::revoke_object_url(&url);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_csv() {
        let datasets = vec![
            ("Series 1".to_string(), vec![[0.0, 1.0], [2.0, 3.0]]),
            ("Series 2".to_string(), vec![[4.0, 5.0]]),
        ];
        let csv = format_csv(&datasets);
        assert!(csv.contains("Dataset,X,Y"));
        assert!(csv.contains("Series 1,0,1"));
        assert!(csv.contains("Series 1,2,3"));
        assert!(csv.contains("Series 2,4,5"));
    }

    #[test]
    fn test_format_json() {
        let datasets = vec![("Series 1".to_string(), vec![[0.0, 1.0]])];
        let json = format_json(&datasets);
        assert!(json.contains("\"dataset\": \"Series 1\""));
        assert!(json.contains("\"x\": 0"));
        assert!(json.contains("\"y\": 1"));
    }

    #[test]
    fn test_color32_image_to_png() {
        let image = egui::ColorImage::new([2, 2], vec![egui::Color32::RED; 4]);
        let png_bytes = color32_image_to_png(&image).expect("PNG encoding failed");
        assert!(!png_bytes.is_empty());
        assert_eq!(&png_bytes[1..4], b"PNG");
    }
}

use std::future::Future;
use std::pin::Pin;

#[allow(missing_docs)]
pub trait VirtualFileSystem {
    #[allow(missing_docs)]
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>>;
    #[allow(missing_docs)]
    fn write_to_file(
        &self,
        path: &str,
        content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>>;
    #[allow(missing_docs)]
    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>>;
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs)]
pub struct DefaultVfs;

#[cfg(target_arch = "wasm32")]
#[allow(missing_docs)]
pub type DefaultVfs = WasmVfs;

#[cfg(not(target_arch = "wasm32"))]
impl VirtualFileSystem for DefaultVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move { std::fs::read_to_string(&path) })
    }

    fn write_to_file(
        &self,
        path: &str,
        content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
        let path = path.to_string();
        let content = content.to_vec();
        Box::pin(async move { std::fs::write(&path, content) })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move {
            let entries = std::fs::read_dir(&path)?;
            let mut files = Vec::new();
            for entry in entries.flatten() {
                if let Some(s) = entry.file_name().to_str() {
                    files.push(s.to_string());
                }
            }
            Ok(files)
        })
    }
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn resolve_wasm_fetch_url(path: &str) -> String {
    if path.starts_with("http://")
        || path.starts_with("https://")
        || path.starts_with('/')
        || path.starts_with("./")
    {
        path.to_string()
    } else {
        format!("./{path}")
    }
}

#[cfg(target_arch = "wasm32")]
#[rustfmt::skip]
include!(concat!(env!("OUT_DIR"), "/vfs_data.rs"));

#[cfg(target_arch = "wasm32")]
pub struct WasmVfs;

#[cfg(target_arch = "wasm32")]
impl VirtualFileSystem for WasmVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move {
            use wasm_bindgen::JsCast;
            use wasm_bindgen_futures::JsFuture;
            let window = web_sys::window().ok_or_else(|| std::io::Error::other("No window"))?;
            let url = resolve_wasm_fetch_url(&path);
            let response_value = JsFuture::from(window.fetch_with_str(&url))
                .await
                .map_err(|_| std::io::Error::other(format!("Fetch failed for {url}")))?;
            let response: web_sys::Response = response_value
                .dyn_into()
                .map_err(|_| std::io::Error::other("Invalid response"))?;
            if !response.ok() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("HTTP {} while fetching {}", response.status(), url),
                ));
            }
            let text_value = JsFuture::from(
                response
                    .text()
                    .map_err(|_| std::io::Error::other("No text in response"))?,
            )
            .await
            .map_err(|_| std::io::Error::other("Text extraction failed"))?;
            let content = text_value.as_string().unwrap_or_default();
            Ok(content)
        })
    }

    fn write_to_file(
        &self,
        path: &str,
        content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
        let path = path.to_string();
        let content = content.to_vec();
        Box::pin(async move {
            trigger_download(&path, &content);
            Ok(())
        })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move {
            if let Some(children) = get_dir_children(&path) {
                Ok(children.iter().map(|s| s.to_string()).collect())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Directory not found in VFS",
                ))
            }
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub fn trigger_download(filename: &str, content: &[u8]) {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let uint8_array = js_sys::Uint8Array::from(content);
    let array = js_sys::Array::new();
    array.push(&uint8_array.buffer());
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/octet-stream");
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&array, &options).unwrap();
    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();

    let a = document
        .create_element("a")
        .unwrap()
        .dyn_into::<web_sys::HtmlElement>()
        .unwrap();
    a.set_attribute("href", &url).unwrap();
    a.set_attribute("download", filename).unwrap();
    a.click();
    web_sys::Url::revoke_object_url(&url).unwrap();
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::Path;

    fn read_build_script_output(out_dir: &Path) -> String {
        let parent_dir = out_dir
            .parent()
            .expect("OUT_DIR must have a parent build directory");
        let build_output_path = [
            parent_dir.join("output"),
            parent_dir.join("run").join("stdout"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .expect("Cargo build script output file must exist");
        fs::read_to_string(&build_output_path)
            .expect("Cargo build script output file must be readable")
    }

    #[test]
    fn test_build_script_emits_normalized_vfs_rerun_directives() {
        let out_dir = Path::new(env!("OUT_DIR"));
        let vfs_data = fs::read_to_string(out_dir.join("vfs_data.rs"))
            .expect("OUT_DIR/vfs_data.rs must be generated by build.rs");
        assert!(
            vfs_data.contains("pub fn get_dir_children"),
            "vfs_data.rs should define get_dir_children"
        );

        let build_output = read_build_script_output(out_dir);
        let rerun_paths: HashSet<&str> = build_output
            .lines()
            .filter_map(|line| line.strip_prefix("cargo:rerun-if-changed="))
            .collect();

        assert!(
            !rerun_paths.is_empty(),
            "oxidize_core build.rs must emit cargo:rerun-if-changed directives"
        );
        for path in &rerun_paths {
            assert!(!path.trim().is_empty(), "rerun directive must not be empty");
            assert!(
                !path.contains('\\'),
                "rerun path must use forward slashes: {path}"
            );
        }

        for expected in [
            "build.rs",
            "src/path_utils.rs",
            "../../papers",
            "../../math_explorer/src",
            "../../math_explorer_gui/src/tabs",
            "../../math_explorer_gui/src/tabs/ai",
            "../../math_explorer_gui/src/tabs/traceability",
        ] {
            assert!(
                rerun_paths.contains(expected),
                "Missing expected cargo:rerun-if-changed={expected} in oxidize_core output"
            );
        }
    }

    #[test]
    fn test_resolve_wasm_fetch_url_prefixes_relative_paths_within_subpath() {
        assert_eq!(
            super::resolve_wasm_fetch_url("traceability_manifest.json"),
            "./traceability_manifest.json"
        );
        assert_eq!(
            super::resolve_wasm_fetch_url("./traceability_manifest.json"),
            "./traceability_manifest.json"
        );
        assert_eq!(
            super::resolve_wasm_fetch_url("papers/essay.tex"),
            "./papers/essay.tex"
        );
        assert_eq!(
            super::resolve_wasm_fetch_url("/OxidizeMath/essay.pdf"),
            "/OxidizeMath/essay.pdf"
        );
        assert_eq!(
            super::resolve_wasm_fetch_url("https://example.com/essay.pdf"),
            "https://example.com/essay.pdf"
        );
    }

    #[test]
    fn test_default_vfs_list_dir_propagates_not_found() {
        use super::{DefaultVfs, VirtualFileSystem};

        let res = futures::executor::block_on(
            DefaultVfs.list_dir("nonexistent_oxidize_vfs_dir_987654321"),
        );
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::NotFound);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "vfs_wasm_tests.rs"]
mod wasm_tests;

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
            let mut files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    if let Some(s) = entry.file_name().to_str() {
                        files.push(s.to_string());
                    }
                }
            }
            Ok(files)
        })
    }
}

#[cfg(target_arch = "wasm32")]
// theory_verification!
#[rustfmt::skip]
include!(concat!(env!("OUT_DIR"), "/vfs_data.rs"));

#[cfg(target_arch = "wasm32")]
// theory_verification!
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
            let window = web_sys::window()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "No window"))?;
            let mut url = path.clone();
            // ensure it can be fetched
            if !url.starts_with("http") && !url.starts_with('/') {
                url = format!("/{}", url);
            }
            let response_value = JsFuture::from(window.fetch_with_str(&url))
                .await
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Fetch failed"))?;
            let response: web_sys::Response = response_value
                .dyn_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid response"))?;
            if !response.ok() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "File not found via fetch",
                ));
            }
            let text_value = JsFuture::from(response.text().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "No text in response")
            })?)
            .await
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "Text extraction failed")
            })?;
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
// theory_verification!
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
// theory_verification!

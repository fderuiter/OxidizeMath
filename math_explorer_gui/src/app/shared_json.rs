#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Reference-counted immutable JSON string for zero-copy workspace state snapshots.

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// Reference-counted serialized tab JSON string avoiding O(N) heap clones in `AppState::extract`.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct SharedTabJson(Arc<str>);

impl SharedTabJson {
    /// Returns the underlying UTF-8 JSON slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SharedTabJson {
    fn default() -> Self {
        Self(Arc::from(""))
    }
}

impl Deref for SharedTabJson {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for SharedTabJson {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for SharedTabJson {
    #[inline]
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<String> for SharedTabJson {
    #[inline]
    fn from(value: String) -> Self {
        Self(Arc::from(value))
    }
}

impl From<&str> for SharedTabJson {
    #[inline]
    fn from(value: &str) -> Self {
        Self(Arc::from(value))
    }
}

impl From<Arc<str>> for SharedTabJson {
    #[inline]
    fn from(value: Arc<str>) -> Self {
        Self(value)
    }
}

impl fmt::Debug for SharedTabJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl fmt::Display for SharedTabJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

impl PartialEq<String> for SharedTabJson {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        &*self.0 == other.as_str()
    }
}

impl PartialEq<SharedTabJson> for String {
    #[inline]
    fn eq(&self, other: &SharedTabJson) -> bool {
        self.as_str() == &*other.0
    }
}

impl PartialEq<str> for SharedTabJson {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for SharedTabJson {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

impl serde::Serialize for SharedTabJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for SharedTabJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(Arc::from(s)))
    }
}

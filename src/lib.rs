//! On-device language models for macOS and Windows.
//!
//! Each model sits behind a Cargo feature. On a platform a model does not
//! support, its module still builds and reports itself unsupported.

#[cfg(feature = "apple")]
pub mod apple;
#[cfg(feature = "bonsai")]
pub mod bonsai;

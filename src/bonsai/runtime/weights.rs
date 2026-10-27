//! Exposes model.safetensors to the GPU as one buffer: memory-mapped without copying on Metal, uploaded on Vulkan.

use super::gpu::{Buffer, Gpu};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct Tensor {
    pub off: usize,
    pub bytes: usize,
    pub shape: Vec<usize>,
}

pub struct Weights {
    pub buf: Buffer,
    tensors: HashMap<String, Tensor>,
    /// After `buf`, so the GPU buffer is released before the file is unmapped.
    map: Mapping,
}

unsafe impl Send for Weights {}

/// The weights file's host mapping (pointer, length); null on Vulkan.
struct Mapping(*const u8, usize);

impl Drop for Mapping {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if !self.0.is_null() {
            unsafe { libc::munmap(self.0 as *mut libc::c_void, self.1) };
        }
    }
}

impl Weights {
    pub fn open(path: &Path, gpu: &Gpu) -> Result<Weights, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut head = [0u8; 8];
        f.read_exact(&mut head).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len() as usize;
        let n = u64::from_le_bytes(head) as usize;
        if n > file_len - 8 {
            return Err(format!("{}: bad safetensors header", path.display()));
        }
        let mut json = vec![0u8; n];
        f.read_exact(&mut json).map_err(|e| format!("{}: {e}", path.display()))?;
        let header: serde_json::Value = serde_json::from_slice(&json).map_err(|e| e.to_string())?;
        let mut names = Vec::new();
        let mut regions = Vec::new();
        let mut shapes = Vec::new();
        for (k, v) in header.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            );
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            if a > b || b > file_len - 8 - n {
                return Err(format!("{}: tensor {k} past the end of the file", path.display()));
            }
            names.push(k.clone());
            regions.push((8 + n + a, b - a));
            shapes.push(shape);
        }
        // the backend may move tensors (Vulkan aligns each one); offsets below are into `buf`
        let (buf, ptr, len) = gpu.map_weights(path, &mut regions)?;
        let tensors = names
            .into_iter()
            .zip(regions)
            .zip(shapes)
            .map(|((k, (off, bytes)), shape)| (k, Tensor { off, bytes, shape }))
            .collect();
        Ok(Weights {
            buf,
            tensors,
            map: Mapping(ptr, len),
        })
    }

    pub fn get(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    pub fn off(&self, name: &str) -> usize {
        self.get(name).off
    }

    /// Host bytes of a tensor (Metal only: the Vulkan backend keeps no host copy).
    pub fn bytes(&self, name: &str) -> &[u8] {
        let t = self.get(name);
        let Mapping(ptr, len) = self.map;
        assert!(!ptr.is_null(), "no host copy of the weights");
        assert!(t.off + t.bytes <= len);
        unsafe { std::slice::from_raw_parts(ptr.add(t.off), t.bytes) }
    }

    pub fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

}

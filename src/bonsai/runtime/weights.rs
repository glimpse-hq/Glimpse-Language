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
    ptr: *const u8,
    len: usize,
    tensors: HashMap<String, Tensor>,
}

unsafe impl Send for Weights {}

impl Weights {
    pub fn open(path: &Path, gpu: &Gpu) -> Result<Weights, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut head = [0u8; 8];
        f.read_exact(&mut head).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = u64::from_le_bytes(head) as usize;
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
            ptr,
            len,
            tensors,
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
        assert!(!self.ptr.is_null(), "no host copy of the weights");
        assert!(t.off + t.bytes <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(t.off), t.bytes) }
    }

    pub fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

}

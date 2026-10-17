//! Memory-maps model.safetensors and exposes it to the GPU as one zero-copy buffer.

use super::gpu::{Buffer, Gpu};
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
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
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let map_len = len.div_ceil(page) * page;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                f.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| e.to_string())?;
        let mut tensors = HashMap::new();
        for (k, v) in header.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            );
            let shape = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            tensors.insert(
                k.clone(),
                Tensor {
                    off: 8 + n + a,
                    bytes: b - a,
                    shape,
                },
            );
        }
        let buf = unsafe { gpu.buffer_no_copy(ptr, map_len) };
        Ok(Weights {
            buf,
            ptr: ptr as *const u8,
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

    pub fn bytes(&self, name: &str) -> &[u8] {
        let t = self.get(name);
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

//! Thin Metal layer: device, queue, precompiled kernels, buffers, one compute encoder per command buffer.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: HashMap<&'static str, Pipeline>,
}

const KERNELS: &[&str] = &[
    "norm_rotate", "swiglu_rotate", "gate_rotate", "gated_norm_rotate", "embed", "qmv4", "qmv3", "qmvx",
    "qmm3", "qmvr", "gdn_ab", "gdn_conv", "conv_state", "gdn_recurrent", "attn_prep", "attn_split", "attn_combine", "argmax1",
    "argmax2", "lora_b", "v_gemm", "v_layernorm", "v_rope", "v_flash",
];

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no queue")?;
        let data = dispatch2::DispatchData::from_bytes(METALLIB);
        let lib = device
            .newLibraryWithData_error(&data)
            .map_err(|e| format!("metallib: {e:?}"))?;
        let mut pipes = HashMap::new();
        for &name in KERNELS {
            let f = lib
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .ok_or_else(|| format!("missing kernel {name}"))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| format!("{name}: {e:?}"))?;
            pipes.insert(name, p);
        }
        Ok(Gpu {
            device,
            queue,
            pipes,
        })
    }

    pub fn buffer(&self, bytes: usize) -> Buffer {
        self.device
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
            .expect("buffer alloc")
    }

    /// Wraps page-aligned memory (an mmap) without copying.
    pub unsafe fn buffer_no_copy(&self, ptr: *mut c_void, len: usize) -> Buffer {
        self.device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(ptr).unwrap(),
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .expect("no-copy buffer")
    }

    pub fn begin(&self) -> Cmd<'_> {
        let cb = self.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        Cmd {
            gpu: self,
            cb,
            enc: Some(enc),
        }
    }
}

pub fn contents<T>(b: &Buffer) -> *mut T {
    b.contents().as_ptr() as *mut T
}

/// A buffer binding: buffer plus byte offset.
#[derive(Clone, Copy)]
pub struct B<'a>(pub &'a Buffer, pub usize);

pub enum Arg<'a> {
    Buf(B<'a>),
    U32(u32),
}

impl<'a> From<B<'a>> for Arg<'a> {
    fn from(b: B<'a>) -> Self {
        Arg::Buf(b)
    }
}
impl<'a> From<u32> for Arg<'a> {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

pub struct Cmd<'g> {
    gpu: &'g Gpu,
    cb: CmdBuf,
    enc: Option<Encoder>,
}

impl<'g> Cmd<'g> {
    /// Dispatch `groups` threadgroups of `threads` threads.
    pub fn run(
        &mut self,
        name: &str,
        args: &[Arg],
        groups: (usize, usize, usize),
        threads: (usize, usize, usize),
    ) {
        let enc = self.enc.as_ref().unwrap();
        let p = &self.gpu.pipes[name];
        enc.setComputePipelineState(p);
        for (i, a) in args.iter().enumerate() {
            unsafe {
                match a {
                    Arg::Buf(B(b, off)) => enc.setBuffer_offset_atIndex(Some(b), *off, i),
                    Arg::U32(v) => enc.setBytes_length_atIndex(
                        NonNull::new(v as *const u32 as *mut c_void).unwrap(),
                        4,
                        i,
                    ),
                }
            }
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.0,
                height: groups.1,
                depth: groups.2,
            },
            MTLSize {
                width: threads.0,
                height: threads.1,
                depth: threads.2,
            },
        );
        if profiling() {
            // Time 20 more repeats of this dispatch alone so launch overhead amortizes (results are garbage).
            self.enc.take().unwrap().endEncoding();
            self.cb.commit();
            self.cb.waitUntilCompleted();
            self.cb = self.gpu.queue.commandBuffer().unwrap();
            let enc = self.cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(p);
            for (i, a) in args.iter().enumerate() {
                unsafe {
                    match a {
                        Arg::Buf(B(b, off)) => enc.setBuffer_offset_atIndex(Some(b), *off, i),
                        Arg::U32(v) => enc.setBytes_length_atIndex(
                            NonNull::new(v as *const u32 as *mut c_void).unwrap(),
                            4,
                            i,
                        ),
                    }
                }
            }
            for _ in 0..20 {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups.0,
                        height: groups.1,
                        depth: groups.2,
                    },
                    MTLSize {
                        width: threads.0,
                        height: threads.1,
                        depth: threads.2,
                    },
                );
            }
            self.enc = Some(enc);
            self.enc.take().unwrap().endEncoding();
            self.cb.commit();
            self.cb.waitUntilCompleted();
            let (a, b): (f64, f64) = unsafe {
                (
                    objc2::msg_send![&*self.cb, GPUStartTime],
                    objc2::msg_send![&*self.cb, GPUEndTime],
                )
            };
            let dt = (b - a) / 20.0;
            let mut p = PROFILE.lock().unwrap();
            let e = p.entry(name.to_string()).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
            self.cb = self.gpu.queue.commandBuffer().unwrap();
            self.enc = Some(self.cb.computeCommandEncoder().unwrap());
        }
    }

    pub fn copy(&mut self, src: B, dst: B, bytes: usize) {
        if let Some(e) = self.enc.take() {
            e.endEncoding();
        }
        let blit = self.cb.blitCommandEncoder().unwrap();
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                src.0, src.1, dst.0, dst.1, bytes,
            )
        };
        blit.endEncoding();
        self.enc = Some(self.cb.computeCommandEncoder().unwrap());
    }

    pub fn commit(mut self) -> Pending {
        if let Some(e) = self.enc.take() {
            e.endEncoding();
        }
        self.cb.commit();
        Pending(self.cb.clone())
    }

    pub fn finish(self) {
        self.commit().wait();
    }
}

static PROFILE: std::sync::Mutex<Vec<(String, (f64, usize))>> = std::sync::Mutex::new(Vec::new());
fn profiling() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var_os("BONSAI_PROFILE").is_some())
}
trait Entry {
    fn entry(&mut self, k: String) -> Entr<'_>;
}
pub struct Entr<'a>(&'a mut (f64, usize));
impl<'a> Entr<'a> {
    fn or_insert(self, _: (f64, usize)) -> &'a mut (f64, usize) {
        self.0
    }
}
impl Entry for Vec<(String, (f64, usize))> {
    fn entry(&mut self, k: String) -> Entr<'_> {
        let i = match self.iter().position(|(n, _)| *n == k) {
            Some(i) => i,
            None => {
                self.push((k, (0.0, 0)));
                self.len() - 1
            }
        };
        Entr(&mut self[i].1)
    }
}
/// Per-kernel GPU time totals when BONSAI_PROFILE is set (each dispatch then runs alone).
pub fn profile_report() {
    if !profiling() {
        return;
    }
    let mut p = PROFILE.lock().unwrap();
    let total: f64 = p.iter().map(|(_, v)| v.0).sum();
    for (k, (t, n)) in p.iter() {
        eprintln!(
            "{k:20} {:9.3} ms total {:7.1} us/call x{n} {:5.1}%",
            t * 1e3,
            t * 1e6 / *n as f64,
            t / total * 100.0
        );
    }
    p.clear();
}

pub struct Pending(CmdBuf);
impl Pending {
    pub fn wait(self) {
        self.0.waitUntilCompleted();
        if let Some(e) = self.0.error() {
            panic!("GPU error: {e:?}");
        }
    }
}

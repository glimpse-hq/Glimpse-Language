//! Thin Vulkan layer with the surface of gpu.rs (Metal): device, queue, kernels precompiled to SPIR-V, buffers
//! passed to kernels by device address in push constants. A Cmd collects dispatches and copies, each followed
//! by a barrier (Metal's serial compute encoder), and at commit replays a cached command buffer when the same
//! stream ran before. Subgroups are fixed at 32 lanes, the width the kernels assume.

use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

mod spv {
    include!(concat!(env!("OUT_DIR"), "/kernels_spv.rs"));
}

/// Verify cost in decode steps for T = 1..=8 rows (the last token plus T - 1 drafts), accept included: a
/// synchronous verify round against a pipelined decode step, measured on an RTX 4000 Ada. The verify matmuls run
/// on cooperative matrices, so rows past the first cost little and the curve is flat.
pub const VERIFY_COST_2B: [f64; 8] = [1.0, 2.2, 2.2, 2.2, 2.25, 2.25, 2.3, 2.35];
pub const VERIFY_COST_4B: [f64; 8] = [1.0, 2.15, 2.15, 2.15, 2.2, 2.2, 2.25, 2.3];

/// All kernels as built, for identifying saved states made with other kernels.
pub static KERNEL_BLOB: &[u8] = spv::ALL;

const PUSH_BYTES: u32 = 256;
const SUBGROUP: u32 = 32;

struct Shared {
    /// Timestamp period in ns, for BONSAI_PROFILE.
    ts_period: f32,
    cache: Mutex<Cache>,
    fences: Mutex<Vec<vk::Fence>>,
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: Mutex<Pool>,
    mem: vk::PhysicalDeviceMemoryProperties,
}

struct Pool {
    pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
}

impl Drop for Shared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let cache = self.cache.get_mut().unwrap();
            for (_, e) in cache.entries.drain().chain(cache.doomed.drain(..)) {
                e.params.free(&self.device);
            }
            for f in self.fences.get_mut().unwrap().drain(..) {
                self.device.destroy_fence(f, None);
            }
            self.device.destroy_command_pool(self.pool.get_mut().unwrap().pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Shared {
    fn take_cmd(&self) -> vk::CommandBuffer {
        let mut p = self.pool.lock().unwrap();
        if let Some(cb) = p.free.pop() {
            return cb;
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(p.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        unsafe { self.device.allocate_command_buffers(&info) }.expect("command buffer")[0]
    }

    fn give_cmd(&self, cb: vk::CommandBuffer) {
        unsafe { self.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) }.expect("reset command buffer");
        self.pool.lock().unwrap().free.push(cb);
    }

    fn take_fence(&self) -> vk::Fence {
        self.fences
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) }.expect("fence"))
    }

    /// Submits `cbs` and waits for them.
    fn submit_wait(&self, cbs: &[vk::CommandBuffer]) {
        let fence = self.take_fence();
        unsafe {
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(cbs)], fence).expect("submit");
            self.device.wait_for_fences(&[fence], true, u64::MAX).expect("GPU error");
            self.device.reset_fences(&[fence]).expect("reset fence");
        }
        self.fences.lock().unwrap().push(fence);
    }

    /// Records `f` into a command buffer, submits it and waits.
    fn oneshot(&self, f: impl FnOnce(vk::CommandBuffer)) {
        let cb = self.take_cmd();
        unsafe {
            let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cb, &info).expect("begin");
            barrier(&self.device, cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS);
            f(cb);
            barrier(&self.device, cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST);
            self.device.end_command_buffer(cb).expect("end");
        }
        self.submit_wait(&[cb]);
        self.give_cmd(cb);
    }

    fn mem_type(&self, bits: u32, want: vk::MemoryPropertyFlags, avoid: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.mem.memory_types[..self.mem.memory_type_count as usize];
        let ok = |i: usize, t: &vk::MemoryType, strict: bool| {
            bits & (1 << i) != 0 && t.property_flags.contains(want) && (!strict || !t.property_flags.intersects(avoid))
        };
        let pick = |strict: bool| types.iter().enumerate().find(|(i, t)| ok(*i, t, strict)).map(|(i, _)| i as u32);
        pick(true).or_else(|| pick(false))
    }

    /// A zeroed buffer. `tracked`: commands may keep its address (staging buffers are untracked, so freeing
    /// them leaves the command cache alone).
    fn alloc(self: &Arc<Self>, bytes: usize, host: bool, tracked: bool) -> Buffer {
        let size = bytes.max(16) as u64;
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&info, None).expect("buffer");
            let req = self.device.get_buffer_memory_requirements(buf);
            let (want, avoid) = if host {
                (vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, vk::MemoryPropertyFlags::empty())
            } else {
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::HOST_VISIBLE)
            };
            let ty = self.mem_type(req.memory_type_bits, want, avoid).expect("no suitable memory type");
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut flags);
            let mem = self.device.allocate_memory(&ai, None).unwrap_or_else(|e| panic!("GPU memory ({bytes} bytes): {e}"));
            self.device.bind_buffer_memory(buf, mem, 0).expect("bind");
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            let props = self.mem.memory_types[ty as usize].property_flags;
            let map = if props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) {
                self.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("map") as *mut u8
            } else {
                std::ptr::null_mut()
            };
            let b = Buffer(Arc::new(BufInner { sh: Arc::clone(self), buf, mem, addr, size: size as usize, map, tracked }));
            zero(&b, size as usize);
            b
        }
    }
}

unsafe fn barrier(d: &ash::Device, cb: vk::CommandBuffer, src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe { d.cmd_pipeline_barrier(cb, src, dst, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
}

struct BufInner {
    sh: Arc<Shared>,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    addr: u64,
    size: usize,
    map: *mut u8,
    tracked: bool,
}

unsafe impl Send for BufInner {}
unsafe impl Sync for BufInner {}

/// Bumped when a tracked buffer is freed: cached command buffers may reference it, so the cache is dropped.
static BUF_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Drop for BufInner {
    fn drop(&mut self) {
        if self.tracked {
            BUF_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        unsafe {
            self.sh.device.destroy_buffer(self.buf, None);
            self.sh.device.free_memory(self.mem, None);
        }
    }
}

/// A GPU buffer. Device-local unless made with `Gpu::io_buffer`; only mapped buffers have `contents`.
#[derive(Clone)]
pub struct Buffer(Arc<BufInner>);

pub fn contents<T>(b: &Buffer) -> *mut T {
    assert!(!b.0.map.is_null(), "contents() of a device-local buffer; use read/write");
    b.0.map as *mut T
}

/// Host to buffer copy (through a staging buffer when the buffer is not mapped). Blocking.
pub fn write(b: &Buffer, off: usize, data: &[u8]) {
    assert!(off + data.len() <= b.0.size);
    if !b.0.map.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.0.map.add(off), data.len()) };
        return;
    }
    let sh = &b.0.sh;
    let stage = sh.alloc(data.len(), true, false);
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), stage.0.map, data.len()) };
    sh.oneshot(|cb| unsafe {
        let r = vk::BufferCopy { src_offset: 0, dst_offset: off as u64, size: data.len() as u64 };
        sh.device.cmd_copy_buffer(cb, stage.0.buf, b.0.buf, &[r]);
    });
}

/// Buffer to host copy. Blocking.
pub fn read(b: &Buffer, off: usize, len: usize) -> Vec<u8> {
    assert!(off + len <= b.0.size);
    let mut out = vec![0u8; len];
    if !b.0.map.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(b.0.map.add(off), out.as_mut_ptr(), len) };
        return out;
    }
    let sh = &b.0.sh;
    let stage = sh.alloc(len, true, false);
    sh.oneshot(|cb| unsafe {
        let r = vk::BufferCopy { src_offset: off as u64, dst_offset: 0, size: len as u64 };
        sh.device.cmd_copy_buffer(cb, b.0.buf, stage.0.buf, &[r]);
    });
    unsafe { std::ptr::copy_nonoverlapping(stage.0.map, out.as_mut_ptr(), len) };
    out
}

/// Zeroes the first `len` bytes. Blocking.
pub fn zero(b: &Buffer, len: usize) {
    if !b.0.map.is_null() {
        unsafe { std::ptr::write_bytes(b.0.map, 0, len) };
        return;
    }
    let sh = &b.0.sh;
    sh.oneshot(|cb| unsafe { sh.device.cmd_fill_buffer(cb, b.0.buf, 0, (len as u64).div_ceil(4) * 4, 0) });
}

/// Why this machine cannot run the kernels, or the chosen device.
fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32, String), String> {
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| format!("Vulkan devices: {e}"))?;
    let mut best: Option<(u32, vk::PhysicalDevice, u32, String)> = None;
    let mut why = String::from("no Vulkan GPU");
    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy().into_owned();
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => continue,
        };
        if props.api_version < vk::API_VERSION_1_3 {
            why = format!("{name}: needs Vulkan 1.3");
            continue;
        }
        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut sgc = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg).push_next(&mut sgc);
        unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut f13);
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        let ops = vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::SHUFFLE
            | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE;
        let sized = sg.subgroup_size == SUBGROUP
            || (f13.subgroup_size_control == vk::TRUE
                && f13.compute_full_subgroups == vk::TRUE
                && sgc.min_subgroup_size <= SUBGROUP
                && sgc.max_subgroup_size >= SUBGROUP
                && sgc.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE));
        if f12.buffer_device_address != vk::TRUE {
            why = format!("{name}: no buffer device address");
        } else if !sg.supported_stages.contains(vk::ShaderStageFlags::COMPUTE) || !sg.supported_operations.contains(ops) {
            why = format!("{name}: missing subgroup operations");
        } else if !sized {
            why = format!("{name}: no 32-wide subgroups");
        } else if props.limits.max_push_constants_size < PUSH_BYTES {
            why = format!("{name}: push constants under {PUSH_BYTES} bytes");
        } else {
            let qfams = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(q) = qfams.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)) else {
                continue;
            };
            if best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, pd, q as u32, name));
            }
        }
    }
    best.map(|(_, pd, q, n)| (pd, q, n)).ok_or(why)
}

fn instance() -> Result<(ash::Entry, ash::Instance), String> {
    let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("Vulkan loader: {e}"))?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&info, None) }.map_err(|e| format!("Vulkan instance: {e}"))?;
    Ok((entry, instance))
}

/// Whether a suitable Vulkan GPU exists (cached; creates and drops an instance on the first call).
pub fn available() -> Result<String, String> {
    static PROBE: OnceLock<Result<String, String>> = OnceLock::new();
    PROBE
        .get_or_init(|| {
            let (_entry, inst) = instance()?;
            let r = pick_device(&inst).map(|(_, _, n)| n);
            unsafe { inst.destroy_instance(None) };
            r
        })
        .clone()
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

pub struct Gpu {
    sh: Arc<Shared>,
    pipes: HashMap<&'static str, Pipe>,
    name: String,
    /// 16x16x16 fp16 cooperative matrices with fp32 accumulation: T > 1 matmuls run on qmm_cm.
    coopmat: bool,
    /// Split-K partials of qmm_cm.
    part: Option<Buffer>,
}

/// Workgroups a T > 1 matmul should have before it splits K (several per SM on the GPUs we target).
const CM_TARGET_GROUPS: usize = 256;
const PART_BYTES: usize = 8 << 20;

/// Whether `pd` has 16x16x16 fp16 x fp16 -> fp32 subgroup cooperative matrices (and the env does not turn
/// them off).
fn has_coopmat(entry: &ash::Entry, instance: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    if std::env::var_os("BONSAI_NO_COOPMAT").is_some() {
        return false;
    }
    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let has = exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::khr::cooperative_matrix::NAME);
    if !has {
        return false;
    }
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut fcm);
    unsafe { instance.get_physical_device_features2(pd, &mut f2) };
    if f12.shader_float16 != vk::TRUE || fcm.cooperative_matrix != vk::TRUE {
        return false;
    }
    let cm = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let props = unsafe { cm.get_physical_device_cooperative_matrix_properties(pd) }.unwrap_or_default();
    props.iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.a_type == vk::ComponentTypeKHR::FLOAT16
            && p.b_type == vk::ComponentTypeKHR::FLOAT16
            && p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32
            && p.scope == vk::ScopeKHR::SUBGROUP
    })
}

impl Drop for Gpu {
    fn drop(&mut self) {
        profile_report();
        unsafe {
            let _ = self.sh.device.device_wait_idle();
            for p in self.pipes.values() {
                self.sh.device.destroy_pipeline(p.pipeline, None);
                self.sh.device.destroy_pipeline_layout(p.layout, None);
            }
        }
    }
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let (entry, instance) = instance()?;
        let (pd, qfam, name) = match pick_device(&instance) {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
        unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
        let ts_period = p2.properties.limits.timestamp_period;
        let control = sg.subgroup_size != SUBGROUP;
        let coopmat = has_coopmat(&entry, &instance, pd);
        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfam).queue_priorities(&prio)];
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).shader_float16(coopmat);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(control)
            .compute_full_subgroups(control);
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let ext_names = [ash::khr::cooperative_matrix::NAME.as_ptr()];
        let mut dinfo = vk::DeviceCreateInfo::default().queue_create_infos(&qinfo).push_next(&mut f12).push_next(&mut f13);
        if coopmat {
            dinfo = dinfo.enabled_extension_names(&ext_names).push_next(&mut fcm);
        }
        let device = unsafe { instance.create_device(pd, &dinfo, None) }.map_err(|e| format!("Vulkan device: {e}"))?;
        let queue = unsafe { device.get_device_queue(qfam, 0) };
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfam)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| format!("command pool: {e}"))?;
        let mem = unsafe { instance.get_physical_device_memory_properties(pd) };
        let sh = Arc::new(Shared {
            ts_period,
            cache: Mutex::new(Cache::default()),
            fences: Mutex::new(Vec::new()),
            _entry: entry,
            instance,
            device,
            queue,
            pool: Mutex::new(Pool { pool, free: Vec::new() }),
            mem,
        });
        let pipes = make_pipes(&sh.device, control, coopmat)?;
        let part = coopmat.then(|| sh.alloc(PART_BYTES, false, true));
        Ok(Gpu { sh, pipes, name, coopmat, part })
    }

    pub fn device_name(&self) -> String {
        self.name.clone()
    }

    pub fn buffer(&self, bytes: usize) -> Buffer {
        self.sh.alloc(bytes, false, true)
    }

    /// A small buffer the host reads and writes through `contents` (token ids, positions, results).
    pub fn io_buffer(&self, bytes: usize) -> Buffer {
        self.sh.alloc(bytes, true, true)
    }

    /// Uploads `path` into one device buffer. Each region (file offset, bytes) moves to a 256-byte aligned
    /// offset, written back into `regions`. Returns the buffer; there is no host mapping.
    pub fn map_weights(&self, path: &Path, regions: &mut [(usize, usize)]) -> Result<(Buffer, *const u8, usize), String> {
        use std::io::{Read, Seek, SeekFrom};
        let mut order: Vec<usize> = (0..regions.len()).collect();
        order.sort_by_key(|&i| regions[i].0);
        let mut dst = vec![0usize; regions.len()];
        let mut total = 0usize;
        for &i in &order {
            dst[i] = total;
            total = (total + regions[i].1).div_ceil(256) * 256;
        }
        let buf = self.sh.alloc(total, false, true);
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // staging in 64 MB pieces; a region may span pieces
        const PIECE: usize = 64 << 20;
        let stage = self.sh.alloc(PIECE, true, false);
        let mut copies: Vec<vk::BufferCopy> = Vec::new();
        let mut used = 0usize;
        let flush = |copies: &mut Vec<vk::BufferCopy>| {
            if copies.is_empty() {
                return;
            }
            self.sh.oneshot(|cb| unsafe { self.sh.device.cmd_copy_buffer(cb, stage.0.buf, buf.0.buf, copies) });
            copies.clear();
        };
        for &i in &order {
            let (off, len) = regions[i];
            f.seek(SeekFrom::Start(off as u64)).map_err(|e| e.to_string())?;
            let mut done = 0;
            while done < len {
                if used == PIECE {
                    flush(&mut copies);
                    used = 0;
                }
                let n = (len - done).min(PIECE - used);
                let slice = unsafe { std::slice::from_raw_parts_mut(stage.0.map.add(used), n) };
                f.read_exact(slice).map_err(|e| format!("{}: {e}", path.display()))?;
                copies.push(vk::BufferCopy { src_offset: used as u64, dst_offset: (dst[i] + done) as u64, size: n as u64 });
                used += n;
                done += n;
            }
        }
        flush(&mut copies);
        for (i, r) in regions.iter_mut().enumerate() {
            r.0 = dst[i];
        }
        Ok((buf, std::ptr::null(), 0))
    }

    pub fn begin(&self) -> Cmd<'_> {
        Cmd { gpu: self, ops: Vec::with_capacity(320), attn_ns: None, labels: profiling().then(Vec::new) }
    }
}

fn make_pipes(device: &ash::Device, control: bool, coopmat: bool) -> Result<HashMap<&'static str, Pipe>, String> {
    let range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(PUSH_BYTES)];
    let mut modules = Vec::new();
    let mut layouts = Vec::new();
    let kernels: Vec<&(&str, &[u8])> = spv::SPV.iter().filter(|(n, _)| coopmat || !n.starts_with("qmm_")).collect();
    for (name, bytes) in kernels.iter().copied() {
        let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).map_err(|e| format!("{name}: {e}"))?;
        let m = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }
            .map_err(|e| format!("{name}: {e}"))?;
        let l = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range), None) }
            .map_err(|e| format!("{name}: {e}"))?;
        modules.push(m);
        layouts.push(l);
    }
    let mut req: Vec<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo> =
        kernels.iter().map(|_| vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(SUBGROUP)).collect();
    let infos: Vec<vk::ComputePipelineCreateInfo> = req
        .iter_mut()
        .enumerate()
        .map(|(i, r)| {
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(modules[i])
                .name(c"main");
            if control {
                stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(r);
            }
            vk::ComputePipelineCreateInfo::default().stage(stage).layout(layouts[i])
        })
        .collect();
    let pipelines = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None) }
        .map_err(|(_, e)| format!("compute pipelines: {e}"))?;
    for m in modules {
        unsafe { device.destroy_shader_module(m, None) };
    }
    Ok(kernels
        .into_iter()
        .zip(pipelines.into_iter().zip(layouts))
        .map(|((name, _), (pipeline, layout))| (*name, Pipe { pipeline, layout }))
        .collect())
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
fn copy<'a>(a: &Arg<'a>) -> Arg<'a> {
    match a {
        Arg::Buf(b) => Arg::Buf(*b),
        Arg::U32(v) => Arg::U32(*v),
    }
}

impl<'a> From<u32> for Arg<'a> {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

pub struct Cmd<'g> {
    gpu: &'g Gpu,
    ops: Vec<Op>,
    /// Split count chosen for the last attn_split, for the attn_combine after it.
    attn_ns: Option<u32>,
    /// Kernel labels per op when profiling.
    labels: Option<Vec<String>>,
}

/// One recorded operation. Vulkan commands are written at commit, into a cached command buffer when the same
/// stream was seen before.
struct Op {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push: [u8; PUSH_BYTES as usize],
    plen: usize,
    groups: [u32; 3],
    dynamic: Option<Dyn>,
    copy: Option<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,
}

/// Per-step values of an attention dispatch (KV offset, split count; attn_split's grid y is the split count),
/// read from a parameter slot in cached command buffers so one recording serves every step. `at`: byte
/// offsets of the values in the push constants, zeroed for the stream signature.
#[derive(Clone, Copy)]
struct Dyn {
    off: u32,
    ns: u32,
    at: [Option<usize>; 2],
    indirect: bool,
    /// Byte offset of the parameter-slot pointer in the push constants.
    slot_at: usize,
}

/// Parameter slot layout (u32): off, ns, then the indirect grid x, y, z.
const SLOT_BYTES: usize = 32;
const CACHE_MAX: usize = 96;

/// A host-visible buffer owned by the cache (not a `Buffer`, which would keep `Shared` alive).
struct RawBuf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    addr: u64,
    map: *mut u8,
}

impl RawBuf {
    unsafe fn free(&self, d: &ash::Device) {
        unsafe {
            d.destroy_buffer(self.buf, None);
            d.free_memory(self.mem, None);
        }
    }
}

struct Entry {
    cb: vk::CommandBuffer,
    params: RawBuf,
    busy: bool,
    used: u64,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<u64, Entry>,
    /// Streams seen once; a stream is cached the second time it shows up.
    seen: HashMap<u64, u32>,
    tick: u64,
    generation: u64,
    /// Recordings dropped while in flight; freed when their submission completes.
    doomed: Vec<(u64, Entry)>,
}

unsafe impl Send for Cache {}

/// BONSAI_PROFILE: a timestamp pair around every dispatch, summed per kernel when the command buffer completes.
struct Prof {
    pool: vk::QueryPool,
    names: Vec<String>,
}

const PROF_MAX: u32 = 4096;

fn profiling() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| std::env::var_os("BONSAI_PROFILE").is_some())
}

static PROFILE: Mutex<Vec<(String, f64, usize)>> = Mutex::new(Vec::new());

fn fnv(h: u64, b: &[u8]) -> u64 {
    b.iter().fold(h, |h, &c| (h ^ c as u64).wrapping_mul(0x100000001b3))
}

/// Signature of an op stream with the per-step attention values left out.
fn signature(ops: &[Op]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for op in ops {
        if let Some((s, d, r)) = op.copy {
            h = fnv(h, &vk::Handle::as_raw(s).to_le_bytes());
            h = fnv(h, &vk::Handle::as_raw(d).to_le_bytes());
            h = fnv(h, &r.src_offset.to_le_bytes());
            h = fnv(h, &r.dst_offset.to_le_bytes());
            h = fnv(h, &r.size.to_le_bytes());
            continue;
        }
        let mut push = op.push;
        let mut groups = op.groups;
        if let Some(d) = op.dynamic {
            for at in d.at.into_iter().flatten() {
                push[at..at + 4].fill(0);
            }
            if d.indirect {
                groups[1] = 0;
            }
        }
        h = fnv(h, &vk::Handle::as_raw(op.pipeline).to_le_bytes());
        h = fnv(h, &push[..op.plen]);
        for g in groups {
            h = fnv(h, &g.to_le_bytes());
        }
    }
    h
}

impl Shared {
    fn raw_host(&self, bytes: usize) -> RawBuf {
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(bytes as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::INDIRECT_BUFFER);
            let buf = self.device.create_buffer(&info, None).expect("buffer");
            let req = self.device.get_buffer_memory_requirements(buf);
            let ty = self
                .mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, vk::MemoryPropertyFlags::empty())
                .expect("host memory");
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut flags);
            let mem = self.device.allocate_memory(&ai, None).expect("host memory");
            self.device.bind_buffer_memory(buf, mem, 0).expect("bind");
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            let map = self.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("map") as *mut u8;
            RawBuf { buf, mem, addr, map }
        }
    }

    /// Writes `ops` into `cb`. With `params`, attention values come from its slots (pointer in the push
    /// constants, attn_split's grid by indirect dispatch); without, they are the recorded values.
    unsafe fn record(&self, cb: vk::CommandBuffer, ops: &[Op], params: Option<&RawBuf>, flags: vk::CommandBufferUsageFlags, prof: Option<&Prof>) {
        let d = &self.device;
        unsafe {
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(flags)).expect("begin");
            if let Some(pr) = prof {
                d.cmd_reset_query_pool(cb, pr.pool, 0, PROF_MAX);
            }
            barrier(d, cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS);
            let mut slot = 0usize;
            let mut stamp = 0u32;
            for op in ops {
                if let Some((s, dst, r)) = op.copy {
                    d.cmd_copy_buffer(cb, s, dst, &[r]);
                    barrier(d, cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER);
                    continue;
                }
                let mut push = op.push;
                let mut indirect = None;
                if let (Some(dy), Some(p)) = (op.dynamic, params) {
                    let a = p.addr + (slot * SLOT_BYTES) as u64;
                    push[dy.slot_at..dy.slot_at + 8].copy_from_slice(&a.to_le_bytes());
                    if dy.indirect {
                        indirect = Some((p.buf, (slot * SLOT_BYTES + 8) as u64));
                    }
                    slot += 1;
                }
                let timed = prof.filter(|pr| (stamp as usize) < pr.names.len()).map(|pr| pr.pool);
                if let Some(pool) = timed {
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 2 * stamp);
                }
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, op.pipeline);
                d.cmd_push_constants(cb, op.layout, vk::ShaderStageFlags::COMPUTE, 0, &push[..op.plen]);
                match indirect {
                    Some((b, off)) => d.cmd_dispatch_indirect(cb, b, off),
                    None => d.cmd_dispatch(cb, op.groups[0], op.groups[1], op.groups[2]),
                }
                if let Some(pool) = timed {
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, 2 * stamp + 1);
                }
                stamp += 1;
                barrier(d, cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER);
            }
            barrier(d, cb, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST);
            d.end_command_buffer(cb).expect("end");
        }
    }
}

/// Writes this step's attention values into a cached stream's parameter slots.
fn fill_params(p: &RawBuf, ops: &[Op]) {
    let mut slot = 0;
    for op in ops {
        if let Some(d) = op.dynamic {
            let v = [d.off, d.ns, op.groups[0], op.groups[1], op.groups[2]];
            for (i, x) in v.iter().enumerate() {
                unsafe { (p.map.add(slot * SLOT_BYTES + 4 * i) as *mut u32).write(*x) };
            }
            slot += 1;
        }
    }
}

impl<'g> Cmd<'g> {
    /// Dispatch `groups` workgroups. The workgroup size is the kernel's own; `threads.0` is passed after the
    /// arguments for kernels that size their work by it.
    pub fn run(&mut self, name: &str, args: &[Arg], groups: (usize, usize, usize), threads: (usize, usize, usize)) {
        if self.gpu.coopmat && matches!(name, "qmm3" | "qmvx" | "qmvr") {
            return self.run_cm(args);
        }
        let u = |i: usize| match args.get(i) {
            Some(Arg::U32(v)) => *v as usize,
            _ => unreachable!("{name}: argument {i} is not a scalar"),
        };
        // verify batches: 64-key attention chunks (as decode) instead of the prefill's 512, so the few query rows
        // still spread over enough workgroups
        if name == "attn_split" && groups.2 <= 16 && u(5) == 512 {
            let n = u(4) + groups.2;
            let ns = n.div_ceil(64);
            self.attn_ns = Some(ns as u32);
            let a: Vec<Arg> = args
                .iter()
                .enumerate()
                .map(|(i, a)| match (i, a) {
                    (5, _) => Arg::U32(64),
                    (6, _) => Arg::U32(ns as u32),
                    (_, Arg::Buf(b)) => Arg::Buf(*b),
                    (_, Arg::U32(v)) => Arg::U32(*v),
                })
                .collect();
            return self.dispatch(name, &a, (groups.0, ns, groups.2), threads);
        }
        if name == "attn_combine"
            && let Some(ns) = self.attn_ns.take()
        {
            let a = [copy(&args[0]), copy(&args[1]), Arg::U32(ns), copy(&args[3])];
            return self.dispatch(name, &a, groups, threads);
        }
        // K = 2048: 2 subgroups x 4 rows per workgroup (as Metal); longer K (the MLP down projection): 4 rows per
        // workgroup with K split over 4 subgroups, which halves its latency on the RTX 4000 Ada
        let (name, groups) = match name {
            "qmv4" if u(4) > 2048 => ("qmv4s", (u(5) / 4, 1, 1)),
            _ => (name, groups),
        };
        self.dispatch(name, args, groups, threads);
    }

    /// A T > 1 matmul (qmm3 / qmvx / qmvr arguments) on qmm_cm. qmvr has no LoRA arguments: its LoRA runs in
    /// the lora_b pass after it.
    fn run_cm(&mut self, args: &[Arg]) {
        let u = |i: usize| match args[i] {
            Arg::U32(v) => v as usize,
            Arg::Buf(_) => unreachable!("matmul argument {i} is not a scalar"),
        };
        let (k, o, add, t) = (u(4), u(5), u(6), u(7));
        let buf = |i: usize| match args[i] {
            Arg::Buf(b) => b,
            Arg::U32(_) => unreachable!("matmul argument {i} is not a buffer"),
        };
        // qmvr has no LoRA arguments (its LoRA runs in the lora_b pass after it)
        let (bm, uu, lr, ur) = if args.len() >= 12 { (buf(8), buf(9), u(10), u(11)) } else { (buf(1), buf(1), 0, 0) };
        assert!(lr <= 128, "LoRA rank {lr} over 128");
        // verify batches (T <= 16): qmv_cm, 128 rows per workgroup; longer: qmm_cm, 64 rows x 32 tokens
        let (kernel, rows, tile) = if t <= 16 && o % 128 == 0 { ("qmv_cm", 128, 16) } else { ("qmm_cm", 64, 32) };
        let groups = (o / rows) * t.div_ceil(tile);
        let mut split = CM_TARGET_GROUPS.div_ceil(groups).min(k / 128).max(1);
        while split > 1 && split * t * o * 4 > PART_BYTES {
            split -= 1;
        }
        let kslice = (k / split).div_ceil(128) * 128;
        let split = k.div_ceil(kslice);
        let part = self.gpu.part.clone().expect("coopmat scratch");
        let pb = B(&part, 0);
        let full: [Arg; 14] = [
            buf(0).into(), buf(1).into(), buf(2).into(), buf(3).into(),
            (k as u32).into(), (o as u32).into(), (add as u32).into(), (t as u32).into(),
            bm.into(), uu.into(), (lr as u32).into(), (ur as u32).into(),
            pb.into(), (kslice as u32).into(),
        ];
        self.dispatch(kernel, &full, (o / rows, t.div_ceil(tile), split), (128, 1, 1));
        if split > 1 {
            let red: [Arg; 6] = [pb.into(), buf(3).into(), (t as u32).into(), (o as u32).into(), (split as u32).into(), (add as u32).into()];
            self.dispatch("qmm_reduce", &red, ((t * o).div_ceil(256), 1, 1), (256, 1, 1));
        }
    }

    fn dispatch(&mut self, name: &str, args: &[Arg], groups: (usize, usize, usize), threads: (usize, usize, usize)) {
        let p = self.gpu.pipes.get(name).unwrap_or_else(|| panic!("no Vulkan kernel {name}"));
        let mut push = [0u8; PUSH_BYTES as usize];
        let mut n = 0usize;
        let mut at = Vec::with_capacity(args.len());
        for a in args {
            match a {
                Arg::Buf(B(b, off)) => {
                    n = n.div_ceil(8) * 8;
                    push[n..n + 8].copy_from_slice(&(b.0.addr + *off as u64).to_le_bytes());
                    at.push(n);
                    n += 8;
                }
                Arg::U32(v) => {
                    n = n.div_ceil(4) * 4;
                    push[n..n + 4].copy_from_slice(&v.to_le_bytes());
                    at.push(n);
                    n += 4;
                }
            }
        }
        push[n..n + 4].copy_from_slice(&(threads.0 as u32).to_le_bytes());
        n += 4;
        // attention kernels: the slot pointer follows `threads`
        let val = |i: usize| match args[i] {
            Arg::U32(v) => v,
            Arg::Buf(_) => unreachable!(),
        };
        let dynamic = match name {
            "attn_prep" => Some((val(11), 0, [Some(at[11]), None], false)),
            "attn_split" => Some((val(4), val(6), [Some(at[4]), Some(at[6])], true)),
            "attn_combine" => Some((0, val(2), [None, Some(at[2])], false)),
            _ => None,
        }
        .map(|(off, ns, at, indirect)| {
            n = n.div_ceil(8) * 8;
            let slot_at = n;
            n += 8;
            Dyn { off, ns, at, indirect, slot_at }
        });
        assert!(n <= PUSH_BYTES as usize, "{name}: {n} bytes of arguments");
        if let Some(l) = self.labels.as_mut() {
            l.push(prof_label(name, args));
        }
        self.ops.push(Op {
            pipeline: p.pipeline,
            layout: p.layout,
            push,
            plen: n,
            groups: [groups.0 as u32, groups.1 as u32, groups.2 as u32],
            dynamic,
            copy: None,
        });
    }

    pub fn copy(&mut self, src: B, dst: B, bytes: usize) {
        let r = vk::BufferCopy { src_offset: src.1 as u64, dst_offset: dst.1 as u64, size: bytes as u64 };
        self.ops.push(Op {
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            push: [0; PUSH_BYTES as usize],
            plen: 0,
            groups: [0; 3],
            dynamic: None,
            copy: Some((src.0 .0.buf, dst.0 .0.buf, r)),
        });
    }

    pub fn commit(self) -> Pending {
        let sh = &self.gpu.sh;
        let mut transient = Vec::new();
        let mut cached = Vec::new();
        let mut prof = None;
        let mut cbs = Vec::new();
        if let Some(labels) = self.labels {
            // profiling: one transient command buffer with timestamps, no cache
            let cb = sh.take_cmd();
            let pool = unsafe {
                sh.device
                    .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(PROF_MAX), None)
                    .expect("query pool")
            };
            let names: Vec<String> = labels.into_iter().take(PROF_MAX as usize / 2).collect();
            let pr = Prof { pool, names };
            unsafe { sh.record(cb, &self.ops, None, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, Some(&pr)) };
            prof = Some(pr);
            transient.push(cb);
            cbs.push(cb);
        } else {
            // a pending accept (ops before the first embed) and the forward pass are cached separately
            let embed = self.gpu.pipes["embed"].pipeline;
            let cut = self.ops.iter().position(|o| o.pipeline == embed).unwrap_or(0);
            let segments: Vec<&[Op]> = if cut > 0 { vec![&self.ops[..cut], &self.ops[cut..]] } else { vec![&self.ops[..]] };
            let mut cache = sh.cache.lock().unwrap();
            let generation = BUF_GEN.load(std::sync::atomic::Ordering::Relaxed);
            if cache.generation != generation {
                // a buffer was freed: idle recordings may point at it
                cache.generation = generation;
                cache.seen.clear();
                let keys: Vec<u64> = cache.entries.keys().copied().collect();
                for k in keys {
                    let e = cache.entries.remove(&k).unwrap();
                    if e.busy {
                        cache.doomed.push((k, e));
                    } else {
                        unsafe {
                            sh.device.free_command_buffers(sh.pool.lock().unwrap().pool, &[e.cb]);
                            e.params.free(&sh.device);
                        }
                    }
                }
            }
            for seg in segments {
                if seg.is_empty() {
                    continue;
                }
                let sig = signature(seg);
                cache.tick += 1;
                let tick = cache.tick;
                if let Some(e) = cache.entries.get_mut(&sig).filter(|e| !e.busy) {
                    fill_params(&e.params, seg);
                    e.busy = true;
                    e.used = tick;
                    cbs.push(e.cb);
                    cached.push(sig);
                    continue;
                }
                if cache.seen.len() > 4096 {
                    cache.seen.clear();
                }
                let seen = cache.seen.entry(sig).or_insert(0);
                *seen += 1;
                if *seen >= 2 && !cache.entries.contains_key(&sig) {
                    if cache.entries.len() >= CACHE_MAX {
                        let old = cache.entries.iter().filter(|(_, e)| !e.busy).min_by_key(|(_, e)| e.used).map(|(k, _)| *k);
                        if let Some(k) = old {
                            let e = cache.entries.remove(&k).unwrap();
                            unsafe {
                                sh.device.free_command_buffers(sh.pool.lock().unwrap().pool, &[e.cb]);
                                e.params.free(&sh.device);
                            }
                        }
                    }
                    if cache.entries.len() < CACHE_MAX {
                        let cb = sh.take_cmd();
                        let ndyn = seg.iter().filter(|o| o.dynamic.is_some()).count().max(1);
                        let params = sh.raw_host(ndyn * SLOT_BYTES);
                        unsafe { sh.record(cb, seg, Some(&params), vk::CommandBufferUsageFlags::empty(), None) };
                        fill_params(&params, seg);
                        cache.entries.insert(sig, Entry { cb, params, busy: true, used: tick });
                        cbs.push(cb);
                        cached.push(sig);
                        continue;
                    }
                }
                let cb = sh.take_cmd();
                unsafe { sh.record(cb, seg, None, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, None) };
                transient.push(cb);
                cbs.push(cb);
            }
        }
        let fence = sh.take_fence();
        unsafe {
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            sh.device.queue_submit(sh.queue, &[submit], fence).expect("submit");
        }
        Pending { sh: Arc::clone(sh), fence, transient, cached, prof }
    }

    pub fn finish(self) {
        self.commit().wait();
    }
}
/// Profile label: matmuls by shape (O x K, T) so their bandwidth can be read off.
fn prof_label(name: &str, args: &[Arg]) -> String {
    let u: Vec<u32> = args.iter().filter_map(|a| if let Arg::U32(v) = a { Some(*v) } else { None }).collect();
    match name {
        "qmv4" | "qmv4s" => format!("{name} {}x{}", u[1], u[0]),
        "qmv3" => format!("qmv3 {}x{}", u[5], u[0]),
        "qmvr" | "qmvx" | "qmm3" => format!("{name} {}x{} T{}", u[1], u[0], u[3]),
        "qmm_cm" | "qmv_cm" => format!("{name} {}x{} T{} s{}", u[1], u[0], u[3], u[0].div_ceil(u[6].max(1))),
        _ => name.to_string(),
    }
}

/// Weight bytes a matmul label reads (codes plus scales), for GB/s.
fn label_bytes(label: &str) -> Option<f64> {
    let shape = label.split(' ').nth(1)?;
    let (o, k) = shape.split_once('x')?;
    let (o, k): (f64, f64) = (o.parse().ok()?, k.parse().ok()?);
    Some(o * k / 4.0 + o * k / 128.0 * 2.0)
}

/// Per-kernel GPU time totals when BONSAI_PROFILE is set (also printed when the Gpu is dropped).
pub fn profile_report() {
    let mut p = PROFILE.lock().unwrap();
    if p.is_empty() {
        return;
    }
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = p.iter().map(|e| e.1).sum();
    for (k, t, n) in p.iter() {
        let per = t / *n as f64;
        let bw = label_bytes(k).map(|b| format!("{:7.1} GB/s", b / per / 1e9)).unwrap_or_default();
        eprintln!("{k:28} {:9.3} ms total {:8.2} us/call x{n:<7} {:5.1}% {bw}", t * 1e3, per * 1e6, t / total * 100.0);
    }
    p.clear();
}

pub struct Pending {
    sh: Arc<Shared>,
    fence: vk::Fence,
    transient: Vec<vk::CommandBuffer>,
    cached: Vec<u64>,
    prof: Option<Prof>,
}

impl Pending {
    pub fn wait(self) {
        let sh = &self.sh;
        unsafe {
            sh.device.wait_for_fences(&[self.fence], true, u64::MAX).expect("GPU error");
            sh.device.reset_fences(&[self.fence]).expect("reset fence");
        }
        sh.fences.lock().unwrap().push(self.fence);
        for cb in &self.transient {
            sh.give_cmd(*cb);
        }
        if !self.cached.is_empty() {
            let mut cache = sh.cache.lock().unwrap();
            for k in &self.cached {
                if let Some(e) = cache.entries.get_mut(k) {
                    e.busy = false;
                } else if let Some(i) = cache.doomed.iter().position(|(d, _)| d == k) {
                    let (_, e) = cache.doomed.swap_remove(i);
                    unsafe {
                        sh.device.free_command_buffers(sh.pool.lock().unwrap().pool, &[e.cb]);
                        e.params.free(&sh.device);
                    }
                }
            }
        }
        if let Some(pr) = self.prof {
            let d = &sh.device;
            let mut ts = vec![0u64; 2 * pr.names.len()];
            unsafe {
                d.get_query_pool_results(pr.pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                    .expect("timestamps");
                d.destroy_query_pool(pr.pool, None);
            }
            let mut p = PROFILE.lock().unwrap();
            for (i, name) in pr.names.iter().enumerate() {
                let dt = ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * sh.ts_period as f64 * 1e-9;
                match p.iter_mut().find(|e| e.0 == *name) {
                    Some(e) => {
                        e.1 += dt;
                        e.2 += 1;
                    }
                    None => p.push((name.clone(), dt, 1)),
                }
            }
        }
    }
}

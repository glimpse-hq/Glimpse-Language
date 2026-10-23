// C surface over libllama for the Rust side (bonsai/ggml.rs): plain arguments, no llama.h structs.
// One sequence (id 0) per context; greedy argmax per output row.

#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex g_log_mutex;
std::string g_last_error;
bool g_verbose = false;

void log_cb(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_last_error = text;
        while (!g_last_error.empty() && (g_last_error.back() == '\n' || g_last_error.back() == ' ')) {
            g_last_error.pop_back();
        }
    }
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

void set_err(char * err, int len, const std::string & msg) {
    if (err && len > 0) {
        snprintf(err, (size_t) len, "%s", msg.c_str());
    }
}

std::string last_error(const char * fallback) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_last_error.empty() ? std::string(fallback) : g_last_error;
}

std::vector<ggml_backend_dev_t> gpus() {
    std::vector<ggml_backend_dev_t> out;
    // discrete first, then integrated
    for (auto want : {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU}) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == want) {
                out.push_back(dev);
            }
        }
    }
    return out;
}

} // namespace

struct gb_model {
    llama_model * model = nullptr;
    llama_adapter_lora * lora = nullptr;
    int32_t n_vocab = 0;
};

struct gb_ctx {
    gb_model * m = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    int32_t n_batch = 0;
    int32_t n_out_max = 0;
};

extern "C" {

void gb_init(int verbose) {
    static std::once_flag once;
    g_verbose = verbose != 0;
    std::call_once(once, [] {
        // llama_log_set also replaces ggml's logger, which the host (transcribe.cpp) may have set
        ggml_log_callback prev_cb = nullptr;
        void * prev_ud = nullptr;
        ggml_log_get(&prev_cb, &prev_ud);
        llama_log_set(log_cb, nullptr);
        ggml_log_set(prev_cb, prev_ud);
        llama_backend_init();
    });
}

// GPU devices in the order gb_model_load indexes them. kind: 1 discrete, 2 integrated.
int gb_device_count(void) {
    return (int) gpus().size();
}

int gb_device_info(int i, int * kind, char * name, int len, uint64_t * mem_free, uint64_t * mem_total) {
    auto list = gpus();
    if (i < 0 || i >= (int) list.size()) {
        return -1;
    }
    ggml_backend_dev_t dev = list[i];
    *kind = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ? 1 : 2;
    set_err(name, len, std::string(ggml_backend_dev_description(dev)));
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(dev, &f, &t);
    *mem_free = f;
    *mem_total = t;
    return 0;
}

// device: index from gb_device_info. Every layer goes to that GPU.
gb_model * gb_model_load(const char * path, int device, char * err, int len) {
    auto list = gpus();
    if (device < 0 || device >= (int) list.size()) {
        set_err(err, len, "no such GPU");
        return nullptr;
    }
    ggml_backend_dev_t devs[2] = {list[device], nullptr};
    auto params = llama_model_default_params();
    params.devices = devs;
    params.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(path, params);
    if (!model) {
        set_err(err, len, last_error("failed to load the model"));
        return nullptr;
    }
    auto * m = new gb_model();
    m->model = model;
    m->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return m;
}

int gb_model_n_vocab(const gb_model * m) {
    return m->n_vocab;
}

// Loads a GGUF LoRA adapter for every context made afterwards; NULL removes it.
int gb_model_set_lora(gb_model * m, const char * path, char * err, int len) {
    if (m->lora) {
        llama_adapter_lora_free(m->lora);
        m->lora = nullptr;
    }
    if (!path) {
        return 0;
    }
    m->lora = llama_adapter_lora_init(m->model, path);
    if (!m->lora) {
        set_err(err, len, last_error("failed to load the adapter"));
        return -1;
    }
    return 0;
}

void gb_model_free(gb_model * m) {
    if (!m) {
        return;
    }
    if (m->lora) {
        llama_adapter_lora_free(m->lora);
    }
    llama_model_free(m->model);
    delete m;
}

gb_ctx * gb_ctx_new(gb_model * m, int n_ctx, int n_batch, int n_out_max, int n_rs, char * err, int len) {
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) n_batch;
    cp.n_seq_max = 1;
    cp.n_rs_seq = (uint32_t) n_rs;
    cp.n_outputs_max = (uint32_t) n_out_max;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(m->model, cp);
    if (!ctx) {
        set_err(err, len, last_error("failed to create the context"));
        return nullptr;
    }
    if (m->lora) {
        float scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &m->lora, 1, &scale) != 0) {
            llama_free(ctx);
            set_err(err, len, last_error("failed to apply the adapter"));
            return nullptr;
        }
    }
    auto * c = new gb_ctx();
    c->m = m;
    c->ctx = ctx;
    c->n_batch = n_batch;
    c->n_out_max = n_out_max;
    c->batch = llama_batch_init(n_batch, 0, 1);
    return c;
}

void gb_ctx_free(gb_ctx * c) {
    if (!c) {
        return;
    }
    llama_batch_free(c->batch);
    llama_free(c->ctx);
    delete c;
}

// Decodes ids[0..n) at positions pos0.. and writes the argmax after each of the last n_out tokens.
// Returns 0, or llama_decode's error code.
int gb_decode(gb_ctx * c, const int32_t * ids, int n, int pos0, int n_out, int32_t * out) {
    if (n_out > n || n_out > c->n_out_max) {
        return -100;
    }
    // chunks of n_batch, the last one holding every output row
    int done = 0;
    while (done < n) {
        int left = n - done;
        int take = left <= c->n_batch ? left : (left - n_out < c->n_batch ? left - n_out : c->n_batch);
        auto & b = c->batch;
        b.n_tokens = take;
        for (int i = 0; i < take; i++) {
            int k = done + i;
            b.token[i] = ids[k];
            b.pos[i] = pos0 + k;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = k >= n - n_out;
        }
        int ret = llama_decode(c->ctx, b);
        if (ret != 0) {
            return ret;
        }
        done += take;
    }
    const int nv = c->m->n_vocab;
    const int first = c->batch.n_tokens - n_out;
    for (int j = 0; j < n_out; j++) {
        const float * logits = llama_get_logits_ith(c->ctx, first + j);
        if (!logits) {
            return -101;
        }
        int best = 0;
        float bv = logits[0];
        for (int t = 1; t < nv; t++) {
            if (logits[t] > bv) {
                bv = logits[t];
                best = t;
            }
        }
        out[j] = best;
    }
    return 0;
}

// Drops positions >= p0. False when the recurrent state cannot go back that far.
int gb_seq_rm(gb_ctx * c, int p0) {
    return llama_memory_seq_rm(llama_get_memory(c->ctx), 0, p0, -1) ? 1 : 0;
}

int gb_pos_max(gb_ctx * c) {
    return (int) llama_memory_seq_pos_max(llama_get_memory(c->ctx), 0);
}

void gb_clear(gb_ctx * c) {
    llama_memory_clear(llama_get_memory(c->ctx), true);
}

// flags: LLAMA_STATE_SEQ_FLAGS_* (1 recurrent part only, 2 kept on the device).
uint64_t gb_state_size(gb_ctx * c, uint32_t flags) {
    return llama_state_seq_get_size_ext(c->ctx, 0, flags);
}

uint64_t gb_state_get(gb_ctx * c, uint8_t * dst, uint64_t size, uint32_t flags) {
    return llama_state_seq_get_data_ext(c->ctx, dst, (size_t) size, 0, flags);
}

uint64_t gb_state_set(gb_ctx * c, const uint8_t * src, uint64_t size, uint32_t flags) {
    return llama_state_seq_set_data_ext(c->ctx, src, (size_t) size, 0, flags);
}

} // extern "C"

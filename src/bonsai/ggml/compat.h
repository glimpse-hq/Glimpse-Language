// What vendor/llama.cpp (ggml 0.27 era) calls that transcribe.cpp's ggml 0.26 lacks. Force-included into
// libllama; the stubs in compat.cpp drop two scheduler hooks (logging and MoE expert copies) and a sparse
// attention op no Bonsai graph uses.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef bool (*ggml_backend_sched_copy_callback)(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst, struct ggml_cgraph * graph, void * user_data);
typedef void (*ggml_backend_sched_realloc_callback)(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool backend_ids_changed, bool graph_size_changed, void * user_data);
void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data);
void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data);
struct ggml_tensor * ggml_top_k_mask(struct ggml_context * ctx, struct ggml_tensor * mask, struct ggml_tensor * top_k, struct ggml_tensor * score, struct ggml_tensor * blocks, struct ggml_tensor * extra);
#ifdef __cplusplus
}
#endif

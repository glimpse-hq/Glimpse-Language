#include "compat.h"
#include <cstdlib>

extern "C" {
void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t, ggml_backend_sched_copy_callback, void *) {}
void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t, ggml_backend_sched_realloc_callback, void *) {}
struct ggml_tensor * ggml_top_k_mask(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *) {
    abort();
}
}

#ifndef INSTORAGE_ATTENTION_H
#define INSTORAGE_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

/**
 * Freestanding Embedded C Kernel Header for In-Storage Attention Filtering.
 * Zero libc dependencies: no malloc, no stdlib, no math.h.
 * Tailored for embedded SSD controller execution.
 */

EXPORT_API float compute_block_score(
    const float* query,
    const float* k_block,
    int tokens,
    int heads,
    int head_dim,
    float scale
);

EXPORT_API int instorage_topk_filter(
    const float* query,
    const float* k_blocks,
    int num_blocks,
    int tokens,
    int heads,
    int head_dim,
    int top_k,
    float scale,
    int* out_topk_indices,
    float* out_topk_scores
);

/* High-performance AVX2 SIMD GQA-aware In-Storage Scoring & Filtering */
EXPORT_API float compute_block_score_gqa_avx2(
    const float* query,
    const float* k_block,
    int actual_tokens,
    int q_heads,
    int kv_heads,
    int head_dim,
    float scale
);

EXPORT_API int instorage_topk_filter_gqa_avx2(
    const float* query,
    const float* const* k_blocks,
    const int* actual_tokens,
    int num_blocks,
    int q_heads,
    int kv_heads,
    int head_dim,
    int top_k,
    float scale,
    int* out_topk_indices,
    float* out_topk_scores
);

#ifdef __cplusplus
}
#endif

#endif /* INSTORAGE_ATTENTION_H */

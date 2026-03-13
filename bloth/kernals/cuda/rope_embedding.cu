/*
 * Bloth: Rotary Position Embedding (RoPE) Kernel
 * 
 * Optimized RoPE implementation for transformer models.
 * RoPE encodes position information by rotating the query/key vectors.
 * 
 * Copyright 2026 Bloth Team
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace bloth {
namespace rope {

// Precompute cos/sin cache for RoPE
__global__ void precompute_rope_cache(
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    int max_seq_len,
    int head_dim,
    float base = 10000.0f
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq_len * (head_dim / 2)) return;
    
    const int pos = idx / (head_dim / 2);
    const int dim_idx = idx % (head_dim / 2);
    
    // Compute theta = pos / (base^(2*dim_idx/head_dim))
    const float theta = pos / powf(base, 2.0f * dim_idx / head_dim);
    
    cos_cache[idx] = cosf(theta);
    sin_cache[idx] = sinf(theta);
}

// Apply RoPE to Q and K tensors
template<typename T>
__global__ void __launch_bounds__(256, 2)
apply_rope(
    T* __restrict__ q,
    T* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ position_ids,  // Optional, can be NULL for sequential
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    const int d = idx % head_dim;
    const int seq = (idx / head_dim) % seq_len;
    const int head = (idx / (head_dim * seq_len)) % num_heads;
    const int batch = idx / (head_dim * seq_len * num_heads);
    
    // Get position (use position_ids if provided, else sequential)
    const int pos = position_ids ? position_ids[batch * seq_len + seq] : seq;
    
    // Only process even dimensions (pairs)
    if (d % 2 != 0) return;
    
    // Get cos/sin for this position and dimension pair
    const int cache_idx = pos * (head_dim / 2) + d / 2;
    const float cos_val = cos_cache[cache_idx];
    const float sin_val = sin_cache[cache_idx];
    
    // Apply rotation to Q
    const float q_even = static_cast<float>(q[idx]);
    const float q_odd = static_cast<float>(q[idx + 1]);
    q[idx] = static_cast<T>(q_even * cos_val - q_odd * sin_val);
    q[idx + 1] = static_cast<T>(q_even * sin_val + q_odd * cos_val);
    
    // Apply rotation to K
    const float k_even = static_cast<float>(k[idx]);
    const float k_odd = static_cast<float>(k[idx + 1]);
    k[idx] = static_cast<T>(k_even * cos_val - k_odd * sin_val);
    k[idx + 1] = static_cast<T>(k_even * sin_val + k_odd * cos_val);
}

// Fused RoPE + attention score computation
template<typename T>
__global__ void __launch_bounds__(256, 2)
fused_rope_attention_scores(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ scores,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Each block computes one attention score matrix element
    const int batch = blockIdx.z / num_heads;
    const int head = blockIdx.z % num_heads;
    const int q_seq = blockIdx.y;
    const int k_seq = blockIdx.x;
    
    if (q_seq >= seq_len || k_seq >= seq_len) return;
    
    // Compute dot product with RoPE applied on-the-fly
    float sum = 0.0f;
    
    for (int d = 0; d < head_dim; d += 2) {
        // Get Q values
        const int q_idx = ((batch * num_heads + head) * seq_len + q_seq) * head_dim + d;
        const float q_even = static_cast<float>(q[q_idx]);
        const float q_odd = static_cast<float>(q[q_idx + 1]);
        
        // Get K values
        const int k_idx = ((batch * num_heads + head) * seq_len + k_seq) * head_dim + d;
        const float k_even = static_cast<float>(k[k_idx]);
        const float k_odd = static_cast<float>(k[k_idx + 1]);
        
        // Get cos/sin for Q position
        const int q_cache_idx = q_seq * (head_dim / 2) + d / 2;
        const float q_cos = cos_cache[q_cache_idx];
        const float q_sin = sin_cache[q_cache_idx];
        
        // Get cos/sin for K position
        const int k_cache_idx = k_seq * (head_dim / 2) + d / 2;
        const float k_cos = cos_cache[k_cache_idx];
        const float k_sin = sin_cache[k_cache_idx];
        
        // Apply RoPE and accumulate
        // q_rot = [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos]
        // k_rot = [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos]
        // dot = q_rot[0] * k_rot[0] + q_rot[1] * k_rot[1]
        
        const float q_rot_even = q_even * q_cos - q_odd * q_sin;
        const float q_rot_odd = q_even * q_sin + q_odd * q_cos;
        const float k_rot_even = k_even * k_cos - k_odd * k_sin;
        const float k_rot_odd = k_even * k_sin + k_odd * k_cos;
        
        sum += q_rot_even * k_rot_even + q_rot_odd * k_rot_odd;
    }
    
    // Store scaled score
    const int score_idx = ((batch * num_heads + head) * seq_len + q_seq) * seq_len + k_seq;
    scores[score_idx] = sum * scale;
}

// RoPE with NTK-aware scaling (for longer contexts)
template<typename T>
__global__ void __launch_bounds__(256, 2)
apply_rope_ntk(
    T* __restrict__ q,
    T* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scaling_factor
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    if (idx >= total_elements) return;
    
    const int d = idx % head_dim;
    const int seq = (idx / head_dim) % seq_len;
    const int head = (idx / (head_dim * seq_len)) % num_heads;
    const int batch = idx / (head_dim * seq_len * num_heads);
    
    if (d % 2 != 0) return;
    
    // Apply NTK scaling to position
    const float scaled_pos = seq * scaling_factor;
    
    // Interpolate in cache
    const int pos_low = static_cast<int>(scaled_pos);
    const int pos_high = pos_low + 1;
    const float t = scaled_pos - pos_low;
    
    const int cache_idx_low = pos_low * (head_dim / 2) + d / 2;
    const int cache_idx_high = pos_high * (head_dim / 2) + d / 2;
    
    const float cos_val = (1 - t) * cos_cache[cache_idx_low] + t * cos_cache[cache_idx_high];
    const float sin_val = (1 - t) * sin_cache[cache_idx_low] + t * sin_cache[cache_idx_high];
    
    // Apply rotation
    const float q_even = static_cast<float>(q[idx]);
    const float q_odd = static_cast<float>(q[idx + 1]);
    q[idx] = static_cast<T>(q_even * cos_val - q_odd * sin_val);
    q[idx + 1] = static_cast<T>(q_even * sin_val + q_odd * cos_val);
    
    const float k_even = static_cast<float>(k[idx]);
    const float k_odd = static_cast<float>(k[idx + 1]);
    k[idx] = static_cast<T>(k_even * cos_val - k_odd * sin_val);
    k[idx + 1] = static_cast<T>(k_even * sin_val + k_odd * cos_val);
}

// Kernel launcher
template<typename T>
class RoPELauncher {
public:
    static void precompute_cache(
        float* cos_cache,
        float* sin_cache,
        int max_seq_len,
        int head_dim,
        float base,
        cudaStream_t stream
    ) {
        const int total_elements = max_seq_len * (head_dim / 2);
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        
        precompute_rope_cache<<<grid_size, block_size, 0, stream>>>(
            cos_cache, sin_cache, max_seq_len, head_dim, base
        );
    }
    
    static void apply(
        T* q,
        T* k,
        const float* cos_cache,
        const float* sin_cache,
        const int* position_ids,
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        cudaStream_t stream
    ) {
        const int total_elements = batch_size * num_heads * seq_len * head_dim;
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        
        apply_rope<T><<<grid_size, block_size, 0, stream>>>(
            q, k, cos_cache, sin_cache, position_ids,
            batch_size, num_heads, seq_len, head_dim
        );
    }
};

// Explicit instantiations
template class RoPELauncher<__half>;
template class RoPELauncher<__nv_bfloat16>;

} // namespace rope
} // namespace bloth

// C API for Python bindings
extern "C" {

void bloth_precompute_rope_cache(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    cudaStream_t stream
) {
    bloth::rope::RoPELauncher<__half>::precompute_cache(
        cos_cache, sin_cache, max_seq_len, head_dim, base, stream
    );
}

void bloth_apply_rope_f16(
    void* q,
    void* k,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    bloth::rope::RoPELauncher<__half>::apply(
        static_cast<__half*>(q),
        static_cast<__half*>(k),
        cos_cache, sin_cache, position_ids,
        batch_size, num_heads, seq_len, head_dim,
        stream
    );
}

} // extern "C"

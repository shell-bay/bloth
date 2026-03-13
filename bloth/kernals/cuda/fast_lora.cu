/*
 * Bloth: Fast LoRA Kernels
 * 
 * Optimized kernels for LoRA (Low-Rank Adaptation) operations:
 * - Fused LoRA linear layer: x @ W + x @ A @ B * scaling
 * - Batched LoRA for multiple adapters
 * - Gradient computation for LoRA weights
 * - Memory-efficient implementations
 * 
 * Copyright 2026 Bloth Team
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace bloth {
namespace lora {

// Fused LoRA forward pass
// Computes: output = x @ W + x @ A @ B * scaling
template<typename T, int RANK>
__global__ void __launch_bounds__(256, 2)
fused_lora_forward(
    const T* __restrict__ x,
    const T* __restrict__ base_weight,
    const T* __restrict__ lora_a,
    const T* __restrict__ lora_b,
    T* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float scaling
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= batch_size || col >= out_features) return;
    
    // Base path: x @ W
    float base_sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < in_features; ++i) {
        base_sum += static_cast<float>(x[row * in_features + i]) * 
                    static_cast<float>(base_weight[col * in_features + i]);
    }
    
    // LoRA path: x @ A @ B
    // First compute intermediate: x @ A
    float lora_intermediate[RANK];
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < in_features; ++i) {
            sum += static_cast<float>(x[row * in_features + i]) * 
                   static_cast<float>(lora_a[r * in_features + i]);
        }
        lora_intermediate[r] = sum;
    }
    
    // Then compute: intermediate @ B
    float lora_sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_sum += lora_intermediate[r] * 
                    static_cast<float>(lora_b[col * RANK + r]);
    }
    
    // Combine with scaling
    float result = base_sum + lora_sum * scaling;
    output[row * out_features + col] = static_cast<T>(result);
}

// Batched LoRA for multiple adapters
// Each sample in batch can use a different LoRA adapter
template<typename T, int RANK>
__global__ void __launch_bounds__(256, 2)
batched_lora_forward(
    const T* __restrict__ x,
    const T* __restrict__ base_weight,
    const T* __restrict__ lora_a,  // [num_adapters, rank, in_features]
    const T* __restrict__ lora_b,  // [num_adapters, out_features, rank]
    const int* __restrict__ adapter_indices,  // [batch_size]
    T* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_adapters,
    float scaling
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= batch_size || col >= out_features) return;
    
    const int adapter_idx = adapter_indices[row];
    
    // Get LoRA weights for this adapter
    const T* adapter_lora_a = lora_a + adapter_idx * RANK * in_features;
    const T* adapter_lora_b = lora_b + adapter_idx * out_features * RANK;
    
    // Base path
    float base_sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < in_features; ++i) {
        base_sum += static_cast<float>(x[row * in_features + i]) * 
                    static_cast<float>(base_weight[col * in_features + i]);
    }
    
    // LoRA path
    float lora_intermediate[RANK];
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < in_features; ++i) {
            sum += static_cast<float>(x[row * in_features + i]) * 
                   static_cast<float>(adapter_lora_a[r * in_features + i]);
        }
        lora_intermediate[r] = sum;
    }
    
    float lora_sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_sum += lora_intermediate[r] * 
                    static_cast<float>(adapter_lora_b[col * RANK + r]);
    }
    
    float result = base_sum + lora_sum * scaling;
    output[row * out_features + col] = static_cast<T>(result);
}

// LoRA backward pass - compute gradients
template<typename T, int RANK>
__global__ void __launch_bounds__(256, 2)
lora_backward(
    const T* __restrict__ grad_output,
    const T* __restrict__ x,
    const T* __restrict__ lora_a,
    const T* __restrict__ lora_b,
    T* __restrict__ grad_lora_a,
    T* __restrict__ grad_lora_b,
    int batch_size,
    int in_features,
    int out_features,
    float scaling
) {
    // Compute gradients for LoRA weights
    // grad_A = x.T @ (grad_output @ B.T) * scaling
    // grad_B = (x @ A).T @ grad_output * scaling
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread handles one element of grad_lora_a or grad_lora_b
    // Simplified implementation - full version would use shared memory
    
    // For grad_lora_a: [rank, in_features]
    if (tid < RANK * in_features) {
        int r = tid / in_features;
        int i = tid % in_features;
        
        float grad = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < batch_size; ++b) {
            // Compute (grad_output @ B.T)[b, r]
            float grad_intermediate = 0.0f;
            #pragma unroll 4
            for (int o = 0; o < out_features; ++o) {
                grad_intermediate += static_cast<float>(grad_output[b * out_features + o]) *
                                      static_cast<float>(lora_b[o * RANK + r]);
            }
            grad += static_cast<float>(x[b * in_features + i]) * grad_intermediate;
        }
        
        grad_lora_a[r * in_features + i] = static_cast<T>(grad * scaling);
    }
    
    // For grad_lora_b: [out_features, rank]
    const int tid_b = tid - RANK * in_features;
    if (tid_b >= 0 && tid_b < out_features * RANK) {
        int o = tid_b / RANK;
        int r = tid_b % RANK;
        
        float grad = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < batch_size; ++b) {
            // Compute (x @ A)[b, r]
            float x_a = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < in_features; ++i) {
                x_a += static_cast<float>(x[b * in_features + i]) *
                       static_cast<float>(lora_a[r * in_features + i]);
            }
            grad += x_a * static_cast<float>(grad_output[b * out_features + o]);
        }
        
        grad_lora_b[o * RANK + r] = static_cast<T>(grad * scaling);
    }
}

// Fused LoRA + Dropout
template<typename T, int RANK>
__global__ void __launch_bounds__(256, 2)
fused_lora_dropout_forward(
    const T* __restrict__ x,
    const T* __restrict__ base_weight,
    const T* __restrict__ lora_a,
    const T* __restrict__ lora_b,
    const bool* __restrict__ dropout_mask,
    T* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float scaling,
    float dropout_prob
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= batch_size || col >= out_features) return;
    
    // Check dropout mask
    bool dropped = dropout_mask && dropout_mask[row * out_features + col];
    
    if (dropped) {
        output[row * out_features + col] = static_cast<T>(0);
        return;
    }
    
    // Scale by 1/(1-dropout) for inverted dropout
    float scale = dropout_prob > 0 ? 1.0f / (1.0f - dropout_prob) : 1.0f;
    
    // Base path
    float base_sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < in_features; ++i) {
        base_sum += static_cast<float>(x[row * in_features + i]) * 
                    static_cast<float>(base_weight[col * in_features + i]);
    }
    
    // LoRA path
    float lora_intermediate[RANK];
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < in_features; ++i) {
            sum += static_cast<float>(x[row * in_features + i]) * 
                   static_cast<float>(lora_a[r * in_features + i]);
        }
        lora_intermediate[r] = sum;
    }
    
    float lora_sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_sum += lora_intermediate[r] * 
                    static_cast<float>(lora_b[col * RANK + r]);
    }
    
    float result = (base_sum + lora_sum * scaling) * scale;
    output[row * out_features + col] = static_cast<T>(result);
}

// Quantized LoRA (QLoRA) - 4-bit base weights
template<typename T, int RANK>
__global__ void __launch_bounds__(256, 2)
qlora_forward(
    const T* __restrict__ x,
    const uint8_t* __restrict__ quantized_weight,
    const float* __restrict__ weight_scales,
    const T* __restrict__ lora_a,
    const T* __restrict__ lora_b,
    T* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int group_size,
    float scaling
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= batch_size || col >= out_features) return;
    
    // Dequantize and compute base path
    float base_sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < in_features; ++i) {
        // Dequantize weight
        int group_idx = i / group_size;
        float scale = weight_scales[col * (in_features / group_size) + group_idx];
        
        uint8_t qval = quantized_weight[col * in_features + i];
        float wval = (static_cast<float>(qval) - 8.0f) * scale;  // NF4 dequantization
        
        base_sum += static_cast<float>(x[row * in_features + i]) * wval;
    }
    
    // LoRA path (same as before)
    float lora_intermediate[RANK];
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < in_features; ++i) {
            sum += static_cast<float>(x[row * in_features + i]) * 
                   static_cast<float>(lora_a[r * in_features + i]);
        }
        lora_intermediate[r] = sum;
    }
    
    float lora_sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_sum += lora_intermediate[r] * 
                    static_cast<float>(lora_b[col * RANK + r]);
    }
    
    float result = base_sum + lora_sum * scaling;
    output[row * out_features + col] = static_cast<T>(result);
}

// Kernel launcher
template<typename T>
class LoRALauncher {
public:
    template<int RANK>
    static void forward(
        const T* x,
        const T* base_weight,
        const T* lora_a,
        const T* lora_b,
        T* output,
        int batch_size,
        int in_features,
        int out_features,
        float scaling,
        cudaStream_t stream
    ) {
        dim3 block(16, 16);
        dim3 grid(
            (batch_size + block.x - 1) / block.x,
            (out_features + block.y - 1) / block.y
        );
        
        fused_lora_forward<T, RANK><<<grid, block, 0, stream>>>(
            x, base_weight, lora_a, lora_b, output,
            batch_size, in_features, out_features, scaling
        );
    }
    
    template<int RANK>
    static void backward(
        const T* grad_output,
        const T* x,
        const T* lora_a,
        const T* lora_b,
        T* grad_lora_a,
        T* grad_lora_b,
        int batch_size,
        int in_features,
        int out_features,
        float scaling,
        cudaStream_t stream
    ) {
        int total_elements = RANK * in_features + out_features * RANK;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        lora_backward<T, RANK><<<grid_size, block_size, 0, stream>>>(
            grad_output, x, lora_a, lora_b,
            grad_lora_a, grad_lora_b,
            batch_size, in_features, out_features, scaling
        );
    }
};

// Explicit instantiations
template class LoRALauncher<__half>;
template class LoRALauncher<__nv_bfloat16>;

} // namespace lora
} // namespace bloth

// C API for Python bindings
extern "C" {

void bloth_lora_forward_f16(
    const void* x,
    const void* base_weight,
    const void* lora_a,
    const void* lora_b,
    void* output,
    int batch_size,
    int in_features,
    int out_features,
    int rank,
    float scaling,
    cudaStream_t stream
) {
    // Dispatch based on rank
    switch (rank) {
        case 4:
            bloth::lora::LoRALauncher<__half>::forward<4>(
                static_cast<const __half*>(x),
                static_cast<const __half*>(base_weight),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(output),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 8:
            bloth::lora::LoRALauncher<__half>::forward<8>(
                static_cast<const __half*>(x),
                static_cast<const __half*>(base_weight),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(output),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 16:
            bloth::lora::LoRALauncher<__half>::forward<16>(
                static_cast<const __half*>(x),
                static_cast<const __half*>(base_weight),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(output),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 32:
            bloth::lora::LoRALauncher<__half>::forward<32>(
                static_cast<const __half*>(x),
                static_cast<const __half*>(base_weight),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(output),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 64:
            bloth::lora::LoRALauncher<__half>::forward<64>(
                static_cast<const __half*>(x),
                static_cast<const __half*>(base_weight),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(output),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        default:
            // Fall back to rank 16
            bloth::lora::LoRALauncher<__half>::forward<16>(
                static_cast<const __half*>(x),
                static_cast<const __half*>(base_weight),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(output),
                batch_size, in_features, out_features, scaling, stream
            );
    }
}

void bloth_lora_backward_f16(
    const void* grad_output,
    const void* x,
    const void* lora_a,
    const void* lora_b,
    void* grad_lora_a,
    void* grad_lora_b,
    int batch_size,
    int in_features,
    int out_features,
    int rank,
    float scaling,
    cudaStream_t stream
) {
    switch (rank) {
        case 4:
            bloth::lora::LoRALauncher<__half>::backward<4>(
                static_cast<const __half*>(grad_output),
                static_cast<const __half*>(x),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(grad_lora_a),
                static_cast<__half*>(grad_lora_b),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 8:
            bloth::lora::LoRALauncher<__half>::backward<8>(
                static_cast<const __half*>(grad_output),
                static_cast<const __half*>(x),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(grad_lora_a),
                static_cast<__half*>(grad_lora_b),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 16:
            bloth::lora::LoRALauncher<__half>::backward<16>(
                static_cast<const __half*>(grad_output),
                static_cast<const __half*>(x),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(grad_lora_a),
                static_cast<__half*>(grad_lora_b),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 32:
            bloth::lora::LoRALauncher<__half>::backward<32>(
                static_cast<const __half*>(grad_output),
                static_cast<const __half*>(x),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(grad_lora_a),
                static_cast<__half*>(grad_lora_b),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        case 64:
            bloth::lora::LoRALauncher<__half>::backward<64>(
                static_cast<const __half*>(grad_output),
                static_cast<const __half*>(x),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(grad_lora_a),
                static_cast<__half*>(grad_lora_b),
                batch_size, in_features, out_features, scaling, stream
            );
            break;
        default:
            bloth::lora::LoRALauncher<__half>::backward<16>(
                static_cast<const __half*>(grad_output),
                static_cast<const __half*>(x),
                static_cast<const __half*>(lora_a),
                static_cast<const __half*>(lora_b),
                static_cast<__half*>(grad_lora_a),
                static_cast<__half*>(grad_lora_b),
                batch_size, in_features, out_features, scaling, stream
            );
    }
}

} // extern "C"

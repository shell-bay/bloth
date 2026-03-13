/*
 * Bloth: Python Bindings for CUDA Kernels
 * 
 * PyTorch C++ extension bindings for all Bloth kernels.
 * 
 * Copyright 2026 Bloth Team
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// External kernel declarations
extern "C" {
    // GEMM
    void bloth_gemm_f16(const void* a, const void* b, void* c,
                        int m, int n, int k, float alpha, float beta,
                        cudaStream_t stream);
    
    // Attention
    void bloth_flash_attention_forward(const void* Q, const void* K, const void* V, void* O,
                                       float softmax_scale, int batch_size, int num_heads,
                                       int seq_len, int head_dim, cudaStream_t stream);
    
    // Layer Norm
    void bloth_rms_norm_forward(const void* input, const void* weight, void* output,
                                float* inv_rms, float eps, int num_tokens, int hidden_size,
                                cudaStream_t stream);
    
    // LoRA
    void bloth_lora_forward_f16(const void* x, const void* base_weight,
                                const void* lora_a, const void* lora_b, void* output,
                                int batch_size, int in_features, int out_features,
                                int rank, float scaling, cudaStream_t stream);
    
    // RoPE
    void bloth_apply_rope_f16(void* q, void* k, const float* cos_cache, const float* sin_cache,
                              const int* position_ids, int batch_size, int num_heads,
                              int seq_len, int head_dim, cudaStream_t stream);
}

// Helper to get CUDA stream from PyTorch
cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream();
}

// GEMM wrapper
torch::Tensor gemm_f16(torch::Tensor a, torch::Tensor b, 
                       int64_t m, int64_t n, int64_t k,
                       double alpha, double beta) {
    auto c = torch::empty({m, n}, a.options());
    
    bloth_gemm_f16(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        m, n, k,
        static_cast<float>(alpha),
        static_cast<float>(beta),
        get_cuda_stream()
    );
    
    return c;
}

// Flash Attention wrapper
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                      double softmax_scale, int64_t batch_size,
                                      int64_t num_heads, int64_t seq_len, int64_t head_dim) {
    auto O = torch::empty_like(Q);
    
    bloth_flash_attention_forward(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        O.data_ptr(),
        static_cast<float>(softmax_scale),
        batch_size, num_heads, seq_len, head_dim,
        get_cuda_stream()
    );
    
    return O;
}

// RMS Norm wrapper
torch::Tensor rms_norm_forward(torch::Tensor input, torch::Tensor weight,
                               double eps, int64_t num_tokens, int64_t hidden_size) {
    auto output = torch::empty_like(input);
    auto inv_rms = torch::empty(num_tokens, torch::dtype(torch::kFloat32).device(input.device()));
    
    bloth_rms_norm_forward(
        input.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
        inv_rms.data_ptr<float>(),
        static_cast<float>(eps),
        num_tokens, hidden_size,
        get_cuda_stream()
    );
    
    return output;
}

// LoRA wrapper
torch::Tensor lora_forward_f16(torch::Tensor x, torch::Tensor base_weight,
                               torch::Tensor lora_a, torch::Tensor lora_b,
                               int64_t batch_size, int64_t in_features, int64_t out_features,
                               int64_t rank, double scaling) {
    auto output = torch::empty({batch_size, out_features}, x.options());
    
    bloth_lora_forward_f16(
        x.data_ptr(),
        base_weight.data_ptr(),
        lora_a.data_ptr(),
        lora_b.data_ptr(),
        output.data_ptr(),
        batch_size, in_features, out_features,
        rank,
        static_cast<float>(scaling),
        get_cuda_stream()
    );
    
    return output;
}

// RoPE wrapper
torch::Tensor apply_rope_f16(torch::Tensor q, torch::Tensor k,
                             torch::Tensor cos_cache, torch::Tensor sin_cache,
                             torch::Tensor position_ids,
                             int64_t batch_size, int64_t num_heads,
                             int64_t seq_len, int64_t head_dim) {
    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    
    // Copy input to output (in-place modification)
    q_out.copy_(q);
    k_out.copy_(k);
    
    bloth_apply_rope_f16(
        q_out.data_ptr(),
        k_out.data_ptr(),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        position_ids.data_ptr<int>(),
        batch_size, num_heads, seq_len, head_dim,
        get_cuda_stream()
    );
    
    return torch::stack({q_out, k_out});
}

// Check CUDA availability
bool cuda_is_available() {
    return torch::cuda::is_available();
}

// Get CUDA device capability
std::tuple<int, int> get_device_capability(int64_t device_id) {
    if (!torch::cuda::is_available()) {
        return std::make_tuple(0, 0);
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    return std::make_tuple(prop.major, prop.minor);
}

// Get GPU memory info
std::tuple<int64_t, int64_t> get_memory_info(int64_t device_id) {
    if (!torch::cuda::is_available()) {
        return std::make_tuple(0, 0);
    }
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    return std::make_tuple(free, total);
}

// Synchronize CUDA
void cuda_synchronize() {
    cudaStreamSynchronize(get_cuda_stream());
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Bloth CUDA Kernels";
    
    // GEMM
    m.def("gemm_f16", &gemm_f16, "FP16 GEMM");
    
    // Attention
    m.def("flash_attention_forward", &flash_attention_forward, "Flash Attention Forward");
    
    // Normalization
    m.def("rms_norm_forward", &rms_norm_forward, "RMS Norm Forward");
    
    // LoRA
    m.def("lora_forward_f16", &lora_forward_f16, "LoRA Forward FP16");
    
    // RoPE
    m.def("apply_rope_f16", &apply_rope_f16, "Apply RoPE FP16");
    
    // Utilities
    m.def("cuda_is_available", &cuda_is_available, "Check CUDA availability");
    m.def("get_device_capability", &get_device_capability, "Get device capability");
    m.def("get_memory_info", &get_memory_info, "Get GPU memory info");
    m.def("cuda_synchronize", &cuda_synchronize, "Synchronize CUDA");
}

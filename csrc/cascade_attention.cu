#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
__global__ void merge_attn_states_kernel(
    scalar_t* __restrict__ output, const scalar_t* __restrict__ prefix_output,
    const float* __restrict__ prefix_lse,
    const scalar_t* __restrict__ suffix_output,
    const float* __restrict__ suffix_lse, int head_size) {
  int thread_id = threadIdx.x;
  int token_id = blockIdx.x;
  int num_tokens = gridDim.x;
  int head_id = blockIdx.y;
  int num_heads = gridDim.y;

  float p_lse = VLLM_LDG(prefix_lse + head_id * num_tokens + token_id);
  float s_lse = VLLM_LDG(suffix_lse + head_id * num_tokens + token_id);
  float max_lse = p_lse > s_lse ? p_lse : s_lse;
  p_lse -= max_lse;
  s_lse -= max_lse;
  float p_scale = __expf(p_lse) / (__expf(p_lse) + __expf(s_lse));
  float s_scale = __expf(s_lse) / (__expf(p_lse) + __expf(s_lse));

  int offset = token_id * num_heads * head_size + head_id * head_size;
  if (thread_id < head_size) {
    float p_out = (float)prefix_output[offset + thread_id];
    float s_out = (float)suffix_output[offset + thread_id];
    output[offset + thread_id] = (scalar_t)(p_out * p_scale + s_out * s_scale);
  }
}
}  // namespace vllm

void merge_attn_states(
    torch::Tensor& output,         // [num_tokens, num_heads, head_size]
    torch::Tensor& prefix_output,  // [num_tokens, num_heads, head_size]
    torch::Tensor& prefix_lse,     // [num_heads, num_tokens]
    torch::Tensor& suffix_output,  // [num_tokens, num_heads, head_size]
    torch::Tensor& suffix_lse      // [num_heads, num_tokens]
) {
  int num_tokens = output.size(0);
  int num_heads = output.size(1);
  int head_size = output.size(2);

  dim3 grid(num_tokens, num_heads);
  dim3 block((head_size + 31) / 32 * 32);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(output.scalar_type(), "merge_attn_states", [&] {
    vllm::merge_attn_states_kernel<scalar_t><<<grid, block, 0, stream>>>(
        output.data_ptr<scalar_t>(), prefix_output.data_ptr<scalar_t>(),
        prefix_lse.data_ptr<float>(), suffix_output.data_ptr<scalar_t>(),
        suffix_lse.data_ptr<float>(), head_size);
  });
}

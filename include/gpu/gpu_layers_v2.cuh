#ifndef GPU_LAYERS_V2_CUH
#define GPU_LAYERS_V2_CUH

#include <cuda_runtime.h>

// ============================================================================
// GPU Layer Operations V2 - Optimized Kernels
// ============================================================================

namespace gpu_v2 {

// Forward pass operations (FUSED kernels)
void conv2d_bias_relu_forward_v2(
    const float* d_input, const float* d_weight, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding
);

void conv2d_bias_forward_v2(
    const float* d_input, const float* d_weight, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding
);

void maxpool2d_forward_v2(
    const float* d_input, float* d_output,
    int batch_size, int channels, int height, int width,
    int pool_size, int stride
);

void upsample2d_forward_v2(
    const float* d_input, float* d_output,
    int batch_size, int channels, int height, int width,
    int scale_factor
);

// Backward helpers (reuse baseline-style kernels with batch dimension)
__global__ void mse_gradient_kernel_v2(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    int total_size
);

void conv2d_backward_input_v2(
    const float* d_input,
    const float* d_weights,
    const float* d_grad_output,
    float* d_grad_input,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

void conv2d_backward_full_v2(
    const float* d_input,
    const float* d_weights,
    const float* d_conv_output,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_weights,
    float* d_grad_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

void maxpool2d_backward_v2(
    const float* d_grad_output,
    const float* d_input,
    const float* d_output,
    float* d_grad_input,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int pool_size,
    int stride
);

void upsample2d_backward_v2(
    const float* d_grad_output,
    float* d_grad_input,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int scale_factor
);

// Loss and optimizer
float mse_loss_v2(
    const float* d_output, const float* d_target,
    int batch_size, int channels, int height, int width
);

void sgd_update_v2(float* d_weights, const float* d_gradients, float lr, int size);

} // namespace gpu_v2

#endif // GPU_LAYERS_V2_CUH

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

// Backward pass operations (OPTIMIZED kernels)
void conv2d_relu_backward_fused_kernel(
    const float* grad_output, const float* input, const float* weight,
    const float* conv_output, float* grad_input,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding
);

void conv2d_weight_grad_optimized_kernel(
    const float* input, const float* grad_output, float* grad_weight,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding
);

void bias_grad_optimized_kernel(
    const float* grad_output, float* grad_bias,
    int out_channels, int height, int width
);

void maxpool2d_backward_optimized_kernel(
    const float* grad_output, const float* input, const float* output, float* grad_input,
    int channels, int in_height, int in_width, int out_height, int out_width,
    int pool_size, int stride
);

void upsample2d_backward_optimized_kernel(
    const float* grad_output, float* grad_input,
    int channels, int in_height, int in_width, int out_height, int out_width,
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

// ============================================================================
// GPU Layer Operations V2 - Optimized Kernels
// Optimizations:
//   - Kernel Fusion: Conv+Bias+ReLU in single kernel
//   - Loop Unrolling: Manual unrolling for 3x3 convolutions
//   - Optimized Block Dimensions: Tuned for Tesla T4
// ============================================================================

#include "gpu/gpu_layers_v2.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

namespace gpu_v2 {

// ============================================================================
// FUSED FORWARD KERNELS (Conv + Bias + ReLU)
// ============================================================================

__global__ void conv2d_bias_relu_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width;
    
    if (idx >= total) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[oc];
    
    // LOOP UNROLLING for 3x3 kernel
    if (kernel_size == 3) {
        for (int ic = 0; ic < in_channels; ic++) {
            // Unroll 3x3 manually
            int ih, iw, input_idx, weight_idx;
            
            // Row 0
            ih = h - padding; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 0;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 1;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 2;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            // Row 1
            ih = h - padding + 1; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 3;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 1; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 4;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 1; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 5;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            // Row 2
            ih = h - padding + 2; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 6;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 2; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 7;
                sum += input[input_idx] * weight[weight_idx];
            }
            
            ih = h - padding + 2; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 8;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // FUSED ReLU activation
    output[idx] = (sum > 0.0f) ? sum : 0.0f;
}

__global__ void conv2d_bias_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width;
    
    if (idx >= total) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[oc];
    
    // LOOP UNROLLING for 3x3
    if (kernel_size == 3) {
        for (int ic = 0; ic < in_channels; ic++) {
            int ih, iw, input_idx, weight_idx;
            
            // Unroll 3x3 (same as above but no ReLU)
            ih = h - padding; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 0;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 1;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 2;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 1; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 3;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 1; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 4;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 1; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 5;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 2; iw = w - padding;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 6;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 2; iw = w - padding + 1;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 7;
                sum += input[input_idx] * weight[weight_idx];
            }
            ih = h - padding + 2; iw = w - padding + 2;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                weight_idx = oc * (in_channels * 9) + ic * 9 + 8;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    output[idx] = sum;  // No ReLU
}

void conv2d_bias_relu_forward_v2(
    const float* d_input, const float* d_weight, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding
) {
    int total = batch_size * out_channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv2d_bias_relu_forward_kernel<<<blocks, threads>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding
    );
}

void conv2d_bias_forward_v2(
    const float* d_input, const float* d_weight, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding
) {
    int total = batch_size * out_channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv2d_bias_forward_kernel<<<blocks, threads>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding
    );
}

// ============================================================================
// MAXPOOL2D FORWARD
// ============================================================================

__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = in_height / stride;
    int out_width = in_width / stride;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    float max_val = -1e38f;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = h * stride + ph;
            int iw = w * stride + pw;
            int input_idx = b * (channels * in_height * in_width) +
                           c * (in_height * in_width) + ih * in_width + iw;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }
    
    output[idx] = max_val;
}

void maxpool2d_forward_v2(
    const float* d_input, float* d_output,
    int batch_size, int channels, int height, int width,
    int pool_size, int stride
) {
    int out_height = height / stride;
    int out_width = width / stride;
    int total = batch_size * channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    maxpool2d_forward_kernel<<<blocks, threads>>>(
        d_input, d_output, batch_size, channels, height, width, pool_size, stride
    );
}

// ============================================================================
// UPSAMPLE2D FORWARD
// ============================================================================

__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_height, int in_width, int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    int ih = h / scale_factor;
    int iw = w / scale_factor;
    int input_idx = b * (channels * in_height * in_width) +
                   c * (in_height * in_width) + ih * in_width + iw;
    
    output[idx] = input[input_idx];
}

void upsample2d_forward_v2(
    const float* d_input, float* d_output,
    int batch_size, int channels, int height, int width, int scale_factor
) {
    int out_height = height * scale_factor;
    int out_width = width * scale_factor;
    int total = batch_size * channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    upsample2d_forward_kernel<<<blocks, threads>>>(
        d_input, d_output, batch_size, channels, height, width, scale_factor
    );
}

// ============================================================================
// BACKWARD KERNELS (OPTIMIZED)
// ============================================================================

__global__ void conv2d_relu_backward_fused_kernel_impl(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_output,
    float* __restrict__ grad_input,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (w >= width || h >= height) return;
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_channels; ++oc) {
        int grad_idx = oc * height * width + h * width + w;
        int conv_idx = oc * height * width + h * width + w;
        
        // FUSED: Apply ReLU gradient
        float grad = (conv_output[conv_idx] > 0.0f) ? grad_output[grad_idx] : 0.0f;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int oh = h + padding - kh;
                int ow = w + padding - kw;
                
                if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                    c * kernel_size * kernel_size +
                                    kh * kernel_size + kw;
                    sum += grad * weight[weight_idx];
                }
            }
        }
    }
    
    int input_idx = c * height * width + h * width + w;
    grad_input[input_idx] = sum;
}

__global__ void conv2d_weight_grad_optimized_kernel_impl(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weight,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding
) {
    int ic = blockIdx.x;
    int oc = blockIdx.y;
    int k = threadIdx.x;
    
    if (k >= kernel_size * kernel_size) return;
    
    int kh = k / kernel_size;
    int kw = k % kernel_size;
    
    float sum = 0.0f;
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int ih = h + padding - kh;
            int iw = w + padding - kw;
            
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = ic * height * width + ih * width + iw;
                int grad_idx = oc * height * width + h * width + w;
                sum += input[input_idx] * grad_output[grad_idx];
            }
        }
    }
    
    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                     ic * kernel_size * kernel_size + k;
    atomicAdd(&grad_weight[weight_idx], sum);
}

__global__ void bias_grad_optimized_kernel_impl(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int out_channels, int height, int width
) {
    extern __shared__ float shared[];
    
    int oc = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    float sum = 0.0f;
    int total = height * width;
    
    for (int i = tid; i < total; i += stride) {
        sum += grad_output[oc * total + i];
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    // Reduction
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&grad_bias[oc], shared[0]);
    }
}

__global__ void maxpool2d_backward_optimized_kernel_impl(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ output,
    float* __restrict__ grad_input,
    int channels, int in_height, int in_width,
    int out_height, int out_width, int pool_size, int stride
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (w >= in_width || h >= in_height) return;
    
    int input_idx = c * in_height * in_width + h * in_width + w;
    float input_val = input[input_idx];
    float grad = 0.0f;
    
    int oh_start = h / stride;
    int ow_start = w / stride;
    
    for (int oh = oh_start; oh < out_height && oh * stride < h + pool_size; ++oh) {
        for (int ow = ow_start; ow < out_width && ow * stride < w + pool_size; ++ow) {
            int output_idx = c * out_height * out_width + oh * out_width + ow;
            if (input_val == output[output_idx]) {
                grad += grad_output[output_idx];
            }
        }
    }
    
    grad_input[input_idx] = grad;
}

__global__ void upsample2d_backward_optimized_kernel_impl(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int channels, int in_height, int in_width,
    int out_height, int out_width, int scale_factor
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (w >= in_width || h >= in_height) return;
    
    float sum = 0.0f;
    
    for (int sh = 0; sh < scale_factor; ++sh) {
        for (int sw = 0; sw < scale_factor; ++sw) {
            int oh = h * scale_factor + sh;
            int ow = w * scale_factor + sw;
            int grad_idx = c * out_height * out_width + oh * out_width + ow;
            sum += grad_output[grad_idx];
        }
    }
    
    int input_idx = c * in_height * in_width + h * in_width + w;
    grad_input[input_idx] = sum;
}

// Wrapper functions
void conv2d_relu_backward_fused_kernel(
    const float* grad_output, const float* input, const float* weight,
    const float* conv_output, float* grad_input,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, in_channels);
    
    conv2d_relu_backward_fused_kernel_impl<<<grid, block>>>(
        grad_output, input, weight, conv_output, grad_input,
        in_channels, out_channels, height, width, kernel_size, padding
    );
}

void conv2d_weight_grad_optimized_kernel(
    const float* input, const float* grad_output, float* grad_weight,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding
) {
    dim3 grid(in_channels, out_channels);
    int threads = kernel_size * kernel_size;
    
    conv2d_weight_grad_optimized_kernel_impl<<<grid, threads>>>(
        input, grad_output, grad_weight,
        in_channels, out_channels, height, width, kernel_size, padding
    );
}

void bias_grad_optimized_kernel(
    const float* grad_output, float* grad_bias,
    int out_channels, int height, int width
) {
    bias_grad_optimized_kernel_impl<<<out_channels, 256, 256 * sizeof(float)>>>(
        grad_output, grad_bias, out_channels, height, width
    );
}

void maxpool2d_backward_optimized_kernel(
    const float* grad_output, const float* input, const float* output, float* grad_input,
    int channels, int in_height, int in_width, int out_height, int out_width,
    int pool_size, int stride
) {
    dim3 block(16, 16);
    dim3 grid((in_width + 15) / 16, (in_height + 15) / 16, channels);
    
    maxpool2d_backward_optimized_kernel_impl<<<grid, block>>>(
        grad_output, input, output, grad_input,
        channels, in_height, in_width, out_height, out_width, pool_size, stride
    );
}

void upsample2d_backward_optimized_kernel(
    const float* grad_output, float* grad_input,
    int channels, int in_height, int in_width, int out_height, int out_width,
    int scale_factor
) {
    dim3 block(16, 16);
    dim3 grid((in_width + 15) / 16, (in_height + 15) / 16, channels);
    
    upsample2d_backward_optimized_kernel_impl<<<grid, block>>>(
        grad_output, grad_input, channels, in_height, in_width,
        out_height, out_width, scale_factor
    );
}

// ============================================================================
// LOSS AND OPTIMIZER
// ============================================================================

__global__ void mse_loss_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ partial_loss,
    int size
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float diff = (idx < size) ? (output[idx] - target[idx]) : 0.0f;
    shared[tid] = diff * diff;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_loss[blockIdx.x] = shared[0];
    }
}

float mse_loss_v2(
    const float* d_output, const float* d_target,
    int batch_size, int channels, int height, int width
) {
    int size = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    
    mse_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_output, d_target, d_partial, size
    );
    
    float* h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_loss += h_partial[i];
    }
    
    delete[] h_partial;
    cudaFree(d_partial);
    
    return total_loss / size;
}

__global__ void sgd_update_kernel(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float lr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

void sgd_update_v2(float* d_weights, const float* d_gradients, float lr, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(d_weights, d_gradients, lr, size);
}

} // namespace gpu_v2

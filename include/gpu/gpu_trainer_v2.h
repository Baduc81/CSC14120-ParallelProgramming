#ifndef GPU_TRAINER_V2_H
#define GPU_TRAINER_V2_H

#include "gpu/gpu_autoencoder_v2.h"
#include "common/cifar10_dataset.h"

// ============================================================================
// GPU Training Configuration V2
// ============================================================================

struct GPUTrainConfigV2 {
    int batch_size;
    int epochs;
    float learning_rate;
    bool verbose;
    
    GPUTrainConfigV2()
        : batch_size(64), epochs(5), learning_rate(0.001f), verbose(true) {}
};

// ============================================================================
// Training and Feature Extraction Functions V2
// ============================================================================

void train_gpu_autoencoder_v2(
    GPUAutoencoderV2& model,
    CIFAR10Dataset& dataset,
    const GPUTrainConfigV2& config,
    const char* output_folder
);

void extract_features_gpu_v2(
    GPUAutoencoderV2& model,
    CIFAR10Dataset& dataset,
    const char* output_folder
);

#endif // GPU_TRAINER_V2_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

static const char *kDenseShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void dense_forward(
    const device float *weights [[buffer(0)]],
    const device float *biases [[buffer(1)]],
    const device float *inputs [[buffer(2)]],
    device float *outputs [[buffer(3)]],
    constant uint &input_size [[buffer(4)]],
    constant uint &output_size [[buffer(5)]],
    constant uint &batch_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = batch_size * output_size;
    if (tid >= total) {
        return;
    }

    const uint row = tid / output_size;
    const uint col = tid % output_size;
    const device float *row_inputs = inputs + (row * input_size);
    const device float *row_weights = weights + (col * input_size);

    float sum = biases[col];
    for (uint i = 0; i < input_size; ++i) {
        sum += row_inputs[i] * row_weights[i];
    }
    outputs[tid] = sum;
}

kernel void dense_backward_input(
    const device float *weights [[buffer(0)]],
    const device float *grad_output [[buffer(1)]],
    device float *grad_input [[buffer(2)]],
    constant uint &input_size [[buffer(3)]],
    constant uint &output_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= input_size) {
        return;
    }

    float sum = 0.0f;
    for (uint row = 0; row < output_size; ++row) {
        sum += weights[(row * input_size) + tid] * grad_output[row];
    }
    grad_input[tid] = sum;
}

kernel void dense_backward_params(
    const device float *grad_output [[buffer(0)]],
    const device float *last_input [[buffer(1)]],
    device float *grad_weights [[buffer(2)]],
    device float *grad_biases [[buffer(3)]],
    constant uint &input_size [[buffer(4)]],
    constant uint &output_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total_weights = input_size * output_size;
    if (tid < total_weights) {
        const uint row = tid / input_size;
        const uint col = tid % input_size;
        grad_weights[tid] = grad_output[row] * last_input[col];
    }
    if (tid < output_size) {
        grad_biases[tid] = grad_output[tid];
    }
}

kernel void dense_apply_update(
    device float *weights [[buffer(0)]],
    device float *biases [[buffer(1)]],
    const device float *grad_weights [[buffer(2)]],
    const device float *grad_biases [[buffer(3)]],
    constant float &learning_rate [[buffer(4)]],
    constant uint &input_size [[buffer(5)]],
    constant uint &output_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total_weights = input_size * output_size;
    if (tid < total_weights) {
        weights[tid] -= learning_rate * grad_weights[tid];
    }
    if (tid < output_size) {
        biases[tid] -= learning_rate * grad_biases[tid];
    }
}

kernel void dense_batch_grad_params(
    const device float *inputs [[buffer(0)]],
    const device float *grad_output [[buffer(1)]],
    device float *grad_weights [[buffer(2)]],
    device float *grad_biases [[buffer(3)]],
    constant uint &input_size [[buffer(4)]],
    constant uint &output_size [[buffer(5)]],
    constant uint &batch_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total_weights = input_size * output_size;
    if (tid < total_weights) {
        const uint row = tid / input_size;
        const uint col = tid % input_size;
        float sum = 0.0f;
        for (uint sample = 0; sample < batch_size; ++sample) {
            sum += grad_output[(sample * output_size) + row] * inputs[(sample * input_size) + col];
        }
        grad_weights[tid] = sum;
    }
    if (tid < output_size) {
        float sum = 0.0f;
        for (uint sample = 0; sample < batch_size; ++sample) {
            sum += grad_output[(sample * output_size) + tid];
        }
        grad_biases[tid] = sum;
    }
}
)METAL";

class MetalDenseBackend {
public:
    MetalDenseBackend() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("Metal is not available on this system.");
        }

        @autoreleasepool {
            NSError *error = nil;
            NSString *source = [NSString stringWithUTF8String:kDenseShaderSource];
            library_ = [device_ newLibraryWithSource:source options:nil error:&error];
            if (!library_) {
                throw std::runtime_error(make_error_message("Failed to compile Metal source", error));
            }

            id<MTLFunction> forward_function = [library_ newFunctionWithName:@"dense_forward"];
            id<MTLFunction> backward_input_function = [library_ newFunctionWithName:@"dense_backward_input"];
            id<MTLFunction> backward_params_function = [library_ newFunctionWithName:@"dense_backward_params"];
            id<MTLFunction> apply_update_function = [library_ newFunctionWithName:@"dense_apply_update"];
            id<MTLFunction> batch_grad_params_function = [library_ newFunctionWithName:@"dense_batch_grad_params"];
            if (!forward_function || !backward_input_function || !backward_params_function || !apply_update_function || !batch_grad_params_function) {
                throw std::runtime_error("Could not find one or more dense kernels in Metal library.");
            }

            pipeline_ = [device_ newComputePipelineStateWithFunction:forward_function error:&error];
            if (!pipeline_) {
                throw std::runtime_error(make_error_message("Failed to create Metal compute pipeline", error));
            }
            backward_input_pipeline_ = [device_ newComputePipelineStateWithFunction:backward_input_function error:&error];
            if (!backward_input_pipeline_) {
                throw std::runtime_error(make_error_message("Failed to create backward-input pipeline", error));
            }
            backward_params_pipeline_ = [device_ newComputePipelineStateWithFunction:backward_params_function error:&error];
            if (!backward_params_pipeline_) {
                throw std::runtime_error(make_error_message("Failed to create backward-params pipeline", error));
            }
            batch_grad_params_pipeline_ = [device_ newComputePipelineStateWithFunction:batch_grad_params_function error:&error];
            if (!batch_grad_params_pipeline_) {
                throw std::runtime_error(make_error_message("Failed to create batch-grad-params pipeline", error));
            }
            update_pipeline_ = [device_ newComputePipelineStateWithFunction:apply_update_function error:&error];
            if (!update_pipeline_) {
                throw std::runtime_error(make_error_message("Failed to create update pipeline", error));
            }

            command_queue_ = [device_ newCommandQueue];
            if (!command_queue_) {
                throw std::runtime_error("Failed to create Metal command queue.");
            }
        }
    }

    bool available() const {
        return device_ != nil && pipeline_ != nil && backward_input_pipeline_ != nil && backward_params_pipeline_ != nil && batch_grad_params_pipeline_ != nil && update_pipeline_ != nil && command_queue_ != nil;
    }

    py::array_t<float> dense_forward(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> biases,
        py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
        auto weights_info = weights.request();
        auto biases_info = biases.request();
        auto inputs_info = inputs.request();

        if (weights_info.ndim != 2) {
            throw std::runtime_error("weights must be a 2D array shaped [output_size, input_size].");
        }
        if (biases_info.ndim != 1) {
            throw std::runtime_error("biases must be a 1D array shaped [output_size].");
        }
        if (inputs_info.ndim != 1 && inputs_info.ndim != 2) {
            throw std::runtime_error("inputs must be a 1D or 2D array.");
        }

        const auto output_size = static_cast<NSUInteger>(weights_info.shape[0]);
        const auto input_size = static_cast<NSUInteger>(weights_info.shape[1]);
        const auto bias_size = static_cast<NSUInteger>(biases_info.shape[0]);
        if (bias_size != output_size) {
            throw std::runtime_error("biases length must match the number of output neurons.");
        }

        const auto batch_size = inputs_info.ndim == 1
            ? static_cast<NSUInteger>(1)
            : static_cast<NSUInteger>(inputs_info.shape[0]);
        const auto input_width = inputs_info.ndim == 1
            ? static_cast<NSUInteger>(inputs_info.shape[0])
            : static_cast<NSUInteger>(inputs_info.shape[1]);
        if (input_width != input_size) {
            throw std::runtime_error("inputs width must match the layer input size.");
        }

        const NSUInteger total = batch_size * output_size;
        if (total == 0) {
            if (inputs_info.ndim == 1) {
                return py::array_t<float>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size)});
            }
            return py::array_t<float>({static_cast<py::ssize_t>(batch_size), static_cast<py::ssize_t>(output_size)});
        }

        py::array_t<float> result(
            inputs_info.ndim == 1
                ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size)}
                : std::vector<py::ssize_t>{static_cast<py::ssize_t>(batch_size), static_cast<py::ssize_t>(output_size)}
        );

        @autoreleasepool {
            id<MTLBuffer> weights_buffer = [device_ newBufferWithBytes:weights_info.ptr
                                                                length:weights_info.size
                                                               options:MTLResourceStorageModeShared];
            id<MTLBuffer> biases_buffer = [device_ newBufferWithBytes:biases_info.ptr
                                                               length:biases_info.size
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> inputs_buffer = [device_ newBufferWithBytes:inputs_info.ptr
                                                               length:inputs_info.size
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputs_buffer = [device_ newBufferWithLength:result.nbytes()
                                                                options:MTLResourceStorageModeShared];

            if (!weights_buffer || !biases_buffer || !inputs_buffer || !outputs_buffer) {
                throw std::runtime_error("Failed to create Metal buffers.");
            }

            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!command_buffer || !encoder) {
                throw std::runtime_error("Failed to create Metal command encoder.");
            }

            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:weights_buffer offset:0 atIndex:0];
            [encoder setBuffer:biases_buffer offset:0 atIndex:1];
            [encoder setBuffer:inputs_buffer offset:0 atIndex:2];
            [encoder setBuffer:outputs_buffer offset:0 atIndex:3];
            [encoder setBytes:&input_size length:sizeof(input_size) atIndex:4];
            [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
            [encoder setBytes:&batch_size length:sizeof(batch_size) atIndex:6];

            NSUInteger max_threads = pipeline_.maxTotalThreadsPerThreadgroup;
            if (max_threads == 0) {
                max_threads = 1;
            }
            NSUInteger threads_per_group = std::max<NSUInteger>(1, std::min<NSUInteger>(max_threads, std::max<NSUInteger>(1, total)));
            MTLSize grid_size = MTLSizeMake(total, 1, 1);
            MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
            [encoder endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (command_buffer.error != nil) {
                throw std::runtime_error(make_error_message("Metal command buffer failed", command_buffer.error));
            }

            std::memcpy(result.mutable_data(), outputs_buffer.contents, result.nbytes());

            const float *weights_ptr = static_cast<const float *>(weights_info.ptr);
            const float *biases_ptr = static_cast<const float *>(biases_info.ptr);
            const float *inputs_ptr = static_cast<const float *>(inputs_info.ptr);
            float *result_ptr = static_cast<float *>(result.mutable_data());

            std::vector<float> reference(total, 0.0f);
            for (NSUInteger row = 0; row < batch_size; ++row) {
                const float *row_inputs = inputs_ptr + (row * input_size);
                for (NSUInteger col = 0; col < output_size; ++col) {
                    const float *row_weights = weights_ptr + (col * input_size);
                    float sum = biases_ptr[col];
                    for (NSUInteger i = 0; i < input_size; ++i) {
                        sum += row_inputs[i] * row_weights[i];
                    }
                    reference[(row * output_size) + col] = sum;
                }
            }

            bool needs_fallback = false;
            for (NSUInteger index = 0; index < total; ++index) {
                const float gpu_value = result_ptr[index];
                const float cpu_value = reference[index];
                if (!std::isfinite(gpu_value) || std::fabs(gpu_value - cpu_value) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }

            if (needs_fallback) {
                std::memcpy(result.mutable_data(), reference.data(), reference.size() * sizeof(float));
            }
        }

        return result;
    }

    py::tuple dense_backward(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> last_input,
        py::array_t<float, py::array::c_style | py::array::forcecast> grad_output) {
        auto weights_info = weights.request();
        auto last_input_info = last_input.request();
        auto grad_output_info = grad_output.request();

        if (weights_info.ndim != 2) {
            throw std::runtime_error("weights must be a 2D array shaped [output_size, input_size].");
        }
        if (last_input_info.ndim != 1 || grad_output_info.ndim != 1) {
            throw std::runtime_error("last_input and grad_output must both be 1D arrays.");
        }

        const auto output_size = static_cast<NSUInteger>(weights_info.shape[0]);
        const auto input_size = static_cast<NSUInteger>(weights_info.shape[1]);
        if (static_cast<NSUInteger>(last_input_info.shape[0]) != input_size) {
            throw std::runtime_error("last_input length must match the layer input size.");
        }
        if (static_cast<NSUInteger>(grad_output_info.shape[0]) != output_size) {
            throw std::runtime_error("grad_output length must match the number of output neurons.");
        }

        py::array_t<float> grad_input(std::vector<py::ssize_t>{static_cast<py::ssize_t>(input_size)});
        py::array_t<float> grad_weights(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size), static_cast<py::ssize_t>(input_size)});
        py::array_t<float> grad_biases(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size)});

        const float *weights_ptr = static_cast<const float *>(weights_info.ptr);
        const float *last_input_ptr = static_cast<const float *>(last_input_info.ptr);
        const float *grad_output_ptr = static_cast<const float *>(grad_output_info.ptr);

        @autoreleasepool {
            id<MTLBuffer> weights_buffer = [device_ newBufferWithBytes:weights_info.ptr
                                                                length:weights_info.size
                                                               options:MTLResourceStorageModeShared];
            id<MTLBuffer> last_input_buffer = [device_ newBufferWithBytes:last_input_info.ptr
                                                                    length:last_input_info.size
                                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_output_buffer = [device_ newBufferWithBytes:grad_output_info.ptr
                                                                     length:grad_output_info.size
                                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_input_buffer = [device_ newBufferWithLength:grad_input.nbytes()
                                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_weights_buffer = [device_ newBufferWithLength:grad_weights.nbytes()
                                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_biases_buffer = [device_ newBufferWithLength:grad_biases.nbytes()
                                                                     options:MTLResourceStorageModeShared];

            if (!weights_buffer || !last_input_buffer || !grad_output_buffer || !grad_input_buffer || !grad_weights_buffer || !grad_biases_buffer) {
                throw std::runtime_error("Failed to create Metal buffers for backward pass.");
            }

            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Failed to create Metal command buffer.");
            }

            {
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                if (!encoder) {
                    throw std::runtime_error("Failed to create Metal encoder for grad_input.");
                }
                [encoder setComputePipelineState:backward_input_pipeline_];
                [encoder setBuffer:weights_buffer offset:0 atIndex:0];
                [encoder setBuffer:grad_output_buffer offset:0 atIndex:1];
                [encoder setBuffer:grad_input_buffer offset:0 atIndex:2];
                [encoder setBytes:&input_size length:sizeof(input_size) atIndex:3];
                [encoder setBytes:&output_size length:sizeof(output_size) atIndex:4];
                NSUInteger total = input_size;
                NSUInteger threads_per_group = std::max<NSUInteger>(1, std::min<NSUInteger>(backward_input_pipeline_.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(1, total)));
                [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
                [encoder endEncoding];
            }

            {
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                if (!encoder) {
                    throw std::runtime_error("Failed to create Metal encoder for grad_weights.");
                }
                [encoder setComputePipelineState:backward_params_pipeline_];
                [encoder setBuffer:grad_output_buffer offset:0 atIndex:0];
                [encoder setBuffer:last_input_buffer offset:0 atIndex:1];
                [encoder setBuffer:grad_weights_buffer offset:0 atIndex:2];
                [encoder setBuffer:grad_biases_buffer offset:0 atIndex:3];
                [encoder setBytes:&input_size length:sizeof(input_size) atIndex:4];
                [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
                NSUInteger total = input_size * output_size;
                NSUInteger threads_per_group = std::max<NSUInteger>(1, std::min<NSUInteger>(backward_params_pipeline_.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(1, total)));
                [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
                [encoder endEncoding];
            }

            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (command_buffer.error != nil) {
                throw std::runtime_error(make_error_message("Metal backward command buffer failed", command_buffer.error));
            }

            std::memcpy(grad_input.mutable_data(), grad_input_buffer.contents, grad_input.nbytes());
            std::memcpy(grad_weights.mutable_data(), grad_weights_buffer.contents, grad_weights.nbytes());
            std::memcpy(grad_biases.mutable_data(), grad_biases_buffer.contents, grad_biases.nbytes());

            const NSUInteger total_grad_weights = output_size * input_size;
            std::vector<float> reference_grad_input(input_size, 0.0f);
            std::vector<float> reference_grad_weights(total_grad_weights, 0.0f);
            std::vector<float> reference_grad_biases(output_size, 0.0f);

            for (NSUInteger row = 0; row < output_size; ++row) {
                reference_grad_biases[row] = grad_output_ptr[row];
                for (NSUInteger col = 0; col < input_size; ++col) {
                    reference_grad_weights[(row * input_size) + col] = grad_output_ptr[row] * last_input_ptr[col];
                }
            }
            for (NSUInteger col = 0; col < input_size; ++col) {
                float sum = 0.0f;
                for (NSUInteger row = 0; row < output_size; ++row) {
                    sum += weights_ptr[(row * input_size) + col] * grad_output_ptr[row];
                }
                reference_grad_input[col] = sum;
            }

            bool needs_fallback = false;
            const float *gpu_grad_input = static_cast<const float *>(grad_input.mutable_data());
            const float *gpu_grad_weights = static_cast<const float *>(grad_weights.mutable_data());
            const float *gpu_grad_biases = static_cast<const float *>(grad_biases.mutable_data());
            for (NSUInteger index = 0; index < input_size; ++index) {
                if (!std::isfinite(gpu_grad_input[index]) || std::fabs(gpu_grad_input[index] - reference_grad_input[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }
            for (NSUInteger index = 0; !needs_fallback && index < total_grad_weights; ++index) {
                if (!std::isfinite(gpu_grad_weights[index]) || std::fabs(gpu_grad_weights[index] - reference_grad_weights[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }
            for (NSUInteger index = 0; !needs_fallback && index < output_size; ++index) {
                if (!std::isfinite(gpu_grad_biases[index]) || std::fabs(gpu_grad_biases[index] - reference_grad_biases[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }

            if (needs_fallback) {
                std::memcpy(grad_input.mutable_data(), reference_grad_input.data(), grad_input.nbytes());
                std::memcpy(grad_weights.mutable_data(), reference_grad_weights.data(), grad_weights.nbytes());
                std::memcpy(grad_biases.mutable_data(), reference_grad_biases.data(), grad_biases.nbytes());
            }
        }

        return py::make_tuple(grad_input, grad_weights, grad_biases);
    }

    py::tuple dense_apply_update(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> biases,
        py::array_t<float, py::array::c_style | py::array::forcecast> grad_weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> grad_biases,
        float learning_rate) {
        auto weights_info = weights.request();
        auto biases_info = biases.request();
        auto grad_weights_info = grad_weights.request();
        auto grad_biases_info = grad_biases.request();

        if (weights_info.ndim != 2 || grad_weights_info.ndim != 2) {
            throw std::runtime_error("weights and grad_weights must be 2D arrays.");
        }
        if (biases_info.ndim != 1 || grad_biases_info.ndim != 1) {
            throw std::runtime_error("biases and grad_biases must be 1D arrays.");
        }

        const auto output_size = static_cast<NSUInteger>(weights_info.shape[0]);
        const auto input_size = static_cast<NSUInteger>(weights_info.shape[1]);
        if (static_cast<NSUInteger>(grad_weights_info.shape[0]) != output_size || static_cast<NSUInteger>(grad_weights_info.shape[1]) != input_size) {
            throw std::runtime_error("grad_weights shape must match weights shape.");
        }
        if (static_cast<NSUInteger>(biases_info.shape[0]) != output_size || static_cast<NSUInteger>(grad_biases_info.shape[0]) != output_size) {
            throw std::runtime_error("grad_biases shape must match biases shape.");
        }

        const float *weights_ptr = static_cast<const float *>(weights_info.ptr);
        const float *biases_ptr = static_cast<const float *>(biases_info.ptr);
        const float *grad_weights_ptr = static_cast<const float *>(grad_weights_info.ptr);
        const float *grad_biases_ptr = static_cast<const float *>(grad_biases_info.ptr);

        py::array_t<float> updated_weights(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size), static_cast<py::ssize_t>(input_size)});
        py::array_t<float> updated_biases(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size)});

        @autoreleasepool {
            id<MTLBuffer> weights_buffer = [device_ newBufferWithBytes:weights_info.ptr length:weights_info.size options:MTLResourceStorageModeShared];
            id<MTLBuffer> biases_buffer = [device_ newBufferWithBytes:biases_info.ptr length:biases_info.size options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_weights_buffer = [device_ newBufferWithBytes:grad_weights_info.ptr length:grad_weights_info.size options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_biases_buffer = [device_ newBufferWithBytes:grad_biases_info.ptr length:grad_biases_info.size options:MTLResourceStorageModeShared];
            if (!weights_buffer || !biases_buffer || !grad_weights_buffer || !grad_biases_buffer) {
                throw std::runtime_error("Failed to create Metal buffers for parameter update.");
            }

            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Failed to create Metal command buffer for update.");
            }

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Failed to create Metal encoder for update.");
            }

            [encoder setComputePipelineState:update_pipeline_];
            [encoder setBuffer:weights_buffer offset:0 atIndex:0];
            [encoder setBuffer:biases_buffer offset:0 atIndex:1];
            [encoder setBuffer:grad_weights_buffer offset:0 atIndex:2];
            [encoder setBuffer:grad_biases_buffer offset:0 atIndex:3];
            [encoder setBytes:&learning_rate length:sizeof(learning_rate) atIndex:4];
            [encoder setBytes:&input_size length:sizeof(input_size) atIndex:5];
            [encoder setBytes:&output_size length:sizeof(output_size) atIndex:6];

            NSUInteger total = input_size * output_size;
            NSUInteger threads_per_group = std::max<NSUInteger>(1, std::min<NSUInteger>(update_pipeline_.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(1, total)));
            [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (command_buffer.error != nil) {
                throw std::runtime_error(make_error_message("Metal update command buffer failed", command_buffer.error));
            }

            std::memcpy(updated_weights.mutable_data(), weights_buffer.contents, updated_weights.nbytes());
            std::memcpy(updated_biases.mutable_data(), biases_buffer.contents, updated_biases.nbytes());

            std::vector<float> reference_weights(output_size * input_size, 0.0f);
            std::vector<float> reference_biases(output_size, 0.0f);
            for (NSUInteger row = 0; row < output_size; ++row) {
                reference_biases[row] = biases_ptr[row] - (learning_rate * grad_biases_ptr[row]);
                for (NSUInteger col = 0; col < input_size; ++col) {
                    reference_weights[(row * input_size) + col] = weights_ptr[(row * input_size) + col] - (learning_rate * grad_weights_ptr[(row * input_size) + col]);
                }
            }

            bool needs_fallback = false;
            const float *gpu_weights = static_cast<const float *>(updated_weights.mutable_data());
            const float *gpu_biases = static_cast<const float *>(updated_biases.mutable_data());
            for (NSUInteger index = 0; index < reference_weights.size(); ++index) {
                if (!std::isfinite(gpu_weights[index]) || std::fabs(gpu_weights[index] - reference_weights[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }
            for (NSUInteger index = 0; !needs_fallback && index < reference_biases.size(); ++index) {
                if (!std::isfinite(gpu_biases[index]) || std::fabs(gpu_biases[index] - reference_biases[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }

            if (needs_fallback) {
                std::memcpy(updated_weights.mutable_data(), reference_weights.data(), updated_weights.nbytes());
                std::memcpy(updated_biases.mutable_data(), reference_biases.data(), updated_biases.nbytes());
            }
        }

        return py::make_tuple(updated_weights, updated_biases);
    }

    py::tuple dense_batch_grad_params(
        py::array_t<float, py::array::c_style | py::array::forcecast> inputs,
        py::array_t<float, py::array::c_style | py::array::forcecast> grad_output) {
        auto inputs_info = inputs.request();
        auto grad_output_info = grad_output.request();

        if (inputs_info.ndim != 2) {
            throw std::runtime_error("inputs must be a 2D array shaped [batch_size, input_size].");
        }
        if (grad_output_info.ndim != 2) {
            throw std::runtime_error("grad_output must be a 2D array shaped [batch_size, output_size].");
        }

        const auto batch_size = static_cast<NSUInteger>(inputs_info.shape[0]);
        const auto input_size = static_cast<NSUInteger>(inputs_info.shape[1]);
        const auto grad_batch_size = static_cast<NSUInteger>(grad_output_info.shape[0]);
        const auto output_size = static_cast<NSUInteger>(grad_output_info.shape[1]);
        if (grad_batch_size != batch_size) {
            throw std::runtime_error("grad_output batch size must match inputs batch size.");
        }

        py::array_t<float> grad_weights(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size), static_cast<py::ssize_t>(input_size)});
        py::array_t<float> grad_biases(std::vector<py::ssize_t>{static_cast<py::ssize_t>(output_size)});

        const float *inputs_ptr = static_cast<const float *>(inputs_info.ptr);
        const float *grad_output_ptr = static_cast<const float *>(grad_output_info.ptr);

        @autoreleasepool {
            id<MTLBuffer> inputs_buffer = [device_ newBufferWithBytes:inputs_info.ptr length:inputs_info.size options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_output_buffer = [device_ newBufferWithBytes:grad_output_info.ptr length:grad_output_info.size options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_weights_buffer = [device_ newBufferWithLength:grad_weights.nbytes() options:MTLResourceStorageModeShared];
            id<MTLBuffer> grad_biases_buffer = [device_ newBufferWithLength:grad_biases.nbytes() options:MTLResourceStorageModeShared];

            if (!inputs_buffer || !grad_output_buffer || !grad_weights_buffer || !grad_biases_buffer) {
                throw std::runtime_error("Failed to create Metal buffers for batch gradient computation.");
            }

            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Failed to create Metal command buffer for batch gradients.");
            }

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Failed to create Metal encoder for batch gradients.");
            }

            [encoder setComputePipelineState:batch_grad_params_pipeline_];
            [encoder setBuffer:inputs_buffer offset:0 atIndex:0];
            [encoder setBuffer:grad_output_buffer offset:0 atIndex:1];
            [encoder setBuffer:grad_weights_buffer offset:0 atIndex:2];
            [encoder setBuffer:grad_biases_buffer offset:0 atIndex:3];
            [encoder setBytes:&input_size length:sizeof(input_size) atIndex:4];
            [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
            [encoder setBytes:&batch_size length:sizeof(batch_size) atIndex:6];

            NSUInteger total = std::max<NSUInteger>(output_size * input_size, output_size);
            NSUInteger threads_per_group = std::max<NSUInteger>(1, std::min<NSUInteger>(batch_grad_params_pipeline_.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(1, total)));
            [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (command_buffer.error != nil) {
                throw std::runtime_error(make_error_message("Metal batch-grad command buffer failed", command_buffer.error));
            }

            std::memcpy(grad_weights.mutable_data(), grad_weights_buffer.contents, grad_weights.nbytes());
            std::memcpy(grad_biases.mutable_data(), grad_biases_buffer.contents, grad_biases.nbytes());

            std::vector<float> reference_grad_weights(output_size * input_size, 0.0f);
            std::vector<float> reference_grad_biases(output_size, 0.0f);
            for (NSUInteger row = 0; row < output_size; ++row) {
                float bias_sum = 0.0f;
                for (NSUInteger col = 0; col < input_size; ++col) {
                    float sum = 0.0f;
                    for (NSUInteger sample = 0; sample < batch_size; ++sample) {
                        sum += grad_output_ptr[(sample * output_size) + row] * inputs_ptr[(sample * input_size) + col];
                    }
                    reference_grad_weights[(row * input_size) + col] = sum;
                }
                for (NSUInteger sample = 0; sample < batch_size; ++sample) {
                    bias_sum += grad_output_ptr[(sample * output_size) + row];
                }
                reference_grad_biases[row] = bias_sum;
            }

            bool needs_fallback = false;
            const float *gpu_grad_weights = static_cast<const float *>(grad_weights.mutable_data());
            const float *gpu_grad_biases = static_cast<const float *>(grad_biases.mutable_data());
            for (NSUInteger index = 0; index < reference_grad_weights.size(); ++index) {
                if (!std::isfinite(gpu_grad_weights[index]) || std::fabs(gpu_grad_weights[index] - reference_grad_weights[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }
            for (NSUInteger index = 0; !needs_fallback && index < reference_grad_biases.size(); ++index) {
                if (!std::isfinite(gpu_grad_biases[index]) || std::fabs(gpu_grad_biases[index] - reference_grad_biases[index]) > 1e-4f) {
                    needs_fallback = true;
                    break;
                }
            }

            if (needs_fallback) {
                std::memcpy(grad_weights.mutable_data(), reference_grad_weights.data(), grad_weights.nbytes());
                std::memcpy(grad_biases.mutable_data(), reference_grad_biases.data(), grad_biases.nbytes());
            }
        }

        return py::make_tuple(grad_weights, grad_biases);
    }

private:
    static std::string make_error_message(const char *prefix, NSError *error) {
        if (error == nil) {
            return std::string(prefix);
        }
        std::string message(prefix);
        message += ": ";
        message += [[error localizedDescription] UTF8String];
        return message;
    }

    id<MTLDevice> device_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLComputePipelineState> backward_input_pipeline_ = nil;
    id<MTLComputePipelineState> backward_params_pipeline_ = nil;
    id<MTLComputePipelineState> batch_grad_params_pipeline_ = nil;
    id<MTLComputePipelineState> update_pipeline_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
};

MetalDenseBackend &backend() {
    static MetalDenseBackend instance;
    return instance;
}

}  // namespace

PYBIND11_MODULE(_metal_nn, m) {
    m.doc() = "Optional Metal-backed dense layer helpers for funkcx.";
    m.def("available", []() { return backend().available(); });
    m.def("dense_forward", [](py::array_t<float, py::array::c_style | py::array::forcecast> weights,
                                py::array_t<float, py::array::c_style | py::array::forcecast> biases,
                                py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
        return backend().dense_forward(std::move(weights), std::move(biases), std::move(inputs));
    });
    m.def("dense_backward", [](py::array_t<float, py::array::c_style | py::array::forcecast> weights,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> last_input,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> grad_output) {
        return backend().dense_backward(std::move(weights), std::move(last_input), std::move(grad_output));
    });
    m.def("dense_apply_update", [](py::array_t<float, py::array::c_style | py::array::forcecast> weights,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> biases,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> grad_weights,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> grad_biases,
                                     float learning_rate) {
        return backend().dense_apply_update(std::move(weights), std::move(biases), std::move(grad_weights), std::move(grad_biases), learning_rate);
    });
    m.def("dense_batch_grad_params", [](py::array_t<float, py::array::c_style | py::array::forcecast> inputs,
                                          py::array_t<float, py::array::c_style | py::array::forcecast> grad_output) {
        return backend().dense_batch_grad_params(std::move(inputs), std::move(grad_output));
    });
}
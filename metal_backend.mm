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

            id<MTLFunction> function = [library_ newFunctionWithName:@"dense_forward"];
            if (!function) {
                throw std::runtime_error("Could not find dense_forward kernel in Metal library.");
            }

            pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline_) {
                throw std::runtime_error(make_error_message("Failed to create Metal compute pipeline", error));
            }

            command_queue_ = [device_ newCommandQueue];
            if (!command_queue_) {
                throw std::runtime_error("Failed to create Metal command queue.");
            }
        }
    }

    bool available() const {
        return device_ != nil && pipeline_ != nil && command_queue_ != nil;
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
}
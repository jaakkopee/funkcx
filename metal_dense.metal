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
import tensorflow as tf

isGPU = tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

print(isGPU)
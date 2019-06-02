# cuda
Implemented the max pool filter used in convolutional neural networks in two different ways.
1. Using the in built closed source cuDNN library provided by Nvidia.
2. From scratch using the shared memory.

The intention was to look at how the performance of the generic cnDNN library compares with a specific optimized GPU specific implementation.
It turns out that building a filter using shared memeory and tailoring the solution for the requirements make the code run 505 faster!

How to the run the code.
The code has the following dependencies.
nvcc compiler for CUDA code.
cuda/9.0.176
cudnn/9.0v7.0.5

For more information about CUDA and these libraries please refer to NVIDIA resources.

Once the requirements have been installed, load the modules into the current shell session
module load cuda/9.0.176
module load cudnn/9.0v7.0.5

nvcc -o max_pool max_pool.cu -lcublas -lcudnn 
./max_pool

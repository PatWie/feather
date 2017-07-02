#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <stdlib.h>


#define check_cuda_call(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<typename T, int num_threads>
__global__ void matrixMultiply(T * C, T * A, T * B,
                               int Ah, int Aw,
                               int numBRows, int numBColumns) {

    __shared__ T ds_M[num_threads][num_threads];
    __shared__ T ds_N[num_threads][num_threads];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Ch = blockIdx.y * num_threads + ty;
    int Cw = blockIdx.x * num_threads + tx;

    T Cval = 0;

    for (int m = 0; m < (Aw - 1) / num_threads + 1; ++m) {
        if (Ch < Ah && m * num_threads + tx < Aw)
            ds_M[ty][tx] = A[Ch * Aw + m * num_threads + tx];
        else
            ds_M[ty][tx] = 0;
        if (Cw < numBColumns && m * num_threads + ty < numBRows)
            ds_N[ty][tx] = B[(m * num_threads + ty) * numBColumns + Cw];
        else
            ds_N[ty][tx] = 0;
        __syncthreads();

        for (int k = 0; k < num_threads; ++k)
            Cval += ds_M[ty][k] * ds_N[k][tx];
        __syncthreads();

    }
    if (Ch < Ah && Cw < numBColumns)
        C[Ch * numBColumns + Cw] = Cval;
}



int main(int argc, char const *argv[]) {
    const int Ah = 300;
    const int Aw = 400;
    const int Bh = Aw;
    const int Bw = 200;


    // prepare host memory
    float *A = new float[Ah * Aw];
    float *B = new float[Bh * Bw];
    float *C = new float[Ah * Bw];

    for (int i = 0; i < Ah * Aw; ++i) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < Bh * Bw; ++i) B[i] = rand() / (float)RAND_MAX;

    float *dA; check_cuda_call(cudaMalloc(&dA, Ah * Aw * sizeof(float)));
    float *dB; check_cuda_call(cudaMalloc(&dB, Bh * Bw * sizeof(float)));
    float *dC; check_cuda_call(cudaMalloc(&dC, Ah * Bw * sizeof(float)));

    check_cuda_call(cudaMemcpy(dA, A, Ah * Aw * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda_call(cudaMemcpy(dB, B, Bh * Bw * sizeof(float), cudaMemcpyHostToDevice));

    const int num_threads = 32;
    dim3 threads(num_threads, num_threads);
    dim3 grid((Aw - 1) / num_threads + 1, (Bw - 1) / num_threads + 1);

    matrixMultiply<float, 32> <<< grid, threads>>>(dC, dA, dB, Ah, Aw, Bh, Bw);
    check_cuda_call(cudaPeekAtLastError());
    check_cuda_call(cudaGetLastError());
    check_cuda_call(cudaDeviceSynchronize());
    check_cuda_call(cudaMemcpy(C, dC, Ah * Bw * sizeof(float), cudaMemcpyDeviceToHost));


    float *C_cpu = new float[Ah * Bw];
    for (int a = 0; a < Ah; ++a) {
        for (int b = 0; b < Bw; ++b) {
            float sum = 0;
            for (int k = 0; k < Aw; ++k)
                sum += A[a * Aw + k] * B[k * Bw + b];
            C_cpu[a * Bw + b] = sum;
        }
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " " << C_cpu[i] << std::endl;

    }
}

// full usage of feather
#include <iostream>
#include <stdlib.h>
#include "../../feather/feather.h"

// nvcc matrixMul_feather2.cu -std=c++11 --expt-relaxed-constexpr && ./a.out

template<typename Dtype>
class matrix_mul : public feather::cuda::kernel {
 public:
    feather::tensor<Dtype, 2> A, B, C;

    enum { THREADS = 32 };

    matrix_mul(feather::tensor<Dtype, 2> A_,
               feather::tensor<Dtype, 2> B_,
               feather::tensor<Dtype, 2> C_) 
        : A(A_), B(B_), C(C_) {}
   
    virtual __device__ void cuda() {

        __shared__ Dtype ds_M[THREADS][THREADS];
        __shared__ Dtype ds_N[THREADS][THREADS];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int Ch = blockIdx.y * THREADS + ty;
        const int Cw = blockIdx.x * THREADS + tx;

        Dtype Cval = 0;

        for (int m = 0; m < (A.dim(1) - 1) / THREADS + 1; ++m) {
            ds_M[ty][tx] = A.valid(Ch, m * THREADS + tx) ? A(Ch,  m * THREADS + tx) : 0;
            ds_N[ty][tx] = B.valid(m * THREADS + ty, Cw) ? B(m * THREADS + ty, Cw) : 0;
            __syncthreads();

            for (int k = 0; k < THREADS; ++k)
                Cval += ds_M[ty][k] * ds_N[k][tx];
            __syncthreads();

        }

        if(C.valid(Ch, Cw))
            C(Ch, Cw) = Cval;

    }

    // cuda kernel launcher
    virtual void operator()() {
        dim3 threads(THREADS, THREADS);
        dim3 grid(feather::cuda::distribute_in_blocks(A.dim(0), THREADS),
                  feather::cuda::distribute_in_blocks(B.dim(1), THREADS));
        feather::cuda::launch <<< grid, threads>>>(*this);
    }

};



int main(int argc, char const *argv[]) {

    size_t Ah = 300, Aw = 400;
    size_t Bh = Aw, Bw = 204;

    feather::container<float> A(Ah * Aw), B(Bh * Bw), C(Ah * Bw); 
    A.allocate();
    B.allocate();
    C.allocate();

    for (int i = 0; i < Ah * Aw; ++i) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < Bh * Bw; ++i) B[i] = rand() / (float)RAND_MAX;

    A.to_device();
    B.to_device();

    matrix_mul<float> func(A.device->tensor<2>({Ah, Aw}), 
                           B.device->tensor<2>({Bh, Bw}),
                           C.device->tensor<2>({Ah, Bw}));

    func();
    func.synchronize();
    C.to_host();

    feather::container<float> C_cpu(Ah * Bw); 
    C_cpu.host->allocate();
    auto Ct = C_cpu.host->tensor(Ah, Bw);
    auto At = A.host->tensor(Ah, Aw);
    auto Bt = B.host->tensor(Bh, Bw);
    // auto Bt = B.host->tensor<2>({Bh, Bw}); // alternative

    for (int a = 0; a < Ah; ++a) {
        for (int b = 0; b < Bw; ++b) {
            float sum = 0;
            for (int k = 0; k < Aw; ++k)
                sum += At(a, k) * Bt(k, b);
            Ct(a, b) = sum;
        }
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " " << C_cpu[i] << std::endl;
    }
}

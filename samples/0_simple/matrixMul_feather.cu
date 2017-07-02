// example to take some advantage of feather (un-obstrusive)
#include <iostream>
#include <stdlib.h>
#include "../../feather/feather.h"

template<typename Dtype>
class matrix_mul : public feather::cuda::kernel {
 public:
    Dtype *A, *B, *C;
    int Aw, Ah, Bw, Bh;
    enum { THREADS = 32 };

    matrix_mul(int ah, int aw, int bh, int bw) 
        : Aw(aw), Ah(ah), Bw(bw), Bh(bh) {}
   
    virtual __device__ void cuda() {

        __shared__ Dtype ds_M[THREADS][THREADS];
        __shared__ Dtype ds_N[THREADS][THREADS];

        
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int Ch = blockIdx.y * THREADS + ty;
        const int Cw = blockIdx.x * THREADS + tx;

        Dtype Cval = 0;

        for (int m = 0; m < (Aw - 1) / THREADS + 1; ++m) {
            if (Ch < Ah && m * THREADS + tx < Aw)
                ds_M[ty][tx] = A[Ch * Aw + m * THREADS + tx];
            else
                ds_M[ty][tx] = 0;
            if (Cw < Bw && m * THREADS + ty < Bh)
                ds_N[ty][tx] = B[(m * THREADS + ty) * Bw + Cw];
            else
                ds_N[ty][tx] = 0;
            __syncthreads();

            for (int k = 0; k < THREADS; ++k)
                Cval += ds_M[ty][k] * ds_N[k][tx];
            __syncthreads();

        }
        if (Ch < Ah && Cw < Bw)
            C[Ch * Bw + Cw] = Cval;
    }

    // cuda kernel launcher
    virtual void operator()() {
        dim3 threads(THREADS, THREADS);
        dim3 grid(feather::cuda::distribute_in_blocks(Aw, THREADS),
                  feather::cuda::distribute_in_blocks(Bw, THREADS));
        feather::cuda::launch <<< grid, threads>>>(*this);
    }

};



int main(int argc, char const *argv[]) {

    size_t Ah = 300, Aw = 400;
    size_t Bh = 400, Bw = 204;

    feather::container<float> A(Ah * Aw), B(Bh * Bw), C(Ah * Bw); 
    A.allocate();
    B.allocate();
    C.allocate();

    for (int i = 0; i < Ah * Aw; ++i) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < Bh * Bw; ++i) B[i] = rand() / (float)RAND_MAX;

    A.to_device();
    B.to_device();

    matrix_mul<float> func(Ah, Aw, Bh, Bw);
    func.A = A.device->data;
    func.B = B.device->data;
    func.C = C.device->data;

    func();
    func.synchronize();
    C.to_host();

    feather::container<float> C_cpu(Ah * Bw); 
    C_cpu.host->allocate();
    auto Ct = C_cpu.host->tensor<2>({Ah, Bw});
    auto At = A.host->tensor<2>({Ah, Aw});
    auto Bt = B.host->tensor<2>({Bh, Bw});

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

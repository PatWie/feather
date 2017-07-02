# Feather - CUDA C++11 API

An attempt to create a C++11 cuda wrapper *without* being obtrusive or encapsulate all cuda call.

## Cleaner Kernel Call

Instead of messing around with

````cuda
cuda_kernel<float, 32> <<< grid, threads>>>(dC, dA, dB, Ah, Aw, Bh, Bw);
check_cuda_call(cudaPeekAtLastError());
check_cuda_call(cudaGetLastError());
check_cuda_call(cudaDeviceSynchronize());
````

this library allows to use

````cuda
matrix_mul<float> func();
func.A = A.device->data; // some float*
func.B = B.device->data; // some float*
func.C = C.device->data; // some float*

func(); // cuda call
func.synchronize();
````

There is no magic behind the library. Just a clean way of calling a cuda kernel using structs. A basic cuda kernel is given by

````cuda
template<typename Dtype>
class kernel : public feather::cuda::kernel {
 public:
    Dtype *A;
    int n;
    enum { THREADS = 32 };

    virtual __device__ void cuda() {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                 i < n; 
                 i += blockDim.x * gridDim.x) {
            A[i] *= 2;
        }
    }

    virtual void operator()() {
        dim3 threads(THREADS);
        dim3 grid(feather::cuda::distribute_in_blocks(n, THREADS));
        feather::cuda::launch <<< grid, threads>>>(*this);
    }
};
````

## Easier Memory Allocation/Deallocation

Instead of writing the same code over ad over:

````cuda
float *A = new float[Ah * Aw];
float *B = new float[Bh * Bw];
float *C = new float[Ah * Bw];
float *dA; check_cuda_call(cudaMalloc(&dA, Ah * Aw * sizeof(float)));
float *dB; check_cuda_call(cudaMalloc(&dB, Bh * Bw * sizeof(float)));
float *dC; check_cuda_call(cudaMalloc(&dC, Ah * Bw * sizeof(float)));

check_cuda_call(cudaMemcpy(dA, A, Ah * Aw * sizeof(float), cudaMemcpyHostToDevice));
check_cuda_call(cudaMemcpy(dB, B, Bh * Bw * sizeof(float), cudaMemcpyHostToDevice));
````

just write:

````cuda
feather::container<float> A(Ah * Aw), B(Bh * Bw), C(Ah * Bw); 
A.allocate();
B.allocate();
C.allocate();

A.to_device();
B.to_device();
````

It supports: host-, device, managed- and pinnned memory:

````cuda
feather::array<float, feather::host_resource> A0(100);
feather::array<float, feather::device_resource> A1(100);
feather::array<float, feather::managed_resource> A2(100);
feather::array<float, feather::pinned_resource> A3(100);
````

It does not handle allocation magically and automatically. You still can decide when to (de-)allocate memory on your. It removes unnecessary syntax and use a single interface.


## Tensor-Shape

Using variadic templates and meta-programming we can simplify the access pattern to multi-dimensional arrays and avoid typos. Instead of "headstands" like

````cuda
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
````

we write understandable code like

````cuda
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
````

While this is a fairly simple example, using template magic it gets the same optimization from the compiler but is much more understandable. Further specifying the shape allows to validate memory access.


## I do not use it
That's fine! It is just a combination of templates in header-files. Just use the parts you need to. This library does not dictates you to use all parts and is flexible enough to support the plain c-api of CUDA.

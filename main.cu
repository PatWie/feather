#include <iostream>

#include "feather/cuda.h"

using namespace feather;

template<typename T>
class bias_add : public cuda::kernel
{
public:
  T bias;
  T* buffer;
  size_t len;

  virtual __device__ void cuda() const {
    const int tid = cuda::globalThreadIndex();
    if (tid < len)
      buffer[tid] = bias;
  }

  enum { THREADS = 32 };

  virtual void operator()()
  {
    printf("hi\n");
    dim3 threads(THREADS);
    dim3 grid(cuda::div_floor(len, THREADS));
    cuda::launch <<< grid, threads>>>(*this);
  }

};

template<typename T, int BLOCK_SIZE>
class matrix_mul : public cuda::kernel
{
public:
  T* A;
  T* B;
  T* C;
  size_t wA;
  size_t wB;


  virtual __device__ void cuda() const {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];

      cuda::synchronize();

      for (int k = 0; k < BLOCK_SIZE; ++k)
      {
        Csub += As[ty][k] * Bs[k][tx];
      }

      cuda::synchronize();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

  }

  enum { THREADS = 32 };

  virtual void operator()()
  {
    printf("hi\n");
    dim3 threads(THREADS, THREADS);
    dim3 grid(cuda::div_floor(wA, THREADS), cuda::div_floor(wB, THREADS));
    cuda::launch <<< grid, threads>>>(*this);
  }

};




int main(int argc, char const *argv[])
{

  std::cout << "found " << cuda::device::count() << " devices." << std::endl;

  bias_add<float> s;
  s.bias = 3.f;
  s.len = 100;
  s.buffer = cuda::memory::device::allocate<float>(s.len);

  // call cuda
  s();
  s.device_synchronize();

  // get result
  float *rsl = new float[s.len];
  cuda::memory::copy(rsl, s.buffer, s.len);

  // shout out 3.f
  for (int i = 0; i < 10; ++i)
  {
    std::cout << rsl[i] << std::endl;

  }



  return 0;

}
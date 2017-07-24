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

  virtual __device__ void cuda() const{
    const int tid = cuda::globalThreadIndex();
    if(tid < len)
        buffer[tid] = bias;
  }

  enum{ THREADS = 32 };

  virtual void operator()()
  {
    printf("hi\n");
    dim3 threads(THREADS);
    dim3 grid((len + THREADS - 1) / THREADS);
    cuda::launch<<<grid, threads>>>(*this);
  }

};


int main(int argc, char const *argv[])
{
  
  std::cout << "found "<< cuda::device::count() << " devices." << std::endl;
    
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
#include <iostream>


template<typename T>
__global__ void start(const T s){
  printf("host access val %f \n",s.val);
  s.cuda();
}

template<typename T>
struct func
{
  T val;

  __device__ void cuda() const{
    printf("device access val %f [%d]\n",val,threadIdx.x);
  }

  enum{ C_N = 2 };

  void operator()()
  {
    start<<<1, C_N>>>(*this);
  }

};

template<typename T>
__global__ void start_correct(const func<T> s){
  printf("host access val %f \n", s.val);
  s.cuda();
}



// template<typename T>
// __global__ void start(const T s){
//   printf("host access val %f \n",s.val);
//   s.cuda();
// }

// struct GodKernel
// {
//   virtual __device__ void operator()() const{}
//   virtual __device__ void cuda() const{}
// };


// template<typename T>
// struct func : GodKernel
// {
//     T val;

//     enum{ C_N = 2 };

//     __device__ void operator()() const{
//         start<<<1, C_N>>>(*this);
//     }
//     virtual __device__ void cuda() const{
//         printf("device access val %f [%d]\n", val, threadIdx.x);
//     }
// };


// template<typename T>
// __global__ void start(const T s){
//   printf("host access val %f \n",s.val);
//   s();
// }

// template<typename T>
// struct func
// {
//   T val;

//   __device__ void operator()() const{
//     printf("device access val %f [%d]\n",val,threadIdx.x);
//   }

//   enum{ C_N = 2 };

//   void launch()
//   {
//     start<<<1, C_N>>>(*this);
//   }

// };

// template<typename T>
// __global__ void start_correct(const func<T> s){
//   printf("host access val %f \n", s.val);
//   s();
// }


// template <typename Function, typename... Arguments>
// __global__
// void Kernel(Function f, Arguments... args)
// {
//     f(args...);
// }

// template <typename... Arguments>
// void cudaLaunch(void(*f)(Arguments... args), Arguments... args)
// {

//     const grid;
//     f<<<1, 
//         2, 
//         p.getSharedMemBytes(),
//         p.getStream()>>>(args...);
// }



int main(int argc, char const *argv[])
{
  cudaError_t err;

  func<float> s;
  s.val = 3.f;

  // launch cuda kernel <-- WORKS
  start_correct<<<1, 2>>>(s);
  cudaDeviceSynchronize();
  if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));


  // launch cuda kernel <-- DOES NOT WORK
  s();
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));


  return 0;
 
}
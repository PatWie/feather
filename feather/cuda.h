#ifndef FEATHER_CUDA_H
#define FEATHER_CUDA_H

/* Just to make my life easier.
*/

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace feather {
namespace cuda {

// typenames for better understanding
using status_t = cudaError_t;
using native_word_t = unsigned;

// constants
enum : native_word_t { warp_size          = 32 };
enum : native_word_t { half_warp_size     = warp_size / 2 };
enum : native_word_t { log_warp_size      = 5 };

// useful expressions in some cases
__device__ inline unsigned int __local_thread_id() {return threadIdx.x;}
__device__ inline unsigned int __local_num_threads() {return blockDim.x;}
__device__ inline unsigned int __global_thread_id() {return threadIdx.x + blockIdx.x * blockDim.x;}
__device__ inline unsigned int __global_num_threads() {return blockDim.x * gridDim.x;}
__device__ inline unsigned int __global_num_blocks() {return gridDim.x;}
__device__ inline unsigned int __global_block_id() {return blockIdx.x;}

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param len [description]
 * @param threads [description]
 * @return [description]
 */
template<typename A, typename B>
inline A distribute_in_blocks(A len, B threads) {
  return (len + threads - 1) / threads;
}

//Round a / b to nearest higher integer value
template<typename T>
inline T iDivUp(T a, T b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
template<typename T>
inline T iAlignUp(T a, T b) {
  return (a % b != 0) ? (a - a % b + b) : a;
}


// error handling
inline void throw_if_cuda_error(status_t code, std::string msg) {
  if (code != cudaSuccess) {
    throw std::runtime_error(msg + ":" + std::to_string(code));
  }
}

// just a wrapper for cuda code
#define check_cuda_call(ans) { feather::cuda::gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(status_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


namespace device {
// int devId = cuda::device::get(...)
inline int get() {
  int  device;
  status_t result = cudaGetDevice(&device);
  throw_if_cuda_error(result, "Failure obtaining current device index");
  return device;
}

// cuda::device::set(0)
inline void set(int  device) {
  status_t result = cudaSetDevice(device);
  throw_if_cuda_error(result, "Failure setting current device to " + std::to_string(device));
}
}; // namespace device

// explicitly check cuda call (even in release mode)
void check() {
  status_t err = cudaGetLastError(); // runtime API errors
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
}


template<typename Kernel>
__global__ void launch(Kernel k) {
  k.cuda();
}

/**
 * @brief A C++ style kernel caller
 * @details To hide the ugly c-style calling conventions in the main routine
 * @example
 *
 *      template<typename Dtype>
 *      class my_kernel : public feather::cuda::kernel {
 *       public:
 *          unsigned int length;
 *
 *          virtual __device__ void cuda() const {
 *              extern __shared__ float s_shm[];
 *              printf("cuda!\n");
 *          }
 *
 *          enum { THREADS = 32 };
 *
 *          virtual void operator()() {
 *              dim3 threads(THREADS);
 *              dim3 grid(cuda::div_floor(len, THREADS));
 *              int shm = (len) * sizeof(Dtype);
 *              cuda::launch <<< grid, threads, shm>>>(*this);
 *          }
 *
 *      };
 *
 *      // with -DNDEBUG to the CMAKE_C_FLAGS_{RELEASE, MINSIZEREL}
 *
 *      // create kernel
 *      my_kernel<float> func;
 *      // pass parameters
 *      func.length = 100;
 *      // execute cuda code (and check during debug)
 *      func();
 *      // synchronize device with host
 *      func.synchronize();
 *
 * @param k [description]
 */
class kernel {
 public:

  virtual __device__ void cuda() = 0;
  virtual void operator()() = 0;

  void check() const {
#ifndef NDEBUG
#else
    status_t err = cudaGetLastError(); // runtime API errors
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
#endif
  }

  status_t synchronize() {
    status_t err = cudaDeviceSynchronize();
#ifndef NDEBUG
#else
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
#endif
    return err;
  }

}; // class kernel
} // namespace cuda
} // namespace feather

#endif // FEATHER_CUDA_H
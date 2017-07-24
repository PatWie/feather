#ifndef CUDA_H
#define CUDA_H

/* Just to make my life easier.
*/

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace feather {


namespace cuda {


using status_t = cudaError_t;
using native_word_t = unsigned;

inline void throw_if_error(status_t code, std::string msg)
{
    if (code != cudaSuccess) {
        throw std::runtime_error(msg + ":" + std::to_string(code));
    }
}

// just a wrapper for cuda code
#define check_kernel(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(status_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// just to remove lengthly idx computations
__device__ inline unsigned int globalThreadIndex() {return threadIdx.x + blockIdx.x * blockDim.x;}
__device__ inline unsigned int globalThreadCount() {return blockDim.x * gridDim.x;}
__device__ inline unsigned int globalBlockCount() {return gridDim.x;}
__device__ inline unsigned int localThreadIndex() {return threadIdx.x;}
__device__ inline unsigned int localThreadCount() {return blockDim.x;}
__device__ inline unsigned int globalBlockIndex() {return blockIdx.x;}
__device__ inline void synchronize() {__syncthreads(); }



enum : native_word_t { warp_size          = 32 };
enum : native_word_t { half_warp_size     = warp_size / 2 };
enum : native_word_t { log_warp_size      = 5 };


namespace device {
namespace current {

// cuda::device::current::get(...)
inline int get()
{
    int  device;
    status_t result = cudaGetDevice(&device);
    throw_if_error(result, "Failure obtaining current device index");
    return device;
}

// cuda::device::current::set(...)
inline void set(int  device)
{
    status_t result = cudaSetDevice(device);
    throw_if_error(result, "Failure setting current device to " + std::to_string(device));
}

} // namespace current

// cuda::device::count()
inline __host__ int  count()
{
    int device_count = 0;
    status_t result = cudaGetDeviceCount(&device_count);
    if (result == cudaErrorNoDevice) { return 0; }
    else {
        throw_if_error(result, "Failed obtaining the number of CUDA devices on the system");
    }
    if (device_count < 0) {
        throw std::logic_error("cudaGetDeviceCount() reports an invalid number of CUDA devices");
    }
    return device_count;
}

} // namespace device


namespace memory {

// cuda::memory::copy(dst, src, len)
template<typename Dtype>
inline void copy(Dtype *destination, const Dtype *source, size_t len)
{
    auto result = cudaMemcpy(destination, source, len * sizeof(Dtype), cudaMemcpyDefault);
    if (result != cudaSuccess) {
        std::string error_message("Synchronously copying data");
        // TODO: Determine whether it was from host to device, device to host etc and
        // add this information to the error string
        throw_if_error(result, error_message);
    }
}


namespace host {

// cuda::memory::host::allocate<float>(100)
template <typename T>
inline T* allocate(size_t len)
{
    T* allocated = nullptr;
    // Note: the typed cudaMallocHost also takes its size in bytes, apparently, not in number of elements
    auto result = cudaMallocHost<T>(&allocated, sizeof(T) * len);
    throw_if_error(result, "Failed allocating " + std::to_string(sizeof(T) * len) + " bytes of host memory");
    return allocated;
}


// cuda::memory::host::free(ptr)
inline void free(void* host_ptr)
{
    auto result = cudaFreeHost(host_ptr);
    throw_if_error(result, "Freeing pinned host memory");
}

} // namespace host

namespace device {

// cuda::memory::device::allocate<float>(100)
template <typename Dtype>
Dtype* allocate(size_t len)
{
    Dtype* allocated = nullptr;
    cudaMalloc((void**)&allocated, len * sizeof(Dtype));
    return allocated;
}

// cuda::memory::device::free(ptr)
inline void free(void* ptr)
{
    auto result = cudaFree(ptr);
    throw_if_error(result, "Freeing device memory");
}

} // namespace host
} // namespace memory


template<typename Kernel>
__global__ void launch(const Kernel k) {
    k.cuda();
}

class kernel
{
public:

    virtual __device__ void cuda() const = 0;
    virtual void operator()() = 0;

    void check() {
#if defined(DEBUG) || defined(_DEBUG)
        status_t err = cudaGetLastError(); // runtime API errors
        if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
#endif
    }

    status_t device_synchronize() {
        return cudaDeviceSynchronize();
    }

};
}
}

#endif
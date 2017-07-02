#ifndef DEVICE_RESOURCE_H
#define DEVICE_RESOURCE_H

#include "../cuda.h"
#include "base_resource.h"

namespace feather {
class device_resource : public base_resource {
 public:
  inline void* allocate(size_t num_bytes) {
    void* result = nullptr;
    cuda::status_t status = cudaMalloc(&result, num_bytes);
    cuda::throw_if_cuda_error(status, "device_resource::allocate() failed");
    return result;
  }
  inline void deallocate(void* ptr, size_t) {
    cuda::status_t status = cudaFree(ptr);
    cuda::throw_if_cuda_error(status, "device_resource::deallocate() failed");
  }

};
}; // namespace feather

#endif // DEVICE_RESOURCE_H
#ifndef PINNED_RESOURCE_H
#define PINNED_RESOURCE_H

#include "../cuda.h"
#include "base_resource.h"

namespace feather {
class pinned_resource : public base_resource {
 public:
  inline void* allocate(size_t num_bytes) {
    void* result = nullptr;
    cuda::status_t status = cudaHostAlloc(&result, num_bytes, cudaHostAllocPortable);
    cuda::throw_if_cuda_error(status, "pinned_resource::allocate() failed");
    return result;
  }
  inline void deallocate(void* ptr, size_t) {
    cuda::status_t status = cudaFreeHost(ptr);
    cuda::throw_if_cuda_error(status, "pinned_resource::deallocate() failed");
  }

};
}; // namespace feather

#endif // PINNED_RESOURCE_H
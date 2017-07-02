#ifndef MANAGED_RESOURCE_H
#define MANAGED_RESOURCE_H

#include "../cuda.h"
#include "base_resource.h"

namespace feather {
class managed_resource : public base_resource {
 public:
  inline void* allocate(size_t num_bytes) {
    void* result = nullptr;
    cuda::status_t status = cudaMallocManaged(&result, num_bytes, cudaMemAttachGlobal);
    cuda::throw_if_cuda_error(status, "managed_resource::allocate() failed");
    return result;
  }
  inline void deallocate(void* ptr, size_t) {
    cuda::status_t status = cudaFree(ptr);
    cuda::throw_if_cuda_error(status, "managed_resource::deallocate() failed");
  }

};
}; // namespace feather

#endif // MANAGED_RESOURCE_H
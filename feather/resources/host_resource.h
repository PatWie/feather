#ifndef HOST_RESOURCE_H
#define HOST_RESOURCE_H

#include "../cuda.h"
#include "base_resource.h"

namespace feather {
class host_resource : public base_resource {
 public:
  inline void* allocate(size_t num_bytes) {
    void* result = new char[num_bytes];
    return result;
  }
  inline void deallocate(void* ptr, size_t) {
    delete[] (char*) ptr;
  }

};
}; // namespace feather

#endif // HOST_RESOURCE_H
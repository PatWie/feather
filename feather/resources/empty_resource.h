#ifndef EMPTY_RESOURCE_H
#define EMPTY_RESOURCE_H

#include "../cuda.h"
#include "base_resource.h"

namespace feather {
class empty_resource : public base_resource {
 public:
  inline void* allocate(size_t num_bytes) {
    return nullptr;
  }
  inline void deallocate(void* ptr, size_t) {
  }

};
}; // namespace feather

#endif // EMPTY_RESOURCE_H
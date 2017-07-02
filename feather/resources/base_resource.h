#ifndef BASE_RESOURCE_H
#define BASE_RESOURCE_H

namespace feather {
class base_resource {
 public:
  virtual inline void* allocate(size_t num_bytes) = 0;
  virtual inline void deallocate(void* ptr, size_t) = 0;

};
}; // namespace feather

#endif // BASE_RESOURCE_H
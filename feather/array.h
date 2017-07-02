#ifndef FEATHER_ARRAY_H
#define FEATHER_ARRAY_H

#include <cuda_runtime.h>
#include <array>

#include "tensor.h"
#include "resource.h"

namespace feather {
/**
 * @brief A container to manage linear arrays between devices.
 * @details This data structures does not know something about memory or
 *          shapes. It just sees the data somewhere as a pointer and the
 *          length of the data. It should be a seemless replacement to
 *          traditional arrays.
 *
 * @tparam Dtype datatype of the underlying data
 */
template<typename Dtype, typename resource = host_resource>
class array : private resource {
  // disable warning
  // tell the compiler, we really want to use both methods (overriden and ours)
  using resource::allocate;
  using resource::deallocate;

  // some aliases
  using value_type      = Dtype;
  using reference       = Dtype&;
  using pointer         = Dtype*;
  using const_value     = const Dtype;
  using const_reference = const Dtype&;
  using const_pointer   = const Dtype*;
 public:

  // avoiding getter and setter, we keep this public
  size_t length;
  pointer data = nullptr;

  array(size_t len_) : length(len_) {}

  array() : length(0) {}

  size_t size() const {
    return length * sizeof(Dtype);
  }

  void allocate() {
    if (data == nullptr)
      data = (pointer) resource::allocate(size());
    else
      printf("skip allocate\n");
  }

  void deallocate() {
    resource::deallocate(data, size());
    data = nullptr;
  }

  reference operator[](size_t off) {
    return data[off];
  }

  const_value operator[](size_t off) const {
    return data[off];
  }

  /**
   * @brief Return tensor = array + shape_information
   * @example 
   * 
   *     feather::array<float> test(9);
   *     test.allocate();
   *     auto t = test.tensor<2>({3, 3});
   * 
   */
  template <int Rank>
  feather::tensor<Dtype, Rank> tensor(std::array<const size_t, Rank> shape) {
    return feather::tensor<Dtype, Rank>(data, shape);
  }

  /**
   * @brief Return tensor = array + shape_information
   * @example 
   * 
   *     feather::array<float> test(9);
   *     test.allocate();
   *     auto t = test.tensor(3, 3);
   * 
   */
  template <typename ... Is>
  auto tensor (Is ... is) -> feather::tensor<Dtype, sizeof...(Is)> {
    return tensor<sizeof...(Is)>({ {std::size_t(is)... } });
  }

};
}; // namespace feather
#endif // FEATHER_ARRAY_H
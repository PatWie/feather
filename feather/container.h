#ifndef FEATHER_CONTAINER_H
#define FEATHER_CONTAINER_H

#include <cuda_runtime.h>

#include "resource.h"

namespace feather {
/**
 * @brief A container to manage linear arrays between two devices.
 * @details This data structures does not know something about tensor or
 *          shapes. It even does not know how to de-/allocate memory. It should behave
 *          transparent like the first array.
 *
 * @tparam Dtype [description]
 */
template<typename Dtype, typename MA = host_resource, typename MB = device_resource>
class container {
 public:
  size_t len;

  using value_type      = Dtype;
  using reference       = Dtype&;
  using pointer         = Dtype*;
  using const_value     = const Dtype;
  using const_reference = const Dtype&;
  using const_pointer   = const Dtype*;

  feather::array<Dtype, MA>* host;
  feather::array<Dtype, MB>* device;

  /**
   * @brief Create a new device pair with given length
   * @details just setup the meta data (no allocation of ressources)
   * 
   * @param l length of each array
   */
  container(size_t l) : len(l) {
    host = new feather::array<Dtype, MA>(len);
    device = new feather::array<Dtype, MB>(len);
  }

  /**
   * @brief Create a new device pair from given devices.
   * @param h first device (usually host)
   * @param d second device (usually gpu)
   */
  container(array<Dtype, MA> *h, array<Dtype, MB> *d) : host(h), device(d) {}

  /**
   * @brief Copy device data to host.
   */
  void to_host() {
    cudaMemcpy(host->data, device->data, size(), cudaMemcpyDeviceToHost);
  }

  /**
   * @brief Copy host data to device.
   */
  void to_device() {
    cudaMemcpy(device->data, host->data, size(), cudaMemcpyHostToDevice);
  }

  /**
   * @brief Wrap allocation for both arrays.
   */
  void allocate(){
    host->allocate();
    device->allocate();
  }

  /**
   * @brief Wrap deallocation for both arrays.
   */
  void deallocate(){
    host->deallocate();
    device->deallocate();
  }

  /**
   * @brief Number of elements.
   */
  size_t length() const {
    return host->length();
  }

  /**
   * @brief Total size in bytes.
   */
  size_t size() const {
    return host->size();
  }

  /**
   * @brief Write access like c-array for first device.
   */
  reference operator[](size_t off) {
    return host->data[off];
  }

  /**
   * @brief Read access like c-array for first device.
   */
  const_value operator[](size_t off) const {
    return host->data[off];
  }
};

/**
 * @brief Use cuda pinned memory (DMA) instead of using page-locked buffer.
 * @details You need to call "to_device()" after host allocation ones for getting the device pointer.
 */
template<typename Dtype, typename MB = empty_resource>
class pinned_container : public container<Dtype, pinned_resource, MB> {
 public:
  using parent = container<Dtype, pinned_resource, MB>;

  pinned_container(size_t l) : parent(l) {}

  void to_host() {}
  void to_device() {
    cuda::status_t status = cudaHostGetDevicePointer( &(parent::device->data), parent::host->data, 0 );
    cuda::throw_if_cuda_error(status, "pinned_container::to_device()::cudaHostGetDevicePointer() failed");
  }
};


}; // end feather

#endif // FEATHER_CONTAINER_H
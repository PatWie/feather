#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>
#include <vector>
#include <type_traits>

#include "misc.h"
#include "index.h"
#include <cuda_runtime.h>

namespace feather {

template<typename Dtype, typename xpu>
class tensor {
  // public:
    Dtype *buffer;
    std::vector<size_t> shape_;
    size_t len;
    bool owner;

public:
    tensor(std::vector<size_t> s)
        : buffer(nullptr), shape_(s), owner(true) {
        _update_len();
    }
    void _update_len() {
        len = 1;
        for (size_t i : shape_)
            len = len * i;
    }

    const std::vector<size_t> shape() const{
        return shape_;
    }

    const Dtype* const_ptr() const{
        return buffer;
    }

    Dtype* ptr(){
        return buffer;
    }

    Dtype& operator[](int i){
        return buffer[i];
    }

    Dtype operator[](int i) const{
        return buffer[i];
    }

    size_t size() const {
        return len;
    }

    template<typename... T>
    inline Dtype operator()(const T&... t) {
        const int n = sizeof...(T);
        return (Dtype) buffer[index<n>(shape_.data())(t...)];
    }

    void allocate(){
        if (std::is_same<xpu, cpu>::value) {
            buffer = new Dtype[len];
        }
        if (std::is_same<xpu, gpu>::value) {
            cudaMalloc( (void**)&buffer, len * sizeof(Dtype) );
        }
    }

    template<typename in_device>
    tensor<Dtype, xpu>& operator=(tensor<Dtype, in_device> &rhs) {

        shape_ = rhs.shape();
        _update_len();

        if (std::is_same<xpu, in_device>::value) {
            buffer = rhs.ptr();
            return *this;
        }

        if (buffer == nullptr) 
            allocate();


        if (std::is_same<gpu, in_device>::value && std::is_same<cpu, xpu>::value)
            cudaMemcpy( buffer, rhs.const_ptr(), len * sizeof(Dtype), cudaMemcpyDeviceToHost );

        if (std::is_same<cpu, in_device>::value && std::is_same<gpu, xpu>::value)
            cudaMemcpy( buffer, rhs.const_ptr(), len * sizeof(Dtype), cudaMemcpyHostToDevice );

        return *this;

    }

};


}; // namespace feather



#endif
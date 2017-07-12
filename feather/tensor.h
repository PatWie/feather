#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>
#include <vector>
#include <type_traits>

#include "misc.h"
#include "index.h"
#include <cuda_runtime.h>

namespace feather {

// template<typename Dtype, typename xpu=gpu>
template<typename Dtype, typename xpu>
class tensor {
  public:
    Dtype *buffer;
    std::vector<size_t> shape;
    size_t len;
    bool owner;

// public:
    tensor(std::vector<size_t> s)
        : buffer(nullptr), shape(s), owner(true) {
        _update_len();
    }
    void _update_len() {
        len = 1;
        for (size_t i : shape)
            len = len * i;
    }

    template<typename... T>
    inline Dtype operator()(const T&... t) {
        const int n = sizeof...(T);
        return (Dtype) buffer[index<n>(shape.data())(t...)];
    }

    template<typename in>
    tensor<Dtype, xpu>& operator=(tensor<Dtype, in> &rhs) {

        shape = rhs.shape;
        _update_len();

        if (std::is_same<xpu, in>::value) {
            buffer = rhs.buffer;
            return *this;
        }
        if (std::is_same<gpu, in>::value && std::is_same<cpu, xpu>::value) {
            // gpu --> cpu
            if (buffer == nullptr) 
                buffer = new Dtype[len];

            cudaMemcpy( buffer, rhs.buffer, len * sizeof(Dtype), cudaMemcpyDeviceToHost );
            return *this;
        }
        if (std::is_same<cpu, in>::value && std::is_same<gpu, xpu>::value) {
            // cpu --> gpu
            shape = rhs.shape;
            if (buffer == nullptr) 
                cudaMalloc( (void**)&buffer, len * sizeof(Dtype) );

            cudaMemcpy( buffer, rhs.buffer, len * sizeof(Dtype), cudaMemcpyHostToDevice );
            return *this;
        }

    }

};


}; // namespace feather



#endif
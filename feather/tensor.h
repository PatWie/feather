#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>
#include <vector>
#include <type_traits>

#include "misc.h"
#include <cuda_runtime.h>

namespace feather {


template<size_t start, size_t AXES=1>
struct stride {
    constexpr inline size_t operator()(size_t const* arr) const {
        return arr[start] * stride < start + 1, AXES > ()(arr);
    }
} ;

template<size_t AXES>
struct stride<AXES, AXES> {
    constexpr inline size_t operator()(size_t const*  arr) const {
        return 1;
    }
} ;

template<typename Dtype, typename xpu=cpu>
class container
{
public:
    Dtype *buffer;
    size_t len;
    
};

template<typename Dtype, size_t AXES=1, typename xpu=cpu>
class tensor {
public:
    size_t shapes[AXES];
    
    container<Dtype, xpu> data;

    template <typename... Dims>
    tensor(Dims... dims)  : shapes {dims...} {

    }

    template <typename... Dims>
    constexpr inline size_t _i(size_t off, Dims... dims) const {
        return off * (stride < AXES - (sizeof...(Dims)), AXES > ()(shapes)) + _i(dims...);
    }

    constexpr inline size_t _i(size_t t) const {
        return t;
    }
};





// template<typename Dtype, typename xpu>
// class tensor {
//   // public:
//     Dtype *buffer;
//     std::vector<size_t> shape_;
//     size_t len;

// public:
//     tensor(std::vector<size_t> s)
//         : buffer(nullptr), shape_(s){
//         _update_len();
//     }

//     tensor(Dtype* buf, std::vector<size_t> s)
//         : buffer(buf), shape_(s){
//         _update_len();


//     }
//     tensor(Dtype* buf, size_t s)
//         : buffer(buf), shape_({s}), len(s) {
//     }


//     void _update_len() {
//         len = 1;
//         for (size_t i : shape_)
//             len = len * i;
//     }

//     const std::vector<size_t> shape() const{
//         return shape_;
//     }

//     const Dtype* const_ptr() const{
//         return buffer;
//     }

//     Dtype* ptr(){
//         return *((Dtype**)&buffer);
//     }

//     Dtype& operator[](int i){
//         return buffer[i];
//     }

//     Dtype operator[](int i) const{
//         return buffer[i];
//     }

//     size_t size() const {
//         return len;
//     }

//     template<typename... T>
//     inline Dtype& operator()(const T&... t) {
//         const int n = sizeof...(T);
//         return buffer[index<n>(shape_)(t...)];
//     }

//     void allocate(){
//         runtime_assert(!buffer, "buffer already exists");
//         if (std::is_same<xpu, cpu>::value) {
//             buffer = new Dtype[len];
//         }
//         if (std::is_same<xpu, gpu>::value) {
//             cudaMalloc( (void**)&buffer, len * sizeof(Dtype) );
//         }
//     }

//     void free(){

//     }

//     // only move constructor
//     template<typename in_device>
//     tensor<Dtype, xpu>& operator=(tensor<Dtype, in_device> &&rhs) {

//         if (std::is_same<xpu, in_device>::value) {
//             return *this;
//         }

//         shape_ = rhs.shape();
//         _update_len();


//         if (std::is_same<gpu, in_device>::value && std::is_same<cpu, xpu>::value)
//             cudaMemcpy( buffer, rhs.const_ptr(), len * sizeof(Dtype), cudaMemcpyDeviceToHost );

//         if (std::is_same<cpu, in_device>::value && std::is_same<gpu, xpu>::value)
//             cudaMemcpy( buffer, rhs.const_ptr(), len * sizeof(Dtype), cudaMemcpyHostToDevice );

//         return *this;

//     }

// };


}; // namespace feather



#endif
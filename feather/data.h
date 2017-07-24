#ifndef DATA_H
#define DATA_H

#include <stddef.h>
#include <cuda_runtime.h>
#include "misc.h"


namespace feather {
template<typename Dtype, typename xpu = cpu>
class data
{
    Dtype* buffer;
    size_t len;

public:

    data(Dtype* b, size_t l) : buffer(b), len(l) {}

    data(size_t l) : buffer(nullptr), len(l) {}

    void allocate() {
        if (std::is_same<xpu, cpu>::value) {
            buffer = new Dtype[len];
        }else{
            cudaMalloc( (void**)&buffer, size() );
        }
    }

    void deallocate() {
        delete[] buffer;
    }

    const Dtype* const_ptr() const {
        return buffer;
    }

    Dtype* ptr() {
        return *((Dtype**)&buffer);
    }

    size_t size() const {
        return len * sizeof(Dtype);
    }

    size_t length() const {
        return len;
    }

    Dtype& operator[](size_t i) {
        return buffer[i];
    }

    Dtype operator[](size_t i) const {
        return buffer[i];
    }

    Dtype operator()(size_t i) const {
        return buffer[i];
    }



    // only move constructor
    template<typename in_device>
    data<Dtype, xpu>& operator=(data<Dtype, in_device> &&rhs) {

        if (std::is_same<xpu, in_device>::value) {
            return *this;
        }

        if (std::is_same<gpu, in_device>::value && std::is_same<cpu, xpu>::value)
            cudaMemcpy( buffer, rhs.const_ptr(), rhs.size(), cudaMemcpyDeviceToHost );

        if (std::is_same<cpu, in_device>::value && std::is_same<gpu, xpu>::value)
            cudaMemcpy( buffer, rhs.const_ptr(), rhs.size(), cudaMemcpyHostToDevice );

        return *this;

    }
};


// template<>
// void data<int, gpu>::allocate()  { cudaMalloc( (void**)&buffer, size() );}


// template<typename Dtype>
// class data<Dtype, gpu>
// {
//     Dtype* buffer;
//     size_t len;

// public:

//     data(Dtype* b, size_t l) : buffer(b), len(l) {}

//     void allocate() {
//         cudaMalloc( (void**)&buffer, size() );
//     }

//     void deallocate() {

//     }


// };


};

#endif
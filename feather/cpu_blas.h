#ifndef CPU_BLAS_BASE_H
#define CPU_BLAS_BASE_H

extern "C"
{
#include <cblas.h>
}

#include "misc.h"
#include "tensor.h"

namespace feather {


class blas {
    blas() {}
public:
    static blas& hnd() {
        static blas instance;
        return instance;
    }

    // ---------------------- level 1 ---------------------------------------------
    float sum(const int n, const float* x) {
        return cblas_sasum(n, x, 1);
    }
    
    float sum(const tensor<float, cpu> &t) {
        return sum(t.size(), t.const_ptr());
    }

    
    // void axpy(const int n, const float a, const float* x, float* y) {
    //     cblas_saxpy(n, a, x, 1, y, 1);
    // }

    // // void axpy(const float a, const arr<float, cpu>  x, arr<float, cpu>  *y) {
    // void axpy(const float a, const tensor<float, cpu>  x, tensor<float, cpu>  *y) {
    //     assert(x.size()() == y->size()());
    //     cblas_saxpy(x.size()(), a, x.cpu(), 1, y->mutable_cpu(), 1);
    // }

    void copy( const int num_elements, const float* input, float* output) {
        cblas_scopy(num_elements, input, 1, output, 1);
    }

    void copy(const tensor<float, cpu>  &x, tensor<float, cpu>  *y) {
        copy(x.size(), x.const_ptr(), y->ptr());
    }

    // float dot(const int n, const float* a, const float* b) {
    //     return cblas_sdot(n, a, 1, b, 1);
    // }

    // float dot(const arr<float, cpu> &a, const arr<float, cpu> &b) {
    //     return cblas_sdot(a.len(), a.cpu(), 1, b.cpu(), 1);
    // }

    // float norm2(const int n, const float* x) {
    //     return cblas_snrm2(n, x, 1);
    // }

    // float norm2(const arr<float, cpu> &x) {
    //     return cblas_snrm2(x.len(), x.cpu(), 1);
    // }

    // void scale( const int num_elements, const float alpha, float* output) {
    //     cblas_sscal(num_elements, alpha, output, 1);
    // }

    // void scale(const float alpha, arr<float, cpu> *x) {
    //     cblas_sscal(x->len(), alpha, x->mutable_cpu(), 1);
    // }

    // // y = x
    // void swap( const int num_elements, float* input, float* output) {
    //     cblas_sswap(num_elements, input, 1, output, 1);
    // }

    // void swap(arr<float, cpu>  *x, arr<float, cpu>  *y) {
    //     cblas_sswap(x->len() , x->mutable_cpu(), 1, y->mutable_cpu(), 1);
    // }

};



};

#endif
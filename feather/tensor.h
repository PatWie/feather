#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>
#include <vector>

#include "misc.h"
#include "index.h"

namespace feather {

template<typename Dtype, typename xpu=gpu>
class tensor
{
public:
    Dtype *buffer;
    std::vector<size_t> shape;

// public:
    tensor(Dtype *b, std::vector<size_t> s)
        : buffer(b), shape(s){}

    inline Dtype operator()(size_t a) { return (Dtype) buffer[index<1>(shape.data())(a)];}
    inline Dtype operator()(size_t a, size_t b) { return (Dtype) buffer[index<2>(shape.data())(a,b)];}
    inline Dtype operator()(size_t a, size_t b, size_t c) { return (Dtype) buffer[index<3>(shape.data())(a,b,c)];}
    inline Dtype operator()(size_t a, size_t b, size_t c, size_t d) { return (Dtype) buffer[index<4>(shape.data())(a,b,c,d)];}

    template<typename in>
    tensor<Dtype, xpu>& operator=(const tensor<Dtype, in> & rhs)
    {
        std::cout << "same" << std::endl;
          
        buffer = rhs.buffer;
        return *this;
    }

    inline const Dtype test(){
        return 1;
    }

    
};

// template<typename Dtype, typename xpu>
// tensor<Dtype, cpu>& tensor<Dtype, gpu>::tester<gpu>(const tensor<Dtype, gpu> &rhs) { return *this; }


template <>
inline const float tensor<float, gpu>::test() { return 1; }

template <>
inline const float tensor<float, cpu>::test() { return 3; }

// template<typename Dtype>
// tensor<Dtype, cpu>&  tensor<Dtype, cpu>::operator=(const tensor<Dtype, gpu> & rhs){ 
//     // std::cout << "same" << std::endl;
          
//     // buffer = rhs.buffer;
//     return *this;
// }

// template<typename Dtype>
// class tensor<Dtype, gpu>
// {
// public:
//     tensor<Dtype, gpu>& operator=(const tensor<Dtype, cpu> & rhs)
//     {
//         std::cout << "cpu --> gpu" << std::endl;
//         buffer = rhs.buffer;
//         return *this;
//     }
    
// };

// template<typename Dtype>
// class tensor<Dtype, cpu>
// {
// public:
//     tensor<Dtype, cpu>& operator=(const tensor<Dtype, gpu> & rhs)
//     {
//         std::cout << "gpu --> cpu" << std::endl;
//         buffer = rhs.buffer;
//         return *this;
//     }
    
// };

}; // namespace feather



#endif
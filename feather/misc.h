#ifndef MISC_H
#define MISC_H

#include <array>

namespace feather {
struct cpu
{
    static const bool CpuDevice = true;
};

struct gpu
{
    static const bool CpuDevice = false;
};



// meta functions
template<typename T>
constexpr T meta_prod(T x) { return x; }

template<typename T, typename... Ts>
constexpr T meta_prod(T x, Ts... xs) { return x * meta_prod(xs...); }





}

#endif
#ifndef INDEX_H
#define INDEX_H

#include <array>
#include <iostream>

#include "misc.h"


// namespace feather {


// namespace index_helper {
// template<size_t start, size_t AXES>
// struct prod_func
// {
//   constexpr size_t operator()(size_t const * arr) const
//   {
//     return arr[start] * prod_func < start + 1, AXES > ()(arr);
//   }
// } ;

// template<size_t AXES>
// struct prod_func<AXES, AXES>
// {
//   constexpr size_t operator()(size_t const * arr) const
//   {
//     return 1;
//   }
// } ;

// }


// template<int AXES>
// class index
// {
//   const size_t *shapes;
// public:

//   index(size_t *s) : shapes(s) {}
//   index(std::vector<size_t> s) : shapes(s.data()) {}

//   template <typename... Dims>
//   const inline size_t operator()(int off, Dims... dims) const {
//     constexpr size_t num_dims = sizeof...(Dims);
//     static_assert(num_dims < AXES, "num_dims should be smaller than axis");
//     const size_t stride =  index_helper::prod_func < AXES - num_dims, AXES > ()(shapes);
//     return off * stride + operator()(dims...);
//   }

//   constexpr inline size_t operator()(int t) const {
//     return t;
//   }

//   constexpr inline size_t operator[](size_t i) const
//   {
//     return shapes[i];
//   }

// };

// }; // namespace feather

namespace feather {




// for shape (A,B,C,D) and offsets (a,b,c,d)
namespace index_helper {

template<size_t start, size_t AXES>
struct prod_func
{
  constexpr inline size_t operator()(const std::array<const size_t, AXES> arr) const
  {
    return arr[start] * prod_func < start + 1, AXES > ()(arr);
  }
} ;

template<size_t AXES>
struct prod_func<AXES, AXES>
{
  constexpr inline size_t operator()(const std::array<const size_t, AXES> arr) const
  {
    return 1;
  }
} ;

}


template<int AXES>
class index
{
  const std::array<const size_t, AXES> shapes;

public:

  index(std::array<const size_t, AXES> s) : shapes(s) {}

  template <typename... Dims>
  constexpr inline size_t operator()(int off, Dims... dims) const {
    return off * (index_helper::prod_func < AXES - (sizeof...(Dims)), AXES > ()(shapes)) + operator()(dims...);
  }

  constexpr inline size_t operator()(int t) const {
    return t;
  }


};

}; // namespace feather

#endif
  // constexpr inline size_t operator[](size_t i) const
  // {
  //   return shapes[i];
  // }
  // template <typename... Dims>
  // const inline size_t operator()(int off, Dims... dims) const {
  //   constexpr size_t num_dims = sizeof...(Dims);
  //   static_assert(num_dims < AXES, "num_dims should be smaller than axis");
  //   // const size_t stride =  index_helper::prod_func < AXES - num_dims, AXES > ()(shapes);
  //   const size_t stride = index_helper::static_accumulate<AXES - num_dims, AXES, shapes.data()>();
  //   return off * stride + operator()(dims...);
  // }
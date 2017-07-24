#ifndef INDEX_H
#define INDEX_H

#include <array>
#include <iostream>

namespace feather {

// template<size_t start, size_t AXES>
// struct prod_func
// {
//   constexpr inline size_t operator()(size_t const* arr) const
//   {
//     return arr[start] * prod_func < start + 1, AXES > ()(arr);
//   }
// } ;

// template<size_t AXES>
// struct prod_func<AXES, AXES>
// {
//   constexpr inline size_t operator()(size_t const*  arr) const
//   {
//     return 1;
//   }
// } ;

template<int AXES>
class index
{
  size_t shapes[AXES];

public:

  template <typename... Dims>
  index(Dims... dims)  : shapes {dims...} {}

  template <typename... Dims>
  constexpr inline size_t operator()(int off, Dims... dims) const {
    return off * (prod_func < AXES - (sizeof...(Dims)), AXES > ()(shapes)) + operator()(dims...);
  }

  constexpr inline size_t operator()(int t) const {
    return t;
  }


};



// template<size_t start, size_t AXES>
// struct prod_func
// {
//   constexpr inline size_t operator()(const std::array<const size_t, AXES> arr) const
//   {
//     return arr[start] * prod_func < start + 1, AXES > ()(arr);
//   }
// } ;

// template<size_t AXES>
// struct prod_func<AXES, AXES>
// {
//   constexpr inline size_t operator()(const std::array<const size_t, AXES> arr) const
//   {
//     return 1;
//   }
// } ;

// template<int AXES>
// class index
// {
//   const std::array<const size_t, AXES> shapes;

// public:

//   index(std::array<const size_t, AXES> s) : shapes(s) {}

//   template <typename... Dims>
//   constexpr inline size_t operator()(int off, Dims... dims) const {
//     return off * (prod_func < AXES - (sizeof...(Dims)), AXES > ()(shapes)) + operator()(dims...);
//   }

//   constexpr inline size_t operator()(int t) const {
//     return t;
//   }


// };


}; // namespace feather

#endif
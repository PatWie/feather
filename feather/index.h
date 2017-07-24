#ifndef INDEX_H
#define INDEX_H

#include <array>
#include "misc.h"

namespace feather {


namespace index_helper {
template<size_t start, size_t AXES>
struct coords_to_line
{
  constexpr inline size_t operator()(const std::array<const size_t, AXES> arr) const
  {
    return arr[start] * coords_to_line < start + 1U, AXES > ()(arr);
  }
} ;

template<size_t AXES>
struct coords_to_line<AXES, AXES>
{
  constexpr inline size_t operator()(const std::array<const size_t, AXES> arr) const
  {
    return 1U;
  }
} ;

};



template<int AXES>
class index
{
  const std::array<const size_t, AXES> shapes;

public:

  index(std::array<const size_t, AXES> s) : shapes(s) {}

  template <typename... Coords>
  constexpr inline size_t _i(int off, Coords... coords) const {
    return off * (index_helper::coords_to_line < AXES - (sizeof...(Coords)), AXES > ()(shapes)) + operator()(coords...);
  }

  constexpr inline size_t _i(int t) const {
    return t;
  }

  template <typename... Coords>
  constexpr inline size_t operator()(Coords... coords) const {
    static_assert(check_same<std::size_t, Coords...>::value || check_same<int, Coords...>::value, "THE COORDINATE TYPE MUST BE STD::SIZE_T OR INT");
    return _i(coords...);
  }

};
};

#endif
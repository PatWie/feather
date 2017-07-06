#ifndef INDEX_H
#define INDEX_H

#include <array>
#include <iostream>

namespace feather {

template<int AXES>
class index
{
  size_t *shapes;
  size_t offsets[AXES];
public:
  index(size_t *s) : shapes(s) {
    _cache();
  }
  index(std::array<size_t, AXES> s) : shapes(s.data()) {
    _cache();
  }

  void _cache(){
    offsets[AXES - 1] = 1;
    for (int i = AXES - 2; i >= 0; --i)
      offsets[i] = offsets[i + 1] * shapes[i + 1];
  }

  constexpr inline size_t operator[](size_t i) const
  {
    return shapes[i];
  }

  constexpr inline size_t offset(int d) const {
    return offsets[AXES - d - 1];
  }

  constexpr inline size_t i_(int t) const{
    return t;
  }

  template <typename... Rest>
  constexpr inline size_t i_(int axis, int pos, Rest... rest) const{
    return pos * offset(axis - 1) + i_(axis - 1, rest...);
  }

  template <typename... Rest>
  constexpr inline size_t operator()(int pos, Rest... rest) const {
    return pos * offset(AXES - 1) + i_(AXES - 1, rest...);
  }
};

}; // namespace feather

#endif
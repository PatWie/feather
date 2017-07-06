#ifndef INDEX_H
#define INDEX_H

#include <array>

namespace feather {

template<int AXES>
class index
{
  size_t *shapes;
public:
  index(size_t *s) : shapes(s) {}
  index(std::array<size_t, AXES> s) : shapes(s.data()) {}

  const inline size_t operator[](size_t i) const
  {
    return shapes[i];
  }

  const inline size_t helper(int d) const {
    size_t ans = 1;
    for (int dd = 0; dd < d ; ++dd)
      ans *= shapes[dd];
    return ans;
  }

  const inline size_t i_(int t) const{
    return t;
  }

  template <typename... Rest>
  const inline size_t i_(int axis, int pos, Rest... rest) const{
    return pos * helper(axis - 1) + i_(axis - 1, rest...);
  }

  template <typename... Rest>
  const inline size_t operator()(int pos, Rest... rest) const {
    return pos * helper(AXES - 1) + i_(AXES - 1, rest...);
  }
};

}; // namespace feather

#endif
#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include "array.h"
#include "misc.h"

namespace feather {

/**
 * @brief efficient meta-template programming for projecting multi-dim to line
 * @details see my post on https://stackoverflow.com/questions/45269319
 *          waiting for c++14 and loops in constexpr
 */
namespace index_helper {
template<size_t start, size_t Rank>
struct coords_to_line {
  _HD_ constexpr inline size_t operator()(const std::array<const size_t, Rank> arr) const {
    return arr[start] * coords_to_line < start + 1U, Rank > ()(arr);
  }
} ;

template<size_t Rank>
struct coords_to_line<Rank, Rank> {
  _HD_ constexpr inline size_t operator()(const std::array<const size_t, Rank> arr) const {
    return 1U;
  }
} ;
}; // namespace index_helper

template <typename Arg, typename... Args>
void doPrint(std::ostream& out, Arg&& arg, Args&&... args) {
  out << std::forward<Arg>(arg);
  using expander = int[];
  (void)expander{0, (void(out << ',' << std::forward<Args>(args)), 0)...};
}


/**
 * @brief An array with information about shape
 * @details This provides overloaded access operator(....).
 * @example
 *
 *     feather::array<float> test(9);
 *     test.allocate();
 *     auto t1 = test.tensor<2>({3, 3});
 *     auto t2 = test.tensor(3, 3);
 *
 *     double *a = new double[9];
 *     auto t3 = feather::tensor<2>(a, {3, 3});
 *
 * @tparam Dtype datatype of underlying data
 * @tparam shape array containing the shape information
 */
// template<int Rank, template<typename, typename> array_t>
template<typename Dtype, int Rank>
class tensor {

  const std::array<const size_t, Rank> shp_info;
  Dtype *array;

  // helper to compute point2line
  template <typename... Coords>
  _HD_ constexpr inline size_t _i(int off, Coords... coords) const {
    return off * (index_helper::coords_to_line < Rank - (sizeof...(Coords)), Rank > ()(shp_info)) + _i(coords...);
  }

  _HD_ constexpr inline size_t _i(int t) const {
    return t;
  }

  using value_type      = Dtype;
  using reference       = Dtype&;
  using pointer         = Dtype*;
  using const_value     = const Dtype;
  using const_reference = const Dtype&;
  using const_pointer   = const Dtype*;

 public:
  /**
   * @brief create a new tensor (without managing the data)
   * @example
   * 
   *    float *a_raw = new ...;
   *    feather::tensor<float, 2> A(a_raw, {9, 10});
   * 
   * @param a raw data buffer
   * @param s tensor shape nd-array
   */
  _HD_ tensor(Dtype *a, std::array<const size_t, Rank> s)
    : array(a), shp_info(s) {

  }
  _HD_ ~tensor() {}

  _HD_ constexpr inline size_t dim(size_t dim) const{
    return shp_info[dim];
  }

  _HD_ pointer data(){
    return array;
  }

  /**
   * @brief check if arguments are within bounds of shape.
   * @details Check if each "0<= coordinate AND coordinate < shape[pos]" holds 
   * @example
   * 
   *    feather::tensor<Dtype, 2> A(A, {9, 10});
   *    if (A.valid(i, j))
   *      ...
   *    // same as
   *    if ((0<=i) && (i<9) && (0<=j) && (j<10))
   *      ...
   * 
   * 
   * @return whether within bounds
   */
  template <typename... Coords>
  _HD_ constexpr inline bool valid(int off, Coords... coords) const {
    return (off >= 0) && (off < shp_info[Rank - (sizeof...(Coords)) - 1]) && valid(coords...);
  }

  _HD_ constexpr inline bool valid(int off) const {
    return (off >= 0) && (off < shp_info[Rank - 1]);
  }

  template <typename... Coords>
  _HD_ constexpr inline size_t pos(Coords... coords) const {
    static_assert(check_same<std::size_t, Coords...>::value || check_same<int, Coords...>::value, "THE COORDINATE TYPE MUST BE STD::SIZE_T OR INT");
    return _i(coords...);
  }

  /**
   * @brief access element at given position
   * @example
   * 
   *    feather::tensor<Dtype, 2> A(A, {9, 10});
   *    A(i, j) = 7;
   *      ...
   *    // same as
   *    A[i * 10 + j] = 7;
   *      ...
   */
  template <typename... Coords>
  _HD_ value_type operator()(Coords... coords) const {
    const size_t p = pos(coords...);
    return array[p];
  }

  template <typename... Coords>
  _HD_ reference operator()(Coords... coords) {
    const size_t p = pos(coords...);
    return array[p];
  }

  reference operator[](int off) {
    return array[off];
  }

  const_value operator[](size_t off) const {
    return array[off];
  }

};

}; // namespace feather

#endif // TENSOR_H
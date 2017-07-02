#ifndef FEATHER_MATH_H
#define FEATHER_MATH_H

namespace feather
{
  namespace math
  {
    template <typename T>
    constexpr T ipow(T base, unsigned exponent, T coefficient) {
      return exponent == 0 ? coefficient :
        ipow(base * base, exponent >> 1, (exponent & 0x1) ? coefficient * base : coefficient);
    }


    template <typename T>
    constexpr T ipow(T base, unsigned exponent)
    {
      return ipow(base, exponent, 1);
    }
  }; // namespace math
}; // namespace feather

#endif // FEATHER_MATH_H
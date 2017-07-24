#include <iostream>
#include <cstdlib>
#include <array>


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

template<int AXES>
class index
{
  const std::array<const size_t, AXES> shapes;

public:

  index(std::array<const size_t, AXES> s) : shapes(s) {}

  template <typename... Dims>
  constexpr inline size_t operator()(int off, Dims... dims) const {
    return off * (prod_func < AXES - (sizeof...(Dims)), AXES > ()(shapes)) + operator()(dims...);
  }

  constexpr inline size_t operator()(int t) const {
    return t;
  }


};


int main()
{
    size_t cA=2, cB=3, cC=6, cD=7;
    int ca=1, cb=1, cc=1, cd=1;

    volatile size_t A = cA;
    volatile size_t B = cB;
    volatile size_t C = cC;
    volatile size_t D = cD;

    volatile int a = ca;
    volatile int b = cb;
    volatile int c = cc;
    volatile int d = cd;

    asm ("#idx");
    auto idx = index<4>({A,B,C,D});
    size_t result =  idx(a,b,c,d);
    asm ("#output"); 
    std::cout << result << std::endl;
    asm ("#traditional"); 
    result = (a*B*C*D + b*C*D + c*D + d);
    asm ("#output");
    std::cout << result << std::endl;


    asm ("#idx");
    auto idx2 = index2<4>({A,B,C,D});
    result =  idx2(a,b,c,d);
    asm ("#output"); 
    std::cout << result << std::endl;


    return 0;

}
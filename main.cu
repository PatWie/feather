#include <iostream>
#include <vector>
#include <array>

#include "feather/misc.h"
// #include "feather/index.h"
#include "feather/tensor.h"
// #include "feather/cpu_blas.h"


using cpu = feather::cpu;
using gpu = feather::gpu;
template<typename Dtype, int AXES=1, typename xpu=cpu>
using tensor = feather::tensor<Dtype, AXES, xpu>;

int main(int argc, char const *argv[]) {

  auto h = tensor<float, 2>(9, 8);

  std::cout << h.shapes[0] << std::endl;
  std::cout << h.shapes[1] << std::endl;
  std::cout << h._i(1,1) << std::endl;
    
  // tensor<float, cpu> bc = tensor<float, cpu>({9, 9});
  // bc.allocate();
  // tensor<float, gpu> bg = tensor<float, gpu>({9, 9});
  // tensor<float, cpu> bcc= tensor<float, cpu>({9, 9});
  // bcc = bc;

  // float *t = bc.ptr();
  // t[0] = 19;
  // bc(0) = 17;
  // bcc(0) = 18;
  // std::cout << bc[0] << std::endl;
  // std::cout << bc(0) << std::endl;
  // std::cout << t[0] << std::endl;
  // std::cout << bcc(0) << std::endl;


    
    
  // t = new float[9];
  // for (int i = 0; i < 9 * 9; ++i) bc[i] = i;

  // bg = bc;
  // bcc = bg;

  // std::cout << bcc(1,0) << std::endl;
  // std::cout << bcc(1,1) << std::endl;

  // float *test = new float[10];
  // float expected = 0;
  // for (int i = 0; i < 10; ++i)
  // {
  //   test[i] = i;
  //   expected += i;
  // }

  // auto blas = feather::blas::hnd();


  // std::cout << blas.sum(10, test) << std::endl;
  // std::cout << expected << std::endl;
  // std::cout << blas.sum(bc) << std::endl;

  // bcc = bc;

  // float *tt = new float[9];
  // if(!tt)
  //   std::cout << "!tt=true" << std::endl;
  // else
  //   std::cout << "!tt=false" << std::endl;
      

  return 0;

}

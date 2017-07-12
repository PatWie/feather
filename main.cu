#include <iostream>
#include <vector>
#include <array>

#include "feather/index.h"
#include "feather/tensor.h"
#include "feather/cpu_blas.h"


using cpu = feather::cpu;
using gpu = feather::gpu;
template<typename Dtype, typename xpu>
using tensor = feather::tensor<Dtype, xpu>;

int main(int argc, char const *argv[]) {


  tensor<float, cpu> bc = tensor<float, cpu>({9, 9});
  tensor<float, gpu> bg = tensor<float, gpu>({9, 9});
  tensor<float, cpu> bcc = tensor<float, cpu>({9, 9});

  bc.allocate();
  for (int i = 0; i < 9 * 9; ++i) bc.ptr()[i] = i;

  bg = bc;
  bcc = bg;

  std::cout << bcc(1,0) << std::endl;
  std::cout << bcc(1,1) << std::endl;

  float *test = new float[10];
  float expected = 0;
  for (int i = 0; i < 10; ++i)
  {
    test[i] = i;
    expected += i;
  }

  auto blas = feather::blas::hnd();


  std::cout << blas.sum(10, test) << std::endl;
  std::cout << expected << std::endl;
  std::cout << blas.sum(bc) << std::endl;

    
 
  return 0;

}

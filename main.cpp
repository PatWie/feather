#include <iostream>
#include <vector>
#include <array>

#include "feather/index.h"
#include "feather/tensor.h"


using cpu = feather::cpu;
using gpu = feather::gpu;
template<typename Dtype, typename xpu>
using tensor = feather::tensor<Dtype, xpu>;

int main(int argc, char const *argv[]) {


  tensor<float, cpu> bc = tensor<float, cpu>({9, 9});
  tensor<float, gpu> bg = tensor<float, gpu>({9, 9});
  tensor<float, cpu> bcc = tensor<float, cpu>({9, 9});

  bc.buffer = new float[9 * 9];
  for (int i = 0; i < 9 * 9; ++i) bc.buffer[i] = i;

  bg = bc;
  bcc = bg;

  std::cout << bcc(1,0) << std::endl;
  std::cout << bcc(1,1) << std::endl;
 
  return 0;

}

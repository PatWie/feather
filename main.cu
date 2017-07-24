#include <iostream>
#include <vector>
#include <array>

#include "feather/index.h"
#include "feather/tensor.h"
// #include "feather/tensor2.h"
#include "feather/cpu_blas.h"
#include "feather/cuda.h"


using cpu = feather::cpu;
using gpu = feather::gpu;
template<typename Dtype, typename xpu>
using tensor = feather::tensor<Dtype, xpu>;
// template<typename Dtype>
// using tensor2 = feather::tensor2<Dtype>;

int main(int argc, char const *argv[]) {

  // auto idx = feather::index<3>({2,3,5});
  // idx(1,1,1);

  tensor<float, cpu> bc = tensor<float, cpu>({9, 9});
  // tensor<float, gpu> bg = tensor<float, gpu>({9, 9});
  // tensor<float, cpu> bcc = tensor<float, cpu>({9, 9});

  bc.allocate();
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

  // cuda<float>* d = new cuda<float>(10);
  // std::cout << d->len() << std::endl;
  // std::cout << d->size() << std::endl;

  // // d[9] = 8;
  // // std::cout << d[9] << std::endl;
  // // std::cout << d[8] << std::endl;
  // // std::cout << d[7] << std::endl;

  // tensor2<float> t({10, 2});

  // std::cout << t.size() << std::endl;
  // std::cout << t.len() << std::endl;

  // float *dd = t.host();

  // dd[3] = 9;
  // std::cout << t.host()[3] << std::endl;
  // // std::cout << t.host(3) << std::endl;

  // std::cout << t.host(1) << std::
  // ndl;
  bc[1*9 + 1] = 42;
  std::cout << bc[1*9 + 1] << std::endl;
  std::cout << bc(1, 1) << std::endl;
  // std::cout << bc(1, 1, 1) << std::endl;
  // std::cout << bc(1, 1, 1, 1) << std::endl;
    
    
    
    
  return 0;

}

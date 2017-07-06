#include <iostream>
#include <vector>
#include <array>

#include "feather/index.h"
#include "feather/tensor.h"




int main(int argc, char const *argv[])
{

  float *data = new float[9*9];
  for (int i = 0; i < 9*9; ++i) data[i] = i;

  auto a = feather::tensor<float>(data, {9,9});


    auto b = feather::tensor<float>(data, {9,9});
    b = a;

  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      std::cout << a(i, j) << std::endl;


  return 0;
  
}

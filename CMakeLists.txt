cmake_minimum_required(VERSION 2.8)
project(FeatherExample)

find_package(CUDA REQUIRED)
include(FindCUDA)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# use ATLAS for BLAS on CPU
message(STATUS ${CMAKE_MODULE_PATH})
find_package(Atlas REQUIRED)
include_directories(SYSTEM ${Atlas_INCLUDE_DIR})

# add google testing framework
# add_subdirectory(gtest)
# enable_testing()

SET(CXX "g++")

# list(APPEND CMAKE_CXX_FLAGS " -O3 -Wall -std=c++11")
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo  -Xptxas -dlcm=cg -O3)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
list(APPEND CUDA_NVCC_FLAGS --cudart static --relocatable-device-code=false)

cuda_include_directories(/usr/local/cuda/samples/common/inc/)
cuda_include_directories(include)

# --------------------------------------------------------------
# main app
# --------------------------------------------------------------


cuda_add_executable(main main.cu )
cuda_add_cublas_to_target(main)
target_link_libraries(main  ${Atlas_LIBRARIES})

# # --------------------------------------------------------------
# # unittest
# # --------------------------------------------------------------
# cuda_add_executable(main main.cu )
# cuda_add_cublas_to_target(main)
# # target_link_libraries(main  gtest gtest_main pthread ${Atlas_LIBRARIES})
# target_link_libraries(main ${Atlas_LIBRARIES})
# # add_test( main main )

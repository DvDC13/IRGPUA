project(IRGPUA)
cmake_minimum_required(VERSION 3.18)

include(FindCUDAToolkit)

enable_language(CUDA)

find_package(OpenMP REQUIRED)

include_directories(src)
include_directories(src/kernels/include)

add_compile_options(-W -Wall -Wextra -fconcepts)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto -funroll-loops -ffast-math")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp --extended-lambda --generate-line-info --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "--ptxas-options=-v --generate-line-info")

add_executable(main src/main.cu 
                    src/fix_cpu.cu 
                    src/fix_gpu.cu
                    src/fix_thrust.cu
                    src/parse.cu
                    src/kernels/src/reduce.cu
                    src/kernels/src/scan.cu
                    src/kernels/src/shift.cu
                    src/kernels/src/histogram.cu
                    src/kernels/src/apply_map.cu
                    src/kernels/src/build_predicate.cu
                    src/kernels/src/scatter_adresses.cu
                    src/kernels/src/find_first_non_zero.cu
                    src/kernels/src/apply_map_transformation.cu
                    src/kernels/src/radix_sort.cu)

set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

# BUILD_TYPE
if (NOT CMAKE_BUILD_TYPE)
    string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
    # Default BUILD_TYPE is Release
    set(CMAKE_BUILD_TYPE Release)
    # If folder is named "build_debug" BUILD_TYPE is set to: Debug
    if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()

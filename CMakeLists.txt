cmake_minimum_required(VERSION 3.12)


#add_definitions(-DWITH_NEON)
add_definitions(-DWITH_AVX2)

add_compile_options(-mavx2)

add_compile_options(-O3)

set(CMAKE_CXX_STANDARD 11)

project(matmul)

ADD_EXECUTABLE(matmul main.c matmul.c)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(matmul PUBLIC OpenMP::OpenMP_C)
endif()


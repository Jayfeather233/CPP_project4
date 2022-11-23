cmake_minimum_required(VERSION 3.12)

#add_definitions(-DWITH_NEON)
add_definitions(-DWITH_AVX2)

add_compile_options(-mavx2)
add_compile_options(-lopenblas)

include_directories(/usr/local/include/openblas)
link_libraries(/usr/local/lib/libopenblas.a)

project(matmul)

ADD_EXECUTABLE(matmul main.c matmul.c)

find_package(OpenMP)
if(OpenMP_C_FOUND)
   target_link_libraries(matmul PUBLIC OpenMP::OpenMP_C)
endif()
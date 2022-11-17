#ifndef _MATMUL_H_
#define _MATMUL_H_

#include <stdlib.h>
#include <stdio.h>

typedef struct __Mat{
    size_t n,m;
    float *data;
}Mat;

//The plain method to calculate
Mat matmul_plain(const Mat a,const Mat b);

//Use SIMD and OpenMP to calculate
Mat matmul_improved(const Mat a,const Mat b);

//transfer a Matrix to make memory access continues.
float* trans(const Mat a);

//free a matrix 
void mat_free(const Mat a);

//create a new matrix with aligned memory
Mat newmat_aligned(size_t m,size_t n);

//output a matrix
void moutput(const Mat a);
#endif
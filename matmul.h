#ifndef _MATMUL_H_
#define _MATMUL_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct __Mat{
    size_t n,m;
    float *data;
}Mat;

//The plain method to calculate
bool matmul_plain_ijk(const Mat *a,const Mat *b, Mat *ans);
//The plain method with order ikj to calculate
bool matmul_plain_ikj(const Mat *a,const Mat *b, Mat *ans);
//The plain method with order ikj and OpenMP to calculate
bool matmul_plain_omp_ikj(const Mat *a,const Mat *b, Mat *ans);

//Use SIMD and OpenMP to calculate
bool matmul_improved(const Mat *a,const Mat *b, Mat *ans);

//Use SIMD and OpenMP and blocking to calculate
bool matmul_improved_blocking(const Mat *a,const Mat *b, Mat *ans);

//transfer a Matrix to make memory access continues.
void trans(const Mat *a, float *dist);

//free a matrix 
void mat_free(Mat *a);

//create a new matrix with aligned memory
Mat* newmat_aligned(const size_t m,const size_t n, const float *src);

//output a matrix
void moutput(const Mat *a);
#endif
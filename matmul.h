#ifndef _MATMUL_H_
#define _MATMUL_H_

#include <stdlib.h>
#include <stdio.h>

typedef struct __Mat{
    size_t n,m;
    float *data;
}Mat;

Mat matmul_plain(const Mat a,const Mat b);
Mat matmul_improved(const Mat a,const Mat b);
float* trans(const Mat a);
void mat_free(const Mat a);
Mat newmat_aligned(size_t m,size_t n);
void moutput(const Mat a);
#endif
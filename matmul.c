#include "matmul.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

//variable validity check
#define CHECK(a,b)\
if(a==NULL){\
    fprintf(stderr, "%s: line %d, %dst parameter is NULL\n",__FUNCTION__,__LINE__,b);\
    return false;\
}else if(a->data==NULL){\
    fprintf(stderr, "%s: line %d, %dst parameter has no valid data\n",__FUNCTION__,__LINE__,b);\
    return false;\
}

bool matmul_plain_ijk(const Mat *a, const Mat *b, Mat *ans)
{
    CHECK(a,1)
    CHECK(b,2)
    CHECK(ans,3)
    if (a->n != b->m || a->m!=ans->m || b->n!=ans->n){
        fprintf(stderr, "%s: Matrixs do not have the same size\n",__FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }
    
    memset(ans->data, 0.0f, sizeof(float)*a->m*b->n);

    //simple ijk
    size_t kk;
    for (size_t i = 0; i < a->m; i++)
    {
        for (size_t j = 0; j < b->n; j++)
        {
            for (size_t k = 0; k < a->n; k++)
            {
                ans->data[i * b->n + j] += a->data[i * a->n + k] * b->data[k * b->n + j];
            }
        }
    }
    return true;
}
bool matmul_plain_omp_ikj(const Mat *a, const Mat *b, Mat *ans)
{
    CHECK(a,1)
    CHECK(b,2)
    CHECK(ans,3)
    if (a->n != b->m || a->m!=ans->m || b->n!=ans->n){
        fprintf(stderr, "%s: Matrixs do not have the same size\n",__FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }

    memset(ans->data, 0.0f, sizeof(float)*a->m*b->n);

    //simple ikj with omp
#pragma omp parallel for
    for (register size_t i = 0; i < a->m; i++)
    {
        for (register size_t k = 0; k < a->n; k++)
        {
            for (register size_t j = 0; j < b->n; j++)
            {
                ans->data[i * b->n + j] += a->data[i * a->n + k] * b->data[k * b->n + j];
            }
        }
    }
    return true;
}
bool matmul_plain_ikj(const Mat *a, const Mat *b, Mat *ans)
{
    CHECK(a,1)
    CHECK(b,2)
    CHECK(ans,3)
    if (a->n != b->m || a->m!=ans->m || b->n!=ans->n){
        fprintf(stderr, "%s: Matrixs do not have the same size\n",__FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }

    memset(ans->data, 0.0f, sizeof(float)*a->m*b->n);

    float *A=a->data,*B=b->data,*ANS=ans->data;

    //simple ikj but change the pointer
    for (register size_t i = 0; i < a->m; i++)
    {
        B = b->data;
        for (register size_t k = 0; k < a->n; k++)
        {
            ANS = ans->data+(i*b->n);
            for (register size_t j = 0; j < b->n; j++)
            {
                (*(ANS++))=(*A)*(*(B++));
                //ans->data[i * b->n + j] += a->data[i * a->n + k] * b->data[k * b->n + j];
            }
            A++;
        }
    }
    return ans;
}

/**
 * C = A * B (matrix)
 * colA: columns of origial A Matrix
 * (luAx,luAy): left upper position in A
 * sizex, sizey, sizez: matrix multiplication size
*/
void blocking_matmul(const float *A, const size_t colA, const size_t luAx, const size_t luAy,
                     const float *B, const size_t colB, const size_t luBx, const size_t luBy,
                     float *C, const size_t colC, const size_t luCx, const size_t luCy,
                     const size_t sizex, const size_t sizey, const size_t sizez)
{
#ifdef WITH_AVX2
    if (sizex * sizey > 1024 * 1024 && !((sizex & 15) || (sizey & 15) || (sizez & 15)))
    {
        size_t sizex2 = sizex >> 1;
        size_t sizey2 = sizey >> 1;
        size_t sizez2 = sizez >> 1;

        //size to big, divide it into smaller blocks
        blocking_matmul(A, colA, luAx, luAy, B, colB, luBx, luBy, C, colC, luCx, luCy, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx, luAy + sizez2, B, colB, luBx, luBy + sizez2, C, colC, luCx, luCy, sizex2, sizey2, sizez2);

        blocking_matmul(A, colA, luAx, luAy, B, colB, luBx + sizey2, luBy, C, colC, luCx, luCy + sizey2, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx, luAy + sizez2, B, colB, luBx + sizey2, luBy + sizez2, C, colC, luCx, luCy + sizey2, sizex2, sizey2, sizez2);

        blocking_matmul(A, colA, luAx + sizex2, luAy, B, colB, luBx, luBy, C, colC, luCx + sizex2, luCy, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx + sizex2, luAy + sizez2, B, colB, luBx, luBy + sizez2, C, colC, luCx + sizex2, luCy, sizex2, sizey2, sizez2);

        blocking_matmul(A, colA, luAx + sizex2, luAy, B, colB, luBx + sizey2, luBy, C, colC, luCx + sizex2, luCy + sizey2, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx + sizex2, luAy + sizez2, B, colB, luBx + sizey2, luBy + sizez2, C, colC, luCx + sizex2, luCy + sizey2, sizex2, sizey2, sizez2);
        return;
    }

    //similar to improved.
#pragma omp parallel for
    for (register size_t i = 0; i < sizex; i++)
    {
        for (register size_t j = 0; j < sizey; j++)
        {
            //float *sum=(float*)aligned_alloc(256,32);
            float sum[8];
            __m256 ax, bx;
            __m256 c = _mm256_setzero_ps();
            register size_t k;
            for (k = 0; k < sizez; k += 8)
            {
                ax = _mm256_load_ps(A + ((i + luAx) * colA + k + luAy));
                bx = _mm256_load_ps(B + ((j + luBx) * colB + k + luBy));
                c = _mm256_add_ps(c, _mm256_mul_ps(ax, bx));
            }
            _mm256_store_ps(sum, c);
            C[(i + luCx) * colC + j + luCy] += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
            //free(sum);
        }
    }
#else
#ifdef WITH_NEON
    if (sizex * sizey > 1024*1024 && !((sizex & 7) || (sizey & 7) || (sizez & 7)))
    {
        size_t sizex2 = sizex >> 1;
        size_t sizey2 = sizey >> 1;
        size_t sizez2 = sizez >> 1;
        blocking_matmul(A, colA, luAx, luAy, B, colB, luBx, luBy, C, colC, luCx, luCy, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx, luAy + sizez2, B, colB, luBx, luBy + sizez2, C, colC, luCx, luCy, sizex2, sizey2, sizez2);

        blocking_matmul(A, colA, luAx, luAy, B, colB, luBx + sizey2, luBy, C, colC, luCx, luCy + sizey2, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx, luAy + sizez2, B, colB, luBx + sizey2, luBy + sizez2, C, colC, luCx, luCy + sizey2, sizex2, sizey2, sizez2);

        blocking_matmul(A, colA, luAx + sizex2, luAy, B, colB, luBx, luBy, C, colC, luCx + sizex2, luCy, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx + sizex2, luAy + sizez2, B, colB, luBx, luBy + sizez2, C, colC, luCx + sizex2, luCy, sizex2, sizey2, sizez2);

        blocking_matmul(A, colA, luAx + sizex2, luAy, B, colB, luBx + sizey2, luBy, C, colC, luCx + sizex2, luCy + sizey2, sizex2, sizey2, sizez2);
        blocking_matmul(A, colA, luAx + sizex2, luAy + sizez2, B, colB, luBx + sizey2, luBy + sizez2, C, colC, luCx + sizex2, luCy + sizey2, sizex2, sizey2, sizez2);
        return;
    }
#pragma omp parallel for
    for (register size_t i = 0; i < sizex; i++)
    {
        for (register size_t j = 0; j < sizey; j++)
        {
            float sum[4];
            float32x4_t ax, bx;
            float32x4_t c = vdupq_n_f32(0);
            for (register size_t k = 0; k < sizez; k += 4)
            {
                ax = vld1q_f32(A + ((i + luAx) * colA + k + luAy));
                bx = vld1q_f32(B + ((j + luBx) * colB + k + luBy));
                c = vaddq_f32(c, vmulq_f32(ax, bx));
            }
            vst1q_f32(sum, c);
            C[(i + luCx) * colC + j + luCy] += sum[0] + sum[1] + sum[2] + sum[3];
        }
    }
#endif
#endif
}
bool matmul_improved_blocking(const Mat *a, const Mat *b, Mat *ans)
{
    CHECK(a,1)
    CHECK(b,2)
    CHECK(ans,3)
    if (a->n != b->m || a->m!=ans->m || b->n!=ans->n){
        fprintf(stderr, "%s: Matrixs do not have the same size\n",__FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }

    float *C;
#ifdef WITH_AVX2
    C = (float *)(aligned_alloc(256, sizeof(float) * b->m * b->n));
#else
#ifdef WITH_NEON
    C = (float *)(aligned_alloc(128, sizeof(float) * b->m * b->n));
#else
    C = (float *)malloc(sizeof(float) * b->m * b->n);
#endif
#endif
    trans(b,C);
    blocking_matmul(a->data, a->n, 0, 0, C, b->m, 0, 0, ans->data, b->n, 0, 0, a->m, b->n, a->n);
    return ans;
}

bool matmul_improved(const Mat *a, const Mat *b, Mat *ans)
{
    CHECK(a,1)
    CHECK(b,2)
    CHECK(ans,3)
    if (a->n != b->m || a->m!=ans->m || b->n!=ans->n){
        fprintf(stderr, "%s: Matrixs do not have the same size\n",__FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }

#ifdef WITH_AVX2
    if (a->m % 8 || a->n % 8 || b->n % 8)
    {
        fprintf(stderr, "%s: Can not use AVX.\n", __FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }

    float *C;
    C = (float *)(aligned_alloc(256, sizeof(float) * b->m * b->n));
    trans(b,C);
    memset(ans->data, 0.0f, sizeof(float)*a->m*b->n);

#pragma omp parallel for
    for (register size_t i = 0; i < a->m; i++)
    {
        for (register size_t j = 0; j < b->n; j++)
        {
            float *sum=(float*)aligned_alloc(256,sizeof(float)*8);
            __m256 ax, bx;
            __m256 c = _mm256_setzero_ps();
            for (register size_t k = 0; k < b->m; k += 8)
            {
                ax = _mm256_load_ps(a->data + (i * b->m + k));
                bx = _mm256_load_ps(C + (j * b->m + k));
                c = _mm256_add_ps(c, _mm256_mul_ps(ax, bx));
            }
            _mm256_store_ps(sum, c);
            ans->data[i * b->n + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
            free(sum);
        }
    }
    free(C);
    return true;
#else
#ifdef WITH_NEON

    if (a->m % 4 || a->n % 4 || b->n % 4)
    {
        fprintf(stderr, "%s: Can not use AVX.\n", __FUNCTION__);
        fprintf(stderr, "1st: (%ld, %ld)\n",a->m,a->n);
        fprintf(stderr, "2st: (%ld, %ld)\n",b->m,b->n);
        fprintf(stderr, "3st: (%ld, %ld)\n",ans->m,ans->n);
        return false;
    }

    memset(ans->data, 0.0f, sizeof(float)*a->m*b->n);

    float *c;
    c = (float *)(aligned_alloc(128, sizeof(float) * b->m * b->n));
    trans(b,c);
#pragma omp parallel for
    for (register size_t i = 0; i < a->m; i++)
    {
        for (register size_t j = 0; j < b->n; j++)
        {
            float sum[4];
            float32x4_t A, B;
            float32x4_t C = vdupq_n_f32(0);

            for (register size_t k = 0; k < a->n; k += 4)
            {
                A = vld1q_f32(a->data + (i * a->n + k));
                B = vld1q_f32(c + (j * a->n + k));
                C = vaddq_f32(C, vmulq_f32(A, B));
            }
            vst1q_f32(sum, C);
            ans->data[i * b->n + j] = sum[0] + sum[1] + sum[2] + sum[3];
        }
    }
    free(c);
    return ans;
#else
    printf("No SIMD support.\n");
    return false;
#endif
#endif
}

//Matrix transpose
void trans(const Mat *a, float *dist)
{
    //Here is not simple memcpy
    if(!dist) return;
    if(!a) return;
    register size_t i, j;
    for (i = 0; i < a->n; i++)
    {
        for (j = 0; j < a->m; j++)
        {
            dist[i * a->m + j] = a->data[j * a->n + i];
        }
    }
}
void mat_free(Mat *a)
{
    if(a){
        free(a->data);
        free(a);
    }
}

// malloc a new matrix
Mat* newmat_aligned(const size_t m, const size_t n, const float *src)
{
    Mat *ans = (Mat *)malloc(sizeof(Mat));
    ans->m = m;
    ans->n = n;
    size_t sz = sizeof(float) * n * m;
#ifdef WITH_AVX2
    ans->data = (float *)(aligned_alloc(256, sz));
#else
#ifdef WITH_NEON
    ans->data = (float *)(aligned_alloc(128, sz));
#else
    ans->data = (float *)malloc(sz);
#endif
#endif
    if(!ans->data){
        fprintf(stderr, "%s: cannot malloc. No Memory.",__FUNCTION__);
        return NULL;
    }
    if(src) memcpy(ans->data, src, sz);
    else memset(ans->data,0.0f,sz);
    return ans;
}
void moutput(const Mat *a)
{
    printf("Mat %ld*%ld:\n", a->m, a->n);
    if (a->m >= 0 && a->n >= 0)
        for (size_t i = 0; i < a->m; i++)
        {
            for (size_t j = 0; j < a->n; j++)
            {
                printf("%0.1f ", a->data[i * a->n + j]);
            }
            printf("\n");
        }
    printf("\n");
}
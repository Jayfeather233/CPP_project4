#include "matmul.h"
#include <stdlib.h>

#ifdef WITH_AVX2
#include <immintrin.h>
#endif 

#ifdef WITH_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

Mat matmul_plain(const Mat a,const Mat b){
    Mat ans;
    if(a.n!=b.m){
        printf("Can not multiply.\n");
        ans.n=ans.m=-1;
        return ans;
    }
    ans.m=a.m;
    ans.n=b.n;
    ans.data=(float*)malloc(sizeof(float)*a.m*b.n);
    size_t kk;
    for(size_t i=0;i<a.m;i++){
        for(size_t j=0;j<b.n;j++){
            kk=i*b.n+j;
            ans.data[kk]=0;
            for(size_t k=0;k<a.n;k++){
                ans.data[kk]+=a.data[i*a.n+k]*b.data[k*b.n+j];
            }
        }
    }
    return ans;
}
inline float vectormul(float *p1,float *p2,size_t nSize){
#ifdef WITH_AVX2
    float sum[8];
    __m256 c = _mm256_setzero_ps();
    for (register size_t i = 0; i < nSize; i+=8){
        c =  _mm256_add_ps(c, _mm256_mul_ps(_mm256_load_ps(p1 + i), _mm256_load_ps(p2 + i)));
    }
    _mm256_store_ps(sum, c);
    return sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
#else
#ifdef WITH_NEON
    float sum[4];
    float32x4_t a, b;
    float32x4_t c = vdupq_n_f32(0);

    for (register size_t i = 0; i < nSize; i+=4)
    {
        a = vld1q_f32(p1 + i);
        b = vld1q_f32(p2 + i);
        c = vaddq_f32(c, vmulq_f32(a, b));
    }
    vst1q_f32(sum, c);
    return sum[0]+sum[1]+sum[2]+sum[3];
#else
    return 0;
#endif
#endif
}
Mat matmul_improved(const Mat a,const Mat b){
    register size_t i,j;
#ifdef WITH_AVX2
#ifdef _OPENMP
    Mat ans;
    if(a.n!=b.m){
        printf("Can not multiply.\n");
        ans.n=ans.m=-1;
        return ans;
    }
    if(a.m%8 || a.n%8 || b.n%8){
        printf("Can not use AVX.\n");
        ans.n=ans.m=-1;
        return ans;
    }
    ans.m=a.m;
    ans.n=b.n;
    ans.data=(float*)(aligned_alloc(256,sizeof(float)*a.m*b.n));
    float *c=trans(b);
    #pragma omp parallel for
    for(i=0;i<a.m;i++){
        for(j=0;j<b.n;j++){
            ans.data[i*b.n+j]=vectormul(a.data+i*a.n,c+j*b.m,a.n);
        }
    }
    free(c);
    return ans;
#else
    printf("No OpenMP support.\n");
    Mat ans;
    ans.n=ans.m=-1;
    return ans;
#endif
#else
#ifdef WITH_NEON
#ifdef _OPENMP
    Mat ans;
    if(a.n!=b.m){
        printf("Can not multiply.\n");
        ans.n=ans.m=-1;
        return ans;
    }
    if(a.m%4 || a.n%4 || b.n%4){
        printf("Can not use NEON.\n");
        ans.n=ans.m=-1;
        return ans;
    }
    ans.m=a.m;
    ans.n=b.n;
    ans.data=(float*)(aligned_alloc(128,sizeof(float)*a.m*b.n));
    float *c=trans(b);
    #pragma omp parallel for
    for(i=0;i<a.m;i++){
        for(j=0;j<b.n;j++){
            ans.data[i*b.n+j]=vectormul(a.data+i*a.n,c+j*b.m,a.n);
        }
    }
    free(c);
    return ans;
#else
    printf("No OpenMP support.\n");
    Mat ans;
    ans.n=ans.m=-1;
    return ans;
#endif
#else
    printf("No AVX or NEON support.\n");
    Mat ans;
    ans.n=ans.m=-1;
    return ans;
#endif
#endif
}

float *trans(const Mat a){
    float* ans;
#ifdef WITH_AVX2
    ans=(float*)(aligned_alloc(256,sizeof(float)*a.m*a.n));
#else
#ifdef WITH_NEON
    ans=(float*)(aligned_alloc(128,sizeof(float)*a.m*a.n));
#else
    ans=(float*)malloc(sizeof(float)*a.m*a.n);
#endif
#endif
    register size_t i,j;
    for(i=0;i<a.n;i++){
        for(j=0;j<a.m;j++){
            ans[i*a.m+j]=a.data[j*a.n+i];
        }
    }
    return ans;
}
void mat_free(Mat a){
    free(a.data);
}
Mat newmat_aligned(const size_t m,const size_t n, float *src){
    Mat ans;
    ans.m=m;
    ans.n=n;
    size_t sz=n*m;
#ifdef WITH_AVX2
    ans.data=(float*)(aligned_alloc(256,sizeof(float)*sz));
    for(size_t i=0;i<sz;i+=8){
        _mm256_store_ps(ans.data+i, _mm256_load_ps(src+i));
    }
#else
#ifdef WITH_NEON
    ans.data=(float*)(aligned_alloc(128,sizeof(float)*sz));
    for(size_t i=0;i<sz;i+=4){
        vst1q_f32(ans.data+i, vld1q_f32(src+i));
    }
#else
    ans.data=(float*)malloc(sizeof(float)*sz);
    for(size_t i=0;i<sz;i++){
        ans.data[i]=src[i];
    }
#endif
#endif
    return ans;
}
void moutput(const Mat a){
    printf("Mat %ld*%ld:\n",a.m,a.n);
    if(a.m>=0&&a.n>=0)
    for(int i=0;i<a.m;i++){
        for(int j=0;j<a.n;j++){
            printf("%0.1f ",a.data[i*a.n+j]);
        }
        printf("\n");
    }
    printf("\n");
}
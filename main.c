#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cblas.h>
#include "matmul.h"

//This code is just for test.

#define TIME_START clock_gettime(CLOCK_REALTIME,&start);
#define TIME_END(NAME) clock_gettime(CLOCK_REALTIME,&end);\
        printf("%s: %ldms\n",NAME, (end.tv_sec-start.tv_sec)*1000+(end.tv_nsec-start.tv_nsec)/1000000);

int N,M;

float *A,*B,*C;

int main(){
    struct timespec start,end;
    int mod1,mod2,mod3;//mod1:预热 mod2:比较 mod3:输出

    printf("Input:M N Preheat:0/1 compare:0/1 output:0/1\n");
    if(scanf("%d%d%d%d%d",&N,&M,&mod1,&mod2,&mod3)==EOF) return 1;

    srand(time(0));

    /// generate random data 

#ifdef WITH_AVX2
    A=(float*)(aligned_alloc(256,sizeof(float)*M*N));
    B=(float*)(aligned_alloc(256,sizeof(float)*M*N));
    C=(float*)(aligned_alloc(256,sizeof(float)*M*N));
#else
#ifdef WITH_NEON
    A=(float*)(aligned_alloc(128,sizeof(float)*M*N));
    B=(float*)(aligned_alloc(128,sizeof(float)*M*N));
    C=(float*)(aligned_alloc(128,sizeof(float)*M*N));
#else
    A=(float*)malloc(sizeof(float)*M*N);
    B=(float*)malloc(sizeof(float)*M*N);
    C=(float*)malloc(sizeof(float)*M*N);
#endif
#endif

    TIME_START
    for(int i=0;i<N*M;i++){
        A[i]=rand()%10;
        B[i]=rand()%10;
    }
    Mat a=newmat_aligned(N,M,A);
    Mat b=newmat_aligned(N,M,B);
    TIME_END("generate");

    Mat c1,c2;
    if(mod2){//The plain method
        if(mod1){
            c1=matmul_plain(a,b);
            mat_free(c1);
            c1=matmul_plain(a,b);
            mat_free(c1);
        }
        TIME_START
        c1=matmul_plain(a,b);
        TIME_END("plain")
    }

    //improved method
    if(mod1){
        c2=matmul_improved(a,b);
        mat_free(c2);
        c2=matmul_improved(a,b);
        mat_free(c2);
    }
    TIME_START
    c2=matmul_improved(a,b);
    TIME_END("improved");
    
    if(mod1){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,N,1.0f,A,N,B,N,0,C,N);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,N,1.0f,A,N,B,N,0,C,N);
    }
    TIME_START
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,N,1.0f,A,N,B,N,0,C,N);
    TIME_END("OpenBLAS")
    
    if(mod3){//output
        if(mod2) moutput(c1);
        moutput(c2);
        for(int i=0;i<N*M;i++){
            printf("%.2f ",C[i]);
        }
    }
    if(mod2){//compare
        int flg=1;
        for(int i=0;i<N*M&&flg;i++){
            flg&=c1.data[i]==c2.data[i];
            flg&=c2.data[i]==C[i];
        }
        printf("Correct?: %c\n",flg?'Y':'N');
    }
    if(mod2) mat_free(c1);
    mat_free(c2);
    mat_free(a);
    mat_free(b);
    return 0;
}
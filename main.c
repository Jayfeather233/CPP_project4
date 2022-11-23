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


int rand_state;
void set_seed(int seed){
    rand_state = seed;
}
int my_rand(){
    return rand_state = (1ll*rand_state*(1000000007)+99907) % 998244353;
}

int main(){
    struct timespec start,end;

    printf("Input:N\n");
    if(scanf("%d",&N)==EOF) return 1;
    M=N;

    //srand(time(0));
    set_seed(time(0));

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
        A[i]=my_rand()%10;
        B[i]=my_rand()%10;
    }
    Mat a=newmat_aligned(N,M,A);
    Mat b=newmat_aligned(N,M,B);
    TIME_END("generate");

    Mat c1,c2;
    // TIME_START
    // c1=matmul_plain_ijk(a,b);
    // TIME_END("plain")
    // mat_free(c1);

    // c1=matmul_plain_omp_ikj(a,b);
    // mat_free(c1);
    TIME_START
    c1=matmul_plain_omp_ikj(a,b);
    TIME_END("plain_omp_ikj")
    mat_free(c1);

    c1=matmul_improved_blocking(a,b);
    mat_free(c1);
    TIME_START
    c1=matmul_improved_blocking(a,b);
    TIME_END("improved_blocking")

    //improved method
    TIME_START
    c2=matmul_improved(a,b);
    TIME_END("improved");
    
    TIME_START
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,N,1.0f,A,M,B,M,0,C,M);
    TIME_END("OpenBLAS")
    
    //moutput(c2);
    
    int flg=1;
    for(int i=0;i<N*M&&flg;i++){
        flg&=c1.data[i]==c2.data[i];
        flg&=c2.data[i]==C[i];
    }
    printf("Correct?: %c\n",flg?'Y':'N');
    
    mat_free(c1);
    mat_free(c2);
    mat_free(a);
    mat_free(b);
    free(A);
    free(B);
    free(C);
    return 0;
}
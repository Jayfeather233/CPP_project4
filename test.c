#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cblas.h>
#include "matmul.h"

//This code is just for test.

#define TIME_START clock_gettime(CLOCK_REALTIME,&start);
#define TIME_END(NAME) clock_gettime(CLOCK_REALTIME,&end);\
        printf("%s: %ldms\n",NAME, (end.tv_sec-start.tv_sec)*1000+(end.tv_nsec-start.tv_nsec)/1000000);

#define mx(a,b) ((a)>(b) ? (a) : (b))

#define eq(a,b) (a>b ? a-b< mx(a,1)*1e-3 : b-a< mx(b,1)*1e-3)

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
        A[i]=1.0f*my_rand()/998244353;
        B[i]=1.0f*my_rand()/998244353;
    }
    Mat *a=NULL,*b=NULL;
    a=newmat_aligned(N,M,A);
    b=newmat_aligned(M,N,B);
    TIME_END("generate");

    Mat *c1=NULL,*c2=NULL;

    // TIME_START
    // c1=newmat_aligned(N,N,NULL);
    // matmul_plain_ijk(a,b,c1);
    // TIME_END("plain_ijk")
    // mat_free(c1);

    // TIME_START
    // c1=newmat_aligned(N,N,NULL);
    // matmul_plain_ikj(a,b,c1);
    // TIME_END("plain_ikj")
    // mat_free(c1);

    // TIME_START
    // c1=newmat_aligned(N,N,NULL);
    // matmul_plain_omp_ikj(a,b,c1);
    // TIME_END("plain_omp_ikj")

    //improved method
    // TIME_START
    // c1=newmat_aligned(N,N,NULL);
    // matmul_improved(a,b,c1);
    // TIME_END("improved");
    //mat_free(c2);

    TIME_START
    c2=newmat_aligned(N,N,NULL);
    matmul_improved_blocking(a,b,c2);
    TIME_END("improved_blocking")
    
    TIME_START
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,N,1.0f,A,M,B,M,0,C,M);
    TIME_END("OpenBLAS")
    
    //moutput(c2);
    
    //correctness check
    int flg=1;
    float maxdiff=0.0f;
    for(int i=0;i<N*M&&flg;i++){
        //flg&=eq(c1->data[i],c2->data[i]);
        flg&=eq(c2->data[i],C[i]);

        maxdiff=mx(mx(maxdiff,c2->data[i]-C[i]),C[i]-c2->data[i]);
    }
    for(int i=0;i<=10;i++) printf("%0.6f ",c2->data[i]);printf("\n");
    for(int i=0;i<=10;i++) printf("%0.6f ",C[i]);printf("\n");
    printf("Correct?: %c\n",flg?'Y':'N');
    printf("Maxdiff: %0.8f\n",maxdiff);
    
    mat_free(c1);
    mat_free(c2);
    mat_free(a);
    mat_free(b);
    free(A);
    free(B);
    free(C);
    return 0;
}
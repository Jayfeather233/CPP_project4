#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmul.h"

#define TIME_START clock_gettime(CLOCK_REALTIME,&start);
#define TIME_END(NAME) clock_gettime(CLOCK_REALTIME,&end);\
        printf("%s: %ldms\n",NAME, (end.tv_sec-start.tv_sec)*1000+(end.tv_nsec-start.tv_nsec)/1000000);

int N,M;

int main(){
    struct timespec start,end;
    int mod1,mod2,mod3;//mod1:预热 mod2:比较 mod3:输出

    printf("Input:M N Preheat:0/1 compare:0/1 output:0/1");
    if(scanf("%d%d%d%d%d",&N,&M,&mod1,&mod2,&mod3)==EOF) return 1;

    srand(time(0));
    TIME_START
    Mat a=newmat_aligned(N,M);
    Mat b=newmat_aligned(N,M);
    for(int i=0;i<N*M;i++){
        a.data[i]=rand()%10;
        b.data[i]=rand()%10;
    }
    TIME_END("generate");

    Mat c1,c2;
    if(mod2){
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

    if(mod1){
        c2=matmul_improved(a,b);
        mat_free(c2);
        c2=matmul_improved(a,b);
        mat_free(c2);
    }
    TIME_START
    c2=matmul_improved(a,b);
    TIME_END("improved");
    
    if(mod3){
        moutput(c1);
        moutput(c2);
    }
    if(mod2){
        int flg=1;
        for(int i=0;i<N*M&&flg;i++){
            flg&=c1.data[i]==c2.data[i];
        }
        printf("Correct?: %c\n",flg?'Y':'N');
    }
    if(mod2) mat_free(c1);
    mat_free(c2);
    mat_free(a);
    mat_free(b);
    return 0;
}
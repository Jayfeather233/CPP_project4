#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <chrono>
#include "matmul.h"
//duration = ((double)(stop-start))/CLK_TCK;

#define TIME_START clock_gettime(CLOCK_REALTIME,&start);
#define TIME_END(NAME) clock_gettime(CLOCK_REALTIME,&end);printf("%s: %ldms\n",NAME, (end.tv_sec-start.tv_sec)*1000+(end.tv_nsec-start.tv_nsec)/1000000);

// #define TIME_START start=std::chrono::steady_clock::now();
// #define TIME_END(NAME) end=std::chrono::steady_clock::now(); \
//              duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();\
//              cout<<(NAME)<<": result="<<result \
//              <<", duration = "<<duration<<"ms"<<endl;



int N,M;

int main(){
    
    // auto start = std::chrono::steady_clock::now();
    // auto end = std::chrono::steady_clock::now();
    // auto duration = 0L;

    struct timespec start,end;

    int mod1,mod2,mod3;//mod1:预热 mod2:比较 mod3:输出

    if(scanf("%d%d%d%d%d",&N,&M,&mod1,&mod2,&mod3)==EOF) return 1;
    srand(time(0));
    Mat a=newmat_aligned(N,M);
    Mat b=newmat_aligned(N,M);
    for(int i=0;i<N*M;i++){
        a.data[i]=rand()%10;
        b.data[i]=rand()%10;
    }
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
    if(mod2) mat_free(c1);
    mat_free(c2);
    mat_free(a);
    mat_free(b);
    return 0;
}
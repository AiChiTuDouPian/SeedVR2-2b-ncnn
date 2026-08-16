// test_awa_unit.cpp — AwaVk 单元自测（极小合成输入，无需模型权重）
// 构造 T=1 H=2 W=6 (12 token) 网格，2 个窗口，TXT=4；随机 qkv/freq/norm。
// 跑 AwaVk，打印输出尺寸与若干值，并用一份 CPU 参考（同算法）对拍 cos。
#include "awa_vk.h"
#include "dit_vk.h"
#include <ncnn/gpu.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <string>

static const int HEADS=20, HEAD_D=128, DIM=2560, QKV=7680, ROPE_ROT=126;
static const float EPS=1e-5f;

static void qk_norm(const float* x, const float* w, float* o){
    for(int h=0;h<HEADS;h++){const float* xp=x+h*HEAD_D;float* op=o+h*HEAD_D;
        float s=0;for(int d=0;d<HEAD_D;d++)s+=xp[d]*xp[d];float r=sqrtf(s/HEAD_D+EPS);
        for(int d=0;d<HEAD_D;d++)op[d]=xp[d]/r*w[d];}
}
static void rope(const float* fr, const float* q, float* o){
    for(int k=0;k<ROPE_ROT;k++){float c=cosf(fr[k]),s=sinf(fr[k]);float rh=(k%2==0)?-q[k+1]:q[k-1];o[k]=q[k]*c+rh*s;}
    for(int k=ROPE_ROT;k<HEAD_D;k++)o[k]=q[k];
}

// CPU 参考：raw vqkv/tqkv -> vid_attn/txt_attn（与 awa_forward 同算法）
static void cpu_awa(const std::vector<float>& vqkv, const std::vector<float>& tqkv,
                    const Win& win, const std::vector<float>& nqv, const std::vector<float>& nkv,
                    const std::vector<float>& nqt, const std::vector<float>& nkt,
                    int Lv, int TXT, std::vector<float>& vout, std::vector<float>& tout){
    fprintf(stderr,"[cpu_awa] start\n");
    int H=win.h,Wd=win.w; int nwin=win.nwin;
    std::vector<int> cumf(nwin+1,0); std::vector<int> vidx;
    for(int wi=0;wi<nwin;wi++){int f=(win.en[wi]-win.st[wi])*(win.eh[wi]-win.sh[wi])*(win.ew[wi]-win.sw[wi]);
        cumf[wi+1]=cumf[wi]+f;
        for(int lt=win.st[wi];lt<win.en[wi];lt++)for(int lh=win.sh[wi];lh<win.eh[wi];lh++)for(int lw=win.sw[wi];lw<win.ew[wi];lw++)
            vidx.push_back(((lt*H)+lh)*Wd+lw);}
    int sumf=(int)vidx.size();
    std::vector<float> vq(Lv*DIM),vk(Lv*DIM),vv(Lv*DIM),tq(TXT*DIM),tk(TXT*DIM),tv(TXT*DIM);
    for(int i=0;i<Lv;i++){qk_norm(&vqkv[i*QKV],nqv.data(),&vq[i*DIM]);qk_norm(&vqkv[i*QKV+2560],nkv.data(),&vk[i*DIM]);memcpy(&vv[i*DIM],&vqkv[i*QKV+5120],DIM*4);}
    for(int i=0;i<TXT;i++){qk_norm(&tqkv[i*QKV],nqt.data(),&tq[i*DIM]);qk_norm(&tqkv[i*QKV+2560],nkt.data(),&tk[i*DIM]);memcpy(&tv[i*DIM],&tqkv[i*QKV+5120],DIM*4);}
    std::vector<float> vout_win(sumf*DIM), tout_win(nwin*TXT*DIM);
    float scale=1.f/sqrtf((float)HEAD_D);
    for(int wi=0;wi<nwin;wi++){int f=cumf[wi+1]-cumf[wi];int S=f+TXT;
        for(int h=0;h<HEADS;h++){std::vector<float> qseg(S*HEAD_D),kseg(S*HEAD_D),vseg(S*HEAD_D);
            for(int j=0;j<f;j++){int tok=(int)vidx[cumf[wi]+j];memcpy(&qseg[j*HEAD_D],&vq[tok*DIM+h*HEAD_D],HEAD_D*4);memcpy(&kseg[j*HEAD_D],&vk[tok*DIM+h*HEAD_D],HEAD_D*4);memcpy(&vseg[j*HEAD_D],&vv[tok*DIM+h*HEAD_D],HEAD_D*4);}
            for(int ii=0;ii<TXT;ii++){int s=f+ii;memcpy(&qseg[s*HEAD_D],&tq[ii*DIM+h*HEAD_D],HEAD_D*4);memcpy(&kseg[s*HEAD_D],&tk[ii*DIM+h*HEAD_D],HEAD_D*4);memcpy(&vseg[s*HEAD_D],&tv[ii*DIM+h*HEAD_D],HEAD_D*4);}
            // rope
            for(int a=0;a<S;a++){bool qv=a<f;int tb;const float* fr;float* qp;
                if(qv){int g=cumf[wi]+a;fr=&win.vid_freq[g*ROPE_ROT];qp=&qseg[a*HEAD_D];}
                else{int ii=a-f;fr=&win.txt_freq[(wi*TXT+ii)*ROPE_ROT];qp=&qseg[a*HEAD_D];}
                float tmp[HEAD_D];rope(fr,qp,tmp);memcpy(qp,tmp,HEAD_D*4);}
            for(int b=0;b<S;b++){bool kv=b<f;int tb;const float* fr;float* kp;
                if(kv){int g=cumf[wi]+b;fr=&win.vid_freq[g*ROPE_ROT];kp=&kseg[b*HEAD_D];}
                else{int ii=b-f;fr=&win.txt_freq[(wi*TXT+ii)*ROPE_ROT];kp=&kseg[b*HEAD_D];}
                float tmp[HEAD_D];rope(fr,kp,tmp);memcpy(kp,tmp,HEAD_D*4);}
            std::vector<float> sc(S*S);float mx=-1e30f;
            for(int a=0;a<S;a++)for(int b=0;b<S;b++){float dot=0;const float* qa=&qseg[a*HEAD_D];const float* kb=&kseg[b*HEAD_D];for(int d=0;d<HEAD_D;d++)dot+=qa[d]*kb[d];dot*=scale;sc[a*S+b]=dot;if(dot>mx)mx=dot;}
            for(int a=0;a<S;a++){float sm=0;std::vector<float> p(S);for(int b=0;b<S;b++){float e=expf(sc[a*S+b]-mx);p[b]=e;sm+=e;}for(int b=0;b<S;b++)p[b]/=sm;
                float o[HEAD_D]={0};for(int b=0;b<S;b++){const float* vb=&vseg[b*HEAD_D];for(int d=0;d<HEAD_D;d++)o[d]+=p[b]*vb[d];}
                if(a<f)memcpy(&vout_win[(cumf[wi]+a)*DIM+h*HEAD_D],o,HEAD_D*4);else memcpy(&tout_win[(wi*TXT+(a-f))*DIM+h*HEAD_D],o,HEAD_D*4);}
        }
    }
    tout.assign(TXT*DIM,0.f);
    for(int wi=0;wi<nwin;wi++)for(int ii=0;ii<TXT;ii++)for(int d=0;d<DIM;d++)tout[ii*DIM+d]+=tout_win[(wi*TXT+ii)*DIM+d];
    for(size_t i=0;i<tout.size();i++)tout[i]/=(float)nwin;
    vout.assign(Lv*DIM,0.f);
    for(int wi=0;wi<nwin;wi++){int local=0;for(int lt=win.st[wi];lt<win.en[wi];lt++)for(int lh=win.sh[wi];lh<win.eh[wi];lh++)for(int lw=win.sw[wi];lw<win.ew[wi];lw++){int idx=((lt*H)+lh)*Wd+lw;memcpy(&vout[idx*DIM],&vout_win[(cumf[wi]+local)*DIM],DIM*4);local++;}}
    fprintf(stderr,"[cpu_awa] end\n");
}

int main(){
    if(ncnn::create_gpu_instance()!=0){fprintf(stderr,"[FAIL] gpu\n");return 1;}
    ncnn::VulkanDevice* vkdev=ncnn::get_gpu_device(0);
    AwaVk awa; if(!awa.init(vkdev,"models/m5/awa.spv"))return 1;

    std::mt19937 rng(123); auto rnd=[&](){return (float)rng()/rng.max()*2-1;};
    int Lv=12, TXT=4;
    Win win; win.t=1; win.h=2; win.w=6; win.nwin=2; win.txt_len=TXT;
    // 两个窗口沿 W 轴切分：窗口0 覆盖 w∈[0,3)，窗口1 覆盖 w∈[3,6)；各 (1*2*3)=6 token
    win.st={0,0}; win.en={1,1}; win.sh={0,0}; win.eh={2,2}; win.sw={0,3}; win.ew={3,6};
    int sumf=(1)*(2)*(3)*2; // 12
    win.vid_freq.assign(sumf*ROPE_ROT,0.f); for(auto&v:win.vid_freq)v=rnd()*0.1f;
    win.txt_freq.assign(win.nwin*TXT*ROPE_ROT,0.f); for(auto&v:win.txt_freq)v=rnd()*0.1f;
    std::vector<float> nqv(HEAD_D),nkv(HEAD_D),nqt(HEAD_D),nkt(HEAD_D);
    for(auto&v:nqv)v=1.f;for(auto&v:nkv)v=1.f;for(auto&v:nqt)v=1.f;for(auto&v:nkt)v=1.f; // 单位 norm
    std::vector<float> vqkv(Lv*QKV),tqkv(TXT*QKV);
    for(auto&v:vqkv)v=rnd()*0.1f; for(auto&v:tqkv)v=rnd()*0.1f;

    std::vector<float> vout,tout;
    if(!awa.forward(vqkv,tqkv,win,nqv,nkv,nqt,nkt,Lv,TXT,vout,tout)){fprintf(stderr,"[FAIL] AwaVk.forward\n");return 1;}
    fprintf(stderr,"[unit] AwaVk done vout.w=%zu tout.w=%zu\n",vout.size(),tout.size());
    auto maxabs=[&](const std::vector<float>&a){double m=0;for(float v:a)m=std::max(m,(double)fabs(v));return m;};
    fprintf(stderr,"[dbg] vout maxabs=%.4e  tout maxabs=%.4e\n", maxabs(vout), maxabs(tout));
    fprintf(stderr,"[dbg] vout[0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n", vout[0],vout[1],vout[2],vout[3],vout[4],vout[5]);
    fprintf(stderr,"[dbg] cvout[0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n", 0.f,0.f,0.f,0.f,0.f,0.f);

    std::vector<float> cvout,ctout; cpu_awa(vqkv,tqkv,win,nqv,nkv,nqt,nkt,Lv,TXT,cvout,ctout);
    fprintf(stderr,"[unit] cpu_awa done  cvout.w=%zu ctout.w=%zu\n", cvout.size(), ctout.size());
    fprintf(stderr,"[dbg] cvout maxabs=%.4e  ctout maxabs=%.4e\n", maxabs(cvout), maxabs(ctout));
    fprintf(stderr,"[dbg] cvout[0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n", cvout[0],cvout[1],cvout[2],cvout[3],cvout[4],cvout[5]);
    auto cos=[&](const std::vector<float>&a,const std::vector<float>&b){double d=0,na=0,nb=0;for(size_t i=0;i<a.size();i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(sqrt(na)*sqrt(nb)+1e-12);};
    fprintf(stderr,"[unit] cos(vid)=%.6f cos(txt)=%.6f\n",cos(vout,cvout),cos(tout,ctout));
    if(cos(vout,cvout)<0.999||cos(tout,ctout)<0.999){fprintf(stderr,"[FAIL] cos 过低\n");return 1;}
    fprintf(stderr,"[unit] PASS ✅\n");
    ncnn::destroy_gpu_instance();
    return 0;
}

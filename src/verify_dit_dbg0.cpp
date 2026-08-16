// 调试：逐步复现 block 0 并与 m5_debug_block0.py 的 dbg_*.bin 对比
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cstdlib>

static const int HEADS=20, HEAD_D=128, DIM=2560, QKV=HEADS*HEAD_D*3;
static const float EPS=1e-5f;
static const int TXT_LEN=8;
static const int ROPE_ROT=126, ROPE_HALF=63;

static inline float fp16_to_float(uint16_t h){
    uint32_t sign=(h>>15)&1, expo=(h>>10)&0x1f, mant=h&0x3ff; uint32_t f;
    if(expo==0){ if(mant==0) f=sign<<31; else { int e=127-15+1; while((mant&0x400)==0){mant<<=1;e--;} mant&=0x3ff; f=(sign<<31)|(e<<23)|(mant<<13);} }
    else if(expo==0x1f){ f=(sign<<31)|0x7f800000|(mant<<13); }
    else { f=(sign<<31)|((expo-15+127)<<23)|(mant<<13); }
    return reinterpret_cast<float&>(f);
}
static std::vector<float> load_raw(const char* p, std::vector<int64_t>& sh){
    std::ifstream f(p,std::ios::binary); if(!f){fprintf(stderr,"[FAIL]%s\n",p);exit(1);}
    int64_t n; f.read((char*)&n,8); sh.resize(n); for(auto&d:sh)f.read((char*)&d,8);
    int64_t t=1; for(auto d:sh)t*=d; std::vector<float> d(t); f.read((char*)d.data(),t*4); return d;
}
static std::vector<float> load_raw_f(const char* p){std::vector<int64_t> s;return load_raw(p,s);}
struct LinearRaw{int in=0,out=0;std::vector<float>W,b;
    void run(const float*x,float*y,int Ln)const{
        #pragma omp parallel for
        for(int t=0;t<Ln;t++){const float*xi=x+t*in;float*yo=y+t*out;
            for(int o=0;o<out;o++){const float*wi=&W[o*in];float a=0;for(int i=0;i<in;i++)a+=xi[i]*wi[i];yo[o]=a+(b.empty()?0.f:b[o]);}}}
    std::vector<float>batch(const std::vector<float>&x,int Ln){std::vector<float>y(Ln*out);run(x.data(),y.data(),Ln);return y;}};
static LinearRaw load_linear(const std::string&base,bool hb){
    std::ifstream pf((base+".param").c_str());std::string line;int no=0,to=0;
    while(std::getline(pf,line)){if(line.find("InnerProduct")!=std::string::npos){std::istringstream ss(line);std::string tk;
        while(ss>>tk){if(tk.rfind("0=",0)==0)no=std::stoi(tk.substr(2));if(tk.rfind("2=",0)==0)to=std::stoi(tk.substr(2));}}}
    int in=to/no;LinearRaw L;L.in=in;L.out=no;
    std::ifstream bf((base+".bin").c_str(),std::ios::binary);uint32_t tag;bf.read((char*)&tag,4);
    int n=no*in;std::vector<uint16_t>hw(n);bf.read((char*)hw.data(),n*2);L.W.resize(n);for(int i=0;i<n;i++)L.W[i]=fp16_to_float(hw[i]);
    if(hb){L.b.resize(no);bf.read((char*)L.b.data(),no*4);}return L;}
static float cosine(const std::vector<float>&a,const std::vector<float>&b){
    double dot=0,na=0,nb=0;size_t n=a.size()<b.size()?a.size():b.size();
    for(size_t i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return(float)(dot/(sqrt(na)*sqrt(nb)+1e-12));}
static float maxd(const std::vector<float>&a,const std::vector<float>&b){float m=0;size_t n=a.size()<b.size()?a.size():b.size();for(size_t i=0;i<n;i++)m=std::max(m,std::fabs(a[i]-b[i]));return m;}
static void rmsnorm(const float*x,int n,const float*w,float*o){double s=0;for(int i=0;i<n;i++)s+=(double)x[i]*x[i];float r=1.f/sqrtf((float)(s/n)+EPS);for(int i=0;i<n;i++)o[i]=x[i]*r*(w?w[i]:1.f);}
static void qk_norm(const float*x,const float*w,float*o){for(int h=0;h<HEADS;h++){const float*xp=x+h*HEAD_D;float*op=o+h*HEAD_D;float sq=0;for(int d=0;d<HEAD_D;d++)sq+=xp[d]*xp[d];float r=sqrtf(sq/HEAD_D+EPS);for(int d=0;d<HEAD_D;d++)op[d]=xp[d]/r*w[d];}}
static void apply_rope(const float*fr,const float*q,float*o){for(int k=0;k<ROPE_ROT;k++){float c=cosf(fr[k]),s=sinf(fr[k]);float rh=(k%2==0)?-q[k+1]:q[k-1];o[k]=q[k]*c+rh*s;}for(int k=ROPE_ROT;k<HEAD_D;k++)o[k]=q[k];}
struct Win{int t,h,w,nwin;std::vector<int>st,en,sh,eh,sw,ew;std::vector<float>vid_freq,txt_freq;};
static Win load_win(const char*p){std::ifstream f(p,std::ios::binary);int64_t v[10];f.read((char*)v,80);Win w;w.t=(int)v[0];w.h=(int)v[1];w.w=(int)v[2];w.nwin=(int)v[9];
    int nt=w.nwin*6;std::vector<int64_t>sl(nt);f.read((char*)sl.data(),nt*8);
    for(int i=0;i<w.nwin;i++){w.st.push_back((int)sl[i*6]);w.en.push_back((int)sl[i*6+1]);w.sh.push_back((int)sl[i*6+2]);w.eh.push_back((int)sl[i*6+3]);w.sw.push_back((int)sl[i*6+4]);w.ew.push_back((int)sl[i*6+5]);}
    f.seekg(0,std::ios::end);int64_t fsz=f.tellg();f.seekg(80+nt*8,std::ios::beg);int64_t rem=fsz-(80+nt*8);int nf=(int)(rem/4);
    std::vector<float>al(nf);f.read((char*)al.data(),nf*4);int vidn=nf-w.nwin*TXT_LEN*ROPE_ROT;w.vid_freq.assign(al.begin(),al.begin()+vidn);w.txt_freq.assign(al.begin()+vidn,al.end());return w;}
struct BranchW{LinearRaw qkv,out,mlp_in,mlp_g,mlp_out;std::vector<float>nq,nk,attn_shift,attn_scale,attn_gate,mlp_shift,mlp_scale,mlp_gate;};
static void load_branch(BranchW&W,const std::string&b){W.qkv=load_linear(b+"_qkv",false);W.out=load_linear(b+"_out",true);W.mlp_in=load_linear(b+"_mlp_in",false);W.mlp_g=load_linear(b+"_mlp_g",false);W.mlp_out=load_linear(b+"_mlp_out",false);
    W.nq=load_raw_f((b+"_nq.bin").c_str());W.nk=load_raw_f((b+"_nk.bin").c_str());W.attn_shift=load_raw_f((b+"_attn_shift.bin").c_str());W.attn_scale=load_raw_f((b+"_attn_scale.bin").c_str());W.attn_gate=load_raw_f((b+"_attn_gate.bin").c_str());
    W.mlp_shift=load_raw_f((b+"_mlp_shift.bin").c_str());W.mlp_scale=load_raw_f((b+"_mlp_scale.bin").c_str());W.mlp_gate=load_raw_f((b+"_mlp_gate.bin").c_str());}
static void chk(const char*tag,const std::vector<float>&a,const char*ref){auto r=load_raw_f(ref);fprintf(stderr,"[cmp %-16s] cos=%.6f max|d|=%.5f\n",tag,cosine(a,r),maxd(a,r));}

int main(){
    const char*M5="models/m5/";
    LinearRaw vid_in_proj=load_linear(std::string(M5)+"vid_in_proj",true);
    LinearRaw txt_in=load_linear(std::string(M5)+"txt_in",true);
    LinearRaw emb_in=load_linear(std::string(M5)+"emb_proj_in",true);
    LinearRaw emb_hid=load_linear(std::string(M5)+"emb_proj_hid",true);
    LinearRaw emb_out=load_linear(std::string(M5)+"emb_proj_out",true);
    auto x_patch=load_raw_f((std::string(M5)+"x_patch.bin").c_str());
    auto x_txt=load_raw_f((std::string(M5)+"x_txt.bin").c_str());
    int Lv=(int)x_patch.size()/(33*1*2*2),TXT=(int)x_txt.size()/5120;
    std::vector<float>vid=vid_in_proj.batch(x_patch,Lv),txt=txt_in.batch(x_txt,TXT);
    // emb
    auto sine=[&](float t){std::vector<float>e(256);int half=128;for(int j=0;j<256;j++){float fr=(j<128)?expf(-logf(10000.f)*j/128):expf(-logf(10000.f)*(j-128)/128);e[j]=(j<128)?sinf(t*fr):cosf(t*fr);}return e;};
    std::vector<float>e=sine(500.f);e=emb_in.batch(e,1);for(auto&v:e)v=v/(1.f+expf(-v));e=emb_hid.batch(e,1);for(auto&v:e)v=v/(1.f+expf(-v));e=emb_out.batch(e,1);
    chk("emb",e,(std::string(M5)+"emb.bin").c_str());
    BranchW vW,tW;load_branch(vW,std::string(M5)+"b0_vid");load_branch(tW,std::string(M5)+"b0_txt");
    Win win=load_win((std::string(M5)+"win_nonshifted.bin").c_str());
    fprintf(stderr,"[info] Lv=%d TXT=%d nwin=%d\n",Lv,TXT,win.nwin);
    // attn_norm + ada in
    std::vector<float>van(Lv*DIM),tan(TXT*DIM);
    for(int t=0;t<Lv;t++)rmsnorm(&vid[t*DIM],DIM,nullptr,&van[t*DIM]);
    for(int t=0;t<TXT;t++)rmsnorm(&txt[t*DIM],DIM,nullptr,&tan[t*DIM]);
    for(int t=0;t<Lv;t++)for(int d=0;d<DIM;d++){float sA=e[d*6],scA=e[d*6+1],gA=e[d*6+2];float shB=vW.attn_shift[d],scB=vW.attn_scale[d],gB=vW.attn_gate[d];van[t*DIM+d]=van[t*DIM+d]*(scA+scB)+(sA+shB);}
    for(int t=0;t<TXT;t++)for(int d=0;d<DIM;d++){float sA=e[d*6],scA=e[d*6+1];float thB=tW.attn_shift[d],tcB=tW.attn_scale[d];tan[t*DIM+d]=tan[t*DIM+d]*(scA+tcB)+(sA+thB);}
    chk("vid_an",van,(std::string(M5)+"dbg_vid_an.bin").c_str());
    chk("txt_an",tan,(std::string(M5)+"dbg_txt_an.bin").c_str());
    // proj_qkv + qk_norm
    std::vector<float>vqkv=vW.qkv.batch(van,Lv),tqkv=tW.qkv.batch(tan,TXT);
    std::vector<float>vq(Lv*DIM),vk(Lv*DIM),vv(Lv*DIM),tq(TXT*DIM),tk(TXT*DIM),tv(TXT*DIM);
    for(int i=0;i<Lv;i++){const float*q=&vqkv[i*QKV];qk_norm(q,vW.nq.data(),&vq[i*DIM]);qk_norm(q+2560,vW.nk.data(),&vk[i*DIM]);memcpy(&vv[i*DIM],q+5120,DIM*4);}
    for(int i=0;i<TXT;i++){const float*q=&tqkv[i*QKV];qk_norm(q,tW.nq.data(),&tq[i*DIM]);qk_norm(q+2560,tW.nk.data(),&tk[i*DIM]);memcpy(&tv[i*DIM],q+5120,DIM*4);}
    chk("vq",vq,(std::string(M5)+"dbg_vq.bin").c_str());
    chk("vk",vk,(std::string(M5)+"dbg_vk.bin").c_str());
    chk("vv",vv,(std::string(M5)+"dbg_vv.bin").c_str());
    // partition
    std::vector<int>cumf(win.nwin+1,0);for(int wi=0;wi<win.nwin;wi++){int f=(win.en[wi]-win.st[wi])*(win.eh[wi]-win.sh[wi])*(win.ew[wi]-win.sw[wi]);cumf[wi+1]=cumf[wi]+f;}
    int sumf=cumf[win.nwin];int T=win.t,H=win.h,Wd=win.w;
    std::vector<float>vqw(sumf*DIM),vkw(sumf*DIM),vvw(sumf*DIM);
    for(int wi=0;wi<win.nwin;wi++){int local=0;for(int lt=win.st[wi];lt<win.en[wi];lt++)for(int lh=win.sh[wi];lh<win.eh[wi];lh++)for(int lw=win.sw[wi];lw<win.ew[wi];lw++){int idx=((lt*H)+lh)*Wd+lw;int dst=cumf[wi]+local;memcpy(&vqw[dst*DIM],&vq[idx*DIM],DIM*4);memcpy(&vkw[dst*DIM],&vk[idx*DIM],DIM*4);memcpy(&vvw[dst*DIM],&vv[idx*DIM],DIM*4);local++;}}
    chk("vqw",vqw,(std::string(M5)+"dbg_vqw.bin").c_str());
    chk("vkw",vkw,(std::string(M5)+"dbg_vkw.bin").c_str());
    chk("vvw",vvw,(std::string(M5)+"dbg_vvw.bin").c_str());
    // RoPE
    std::vector<float>vqw_r(sumf*DIM),vkw_r(sumf*DIM);
    for(int dst=0;dst<sumf;dst++){const float*fr=&win.vid_freq[dst*ROPE_ROT];for(int hh=0;hh<HEADS;hh++){float inb[HEAD_D],ob[HEAD_D];memcpy(inb,&vqw[dst*DIM+hh*HEAD_D],HEAD_D*4);apply_rope(fr,inb,ob);memcpy(&vqw_r[dst*DIM+hh*HEAD_D],ob,HEAD_D*4);memcpy(inb,&vkw[dst*DIM+hh*HEAD_D],HEAD_D*4);apply_rope(fr,inb,ob);memcpy(&vkw_r[dst*DIM+hh*HEAD_D],ob,HEAD_D*4);}}
    chk("vqw_rope",vqw_r,(std::string(M5)+"dbg_vqw_rope.bin").c_str());
    chk("vkw_rope",vkw_r,(std::string(M5)+"dbg_vkw_rope.bin").c_str());
    // SDPA + reverse (only need vout_rev)
    std::vector<float>tqw(win.nwin*TXT*DIM),tkw(win.nwin*TXT*DIM),tvw(win.nwin*TXT*DIM);
    for(int wi=0;wi<win.nwin;wi++)for(int i=0;i<TXT;i++){int dst=wi*TXT+i;memcpy(&tqw[dst*DIM],&tq[i*DIM],DIM*4);memcpy(&tkw[dst*DIM],&tk[i*DIM],DIM*4);memcpy(&tvw[dst*DIM],&tv[i*DIM],DIM*4);const float*fr=&win.txt_freq[dst*ROPE_ROT];for(int hh=0;hh<HEADS;hh++){float inb[HEAD_D],ob[HEAD_D];memcpy(inb,&tqw[dst*DIM+hh*HEAD_D],HEAD_D*4);apply_rope(fr,inb,ob);memcpy(&tqw[dst*DIM+hh*HEAD_D],ob,HEAD_D*4);memcpy(inb,&tkw[dst*DIM+hh*HEAD_D],HEAD_D*4);apply_rope(fr,inb,ob);memcpy(&tkw[dst*DIM+hh*HEAD_D],ob,HEAD_D*4);}}
    std::vector<float>vout_win(sumf*DIM),tout_win(win.nwin*TXT*DIM);float scale=1.f/sqrtf((float)HEAD_D);
    for(int wi=0;wi<win.nwin;wi++){int f_i=cumf[wi+1]-cumf[wi];int S=f_i+TXT;
        for(int hh=0;hh<HEADS;hh++){std::vector<float>qseg(S*HEAD_D),kseg(S*HEAD_D),vseg(S*HEAD_D);
            for(int j=0;j<f_i;j++){memcpy(&qseg[j*HEAD_D],&vqw_r[(cumf[wi]+j)*DIM+hh*HEAD_D],HEAD_D*4);memcpy(&kseg[j*HEAD_D],&vkw_r[(cumf[wi]+j)*DIM+hh*HEAD_D],HEAD_D*4);memcpy(&vseg[j*HEAD_D],&vvw[(cumf[wi]+j)*DIM+hh*HEAD_D],HEAD_D*4);}
            for(int i=0;i<TXT;i++){int s=f_i+i;memcpy(&qseg[s*HEAD_D],&tqw[(wi*TXT+i)*DIM+hh*HEAD_D],HEAD_D*4);memcpy(&kseg[s*HEAD_D],&tkw[(wi*TXT+i)*DIM+hh*HEAD_D],HEAD_D*4);memcpy(&vseg[s*HEAD_D],&tvw[(wi*TXT+i)*DIM+hh*HEAD_D],HEAD_D*4);}
            std::vector<float>sc(S*S);float mx=-1e30f;for(int a=0;a<S;a++){const float*qa=&qseg[a*HEAD_D];for(int b=0;b<S;b++){const float*kb=&kseg[b*HEAD_D];float dot=0;for(int dd=0;dd<HEAD_D;dd++)dot+=qa[dd]*kb[dd];dot*=scale;sc[a*S+b]=dot;if(dot>mx)mx=dot;}}
            for(int a=0;a<S;a++){float sm=0;std::vector<float>p(S);for(int b=0;b<S;b++){float ex=expf(sc[a*S+b]-mx);p[b]=ex;sm+=ex;}for(int b=0;b<S;b++)p[b]/=sm;float o[HEAD_D]={0};for(int b=0;b<S;b++){const float*vb=&vseg[b*HEAD_D];for(int dd=0;dd<HEAD_D;dd++)o[dd]+=p[b]*vb[dd];}if(a<f_i)memcpy(&vout_win[(cumf[wi]+a)*DIM+hh*HEAD_D],o,HEAD_D*4);else memcpy(&tout_win[(wi*TXT+(a-f_i))*DIM+hh*HEAD_D],o,HEAD_D*4);}}}
    std::vector<float>vout_rev(Lv*DIM,0.f);
    for(int wi=0;wi<win.nwin;wi++){int local=0;for(int lt=win.st[wi];lt<win.en[wi];lt++)for(int lh=win.sh[wi];lh<win.eh[wi];lh++)for(int lw=win.sw[wi];lw<win.ew[wi];lw++){int idx=((lt*H)+lh)*Wd+lw;memcpy(&vout_rev[idx*DIM],&vout_win[(cumf[wi]+local)*DIM],DIM*4);local++;}}
    chk("vout_rev",vout_rev,(std::string(M5)+"dbg_vout_rev.bin").c_str());
    // proj_out
    std::vector<float>vid_awa=vW.out.batch(vout_rev,Lv);
    chk("vid_awa",vid_awa,(std::string(M5)+"dbg_vid_awa.bin").c_str());
    // ada out + residual
    std::vector<float>vid_at(Lv*DIM);
    for(int t=0;t<Lv;t++)for(int d=0;d<DIM;d++){float gA=e[d*6+2],gB=vW.attn_gate[d];vid_at[t*DIM+d]=vid_awa[t*DIM+d]*(gA+gB);}
    for(int k=0;k<Lv*DIM;k++)vid_at[k]+=vid[k];
    chk("vid_b0_attn",vid_at,(std::string(M5)+"dbg_vid_b0_attn.bin").c_str());
    return 0;
}

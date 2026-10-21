/*
 * conv_qnn_fp16.c — QNN HTP FP16 direct 2D convolution (roofline mid-AI, NPU).
 *
 * out = conv2d(in, W), dense KxK, stride 1, same padding — via the native QNN
 * `Conv2d` op. Convolution is the NPU's home turf, so (per Discovery 58's R-ii:
 * native ops finalize fast) finalize_us should be small and throughput high; this
 * kernel measures the NPU on the workload it was actually designed for. Channels
 * parameterized for the roofline AI sweep.
 *
 * Layout: QNN Conv2d wants NHWC activations [1,H,W,Cin] and HWIO filter
 * [Kh,Kw,Cin,Cout]. The shared splitmix64 stream (input HWC first, then weights
 * OHWI — matching the CPU/GPU/DSP kernels + reference.py) is reordered OHWI→HWIO
 * here so the fingerprint cross-validates. A zero bias is supplied (HTP Conv2d
 * takes a bias input; zero keeps the math identical to the no-bias reference).
 *
 * Run: ./conv_qnn_fp16 [--H 256] [--W 256] [--cin 8] [--cout 8] [--k 3]
 *                      [--seed 42] [--reps 5] [--perf-mode sustained]
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "QNN/QnnInterface.h"
#include "QNN/QnnProfile.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnOpDef.h"
#include "QNN/QnnDevice.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

static int64_t now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec*1000000000LL + ts.tv_nsec; }

static uint64_t sm_state;
static void sm_seed(uint64_t s) { sm_state = s; }
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static float sm_float(void) { return (float)(sm_next() >> 11) * (1.0f / (float)(1ULL << 53)); }

typedef uint16_t fp16_t;
static fp16_t f2h(float f) {
#if defined(__aarch64__)
    __fp16 h = (__fp16)f; fp16_t b; memcpy(&b, &h, 2); return b;
#else
    uint32_t bits; memcpy(&bits,&f,4); uint32_t s=(bits>>31)&1,e=(bits>>23)&0xFF,m=bits&0x7FFFFF;
    if(e==0xFF)return (fp16_t)((s<<15)|0x7C00|(m?0x200:0)); int ne=(int)e-127+15;
    if(ne>=31)return (fp16_t)((s<<15)|0x7C00); if(ne<=0)return (fp16_t)(s<<15);
    return (fp16_t)((s<<15)|((uint32_t)ne<<10)|(m>>13));
#endif
}
static float h2f(fp16_t h) {
#if defined(__aarch64__)
    __fp16 v; memcpy(&v,&h,2); return (float)v;
#else
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF; if(e==0x1F)return s?-__builtin_inff():__builtin_inff();
    if(e==0)return (s?-1.0f:1.0f)*(float)m/(float)(1<<24); float f; uint32_t b=(s<<31)|((e-15+127)<<23)|(m<<13); memcpy(&f,&b,4); return f;
#endif
}

typedef Qnn_ErrorHandle_t (*QnnInterface_GetProvidersFn_t)(const QnnInterface_t ***, uint32_t *);
static QnnInterface_t g_iface;
#define QNN(fn, ...) do { Qnn_ErrorHandle_t _rc = g_iface.QNN_INTERFACE_VER_NAME.fn(__VA_ARGS__); \
    if (_rc != QNN_SUCCESS) { fprintf(stderr, "[qnn] " #fn " failed: 0x%x\n", (unsigned)_rc); goto fail; } } while(0)

static Qnn_DeviceHandle_t g_device_handle = NULL;
static QnnHtpDevice_Infrastructure_t *g_htp_infra = NULL;
static uint32_t g_power_cfg_id = 0;
static int htp_perf_begin(void) {
    Qnn_ErrorHandle_t rc = g_iface.QNN_INTERFACE_VER_NAME.deviceCreate(NULL, NULL, &g_device_handle);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] deviceCreate 0x%x\n", (unsigned)rc); return -1; }
    QnnDevice_Infrastructure_t raw = NULL;
    rc = g_iface.QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&raw);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] getInfra 0x%x\n", (unsigned)rc); return -1; }
    g_htp_infra = (QnnHtpDevice_Infrastructure_t *)raw;
    if (!g_htp_infra || !g_htp_infra->perfInfra.createPowerConfigId) { fprintf(stderr, "[perf] infra NULL\n"); return -1; }
    rc = g_htp_infra->perfInfra.createPowerConfigId(0, 0, &g_power_cfg_id);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] cfgId 0x%x\n", (unsigned)rc); return -1; }
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = { .contextId=g_power_cfg_id, .setDcvsEnable=1, .dcvsEnable=0,
            .powerMode=QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            .setSleepLatency=1, .sleepLatency=40, .setSleepDisable=1, .sleepDisable=1,
            .setBusParams=1, .busVoltageCornerMin=DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerTarget=DCVS_VOLTAGE_VCORNER_TURBO, .busVoltageCornerMax=DCVS_VOLTAGE_VCORNER_TURBO,
            .setCoreParams=1, .coreVoltageCornerMin=DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerTarget=DCVS_VOLTAGE_VCORNER_TURBO, .coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_TURBO, } };
    QnnHtpPerfInfrastructure_PowerConfig_t rpc = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
        .rpcPollingTimeConfig = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME };
    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &dcvs, &rpc, NULL };
    rc = g_htp_infra->perfInfra.setPowerConfig(g_power_cfg_id, cfgs);
    if (rc != QNN_SUCCESS) { fprintf(stderr, "[perf] setPowerConfig 0x%x\n", (unsigned)rc); return -1; }
    fprintf(stderr, "[perf] sustained_high_performance active (id=%u)\n", g_power_cfg_id);
    return 0;
}
static void htp_perf_end(void) {
    if (g_htp_infra && g_power_cfg_id) { g_htp_infra->perfInfra.destroyPowerConfigId(g_power_cfg_id); g_power_cfg_id = 0; }
    if (g_device_handle) { g_iface.QNN_INTERFACE_VER_NAME.deviceFree(g_device_handle); g_device_handle = NULL; }
}

static int g_H=256, g_W=256, g_Cin=8, g_Cout=8, g_K=3, g_seed=42, g_reps=5, g_sustained=0;
static void parse_args(int argc, char **argv) {
    for (int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--H")&&i+1<argc)g_H=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--W")&&i+1<argc)g_W=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cin")&&i+1<argc)g_Cin=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cout")&&i+1<argc)g_Cout=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--k")&&i+1<argc)g_K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc)g_seed=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--reps")&&i+1<argc)g_reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--perf-mode")&&i+1<argc){const char*m=argv[++i];g_sustained=(!strcmp(m,"sustained")||!strcmp(m,"sustained_high_performance"))?1:0;}
    }
}

static void mk(Qnn_Tensor_t*t,const char*name,Qnn_TensorType_t type,Qnn_DataType_t dt,uint32_t rank,uint32_t*dims){
    memset(t,0,sizeof(*t)); t->version=QNN_TENSOR_VERSION_1; Qnn_TensorV1_t*v=&t->v1;
    v->name=name; v->type=type; v->dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; v->dataType=dt;
    v->quantizeParams.encodingDefinition=QNN_DEFINITION_UNDEFINED;
    v->quantizeParams.quantizationEncoding=QNN_QUANTIZATION_ENCODING_UNDEFINED;
    v->rank=rank; v->dimensions=dims; v->memType=QNN_TENSORMEMTYPE_RAW;
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    int H=g_H, W=g_W, Cin=g_Cin, Cout=g_Cout, K=g_K;
    int64_t n_in=(int64_t)H*W*Cin, n_w=(int64_t)Cout*K*K*Cin, n_out=(int64_t)H*W*Cout;

    /* input fp16 (NHWC), weights OHWI drawn then reordered to HWIO, zero bias. */
    fp16_t *X = malloc(n_in*sizeof(fp16_t));
    fp16_t *Wq = malloc(n_w*sizeof(fp16_t));           /* HWIO for QNN */
    fp16_t *bias = calloc(Cout, sizeof(fp16_t));       /* zeros */
    fp16_t *C = calloc(n_out, sizeof(fp16_t));
    if(!X||!Wq||!bias||!C){fprintf(stderr,"malloc failed\n");return 1;}
    sm_seed((uint64_t)g_seed);
    for(int64_t i=0;i<n_in;i++) X[i]=f2h(2.0f*sm_float()-1.0f);
    /* draw weights in OHWI order (matching reference), place into HWIO buffer */
    for(int oc=0;oc<Cout;oc++)for(int kh=0;kh<K;kh++)for(int kw=0;kw<K;kw++)for(int ic=0;ic<Cin;ic++){
        float val=2.0f*sm_float()-1.0f;
        int64_t hwio=(((int64_t)kh*K+kw)*Cin+ic)*Cout+oc;
        Wq[hwio]=f2h(val);
    }

    void *lib=dlopen("libQnnHtp.so", RTLD_LAZY|RTLD_GLOBAL);
    if(!lib){fprintf(stderr,"[qnn] dlopen libQnnHtp.so: %s\n",dlerror());return 1;}
    QnnInterface_GetProvidersFn_t gp=(QnnInterface_GetProvidersFn_t)dlsym(lib,"QnnInterface_getProviders");
    if(!gp){fprintf(stderr,"[qnn] dlsym: %s\n",dlerror());return 1;}
    const QnnInterface_t **provs=NULL; uint32_t np=0;
    if(gp(&provs,&np)!=QNN_SUCCESS||np==0){fprintf(stderr,"[qnn] no providers\n");return 1;}
    g_iface=provs[0][0];

    Qnn_BackendHandle_t backend=NULL; Qnn_ContextHandle_t context=NULL; Qnn_GraphHandle_t graph=NULL;
    Qnn_ProfileHandle_t prof=NULL; int64_t t0,t1,finalize_us=0,*exec_times=NULL;

    QNN(backendCreate, NULL, NULL, &backend);
    if(g_sustained){ if(htp_perf_begin()!=0) goto fail; }
    QNN(contextCreate, backend, NULL, NULL, &context);
    QNN(graphCreate, context, "conv2d_fp16", NULL, &graph);

    /* tensors */
    uint32_t d_in[4]={1,(uint32_t)H,(uint32_t)W,(uint32_t)Cin};
    uint32_t d_w[4]={(uint32_t)K,(uint32_t)K,(uint32_t)Cin,(uint32_t)Cout};   /* HWIO */
    uint32_t d_bias[1]={(uint32_t)Cout};
    uint32_t d_out[4]={1,(uint32_t)H,(uint32_t)W,(uint32_t)Cout};
    Qnn_Tensor_t tX,tW,tB,tOut;
    mk(&tX,"input",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_16,4,d_in);
    mk(&tW,"weights",QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_FLOAT_16,4,d_w);
    tW.v1.clientBuf.data=Wq; tW.v1.clientBuf.dataSize=(uint32_t)(n_w*sizeof(fp16_t));
    mk(&tB,"bias",QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_FLOAT_16,1,d_bias);
    tB.v1.clientBuf.data=bias; tB.v1.clientBuf.dataSize=(uint32_t)(Cout*sizeof(fp16_t));
    mk(&tOut,"output",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_16,4,d_out);
    QNN(tensorCreateGraphTensor, graph, &tX);
    QNN(tensorCreateGraphTensor, graph, &tW);
    QNN(tensorCreateGraphTensor, graph, &tB);
    QNN(tensorCreateGraphTensor, graph, &tOut);

    /* param tensors: stride[2]={1,1}, pad[2][2]={{p,p},{p,p}}, dilation[2]={1,1} */
    int pad=K/2;
    uint32_t stride_data[2]={1,1}; uint32_t dil_data[2]={1,1};
    uint32_t pad_data[4]={(uint32_t)pad,(uint32_t)pad,(uint32_t)pad,(uint32_t)pad};
    uint32_t d_s2[1]={2}; uint32_t d_p22[2]={2,2};
    Qnn_Tensor_t tStride,tPad,tDil;
    mk(&tStride,"stride",QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_UINT_32,1,d_s2);
    tStride.v1.clientBuf.data=stride_data; tStride.v1.clientBuf.dataSize=sizeof(stride_data);
    mk(&tPad,"pad_amount",QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_UINT_32,2,d_p22);
    tPad.v1.clientBuf.data=pad_data; tPad.v1.clientBuf.dataSize=sizeof(pad_data);
    mk(&tDil,"dilation",QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_UINT_32,1,d_s2);
    tDil.v1.clientBuf.data=dil_data; tDil.v1.clientBuf.dataSize=sizeof(dil_data);
    QNN(tensorCreateGraphTensor, graph, &tStride);
    QNN(tensorCreateGraphTensor, graph, &tPad);
    QNN(tensorCreateGraphTensor, graph, &tDil);

    Qnn_Param_t params[4]; memset(params,0,sizeof(params));
    params[0].paramType=QNN_PARAMTYPE_TENSOR; params[0].name=QNN_OP_CONV_2D_PARAM_STRIDE;   params[0].tensorParam=tStride;
    params[1].paramType=QNN_PARAMTYPE_TENSOR; params[1].name=QNN_OP_CONV_2D_PARAM_PAD_AMOUNT;params[1].tensorParam=tPad;
    params[2].paramType=QNN_PARAMTYPE_TENSOR; params[2].name=QNN_OP_CONV_2D_PARAM_DILATION;  params[2].tensorParam=tDil;
    params[3].paramType=QNN_PARAMTYPE_SCALAR; params[3].name=QNN_OP_CONV_2D_PARAM_GROUP;
    params[3].scalarParam.dataType=QNN_DATATYPE_UINT_32; params[3].scalarParam.uint32Value=1;

    Qnn_Tensor_t conv_in[3]={tX,tW,tB};
    Qnn_Tensor_t conv_out[1]={tOut};
    Qnn_OpConfig_t op={ .version=QNN_OPCONFIG_VERSION_1, .v1={ .name="conv2d",
        .packageName=QNN_OP_PACKAGE_NAME_QTI_AISW, .typeName=QNN_OP_CONV_2D,
        .numOfParams=4, .params=params, .numOfInputs=3, .inputTensors=conv_in,
        .numOfOutputs=1, .outputTensors=conv_out } };
    QNN(graphAddNode, graph, op);

    QNN(profileCreate, backend, QNN_PROFILE_LEVEL_DETAILED, &prof);
    t0=now_ns(); QNN(graphFinalize, graph, prof, NULL); t1=now_ns(); finalize_us=(t1-t0)/1000;

    Qnn_Tensor_t ein[1]={tX}, eout[1]={tOut};
    ein[0].v1.clientBuf.data=X; ein[0].v1.clientBuf.dataSize=(uint32_t)(n_in*sizeof(fp16_t));
    eout[0].v1.clientBuf.data=C; eout[0].v1.clientBuf.dataSize=(uint32_t)(n_out*sizeof(fp16_t));
    { Qnn_ErrorHandle_t wrc=g_iface.QNN_INTERFACE_VER_NAME.graphExecute(graph,ein,1,eout,1,NULL,NULL);
      if(wrc!=QNN_SUCCESS) fprintf(stderr,"[qnn] warm-up execute 0x%x (continuing)\n",(unsigned)wrc); }

    exec_times=malloc((size_t)g_reps*sizeof(int64_t));
    if(!exec_times){fprintf(stderr,"malloc exec_times failed\n");goto fail;}
    for(int r=0;r<g_reps;r++){
        t0=now_ns();
        Qnn_ErrorHandle_t rc=g_iface.QNN_INTERFACE_VER_NAME.graphExecute(graph,ein,1,eout,1,NULL,NULL);
        t1=now_ns();
        if(rc!=QNN_SUCCESS){fprintf(stderr,"[qnn] execute rep %d 0x%x\n",r,(unsigned)rc);goto fail;}
        exec_times[r]=(t1-t0)/1000;
    }
    double sum_t=0; int64_t mn=exec_times[0],mx=exec_times[0];
    for(int r=0;r<g_reps;r++){sum_t+=exec_times[r]; if(exec_times[r]<mn)mn=exec_times[r]; if(exec_times[r]>mx)mx=exec_times[r];}
    double mean_us=sum_t/g_reps, var=0;
    for(int r=0;r<g_reps;r++){double d=exec_times[r]-mean_us; var+=d*d;}
    double cov=(mean_us>0)?sqrt(var/g_reps)/mean_us*100.0:0;

    double sumc=0,sq=0;
    for(int64_t i=0;i<n_out;i++){float v=h2f(C[i]); sumc+=v; sq+=(double)v*v;}
    double fingerprint=sumc+sqrt(sq);

    int64_t flop=(int64_t)2*H*W*Cout*K*K*Cin;
    int64_t bytes=(int64_t)2*(n_in+n_w+n_out); /* fp16 */
    printf("NPU_DISPATCH_RESULT\nstatus: SUCCESS\n");
    printf("power_mode: %s\n", g_sustained?"sustained_high_performance":"default");
    printf("reps: %d\nexecute_us_mean: %.1f\nexecute_us_min: %lld\nexecute_us_max: %lld\nexecute_us_cov_pct: %.3f\n",
           g_reps, mean_us, (long long)mn, (long long)mx, cov);
    printf("finalize_us: %lld\nprecision: fp16\n", (long long)finalize_us);
    printf("H: %d\nW: %d\ncin: %d\ncout: %d\nk: %d\n", H,W,Cin,Cout,K);
    printf("flop: %lld\nbytes: %lld\nfingerprint: %.6e\n", (long long)flop,(long long)bytes,fingerprint);
    for(int r=0;r<g_reps;r++) printf("rep_%d_us: %lld\n", r,(long long)exec_times[r]);
    printf("NPU_DISPATCH_END\n");

    free(exec_times); htp_perf_end();
    if(prof) g_iface.QNN_INTERFACE_VER_NAME.profileFree(prof);
    if(context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context,NULL);
    if(backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    dlclose(lib); free(X);free(Wq);free(bias);free(C);
    return 0;
fail:
    printf("NPU_DISPATCH_RESULT\nstatus: FAILED\nNPU_DISPATCH_END\n");
    free(exec_times); htp_perf_end();
    if(prof) g_iface.QNN_INTERFACE_VER_NAME.profileFree(prof);
    if(context) g_iface.QNN_INTERFACE_VER_NAME.contextFree(context,NULL);
    if(backend) g_iface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    if(lib) dlclose(lib); free(X);free(Wq);free(bias);free(C);
    return 1;
}

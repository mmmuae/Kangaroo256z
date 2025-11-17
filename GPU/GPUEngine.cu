/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#include <stdint.h>
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_kangaroos(uint64_t *kangaroos,uint32_t maxFound,uint32_t *found,uint64_t *dpMask) {

  int xPtr = (blockIdx.x*blockDim.x*GPU_GRP_SIZE) * KSIZE; // x[4] , y[4] , d[2], lastJump
  ComputeKangaroos(kangaroos + xPtr,maxFound,found,dpMask);

}

// ---------------------------------------------------------------------------------------
//#define GPU_CHECK
#ifdef GPU_CHECK
__global__ void check_gpu() {

  // Check ModInv
  uint64_t N[5] = { 0x0BE3D7593BE1147CULL,0x4952AAF512875655ULL,0x08884CCAACCB9B53ULL,0x9EAE2E2225044292ULL,0ULL };
  uint64_t I[5];
  uint64_t R[5];
  bool ok = true;

  /*
  for(uint64_t i=0;i<10000 && ok;i++) {

    Load(R,N);
    _ModInv(R);
    Load(I,R);
    _ModMult(R,N);
    SubP(R);
    if(!_IsOne(R)) {
      ok = false;
      printf("ModInv wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }
  */
  I[4] = 0;
  R[4] = 0;
  for(uint64_t i = 0; i < 100000 && ok; i++) {

    _ModSqr(I,N);
    _ModMult(R,N,N);
    if(!_IsEqual(I,R)) {
      ok = false;
      printf("_ModSqr wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }

}
#endif

// ---------------------------------------------------------------------------------------

using namespace std;

static bool WaitForStream(cudaStream_t stream,bool spinWait) {

  if(stream == 0)
    return true;

  if(spinWait) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if(err != cudaSuccess) {
      printf("GPUEngine: cudaStreamSynchronize %s\n",cudaGetErrorString(err));
      return false;
    }
    return true;
  }

  while(true) {
    cudaError_t err = cudaStreamQuery(stream);
    if(err == cudaSuccess) {
      return true;
    }
    if(err != cudaErrorNotReady) {
      printf("GPUEngine: cudaStreamQuery %s\n",cudaGetErrorString(err));
      return false;
    }
    Timer::SleepMillis(1);
  }

  return true;

}

int _ConvertSMVer2Cores(int major,int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
             // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    { 0x30, 192 },
    { 0x32, 192 },
    { 0x35, 192 },
    { 0x37, 192 },
    { 0x50, 128 },
    { 0x52, 128 },
    { 0x53, 128 },
    { 0x60,  64 },
    { 0x61, 128 },
    { 0x62, 128 },
    { 0x70,  64 },
    { 0x72,  64 },
    { 0x75,  64 },
    { 0x80,  64 },
    { 0x86, 128 },
    { 0x87, 128 },
    { 0x89, 128 },
    { 0x90, 192 },
    { 0x91, 192 },
    { -1, -1 } };

  int index = 0;

  while(nGpuArchCoresPerSM[index].SM != -1) {
    if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

void GPUEngine::SetWildOffset(Int* offset) {
  wildOffset.Set(offset);
}

GPUEngine::GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound) {

  // Initialise CUDA
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;
  computeStream = 0;
  copyStream = 0;
  kernelFinished[0] = 0;
  kernelFinished[1] = 0;
  outputItem[0] = NULL;
  outputItem[1] = NULL;
  nextOutputBuffer = 0;
  lastLaunchedBuffer = -1;
  kernelInFlight = false;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if(err != cudaSuccess) {
    printf("GPUEngine: %s\n",cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,gpuId);

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,deviceProp.name,deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
    nbThread / nbThreadPerGroup,
    nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if(err != cudaSuccess) {
    printf("GPUEngine: %s\n",cudaGetErrorString(err));
    return;
  }
#if defined(cudaFuncAttributePreferredSharedMemoryCarveout) && defined(cudaSharedmemCarveoutMaxL1)
  cudaFuncSetAttribute(comp_kangaroos,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxL1);
#endif

  // Allocate memory
  inputKangaroo = NULL;
  inputKangarooPinned = NULL;
  outputItemPinned = NULL;
  jumpPinned = NULL;
  dpMask=NULL;

  // dpMask
  err = cudaMalloc((void **)&dpMask,32);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate dpmask memory: %s\n",cudaGetErrorString(err));
    return;
  }

  // Input kangaroos
  kangarooSize = nbThread * GPU_GRP_SIZE * KSIZE * 8;
  err = cudaMalloc((void **)&inputKangaroo,kangarooSize);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n",cudaGetErrorString(err));
    return;
  }
  kangarooSizePinned = nbThreadPerGroup * GPU_GRP_SIZE *  KSIZE * 8;
  err = cudaHostAlloc(&inputKangarooPinned,kangarooSizePinned,cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }

  // Output buffers (double buffered)
  for(int i = 0; i < 2; i++) {
    err = cudaMalloc((void **)&outputItem[i],outputSize);
    if(err != cudaSuccess) {
      printf("GPUEngine: Allocate output memory #%d: %s\n",i,cudaGetErrorString(err));
      return;
    }
  }
  err = cudaHostAlloc(&outputItemPinned,outputSize,cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }

  // Jump array
  jumpSize = NB_JUMP * 8 * 4;
  err = cudaHostAlloc(&jumpPinned,jumpSize,cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate jump pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }

  err = cudaStreamCreateWithFlags(&computeStream,cudaStreamNonBlocking);
  if(err != cudaSuccess) {
    printf("GPUEngine: Create compute stream: %s\n",cudaGetErrorString(err));
    return;
  }
  err = cudaStreamCreateWithFlags(&copyStream,cudaStreamNonBlocking);
  if(err != cudaSuccess) {
    printf("GPUEngine: Create copy stream: %s\n",cudaGetErrorString(err));
    return;
  }
  for(int i = 0; i < 2; i++) {
    err = cudaEventCreateWithFlags(&kernelFinished[i],cudaEventDisableTiming);
    if(err != cudaSuccess) {
      printf("GPUEngine: Create event #%d: %s\n",i,cudaGetErrorString(err));
      return;
    }
  }

  lostWarning = false;
  initialised = true;
  wildOffset.SetInt32(0);

#ifdef GPU_CHECK

  double minT = 1e9;
  for(int i=0;i<5;i++) {
    double t0 = Timer::get_tick();
    check_gpu<<<1,1>>>();
    cudaThreadSynchronize();
    double t1 = Timer::get_tick();
    if( (t1-t0)<minT ) minT = (t1-t0);
  }
  printf("Cuda: %.3f ms\n",minT*1000.0);
  exit(0);

#endif

}

GPUEngine::~GPUEngine() {

  if(dpMask) cudaFree(dpMask);
  if(inputKangaroo) cudaFree(inputKangaroo);
  for(int i = 0; i < 2; i++)
    if(outputItem[i]) cudaFree(outputItem[i]);
  if(inputKangarooPinned) cudaFreeHost(inputKangarooPinned);
  if(outputItemPinned) cudaFreeHost(outputItemPinned);
  if(jumpPinned) cudaFreeHost(jumpPinned);
  for(int i = 0; i < 2; i++)
    if(kernelFinished[i]) cudaEventDestroy(kernelFinished[i]);
  if(computeStream) cudaStreamDestroy(computeStream);
  if(copyStream) cudaStreamDestroy(copyStream);

}


int GPUEngine::GetMemory() {
  return kangarooSize + 2 * outputSize + jumpSize;
}


int GPUEngine::GetGroupSize() {
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId,int *x,int *y) {

  if(*x <= 0 || *y <= 0) {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess) {
      printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
      return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if(deviceCount == 0) {
      printf("GPUEngine: There are no available device(s) that support CUDA\n");
      return false;
    }

    if(gpuId >= deviceCount) {
      printf("GPUEngine::GetGridSize() Invalid gpuId\n");
      return false;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,gpuId);

    int warpSize = (deviceProp.warpSize > 0 ? deviceProp.warpSize : 32);
    int minGridSize = 0;
    int suggestedBlockSize = 0;
    cudaError_t occ = cudaOccupancyMaxPotentialBlockSize(&minGridSize,&suggestedBlockSize,comp_kangaroos,0,0);
    if(occ != cudaSuccess) {
      suggestedBlockSize = 0;
      minGridSize = 0;
    }

    if(*y <= 0) {
      if(suggestedBlockSize <= 0) {
        suggestedBlockSize = 2 * _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor);
      }
      if(suggestedBlockSize <= 0) {
        suggestedBlockSize = warpSize * 4;
      }
      suggestedBlockSize = std::min(suggestedBlockSize,deviceProp.maxThreadsPerBlock);
      if(suggestedBlockSize < warpSize) suggestedBlockSize = warpSize;
      suggestedBlockSize = ((suggestedBlockSize + warpSize - 1) / warpSize) * warpSize;
      *y = suggestedBlockSize;
    }

    if(*x <= 0) {
      int blocks = std::max(2 * deviceProp.multiProcessorCount,minGridSize);
      if(deviceProp.major >= 9)
        blocks = std::max(blocks,deviceProp.multiProcessorCount * 6);
      if(blocks <= 0)
        blocks = deviceProp.multiProcessorCount;
      *x = blocks;
    }

  }

  return true;

}

void *GPUEngine::AllocatePinnedMemory(size_t size) {

  void *buff;

  cudaError_t err = cudaHostAlloc(&buff,size,cudaHostAllocPortable);
  if(err != cudaSuccess) {
    printf("GPUEngine: AllocatePinnedMemory: %s\n",cudaGetErrorString(err));
    return NULL;
  }

  return buff;

}

void GPUEngine::FreePinnedMemory(void *buff) {
  cudaFreeHost(buff);
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
    NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i = 0; i<deviceCount; i++) {

    err = cudaSetDevice(i);
    if(err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n",i,cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
      deviceProp.major,deviceProp.minor,(double)deviceProp.totalGlobalMem / 1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetKangaroos(Int *px,Int *py,Int *d) {

  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
	Int tpx = px[idx];
        inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup] = tpx.bits64[0];
        inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup] = tpx.bits64[1];
        inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup] = tpx.bits64[2];
        inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup] = tpx.bits64[3];

        // Y
	Int tpy = py[idx];
        inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup] = tpy.bits64[0];
        inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup] = tpy.bits64[1];
        inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup] = tpy.bits64[2];
        inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup] = tpy.bits64[3];

        // Distance
        Int dOff;
        dOff.Set(&d[idx]);
        if(idx % 2 == WILD) dOff.ModAddK1order(&wildOffset);
        inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup] = dOff.bits64[0];
        inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup] = dOff.bits64[1];
        inputKangarooPinned[g * strideSize + t + 10 * nbThreadPerGroup] = dOff.bits64[2];
	inputKangarooPinned[g * strideSize + t + 11 * nbThreadPerGroup] = dOff.bits64[3];
#ifdef USE_SYMMETRY
        // Last jump
        inputKangarooPinned[t + 12 * nbThreadPerGroup] = (uint64_t)NB_JUMP;
#endif

        idx++;
      }

    }

    uint32_t offset = b * blockSize;
    cudaMemcpy(inputKangaroo + offset,inputKangarooPinned,kangarooSizePinned,cudaMemcpyHostToDevice);

  }

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: SetKangaroos: %s\n",cudaGetErrorString(err));
  }

}

void GPUEngine::GetKangaroos(Int *px,Int *py,Int *d) {

  if(inputKangarooPinned==NULL ) {
    printf("GPUEngine: GetKangaroos: Cannot retreive kangaroos, mem has been freed\n");
    return;
  }

  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {

    uint32_t offset = b * blockSize;
    cudaMemcpy(inputKangarooPinned,inputKangaroo + offset,kangarooSizePinned,cudaMemcpyDeviceToHost);

    for(int g = 0; g < GPU_GRP_SIZE; g++) {

      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        px[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup];
        px[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup];
        px[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup];
        px[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup];
        px[idx].bits64[4] = 0;

        // Y
        py[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup];
        py[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup];
        py[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup];
        py[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup];
        py[idx].bits64[4] = 0;

        // Distance
        Int dOff;
        dOff.SetInt32(0);
        dOff.bits64[0] = inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup];
        dOff.bits64[1] = inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup];
        if(idx % 2 == WILD) dOff.ModSubK1order(&wildOffset);
        d[idx].Set(&dOff);

        idx++;
      }

    }
  }

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: GetKangaroos: %s\n",cudaGetErrorString(err));
  }

}

void GPUEngine::SetKangaroo(uint64_t kIdx,Int *px,Int *py,Int *d) {

  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int blockSize = nbThreadPerGroup * gSize;

  uint64_t t = kIdx % nbThreadPerGroup;
  uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
  uint64_t b = kIdx / (nbThreadPerGroup*GPU_GRP_SIZE);

  // X
  inputKangarooPinned[0] = px->bits64[0];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 0 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = px->bits64[1];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 1 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = px->bits64[2];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 2 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = px->bits64[3];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 3 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);

  // Y
  inputKangarooPinned[0] = py->bits64[0];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 4 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = py->bits64[1];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 5 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = py->bits64[2];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 6 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = py->bits64[3];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 7 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);

  // D
  Int dOff;
  dOff.Set(d);
  if(kIdx % 2 == WILD) dOff.ModAddK1order(&wildOffset);
  inputKangarooPinned[0] = dOff.bits64[0];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 8 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = dOff.bits64[1];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 9 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = dOff.bits64[2];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 10 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
  inputKangarooPinned[0] = dOff.bits64[3];
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 11 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);

#ifdef USE_SYMMETRY
  // Last jump
  inputKangarooPinned[0] = (uint64_t)NB_JUMP;
  cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 12 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice);
#endif

}

bool GPUEngine::callKernel() {

  if(!initialised)
    return false;

  int writeBuffer = nextOutputBuffer;
  nextOutputBuffer ^= 1;

  cudaError_t err = cudaMemsetAsync(outputItem[writeBuffer],0,4,computeStream);
  if(err != cudaSuccess) {
    printf("GPUEngine: cudaMemsetAsync: %s\n",cudaGetErrorString(err));
    return false;
  }

  // Call the kernel (Perform STEP_SIZE keys per thread)
  comp_kangaroos<<<nbThread / nbThreadPerGroup,nbThreadPerGroup,0,computeStream>>>
      (inputKangaroo,maxFound,outputItem[writeBuffer],dpMask);

  err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n",cudaGetErrorString(err));
    return false;
  }

  err = cudaEventRecord(kernelFinished[writeBuffer],computeStream);
  if(err != cudaSuccess) {
    printf("GPUEngine: cudaEventRecord: %s\n",cudaGetErrorString(err));
    return false;
  }

  lastLaunchedBuffer = writeBuffer;
  kernelInFlight = true;

  return true;

}

void GPUEngine::SetParams(Int *dpMask,Int *distance,Int *px,Int *py) {

  if(!initialised)
    return;

  if(!WaitForStream(computeStream,true) || !WaitForStream(copyStream,true)) {
    printf("GPUEngine: SetParams: Unable to sync CUDA streams before uploading parameters\n");
    return;
  }

  uint64_t hostDpMask[4];

  hostDpMask[0] = dpMask->bits64[0];
  hostDpMask[1] = dpMask->bits64[1];
  hostDpMask[2] = dpMask->bits64[2];
  hostDpMask[3] = dpMask->bits64[3];

  cudaError_t err = cudaMemcpyAsync(this->dpMask,hostDpMask,32,cudaMemcpyHostToDevice,copyStream);
  if(err != cudaSuccess) {
    printf("GPUEngine: SetParams: Failed to copy dpMask: %s\n",cudaGetErrorString(err));
    return;
  }

  if(!UploadJumpTable(distance,"distance",jD) ||
     !UploadJumpTable(px,"px",jPx) ||
     !UploadJumpTable(py,"py",jPy)) {
    return;
  }

  err = cudaStreamSynchronize(copyStream);
  if(err != cudaSuccess) {
    printf("GPUEngine: SetParams: Failed to synchronize copy stream: %s\n",cudaGetErrorString(err));
    return;
  }

}

bool GPUEngine::UploadJumpTable(const Int *src,const char *label,const void *symbol) {

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,src[i].bits64,32);

  cudaError_t err = cudaMemcpyToSymbolAsync(symbol,jumpPinned,jumpSize,0,cudaMemcpyHostToDevice,copyStream);
  if(err != cudaSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory (%s): %s\n",label,cudaGetErrorString(err));
    return false;
  }

  return true;

}

bool GPUEngine::callKernelAndWait() {

  // Debug function
  if(!callKernel())
    return false;

  cudaError_t err = cudaEventSynchronize(kernelFinished[lastLaunchedBuffer]);
  if(err != cudaSuccess) {
    printf("GPUEngine: callKernelAndWait event: %s\n",cudaGetErrorString(err));
    return false;
  }

  err = cudaMemcpy(outputItemPinned,outputItem[lastLaunchedBuffer],outputSize,cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    printf("GPUEngine: callKernelAndWait memcpy: %s\n",cudaGetErrorString(err));
    return false;
  }

  return true;

}

bool GPUEngine::Launch(std::vector<ITEM> &hashFound,bool spinWait) {


  hashFound.clear();

  if(!kernelInFlight)
    return false;

  int bufferToRead = lastLaunchedBuffer;

  cudaError_t err = cudaEventSynchronize(kernelFinished[bufferToRead]);
  if(err != cudaSuccess) {
    printf("GPUEngine: Launch wait kernel: %s\n",cudaGetErrorString(err));
    return false;
  }

  err = cudaMemcpyAsync(outputItemPinned,outputItem[bufferToRead],4,cudaMemcpyDeviceToHost,copyStream);
  if(err != cudaSuccess) {
    printf("GPUEngine: Launch prefix copy: %s\n",cudaGetErrorString(err));
    return false;
  }

  if(!WaitForStream(copyStream,spinWait))
    return false;

  // Look for prefix found
  uint32_t nbFound = outputItemPinned[0];
  if(nbFound > maxFound) {
    // prefix has been lost
    if(!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less threads (-g) or increse dp (-d)\n",(nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }

  size_t copySize = nbFound * ITEM_SIZE + 4;
  if(copySize > 4) {
    err = cudaMemcpyAsync(outputItemPinned,outputItem[bufferToRead],copySize,cudaMemcpyDeviceToHost,copyStream);
    if(err != cudaSuccess) {
      printf("GPUEngine: Launch copy: %s\n",cudaGetErrorString(err));
      return false;
    }
  }

  bool kernelOk = callKernel();

  if(copySize > 4) {
    if(!WaitForStream(copyStream,true))
      return false;
  }

  for(uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputItemPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;

    it.kIdx = *((uint64_t*)(itemPtr + 12));

    uint64_t *x = (uint64_t *)itemPtr;
    it.x.bits64[0] = x[0];
    it.x.bits64[1] = x[1];
    it.x.bits64[2] = x[2];
    it.x.bits64[3] = x[3];
    it.x.bits64[4] = 0;

    uint64_t *d = (uint64_t *)(itemPtr + 8);
    it.d.bits64[0] = d[0];
    it.d.bits64[1] = d[1];
    it.d.bits64[2] = d[2];
    it.d.bits64[3] = d[3];
    it.d.bits64[4] = 0;
    if(it.kIdx % 2 == WILD) it.d.ModSubK1order(&wildOffset);

    hashFound.push_back(it);
  }

  if(!kernelOk) {
    kernelInFlight = false;
    return false;
  }

  return true;

}

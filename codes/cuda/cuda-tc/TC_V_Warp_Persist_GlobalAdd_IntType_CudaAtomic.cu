#include <cuda/atomic>
typedef cuda::atomic<int> data_type;
typedef int basic_t;
static const int WS = 32;
static const int ThreadsPerBlock = 512;
static const int Device = 0;
#include "indigo_tc_vertex_cuda.h"
static __global__ void d_triCounting(data_type* g_count, const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WS;
  for (int v = idx; v < nodes; v += gridDim.x * (ThreadsPerBlock / WS)) {
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;
    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1 + threadIdx.x % WS; j < end1; j += WS){
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      (*g_count) += (basic_t)d_common(j + 1, end1, start2, end2, nlist);
    }
  }
}
static double GPUtc_vertex(basic_t &count, const int nodes, const int* const nindex, const int* const nlist)
{
  data_type* d_count;
  if (cudaSuccess != cudaMalloc((void **)&d_count, sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  timeval start, end;
  const int ThreadsBound = GPUinfo(Device);
  const int blocks = ThreadsBound / ThreadsPerBlock;
  count = 0;
  gettimeofday(&start, NULL);
  if (cudaSuccess != cudaMemcpy(d_count, &count, sizeof(data_type), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");
  d_triCounting<<<blocks, ThreadsPerBlock>>>(d_count, nodes, nindex, nlist);
  if (cudaSuccess != cudaMemcpy(&count, d_count, sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  cudaFree(d_count);
  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}

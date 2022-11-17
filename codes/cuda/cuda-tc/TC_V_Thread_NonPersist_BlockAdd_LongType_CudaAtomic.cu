#include <cuda/atomic>
typedef cuda::atomic<unsigned long long> data_type;
typedef unsigned long long basic_t;
static const int ThreadsPerBlock = 512;
#include "indigo_tc_vertex_cuda.h"
static __global__ void d_triCounting(data_type* g_count, const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist)
{
  __shared__ int count;
  if (threadIdx.x == 0) count = 0;
  __syncthreads();
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < nodes) {
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;
    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1; j < end1; j++) {
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      atomicAdd_block(&count, (basic_t)d_common(j + 1, end1, start2, end2, nlist));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) (*g_count) += count;
}
static double GPUtc_vertex(basic_t &count, const int nodes, const int* const nindex, const int* const nlist)
{
  data_type* d_count;
  if (cudaSuccess != cudaMalloc((void **)&d_count, sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  timeval start, end;
  const int blocks = (nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;
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

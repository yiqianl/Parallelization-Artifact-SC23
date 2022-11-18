#include <cuda/atomic>
typedef cuda::atomic<unsigned long long> data_type;
typedef unsigned long long basic_t;
static const int ThreadsPerBlock = 512;
#include "indigo_tc_edge_cuda.h"
static __global__ void d_triCounting(data_type* g_count, const int edges, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int* const sp)
{
  __shared__ int b_count;
  if (threadIdx.x == 0) b_count = 0;
  __syncthreads();
  const int e = blockIdx.x;
  if (e < edges) {
    const int src = sp[e];
    const int dst = nlist[e];
    if (src > dst) {
      const int beg1 = nindex[dst];
      const int end1 = nindex[dst + 1];
      basic_t count = 0;
      for (int i = beg1 + threadIdx.x; i < end1 && nlist[i] < dst; i += ThreadsPerBlock){
        const int u = nlist[i];
        int beg2 = nindex[src];
        int end2 = nindex[src + 1];
        if (d_find(u, beg2, end2, nlist)) count++;
      }
      atomicAdd_block(&b_count, count);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) (*g_count) += b_count;
}
static double GPUtc_edge(basic_t &count, const int edges, const int* const nindex, const int* const nlist, const int* const sp)
{
  data_type* d_count;
  if (cudaSuccess != cudaMalloc((void **)&d_count, sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  timeval start, end;
  const int blocks = edges;
  count = 0;
  gettimeofday(&start, NULL);
  if (cudaSuccess != cudaMemcpy(d_count, &count, sizeof(data_type), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");
  d_triCounting<<<blocks, ThreadsPerBlock>>>(d_count, edges, nindex, nlist, sp);
  if (cudaSuccess != cudaMemcpy(&count, d_count, sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  cudaFree(d_count);
  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}

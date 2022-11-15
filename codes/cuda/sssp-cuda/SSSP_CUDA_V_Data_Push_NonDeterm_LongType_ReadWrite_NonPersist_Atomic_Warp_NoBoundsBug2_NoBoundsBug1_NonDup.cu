#include <cuda/atomic>
typedef int flag_t;
typedef unsigned long long data_type;
typedef unsigned long long basic_t;
static const int ThreadsPerBlock = 512;
static const int WarpSize = 32;

#include "indigo_sssp_vertex_cuda.h"

static __global__ void init(const int src, data_type* const dist, const int size, const ECLgraph g, int* const wl1, int* const wlsize)
{
  // initialize dist array
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
  // initialize worklist
  if (v == 0) {
    wl1[0] = src;
    *wlsize = 1;
  }
}

static __global__ void sssp_vertex_data(const ECLgraph g, data_type* const dist, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, const int iter, int* const time)
{
  int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  if (idx < wl1size) {
    const int src = wl1[idx];
    const data_type s = atomicRead(&dist[src]);
    const int beg = g.nindex[src];
    const int end = g.nindex[src + 1];

    if (s != maxval) {
      __shared__ bool updated[ThreadsPerBlock / WarpSize];
      if (threadIdx.x % WarpSize == 0) updated[threadIdx.x / WarpSize] = false;
      __syncwarp();
      for (int i = beg + threadIdx.x % WarpSize; i < end; i += WarpSize) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + g.eweight[i];

        const data_type d = atomicRead(&dist[dst]);
        if (d > new_dist) {
          atomicWrite(&dist[dst], new_dist);
          if (atomicMax(&time[dst], iter) < iter) {
            wl2[atomicAdd(wl2size, 1)] = dst;
          }
          updated[threadIdx.x / WarpSize] = true;
        }
      }
      __syncwarp();
      if (threadIdx.x % WarpSize == 0) {
        if (updated[threadIdx.x / WarpSize]) {
          if (atomicMax(&time[src], iter) < iter) {
            wl2[atomicAdd(wl2size, 1)] = src;
          }
        }
      }
    }
  }
}
static double GPUsssp_vertex(const int src, const ECLgraph& g, basic_t* const dist)
{
  data_type* d_dist;
  if (cudaSuccess != cudaMalloc((void **)&d_dist, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist\n");
  int* d_wl1;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1, (g.edges + g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1\n");
  int* d_wl1size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1size\n");
  int* d_wl2;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2, (g.edges + g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2\n");
  int* d_wl2size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2size\n");
  int* d_time;
  if (cudaSuccess != cudaMalloc((void **)&d_time, sizeof(int) * g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  cudaMemset(d_time, 0, sizeof(int) * g.nodes);
  int wlsize;

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(src, d_dist, g.nodes, g, d_wl1, d_wl2size);

  if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
  if (cudaSuccess != cudaMemcpy(d_wl1size, &wlsize, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of wl1size to device failed\n");
  // iterate until no more changes
  int iter = 0;

  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    cudaMemset(d_wl2size, 0, sizeof(int));
    const int blocks = ((long)wlsize * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;

    sssp_vertex_data<<<blocks, ThreadsPerBlock>>>(g, d_dist, d_wl1, wlsize, d_wl2, d_wl2size, iter, d_time);

    if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
    std::swap(d_wl1, d_wl2);
    std::swap(d_wl1size, d_wl2size);
  } while (wlsize > 0);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);

  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  CheckCuda();
  printf("iterations: %d\n", iter);

  if (cudaSuccess != cudaMemcpy(dist, d_dist, g.nodes * sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of dist from device failed\n");

  cudaFree(d_dist);
  cudaFree(d_wl1);
  cudaFree(d_wl1size);
  cudaFree(d_wl2);
  cudaFree(d_wl2size);
  cudaFree(d_time);
  return runtime;
}

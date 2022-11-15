#include <cuda/atomic>
typedef cuda::atomic<int> flag_t;
typedef cuda::atomic<int> data_type;
typedef int basic_t;
static const int ThreadsPerBlock = 512;
static const int WarpSize = 32;

#include "indigo_bfs_vertex_cuda.h"

static __global__ void init(const int src, data_type* const dist, const int size, const ECLgraph g, int* const wl1, int* const wlsize, int* const time)
{
  // initialize dist array
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v].store(temp);
    time[v] = 0;
  }
  // initialize worklist
  if (v == 0) {
    int idx = 0;
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      wl1[idx] = g.nlist[i];
      idx++;
    }
    *wlsize = idx;
  }
}
static __global__ void bfs_vertex_data(const ECLgraph g, data_type* const dist, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, const int iter, int* const time)
{
  int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  if (idx < wl1size) {
    const int v = wl1[idx];
    data_type d = dist[v].load();
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    bool updated = false;
    for (int i = beg + threadIdx.x % WarpSize; i < end; i += WarpSize) {
      const int src = g.nlist[i];
      const data_type s = dist[src].load();
      if (s != maxval) {
        const data_type new_dist = s + 1;
        if (d > new_dist) {
          d = new_dist.load();
          updated = true;
        }
      }
    }
    if (updated) {
      dist[v].fetch_min(d, cuda::memory_order_relaxed);
      for (int j = beg; j < end; j++) {
        const int n = g.nlist[j];
        if (atomicMax(&time[n], iter) != iter) {
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
  }
}
static double GPUbfs_vertex(const int src, const ECLgraph& g, basic_t* const dist)
{
  data_type* d_dist;
  if (cudaSuccess != cudaMalloc((void **)&d_dist, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist\n");
  int* d_wl1;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1, std::max(g.edges, g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1\n");
  int* d_wl1size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1size\n");
  int* d_wl2;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2, std::max(g.edges, g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2\n");
  int* d_wl2size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2size\n");
  int* d_time;
  if (cudaSuccess != cudaMalloc((void **)&d_time, sizeof(int) * std::max(g.edges, g.nodes))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  int wlsize;

  timeval start, end;
  gettimeofday(&start, NULL);

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(src, d_dist, g.nodes, g, d_wl1, d_wl2size, d_time);

  if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
  if (cudaSuccess != cudaMemcpy(d_wl1size, &wlsize, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of wl1size to device failed\n");
  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    cudaMemset(d_wl2size, 0, sizeof(int));
    const int blocks = ((long)wlsize * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;

    bfs_vertex_data<<<blocks, ThreadsPerBlock>>>(g, d_dist, d_wl1, wlsize, d_wl2, d_wl2size, iter, d_time);

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
  return runtime;
}

#include <cuda/atomic>
typedef cuda::atomic<int> data_type;
typedef int basic_t;
static const int ThreadsPerBlock = 512;

#include "sssp_edge_cuda.h"

static __global__ void init(const int src, data_type* const dist, const int size, const ECLgraph g, int* const wl1, int* const wlsize)
{
  // initialize dist array
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v].store(temp);
  }
  // initialize worklist
  if (v == 0) {
    int idx = 0;
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      wl1[idx] = i;
      idx++;
    }
    *wlsize = idx;
  }
}
static __global__ void sssp_edge_data(const ECLgraph g, const int* const sp, data_type* const dist, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, const int iter, int* const time)
{
  for (int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock; idx < wl1size; idx += gridDim.x * ThreadsPerBlock) {
    const int e = wl1[idx];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = dist[src].load();
    if (s != maxval) {
      const data_type new_dist = s + g.eweight[e];
      data_type d = dist[dst].load();
      if (d > new_dist) {
        dist[dst].store(new_dist);
        if (atomicMax(&time[e], iter) != iter) {
          wl2[atomicAdd(wl2size, 1)] = e;
        }
        for (int j = g.nindex[dst]; j < g.nindex[dst + 1]; j++) {
          if (atomicMax(&time[j], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = j;
          }
        }
      }
    }
  }
}
static double GPUsssp_edge(const int src, const ECLgraph& g, basic_t* const dist, const int* const sp)
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
  if (cudaSuccess != cudaMalloc((void **)&d_time, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  cudaMemset(d_time, 0, sizeof(int) * g.edges);
  int* d_sp;
  if (cudaSuccess != cudaMalloc((void **)&d_sp, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate d_sp\n"); exit(-1);}
  cudaMemcpy(d_sp, sp, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  int wlsize;
  const int ThreadsBound = GPUinfo(0);
  const int blocks = ThreadsBound / ThreadsPerBlock;

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

    sssp_edge_data<<<blocks, ThreadsPerBlock>>>(g, d_sp, d_dist, d_wl1, wlsize, d_wl2, d_wl2size, iter, d_time);

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

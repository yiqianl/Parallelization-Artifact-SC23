#include <cuda/atomic>
typedef cuda::atomic<int> flag_t;
typedef cuda::atomic<unsigned long long> data_type;
typedef unsigned long long basic_t;
static const int ThreadsPerBlock = 512;
static const int WarpSize = 32;

#include "indigo_cc_vertex_cuda.h"

static __global__ void init(data_type* const label, const int size, const ECLgraph g, int* const wl1, int* const wlsize)
{
  // initialize label array
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    label[v].store(v);
    wl1[v] = v;
  }
  // initialize worklist
  if (v == 0) {
    *wlsize = size;
  }
}

static __global__ void cc_vertex_data(const ECLgraph g, data_type* const label, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, const int iter, int* const time)
{
  int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  if (idx < wl1size) {
    const int v = wl1[idx];
    data_type d = label[v].load();
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    __shared__ bool updated[ThreadsPerBlock / WarpSize];
    const int warp = threadIdx.x / WarpSize;
    const int lane = threadIdx.x % WarpSize;
    if (lane == 0) updated[warp] = false;
    __syncwarp();
    for (int i = beg + threadIdx.x % WarpSize; i < end; i += WarpSize) {
      const int src = g.nlist[i];
      const data_type new_label = label[src].load();
      if (d > new_label) {
        d = new_label.load();
        updated[warp] = true;
      }
    }
    __syncwarp();
    if (updated[warp]) {
      label[v].fetch_min(d);
      if (lane == 0) {
        for (int j = beg; j < end; j++) {
          const int n = g.nlist[j];
          if (atomicMax(&time[n], iter) < iter) {
            wl2[atomicAdd(wl2size, 1)] = n;
          }
        }
      }
    }
  }
}
static double GPUcc_vertex(const ECLgraph& g, basic_t* const label)
{
  data_type* d_label;
  if (cudaSuccess != cudaMalloc((void **)&d_label, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label\n");
  int* d_wl1;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1, std::max(g.edges, g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1\n");
  int* d_wl1size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1size\n");
  int* d_wl2;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2, std::max(g.edges, g.nodes) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2\n");
  int* d_wl2size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2size\n");
  int* d_time;
  if (cudaSuccess != cudaMalloc((void **)&d_time, sizeof(int) * g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  cudaMemset(d_time, 0, sizeof(int) * g.nodes);
  int wlsize;

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_label, g.nodes, g, d_wl1, d_wl2size);

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

    cc_vertex_data<<<blocks, ThreadsPerBlock>>>(g, d_label, d_wl1, wlsize, d_wl2, d_wl2size, iter, d_time);

    if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
    std::swap(d_wl1, d_wl2);
    std::swap(d_wl1size, d_wl2size);
  } while (wlsize > 0);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);

  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  CheckCuda();
  printf("iterations: %d\n", iter);

  if (cudaSuccess != cudaMemcpy(label, d_label, g.nodes * sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of label from device failed\n");

  cudaFree(d_label);
  cudaFree(d_wl1);
  cudaFree(d_wl1size);
  cudaFree(d_wl2);
  cudaFree(d_wl2size);
  return runtime;
}

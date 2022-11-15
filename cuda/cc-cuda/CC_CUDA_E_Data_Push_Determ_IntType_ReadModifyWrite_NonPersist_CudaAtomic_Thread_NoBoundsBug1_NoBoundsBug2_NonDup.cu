#include <cuda/atomic>

typedef cuda::atomic<int> flag_t;
typedef cuda::atomic<int> data_type;
typedef int basic_t;

#include "indigo_cc_edge_cuda.h"

static const int ThreadsPerBlock = 512;

static __global__ void init(data_type* const label, data_type* const label_n, const ECLgraph g, int* const wl1, int* const wlsize)
{
  // initialize label array
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes) {
    label_n[v].store(v);
    label[v].store(v);
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      wl1[i] = i;
    }
  }
  if (v == 0) {
    *wlsize = g.edges;
  }
}

static __global__ void cc_edge_data(const ECLgraph g, const int* const sp, data_type* const label, data_type* const label_n, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, const int iter, int* const time)
{
  int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wl1size) {
    const int e = wl1[idx];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = label[src].load();

    if (label_n[dst].fetch_min(new_label) > new_label) {
      for (int j = g.nindex[dst]; j < g.nindex[dst + 1]; j++) {
        if (atomicMax(&time[j], iter) != iter) {
          wl2[atomicAdd(wl2size, 1)] = j;
        }
      }
    }
    label_n[src].fetch_min(new_label);
  }
}

static double GPUcc_edge(const ECLgraph& g, basic_t* const label, const int* const sp)
{
  data_type* d_label;
  if (cudaSuccess != cudaMalloc((void **)&d_label, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label\n");
  data_type* d_label_new;
  if (cudaSuccess != cudaMalloc((void **)&d_label_new, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label_new\n");
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

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_label, d_label_new, g, d_wl1, d_wl2size);

  if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
  if (cudaSuccess != cudaMemcpy(d_wl1size, &wlsize, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of wl1size to device failed\n");
  // iterate until no more changes
  int iter = 0;

  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    cudaMemset(d_wl2size, 0, sizeof(int));
    const int blocks = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;

    cc_edge_data<<<blocks, ThreadsPerBlock>>>(g, d_sp, d_label, d_label_new, d_wl1, wlsize, d_wl2, d_wl2size, iter, d_time);

    if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
    std::swap(d_wl1, d_wl2);
    std::swap(d_wl1size, d_wl2size);
    std::swap(d_label, d_label_new);
  } while (wlsize > 0);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  CheckCuda();
  printf("iterations: %d\n", iter);

  if (cudaSuccess != cudaMemcpy(label, d_label_new, g.nodes * sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of label from device failed\n");

  cudaFree(d_label);
  cudaFree(d_wl1);
  cudaFree(d_wl1size);
  cudaFree(d_wl2);
  cudaFree(d_wl2size);
  return runtime;
}

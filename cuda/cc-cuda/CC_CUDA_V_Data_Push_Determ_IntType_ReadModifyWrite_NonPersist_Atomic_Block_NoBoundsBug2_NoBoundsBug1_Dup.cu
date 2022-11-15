#include <cuda/atomic>
typedef int flag_t;
typedef int data_type;
typedef int basic_t;
static const int ThreadsPerBlock = 512;

#include "indigo_cc_vertex_cuda.h"

static __global__ void init(data_type* const label, data_type* const label_n, const int size, const ECLgraph g, int* const wl1, int* const wlsize)
{
  // initialize label array
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    label_n[v] = v;
    label[v] = v;
    wl1[v] = v;
  }
  // initialize worklist
  if (v == 0) {
    // wl1[0] = 0;
    *wlsize = size;
  }
}

static __global__ void cc_vertex_data(const ECLgraph g, data_type* const label, data_type* const label_n, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size)
{
  int idx = blockIdx.x;
  if (idx < wl1size) {
    const int src = wl1[idx];
    const data_type new_label = label[src];
    const int beg = g.nindex[src];
    const int end = g.nindex[src + 1];

    for (int i = beg + threadIdx.x; i < end; i += ThreadsPerBlock) {
      const int dst = g.nlist[i];

      if (atomicMin(&label_n[dst], new_label) > new_label) {
        wl2[atomicAdd(wl2size, 1)] = dst;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicMin(&label_n[src], new_label);
    }
  }
}
static double GPUcc_vertex(const ECLgraph& g, basic_t* const label)
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
  int wlsize;

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_label, d_label_new, g.nodes, g, d_wl1, d_wl2size);

  if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device failed\n");
  if (cudaSuccess != cudaMemcpy(d_wl1size, &wlsize, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of wl1size to device failed\n");
  // iterate until no more changes
  int iter = 0;

  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    cudaMemset(d_wl2size, 0, sizeof(int));
    const int blocks = wlsize;

    cc_vertex_data<<<blocks, ThreadsPerBlock>>>(g, d_label, d_label_new, d_wl1, wlsize, d_wl2, d_wl2size);

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

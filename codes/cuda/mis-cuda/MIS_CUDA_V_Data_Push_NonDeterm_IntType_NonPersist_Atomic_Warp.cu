#include <cuda/atomic>
typedef int flag_t;
typedef int data_type;
static const int ThreadsPerBlock = 512;
static const int WarpSize = 32;

#include "mis_vertex_cuda.h"

static __global__ void init(data_type* const priority, flag_t* const status, const int size, int* const wl1, int* const wlsize)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;

    // initialize worklist
    wl1[v] = v;
  }
  if (v == 0) {
    *wlsize = size;
  }
}

static __global__ void mis(const ECLgraph g, const data_type* const priority, flag_t* const status, const int* const wl1, const int wl1size, int* const wl2, int* const wl2size)
{
  const int lane = threadIdx.x % WarpSize;
  // go over all nodes in worklist
  int w = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  if (w < wl1size) {

    int v = wl1[w];
    if (__any_sync(~0, (lane == 0) && (atomicRead(&status[v]) == undecided))) {
      int i = g.nindex[v];
      // try to find a non-excluded neighbor whose priority is higher
      if (lane == 0) {
        while ((i < g.nindex[v + 1]) && ((atomicRead(&status[g.nlist[i]]) == excluded) || (priority[v] > priority[g.nlist[i]]) || ((priority[v] == priority[g.nlist[i]]) && (v > g.nlist[i])))) {
          i++;
        }
      }
      if (__any_sync(~0, (lane == 0) && (i < g.nindex[v + 1]))) {
        // found such a neighbor -> status still unknown
        if (lane == 0) {
          wl2[atomicAdd(wl2size, 1)] = v;
        }
      } else {
        // no such neighbor -> all neighbors are "excluded" and v is "included"
        if (lane == 0) {
          atomicWrite(&status[v], included);
        }
        for (int j = g.nindex[v] + lane; j < g.nindex[v + 1]; j += WarpSize) {
          atomicWrite(&status[g.nlist[j]], excluded);
        }
      }
    }
  }
}

static double GPUmis_vertex(const ECLgraph& g, data_type* const priority, int* const status)
{
  data_type* d_priority;
  if (cudaSuccess != cudaMalloc((void **)&d_priority, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_priority\n");
  flag_t* d_status;
  if (cudaSuccess != cudaMalloc((void **)&d_status, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_status\n");

  int* d_wl1;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1\n");
  int* d_wl1size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1size\n");
  int* d_wl2;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2\n");
  int* d_wl2size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2size\n");
  int wlsize;


  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_priority, d_status, g.nodes, d_wl1, d_wl1size);

  if (cudaSuccess != cudaMemcpy(&wlsize, d_wl1size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wlsize from device d_wl1size failed\n");

  timeval beg, end;
  gettimeofday(&beg, NULL);

  int iter = 0;
  do {
    iter++;
    cudaMemset(d_wl2size, 0, sizeof(int));
    const int blocks = ((long)wlsize * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;

    mis<<<blocks, ThreadsPerBlock>>>(g, d_priority, d_status, d_wl1, wlsize, d_wl2, d_wl2size);

    if (cudaSuccess != cudaMemcpy(&wlsize, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) { fprintf(stderr, "ERROR: copying of wlsize from device failed\n"); break; }
    std::swap(d_wl1, d_wl2);
    std::swap(d_wl1size, d_wl2size);
  } while (wlsize > 0);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(status, d_status, g.nodes * sizeof(flag_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of status from device failed\n");

  // determine and print set size
  int cnt = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == included) cnt++;
  }
  printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

  cudaFree(d_status);
  cudaFree(d_priority);
  cudaFree(d_wl1);
  cudaFree(d_wl1size);
  cudaFree(d_wl2);
  cudaFree(d_wl2size);
  return runtime;
}

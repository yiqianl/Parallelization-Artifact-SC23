#include <cuda/atomic>
typedef int flag_t;
typedef unsigned long long data_type;
typedef unsigned long long basic_t;
static const int ThreadsPerBlock = 512;

#include "indigo_bfs_edge_cuda.h"

static __global__ void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
}

static __global__ void bfs(const ECLgraph g, const int* const sp, data_type* const dist, flag_t* const goagain)
{
  for (int e = threadIdx.x + blockIdx.x * ThreadsPerBlock; e < g.edges; e += gridDim.x * ThreadsPerBlock) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = atomicRead(&dist[src]);

    if (s != maxval) {
      const data_type new_dist = s + 1;
      if (atomicMin(&dist[dst], new_dist) > new_dist) {
        atomicWrite(goagain, 1);
      }
    }
  }
}

static double GPUbfs_edge(const int src, const ECLgraph& g, basic_t* const dist, const int* const sp)
{
  flag_t* d_goagain;
  data_type* d_dist;
  int* d_sp;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  if (cudaSuccess != cudaMalloc((void **)&d_dist, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist\n");
  if (cudaSuccess != cudaMalloc((void **)&d_sp, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate d_sp\n"); exit(-1);}
  cudaMemcpy(d_sp, sp, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  const int ThreadsBound = GPUinfo(0);
  const int blocks = ThreadsBound / ThreadsPerBlock;

  timeval start, end;
  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(src, d_dist, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  gettimeofday(&start, NULL);

  do {
    iter++;
    goagain = 0;
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");

    bfs<<<blocks, ThreadsPerBlock>>>(g, d_sp, d_dist, d_goagain);

    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
  } while (goagain);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(dist, d_dist, g.nodes * sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of dist from device failed\n");

  cudaFree(d_goagain);
  cudaFree(d_dist);
  return runtime;
}

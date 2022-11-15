#include <cuda/atomic>
typedef int flag_t;
typedef int data_type;
typedef int basic_t;
static const int ThreadsPerBlock = 512;

#include "indigo_sssp_vertex_cuda.h"

static __global__ void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
}

static __global__ void sssp(const ECLgraph g, data_type* const dist, flag_t* const goagain)
{
  int tid = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  for (int v = tid; v < g.nodes; v += gridDim.x * ThreadsPerBlock) {

    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = atomicRead(&dist[v]);

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type s = atomicRead(&dist[src]);
      if (s != maxval) {
        const data_type new_dist = s + g.eweight[i];
        if (new_dist < d) {
          d = new_dist;
          atomicWrite(goagain, 1);
        }
      }
    }
    atomicWrite(&dist[v], d);
  }
}

static double GPUsssp_vertex(const int src, const ECLgraph& g, basic_t* const dist)
{
  flag_t* d_goagain;
  data_type* d_dist;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  if (cudaSuccess != cudaMalloc((void **)&d_dist, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist\n");

  const int ThreadsBound = GPUinfo(0);
  const int blocks = ThreadsBound / ThreadsPerBlock;

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(src, d_dist, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;

  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    goagain = 0;
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(flag_t), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");

    sssp<<<blocks, ThreadsPerBlock>>>(g, d_dist, d_goagain);

    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(flag_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
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

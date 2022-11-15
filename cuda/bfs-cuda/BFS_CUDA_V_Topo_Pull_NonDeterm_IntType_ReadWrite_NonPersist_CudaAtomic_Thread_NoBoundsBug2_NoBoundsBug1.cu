#include <cuda/atomic>
typedef cuda::atomic<int> flag_t;
typedef cuda::atomic<int> data_type;
typedef int basic_t;
static const int ThreadsPerBlock = 512;

#include "indigo_bfs_vertex_cuda.h"

static __global__ void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v].store(temp);
  }
}

static __global__ void bfs(const int iter, const ECLgraph g, data_type* const dist, flag_t* const goagain)
{
  int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes) {

    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = dist[v].load();

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type s = dist[src].load();
      if (s != maxval) {
        const data_type new_dist = s + 1;
        if (new_dist < d) {
          d = new_dist.load();
          *goagain = 1;
        }
      }
    }
    dist[v].store(d);
  }
}

static double GPUbfs_vertex(const int src, const ECLgraph& g, basic_t* const dist)
{
  flag_t* d_goagain;
  data_type* d_dist;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  if (cudaSuccess != cudaMalloc((void **)&d_dist, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist\n");

  const int blocks = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;

  timeval start, end;

  init<<<blocks, ThreadsPerBlock>>>(src, d_dist, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  gettimeofday(&start, NULL);

  do {
    goagain = 0;
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(flag_t), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");

    bfs<<<blocks, ThreadsPerBlock>>>(iter, g, d_dist, d_goagain);
    iter++;

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

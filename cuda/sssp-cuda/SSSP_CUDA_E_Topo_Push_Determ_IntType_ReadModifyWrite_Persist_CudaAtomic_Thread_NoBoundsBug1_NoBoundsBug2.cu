#include <cuda/atomic>
typedef cuda::atomic<int> flag_t;
typedef cuda::atomic<int> data_type;
typedef int basic_t;
static const int ThreadsPerBlock = 512;

#include "indigo_sssp_edge_cuda.h"

static __global__ void init(const int src, data_type* const dist, data_type* const dist_n, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v].store(temp);
    dist[v].store(temp);
  }
}

static __global__ void sssp(const ECLgraph g, const int* const sp, data_type* const dist, data_type* const dist_n, flag_t* const goagain)
{
  for (int e = threadIdx.x + blockIdx.x * ThreadsPerBlock; e < g.edges; e += gridDim.x * ThreadsPerBlock) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = dist[src].load();

    if (s != maxval) {
      const data_type new_dist = s + g.eweight[e];
      if (dist_n[dst].fetch_min(new_dist, cuda::memory_order_relaxed) > new_dist) {
        *goagain = 1;
      }
    }
  }
}

static double GPUsssp_edge(const int src, const ECLgraph& g, basic_t* const dist, const int* const sp)
{
  flag_t* d_goagain;
  data_type* d_dist;
  int* d_sp;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  if (cudaSuccess != cudaMalloc((void **)&d_dist, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist\n");
  if (cudaSuccess != cudaMalloc((void **)&d_sp, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate d_sp\n"); exit(-1);}
  data_type* d_dist_new;
  if (cudaSuccess != cudaMalloc((void **)&d_dist_new, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_dist_new\n");
  cudaMemcpy(d_sp, sp, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  const int ThreadsBound = GPUinfo(0);
  const int blocks = ThreadsBound / ThreadsPerBlock;

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(src, d_dist, d_dist_new, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;

  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    goagain = 0;
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");

    sssp<<<blocks, ThreadsPerBlock>>>(g, d_sp, d_dist, d_dist_new, d_goagain);

    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
    std::swap(d_dist, d_dist_new);
  } while (goagain);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(dist, d_dist_new, g.nodes * sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of dist from device failed\n");

  cudaFree(d_goagain);
  cudaFree(d_dist);
  return runtime;
}

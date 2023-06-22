#include <cuda/atomic>

typedef cuda::atomic<int> flag_t;
typedef cuda::atomic<int> data_type;
typedef int basic_t;

#include "cc_edge_cuda.h"

static const int ThreadsPerBlock = 512;

static __global__ void init(data_type* const label, data_type* const label_n, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    label_n[v].store(v);
    label[v].store(v);
  }
}

static __global__ void cc(const ECLgraph g, const int* const sp, data_type* const label, data_type* const label_n, flag_t* const goagain)
{
  for (int e = threadIdx.x + blockIdx.x * ThreadsPerBlock; e < g.edges; e += gridDim.x * ThreadsPerBlock) {
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = label[src].load();

    if (label_n[dst].fetch_min(new_label) > new_label) {
      *goagain = 1;
    }
  }
}

static double GPUcc_edge(const ECLgraph& g, basic_t* const label, const int* const sp)
{
  flag_t* d_goagain;
  data_type* d_label;
  int* d_sp;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  if (cudaSuccess != cudaMalloc((void **)&d_label, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label\n");
  if (cudaSuccess != cudaMalloc((void **)&d_sp, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate d_sp\n"); exit(-1);}
  data_type* d_label_new;
  if (cudaSuccess != cudaMalloc((void **)&d_label_new, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label_new\n");
  cudaMemcpy(d_sp, sp, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  const int ThreadsBound = GPUinfo(0);
  const int blocks = ThreadsBound / ThreadsPerBlock;

  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_label, d_label_new, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;

  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    goagain = 0;
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");

    cc<<<blocks, ThreadsPerBlock>>>(g, d_sp, d_label, d_label_new, d_goagain);

    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
    std::swap(d_label, d_label_new);
  } while (goagain);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(label, d_label_new, g.nodes * sizeof(data_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of label from device failed\n");

  cudaFree(d_goagain);
  cudaFree(d_label);
  return runtime;
}

#include <cuda/atomic>

typedef int flag_t;
typedef int data_type;
typedef int basic_t;

#include "indigo_cc_vertex_cuda.h"

static const int ThreadsPerBlock = 512;

static __global__ void init(data_type* const label, data_type* const label_n, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size) {
    label_n[v] = v;
    label[v] = v;
  }
}

static __global__ void cc(const ECLgraph g, data_type* const label, data_type* const label_n, flag_t* const goagain)
{
  int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes) {

    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = label[v];

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type new_label  = label[src];
      if (new_label < d) {
        d = new_label;
        atomicWrite(goagain, 1);
      }
    }
    atomicWrite(&label_n[v], d);
  }
}

static double GPUcc_vertex(const ECLgraph& g, basic_t* const label)
{
  flag_t* d_goagain;
  data_type* d_label;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  if (cudaSuccess != cudaMalloc((void **)&d_label, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label\n");
  data_type* d_label_new;
  if (cudaSuccess != cudaMalloc((void **)&d_label_new, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_label_new\n");

  const int blocks = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;

  init<<<blocks, ThreadsPerBlock>>>(d_label, d_label_new, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  timeval start, end;
  gettimeofday(&start, NULL);

  do {
    iter++;
    goagain = 0;
    if (cudaSuccess != cudaMemcpy(d_goagain, &goagain, sizeof(flag_t), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of go_again to device failed\n");

    cc<<<blocks, ThreadsPerBlock>>>(g, d_label, d_label_new, d_goagain);

    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(flag_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
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

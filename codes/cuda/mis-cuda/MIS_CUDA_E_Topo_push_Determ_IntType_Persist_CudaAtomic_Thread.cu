#include <cuda/atomic>
typedef cuda::atomic<int> flag_t;
typedef int data_type;
static const int ThreadsPerBlock = 512;

#include "mis_edge_cuda.h"

static __global__ void init(data_type* const priority, flag_t* const status, flag_t* const status_n, flag_t* const lost, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    status_n[v] = undecided;
    lost[v] = 0;
  }
}

static __global__ void mis(const ECLgraph g, const int* const sp, const data_type* const priority, flag_t* const status, flag_t* const status_n, flag_t* const lost)
{
  // go over all edges
  int tid = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  for (int e = tid; e < g.edges; e += gridDim.x * ThreadsPerBlock) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const int srcStatus = status[src].load();
    const int dstStatus = status[dst].load();

    // if one is included, exclude the other
    if (srcStatus == included) {
      status_n[dst].store(excluded);
    }
    else if (dstStatus == included) {
      status_n[src].store(excluded);
    } else if (srcStatus == undecided && dstStatus == undecided) {
      // if both undecided -> mark lower as lost
      if (priority[src] < priority[dst]) {
        lost[src].store(1);
      } else {
        lost[dst].store(1);
      }
    }
  }
}

static __global__ void mis_vertex_pass(const ECLgraph g, const int* const sp, flag_t* const status, flag_t* const status_n, flag_t* const lost, flag_t* const goagain)
{
  // go over all edges
  int tid = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  for (int e = tid; e < g.edges; e += gridDim.x * ThreadsPerBlock) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const int srcStatus = status[src].load();
    const int dstStatus = status[dst].load();

    // if v didn't lose
    // if src won
    if (lost[src] == 0) {
      if (srcStatus == undecided) {
        // and is undecided -> include
        status_n[src].store(included);
      }
    }
    // if dst won
    if (lost[dst] == 0) {
      if (dstStatus == undecided) {
        // and is undecided -> include
        status_n[dst].store(included);
      }
    }
    // if either is still undecided, goagain
    if (srcStatus == undecided || dstStatus == undecided) {
      *goagain = 1;
    }
  }
}

static __global__ void mis_last_pass(flag_t* const status, const int size)
{
  int tid = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  for (int w = tid; w < size; w += gridDim.x * ThreadsPerBlock) {
    if (status[w] == undecided)
    {
      status[w] = included;
    }
  }
}

static double GPUmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, int* const status)
{
  flag_t* d_goagain;
  if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
  data_type* d_priority;
  if (cudaSuccess != cudaMalloc((void **)&d_priority, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_priority\n");
  flag_t* d_status;
  if (cudaSuccess != cudaMalloc((void **)&d_status, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_status\n");
  flag_t* d_lost;
  if (cudaSuccess != cudaMalloc((void **)&d_lost, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_lost\n");
  flag_t* d_status_new;
  if (cudaSuccess != cudaMalloc((void **)&d_status_new, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_status_new\n");

  const int ThreadsBound = GPUinfo(0, false);
  const int blocks = ThreadsBound / ThreadsPerBlock;

  init<<<(g.edges + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_priority, d_status, d_status_new, d_lost, g.nodes);

  timeval beg, end;
  gettimeofday(&beg, NULL);

  flag_t goagain;
  int iter = 0;
  do {
    iter++;
    cudaMemset(d_goagain, 0, sizeof(flag_t));

    // edge pass
    mis<<<blocks, ThreadsPerBlock>>>(g, sp, d_priority, d_status, d_status_new, d_lost);

    if (cudaSuccess != cudaMemcpy(d_status, d_status_new, g.nodes * sizeof(flag_t), cudaMemcpyDeviceToDevice)) fprintf(stderr, "ERROR: copying of d_status_new to d_status on device failed\n");
    // vertex pass
    mis_vertex_pass<<<blocks, ThreadsPerBlock>>>(g, sp, d_status, d_status_new, d_lost, d_goagain);

    cudaMemset(d_lost, 0, g.nodes * sizeof(flag_t));
    if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of go_again from device failed\n");
    if (cudaSuccess != cudaMemcpy(d_status, d_status_new, g.nodes * sizeof(flag_t), cudaMemcpyDeviceToDevice)) fprintf(stderr, "ERROR: copying of d_status_new to d_status on device failed\n");
  } while (goagain);

  // include all remaining nodes that have no edges
  mis_last_pass<<<blocks, ThreadsPerBlock>>>(d_status, g.nodes);

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

  cudaFree(d_status_new);
  cudaFree(d_status);
  cudaFree(d_priority);
  cudaFree(d_lost);
  cudaFree(d_goagain);
  return runtime;
}

#include <cuda/atomic>
typedef int flag_t;
typedef unsigned long long data_type;
static const int ThreadsPerBlock = 512;

#include "indigo_mis_vertex_cuda.h"

static __global__ void init(data_type* const priority, flag_t* const status, flag_t* const status_n, const int size)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < size)
  {
    priority[v] = ((unsigned long)hash(v + 712313887)) | ((unsigned long)hash(v + 683067839) << (sizeof (unsigned int) * 8));
    status[v] = undecided;
    status_n[v] = undecided;
  }
}

static __global__ void mis(const ECLgraph g, const data_type* const priority, flag_t* const status, flag_t* const status_n, flag_t* const goagain)
{
  // go over all the nodes
  int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes) {

    if (atomicRead(&status[v]) == undecided) {
      int i = g.nindex[v];
      // try to find a non-excluded neighbor whose priority is higher
      while ((i < g.nindex[v + 1]) && ((atomicRead(&status[g.nlist[i]]) == excluded) || (priority[v] > priority[g.nlist[i]]) || ((priority[v] == priority[g.nlist[i]]) && (v > g.nlist[i])))) {
        i++;
      }
      if (i < g.nindex[v + 1]) {
        // found such a neighbor -> check if neighbor is included
        if (atomicRead(&status[g.nlist[i]]) == included) {
          // found included neighbor -> exclude self
          atomicWrite(&status_n[v], excluded);
        } else { // v still undecided, go again
        atomicWrite(goagain, 1);
      }
    } else {
      // no such neighbor -> self is "included"
      atomicWrite(&status_n[v], included);
    }
  }
}
}

static double GPUmis_vertex(const ECLgraph& g, data_type* const priority, int* const status)
{
flag_t* d_goagain;
if (cudaSuccess != cudaMalloc((void **)&d_goagain, sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_goagain\n");
data_type* d_priority;
if (cudaSuccess != cudaMalloc((void **)&d_priority, g.nodes * sizeof(data_type))) fprintf(stderr, "ERROR: could not allocate d_priority\n");
flag_t* d_status;
if (cudaSuccess != cudaMalloc((void **)&d_status, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_status\n");
flag_t* d_status_new;
if (cudaSuccess != cudaMalloc((void **)&d_status_new, g.nodes * sizeof(flag_t))) fprintf(stderr, "ERROR: could not allocate d_status_new\n");

const int blocks = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;

init<<<blocks, ThreadsPerBlock>>>(d_priority, d_status, d_status_new, g.nodes);

timeval beg, end;
gettimeofday(&beg, NULL);

flag_t goagain;
int iter = 0;
do {
  iter++;
  cudaMemset(d_goagain, 0, sizeof(flag_t));

  mis<<<blocks, ThreadsPerBlock>>>(g, d_priority, d_status, d_status_new, d_goagain);

  if (cudaSuccess != cudaMemcpy(d_status, d_status_new, g.nodes * sizeof(flag_t), cudaMemcpyDeviceToDevice)) fprintf(stderr, "ERROR: copying of d_status_new to d_status on device failed\n");
  if (cudaSuccess != cudaMemcpy(&goagain, d_goagain, sizeof(flag_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of goagain from device failed\n");
} while (goagain);

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
cudaFree(d_goagain);
cudaFree(d_status);
cudaFree(d_priority);
return runtime;
}

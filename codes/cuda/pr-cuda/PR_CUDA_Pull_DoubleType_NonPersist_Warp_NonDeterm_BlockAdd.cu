#include <algorithm>
#include <sys/time.h>
#include "ECLgraph.h"

typedef double score_type;
static const score_type EPSILON = 0.0001;
static const score_type kDamp = 0.85;
static const int ThreadsPerBlock = 512;
static const int MAX_ITER = 100;
static const int WarpSize = 32;


static int GPUinfo(const int d)
{
  cudaSetDevice(d);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, d);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int SMs = deviceProp.multiProcessorCount;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  return SMs * mTpSM;
}

template <typename T>
static __device__ inline T block_sum_reduction(T val, void* buffer)  // returns sum to all threads
{
  const int lane = threadIdx.x % WarpSize;
  const int warp = threadIdx.x / WarpSize;
  const int warps = ThreadsPerBlock / WarpSize;
  T* const s_carry = (T*)buffer;
  val += __shfl_xor_sync(~0, val, 1);  // MB: use reduction on 8.6 CC
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 16);
  if (lane == 0) s_carry[warp] = val;
  __syncthreads();  // s_carry written
  if (warps > 1) {
    if (warp == 0) {
      val = (lane < warps) ? s_carry[lane] : 0;
      val += __shfl_xor_sync(~0, val, 1);  // MB: use reduction on 8.6 CC
      val += __shfl_xor_sync(~0, val, 2);
      val += __shfl_xor_sync(~0, val, 4);
      val += __shfl_xor_sync(~0, val, 8);
      val += __shfl_xor_sync(~0, val, 16);
      s_carry[lane] = val;
    }
    __syncthreads();  // s_carry updated
  }
  return s_carry[0];
}

__global__ void contrib(int nodes, score_type* scores, int* degree, score_type* outgoing_contrib)
{
  int src = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  if (src < nodes) {
    outgoing_contrib[src] = scores[src] / degree[src];
  }
}

__global__ void pull(int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist, score_type* scores, score_type* outgoing_contrib, score_type* diff, score_type base_score, int* degree)
{
  __shared__ score_type local_diff;
  if (threadIdx.x == 0) local_diff = 0;
  __syncthreads();
  const int lane = threadIdx.x % WarpSize;
  __shared__ score_type buffer[WarpSize];
  score_type error = 0;
  int src = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  if (src < nodes) {
    score_type incoming_total = 0;
    const int beg = nindex[src];
    const int end = nindex[src + 1];
    for (int i = beg + threadIdx.x % WarpSize; i < end; i += WarpSize) {
      int dst = nlist[i];
      incoming_total +=  scores[dst] / degree[dst];
    }
    __syncwarp();
    score_type tmp = __shfl_up_sync(~0, incoming_total, 1);
    if (lane >= 1) incoming_total += tmp;
    tmp = __shfl_up_sync(~0, incoming_total, 2);
    if (lane >= 2) incoming_total += tmp;
    tmp = __shfl_up_sync(~0, incoming_total, 4);
    if (lane >= 4) incoming_total += tmp;
    tmp = __shfl_up_sync(~0, incoming_total, 8);
    if (lane >= 8) incoming_total += tmp;
    tmp = __shfl_up_sync(~0, incoming_total, 16);
    if (lane >= 16) incoming_total += tmp;
    if (lane == 31) {
      score_type old_score = scores[src];
      const score_type value = base_score + kDamp * incoming_total;
      scores[src] = value;
      error = fabs(value - old_score);
      atomicAdd_block(&local_diff, error);
    }
    atomicAdd(diff, local_diff);
  }
}

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

void PR_GPU(const ECLgraph g, score_type *scores, int* degree)
{
  ECLgraph d_g = g;
  int *d_degree;
  score_type *d_scores, *d_sums, *d_contrib;
  score_type *d_diff, h_diff;
  // allocate device memory
  cudaMalloc((void **)&d_degree, g.nodes * sizeof(int));
  cudaMalloc((void **)&d_scores, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_sums, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_contrib, g.nodes * sizeof(score_type));
  cudaMalloc((void **)&d_diff, sizeof(score_type));
  if (cudaSuccess != cudaMalloc((void **)&d_g.nindex, (g.nodes + 1) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nindex\n");
  if (cudaSuccess != cudaMalloc((void **)&d_g.nlist, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nlist\n");
  // copy data to device
  cudaMemcpy(d_degree, degree, g.nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, scores, g.nodes * sizeof(score_type), cudaMemcpyHostToDevice);
  if (cudaSuccess != cudaMemcpy(d_g.nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of index to device failed\n");
  if (cudaSuccess != cudaMemcpy(d_g.nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of nlist to device failed\n");
  const unsigned int blocks = ((unsigned int)g.nodes * (unsigned int)WarpSize + (unsigned int)ThreadsPerBlock - 1) / (unsigned int)ThreadsPerBlock;
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  // timer
  const int runs = 1;
  timeval start, end;
  double runtimes[runs];
  for (int i = 0; i < runs; i++) {
    int iter = 0;
    gettimeofday(&start, NULL);
    do {
      iter++;
      h_diff = 0;
      if (cudaSuccess != cudaMemcpy(d_diff, &h_diff, sizeof(score_type), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of h_diff to device failed\n");
      pull<<<blocks, ThreadsPerBlock>>>(g.nodes, d_g.nindex, d_g.nlist, d_scores, d_contrib, d_diff, base_score, d_degree);
      if (cudaSuccess != cudaMemcpy(&h_diff, d_diff, sizeof(score_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_diff from device failed\n");
    } while (h_diff > EPSILON && iter < MAX_ITER);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    runtimes[i] = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("GPU iterations = %d.\n", iter);
  }
  const double med = median(runtimes, runs);
  printf("GPU runtime: %.6fs\n\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);
  if (cudaSuccess != cudaMemcpy(scores, d_scores, g.nodes * sizeof(score_type), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_scores from device failed\n");
  cudaFree(d_degree);
  cudaFree(d_scores);
  cudaFree(d_sums);
  cudaFree(d_contrib);
  cudaFree(d_diff);
  return;
}
int main(int argc, char *argv[]) {
  printf("PageRank CUDA v0.1 (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");
  if (argc != 2) {printf("USAGE: %s input_graph\n\n", argv[0]);  exit(-1);}
  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);
  // count degree
  int* degree = new int [g.nodes];
  for (int i = 0; i < g.nodes; i++) {
    degree[i] = g.nindex[i + 1] - g.nindex[i];
  }
  // init scores
  const score_type init_score = 1.0f / (score_type)g.nodes;
  score_type* scores = new score_type [g.nodes];
  std::fill(scores, scores + g.nodes, init_score);
  PR_GPU(g, scores, degree);
  // compare and verify
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* incomming_sums = (score_type*)malloc(g.nodes * sizeof(score_type));
  for(int i = 0; i < g.nodes; i++) incomming_sums[i] = 0;
  double error = 0;
  for (int src = 0; src < g.nodes; src++) {
    score_type outgoing_contrib = scores[src] / degree[src];
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      incomming_sums[g.nlist[i]] += outgoing_contrib;
    }
  }
  for (int i = 0; i < g.nodes; i++) {
    score_type new_score = base_score + kDamp * incomming_sums[i];
    error += fabs(new_score - scores[i]);
    incomming_sums[i] = 0;
  }
  if (error < EPSILON) printf("All good.\n");
  else printf("Total Error: %f\n", error);
  return 0;
}

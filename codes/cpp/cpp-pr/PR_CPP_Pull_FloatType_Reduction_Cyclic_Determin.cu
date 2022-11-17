#include <algorithm>
#include <sys/time.h>
#include <math.h>
#include <thread>
#include <atomic>
#include "ECLgraph.h"
typedef float score_type;
static const score_type EPSILON = 0.0001;
static const score_type kDamp = 0.85;
static const int MAX_ITER = 100;

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

template <typename T>
static inline T atomicAdd(T* addr, T val)
{
  T old = ((std::atomic<T>*)addr)->load();
  while (old != old+val && !(((std::atomic<T>*)addr)->compare_exchange_weak(old, old+val))) {}
  return old;
}

static void errorCalc(const ECLgraph g, double& error, score_type* outgoing_contrib, score_type* const scores, const int* const degree, const score_type base_score, const int threadID, const int threadCount)
{
  double local_error = 0;
  const int top = g.nodes;
  for (int i = threadID; i < top; i += threadCount) {
    score_type incoming_total = 0;
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      incoming_total += outgoing_contrib[nei];
    }
    score_type old_score = scores[i];
    scores[i] = base_score + kDamp * incoming_total;
    local_error += fabs(scores[i] - old_score);
  }
  error = local_error;
}

int main(int argc, char *argv[]) {
  printf("PageRank CPP v0.1 (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");
  if (argc != 2 && argc != 3) {printf("USAGE: %s input_graph thread_count(optional)\n\n", argv[0]);  exit(-1);}
  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);
  // count degree
  int* degree = (int*)malloc(g.nodes * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    degree[i] = g.nindex[i + 1] - g.nindex[i];
  }
  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 3)
  if(const int countInt = atoi(argv[2])) //checks for valid int
  threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n\n", threadCount);
  const int nodes = g.nodes;
  // init scores
  const score_type init_score = 1.0f / (score_type)g.nodes;
  score_type* scores = (score_type*)malloc(nodes * sizeof(score_type));
  std::fill(scores, scores + g.nodes, init_score);
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* outgoing_contrib = (score_type*)malloc(nodes * sizeof(score_type));
  std::thread threadHandles[threadCount];
  double localSums[threadCount];
  int iter;
  timeval start, end;
  gettimeofday(&start, NULL);
  for (iter = 0; iter < MAX_ITER; iter++) {
    double error = 0;
    for (int i = 0; i < nodes; i++) outgoing_contrib[i] = scores[i] / degree[i];
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(errorCalc, g, std::ref(localSums[i]), outgoing_contrib, scores, degree, base_score, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
      error += localSums[i]; //sum reduction
    }
    if (error < EPSILON) break;
  }
  gettimeofday(&end, NULL);
  if (iter < MAX_ITER) iter++;
  const float runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("CPU iterations = %d.\n", iter);
  printf("CPU runtime: %.6fs\n\n", runtime);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / runtime);
  // compare and verify
  score_type* incomming_sums = (score_type*)malloc(g.nodes * sizeof(score_type));
  for(int i = 0; i < g.nodes; i++) incomming_sums[i] = 0;
  double error = 0;
  for (int src = 0; src < g.nodes; src++) {
    score_type outgoing = scores[src] / degree[src];
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      incomming_sums[g.nlist[i]] += outgoing;
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

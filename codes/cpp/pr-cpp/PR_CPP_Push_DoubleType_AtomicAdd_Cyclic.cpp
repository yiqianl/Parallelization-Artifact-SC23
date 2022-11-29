typedef double score_type;
#include "indigo_pr_cpp.h"

static void incomingCalc(const ECLgraph g, score_type* outgoing_contrib, score_type* incoming_total, const int threadID, const int threadCount)
{
  const int top = g.nodes;
  for (int i = threadID; i < top; i += threadCount) {
    const score_type outgoing = outgoing_contrib[i];
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      atomicAdd(&incoming_total[nei], outgoing);
    }
  }
}

static void errorCalc(const int nodes, double& error, score_type* outgoing_contrib, score_type* incoming_total, score_type* scores, const score_type base_score, const int threadID, const int threadCount)
{
  double local_error = 0;
  const int top = nodes;
  for (int i = threadID; i < top; i += threadCount) {
    score_type incoming = incoming_total[i];
    score_type old_score = scores[i];
    const score_type value = base_score + kDamp * incoming;
    scores[i] = value;
    local_error += fabs(value - old_score);
  }
  atomicAdd(&error, local_error);
}

static double PR_CPU(const ECLgraph g, score_type *scores, int* degree, const int threadCount)
{
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* outgoing_contrib = (score_type*)malloc(g.nodes * sizeof(score_type));
  score_type* incoming_total = (score_type*)malloc(g.nodes * sizeof(score_type));
  std::thread threadHandles[threadCount];

  int iter;
  timeval start, end;
  gettimeofday(&start, NULL);

  for (iter = 0; iter < MAX_ITER; iter++) {
    double error = 0;
    for (int i = 0; i < g.nodes; i++) {
      outgoing_contrib[i] = scores[i] / degree[i];
      incoming_total[i] = 0;
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(incomingCalc, g, outgoing_contrib, incoming_total, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(errorCalc, g.nodes, std::ref(error), outgoing_contrib, incoming_total, scores, base_score, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }
    if (error < EPSILON) break;
  }

  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  if (iter < MAX_ITER) iter++;
  printf("CPU iterations = %d.\n", iter);

  return runtime;
}

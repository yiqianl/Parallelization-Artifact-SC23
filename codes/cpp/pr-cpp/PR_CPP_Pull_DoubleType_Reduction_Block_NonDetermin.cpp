typedef double score_type;
#include "indigo_pr_cpp.h"

static void errorCalc(const ECLgraph g, double& error, score_type* outgoing_contrib, score_type* const scores, const int* const degree, const score_type base_score, const int threadID, const int threadCount)
{
  double local_error = 0;
  const int begNode = threadID * (long)g.nodes / threadCount;
  const int endNode = (threadID + 1) * (long)g.nodes / threadCount;
  for (int i = begNode; i < endNode; i++) {
    score_type incoming_total = 0;
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      incoming_total += scores[nei] / degree[nei];
    }
    score_type old_score = scores[i];
    scores[i] = base_score + kDamp * incoming_total;
    local_error += fabs(scores[i] - old_score);
  }
  error = local_error;
}

static double PR_CPU(const ECLgraph g, score_type *scores, int* degree, const int threadCount)
{
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* outgoing_contrib = (score_type*)malloc(g.nodes * sizeof(score_type));
  std::thread threadHandles[threadCount];
  double localSums[threadCount];

  int iter;
  timeval start, end;
  gettimeofday(&start, NULL);

  for (iter = 0; iter < MAX_ITER; iter++) {
    double error = 0;
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
  const double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  if (iter < MAX_ITER) iter++;
  printf("CPU iterations = %d.\n", iter);

  return runtime;
}

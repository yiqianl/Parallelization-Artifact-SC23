typedef float score_type;
#include "indigo_pr_omp.h"

void PR_CPU(const ECLgraph g, score_type *scores, int* degree)
{
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* outgoing_contrib = (score_type*)malloc(g.nodes * sizeof(score_type));
  int iter;
  timeval start, end;
  gettimeofday(&start, NULL);
  for (iter = 0; iter < MAX_ITER; iter++) {
    double error = 0;
    #pragma omp parallel for reduction(+: error)
    for (int i = 0; i < g.nodes; i++) {
      score_type incoming_total = 0;
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int nei = g.nlist[j];
        incoming_total += scores[nei] / degree[nei];
      }
      score_type old_score = scores[i];
      scores[i] = base_score + kDamp * incoming_total;
      error += fabs(scores[i] - old_score);
    }
    if (error < EPSILON) break;
  }
  gettimeofday(&end, NULL);
  if (iter < MAX_ITER) iter++;
  printf("CPU iterations = %d.\n", iter);
  const float runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("CPU iterations = %d.\n", iter);
  printf("CPU runtime: %.6fs\n\n", runtime);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / runtime);
}

typedef int basic_t;
#include "tc_vertex_omp.h"

static void triCounting(basic_t& g_count, const int nodes, const int* const nindex, const int* const nlist)
{
  #pragma omp parallel for reduction(+: g_count) schedule(dynamic)
  for (int v = 0; v < nodes; v++) {
    basic_t count = 0;
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;
    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1; j < end1; j++) {
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      count += (basic_t)common(j + 1, end1, start2, end2, nlist);
    }
    g_count += count;
  }
}

static double CPUtc_vertex(basic_t &count, const int nodes, const int* const nindex, const int* const nlist)
{
  timeval start, end;
  count = 0;

  gettimeofday(&start, NULL);

  triCounting(count, nodes, nindex, nlist);

  gettimeofday(&end, NULL);

  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}

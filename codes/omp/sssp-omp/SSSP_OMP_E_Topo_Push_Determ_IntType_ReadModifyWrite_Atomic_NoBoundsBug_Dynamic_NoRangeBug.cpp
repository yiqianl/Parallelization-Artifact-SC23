typedef int data_type;
#include "sssp_edge_omp.h"

static void init(const int src, data_type* const dist, data_type* const dist_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v] = temp;
    dist[v] = temp;
  }
}

static void sssp(const ECLgraph g, const int* const sp, data_type* const dist, data_type* const dist_n, int &goagain)
{
  #pragma omp parallel for schedule(dynamic)
  for (int e = 0; e < g.edges; e++) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = atomicRead(&dist[src]);

    if (s != maxval) {
      const data_type new_dist = s + g.eweight[e];
      if (critical_min(&dist_n[dst], new_dist) > new_dist) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUsssp_edge(const int src, const ECLgraph& g, data_type* dist, const int* const sp)
{
  data_type* dist_new = new data_type [g.nodes];
  timeval start, end;

  init(src, dist, dist_new, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  gettimeofday(&start, NULL);
  do {
    iter++;
    goagain = 0;
    sssp(g, sp, dist, dist_new, goagain);
    std::swap(dist, dist_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

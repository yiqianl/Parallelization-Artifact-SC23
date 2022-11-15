typedef unsigned long long data_type;
#include "indigo_sssp_vertex_omp.h"

static void init(const int src, data_type* const dist, data_type* const dist_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v] = temp;
    dist[v] = temp;
  }
}

static void sssp(const ECLgraph g, data_type* const dist, data_type* const dist_n, int &goagain)
{
  #pragma omp parallel for
  for (int v = 0; v < g.nodes; v++) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = atomicRead(&dist[v]);
    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type s = atomicRead(&dist[src]);
      if (s != maxval) {
        const data_type new_dist = s + g.eweight[i];
        if (new_dist < d) {
          d = new_dist;
          atomicWrite(&goagain, 1);
        }
      }
      atomicWrite(&dist_n[v], d);
    }
  }
}

static double CPUsssp_vertex(const int src, const ECLgraph& g, data_type* dist)
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
    sssp(g, dist, dist_new, goagain);
    std::swap(dist, dist_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

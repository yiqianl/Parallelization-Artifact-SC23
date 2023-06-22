typedef int data_type;
#include "sssp_vertex_omp.h"

static void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
}

static void sssp(const ECLgraph g, data_type* const dist, int &goagain)
{
  #pragma omp parallel for
  for (int v = 0; v < g.nodes; v++) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    const data_type s = atomicRead(&dist[v]);

    if (s != maxval) {
      bool updated = false;
      for (int i = beg; i < end; i++) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + g.eweight[i];
        const data_type d = atomicRead(&dist[dst]);
        if (d > new_dist) {
          atomicWrite(&dist[dst], new_dist);
          updated = true;
        }
      }
      if (updated) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUsssp_vertex(const int src, const ECLgraph& g, data_type* dist)
{
  timeval start, end;

  init(src, dist, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  gettimeofday(&start, NULL);
  do {
    iter++;
    goagain = 0;
    sssp(g, dist, goagain);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

typedef int data_type;
#include "indigo_bfs_vertex_omp.h"

static void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
}

static void bfs(const int iter, const ECLgraph g, data_type* const dist, int &goagain)
{
  #pragma omp parallel for schedule(dynamic)
  for (int v = 0; v < g.nodes; v++) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    const data_type s = atomicRead(&dist[v]);

    if (s != maxval) {
      bool updated = false;
      for (int i = beg; i < end; i++) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + 1;
        if (critical_min(&dist[dst], new_dist) > new_dist) {
          updated = true;
        }
      }
      if (updated) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUbfs_vertex(const int src, const ECLgraph& g, data_type* dist)
{
  init(src, dist, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    goagain = 0;
    bfs(iter, g, dist, goagain);
    iter++;
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

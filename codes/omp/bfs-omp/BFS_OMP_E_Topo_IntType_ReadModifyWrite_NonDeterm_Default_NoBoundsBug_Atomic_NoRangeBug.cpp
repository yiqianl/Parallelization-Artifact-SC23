typedef int data_type;
#include "indigo_bfs_edge_omp.h"

static void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
}

static void bfs(const ECLgraph g, const int* const sp, data_type* const dist, int &goagain)
{
  #pragma omp parallel for
  for (int e = 0; e < g.edges; e++) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = atomicRead(&dist[src]);

    if (s != maxval) {
      const data_type new_dist = s + 1;
      if (critical_min(&dist[dst], new_dist) > new_dist) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUbfs_edge(const int src, const ECLgraph& g, data_type* dist, const int* const sp)
{
  init(src, dist, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;
    bfs(g, sp, dist, goagain);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

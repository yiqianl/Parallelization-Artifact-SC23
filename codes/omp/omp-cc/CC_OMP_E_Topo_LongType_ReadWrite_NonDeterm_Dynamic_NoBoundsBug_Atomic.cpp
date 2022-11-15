typedef unsigned long long data_type;
#include "indigo_cc_edge_omp.h"

static void init(data_type* const label, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    label[v] = v;
  }
}

static void cc(const ECLgraph g, const int* const sp, data_type* const label, int &goagain)
{
  #pragma omp parallel for schedule(dynamic)
  for (int e = 0; e < g.edges; e++) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = atomicRead(&label[src]);

    const data_type d = atomicRead(&label[dst]);
    if (d > new_label) {
      atomicWrite(&label[dst], new_label);
      atomicWrite(&goagain, 1);
    }
  }
}

static double CPUcc_edge(const ECLgraph g, data_type* label, const int* const sp)
{
  timeval start, end;

  init(label, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  gettimeofday(&start, NULL);
  do {
    iter++;
    goagain = 0;
    cc(g, sp, label, goagain);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

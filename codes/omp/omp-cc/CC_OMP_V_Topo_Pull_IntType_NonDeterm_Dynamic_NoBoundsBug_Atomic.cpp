typedef int data_type;
#include "indigo_cc_vertex_omp.h"

static void init(data_type* const label, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    label[v] = v;
  }
}

static void cc(const ECLgraph g, data_type* const label, int &goagain)
{
  #pragma omp parallel for schedule(dynamic)
  for (int v = 0; v < g.nodes; v++) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = atomicRead(&label[v]);
    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type new_label = atomicRead(&label[src]);
      if (new_label < d) {
        d = new_label;
        atomicWrite(&goagain, 1);
      }
      atomicWrite(&label[v], d);
    }
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label)
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
    cc(g, label, goagain);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

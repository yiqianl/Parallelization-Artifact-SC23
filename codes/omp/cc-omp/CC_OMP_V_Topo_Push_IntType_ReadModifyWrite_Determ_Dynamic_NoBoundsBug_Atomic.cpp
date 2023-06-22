typedef int data_type;
#include "cc_vertex_omp.h"

static void init(data_type* const label, data_type* const label_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    label_n[v] = v;
    label[v] = v;
  }
}

static void cc(const ECLgraph g, data_type* const label, data_type* const label_n, int &goagain)
{
  #pragma omp parallel for schedule(dynamic)
  for (int v = 0; v < g.nodes; v++) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    const data_type new_label = atomicRead(&label[v]);

    bool updated = false;
    for (int i = beg; i < end; i++) {
      const int dst = g.nlist[i];
      if (criticalMin(&label_n[dst], new_label) > new_label) {
        updated = true;
      }
    }

    if (updated) {
      atomicWrite(&goagain, 1);
    }
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label)
{
  data_type* label_new = new data_type [g.nodes];
  timeval start, end;

  init(label, label_new, g.nodes);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  gettimeofday(&start, NULL);

  do {
    iter++;
    goagain = 0;
    cc(g, label, label_new, goagain);
    std::swap(label, label_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

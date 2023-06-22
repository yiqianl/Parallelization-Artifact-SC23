typedef int data_type;
#include "cc_edge_omp.h"

static void init(data_type* const label, data_type* const label_n, const int size, const ECLgraph g, int* const wl1, int &wlsize, int* const time)
{
  int idx = 0;
  // initialize label array
  for (int v = 0; v < size; v++) {
    label_n[v] = v;
    label[v] = v;

    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      wl1[idx] = i;
      idx++;
    }
  }
  wlsize = idx;
}

static void cc_edge_data(const ECLgraph g, const int* const sp, data_type* const label, data_type* const label_n, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time)
{
  #pragma omp parallel for
  for (int idx = 0; idx < wl1size; idx ++) {
    const int e = wl1[idx];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = label[src];
    if (criticalMin(&label_n[dst], new_label) > new_label) {
      for (int j = g.nindex[dst]; j < g.nindex[dst + 1]; j++) {
        if (criticalMax(&time[j], iter) != iter) {
          wl2[fetch_and_add(&wl2size)] = j;
        }
      }
    }
    criticalMin(&label_n[src], new_label);
  }
}

static double CPUcc_edge(const ECLgraph g, data_type* label, const int* const sp)
{
  data_type* label_new = new data_type [g.nodes];
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;

  std::fill(time, time + size, 0);

  timeval start, end;
  init(label, label_new, g.nodes, g, wl1, wl1size, time);

  // iterate until no more changes
  int iter = 0;
  gettimeofday(&start, NULL);

  do {
    iter++;
    wl2size = 0;

    cc_edge_data(g, sp, label, label_new, wl1, wl1size, wl2, wl2size, iter, time);

    std::swap(wl1, wl2);
    wl1size = wl2size;
    std::swap(label, label_new);
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] wl1;
  delete [] wl2;
  delete [] time;
  return runtime;
}

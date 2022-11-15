typedef int data_type;
#include "indigo_cc_vertex_omp.h"

static void init(data_type* const label, const int size, int* const wl1, int &wlsize)
{
  // initialize label array
  for (int v = 0; v < size; v++) {
    label[v] = v;
    wl1[v] = v;
  }
  wlsize = size;
}

static void cc_vertex_data(const ECLgraph g, data_type* const label, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time)
{
  #pragma omp parallel for
  for (int idx = 0; idx < wl1size; idx ++) {
    const int v = wl1[idx];
    data_type d = atomicRead(&label[v]);
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    bool updated = false;

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type new_label = atomicRead(&label[src]);
      if (d > new_label) {
        d = new_label;
        updated = true;
      }
    }

    if (updated) {
      atomicWrite(&label[v], d);
      for (int j = beg; j < end; j++) {
        const int n = g.nlist[j];
        if (criticalMax(&time[n], iter) != iter) {
          wl2[fetch_and_add(&wl2size)] = n;
        }
      }
    }
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label)
{
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;

  std::fill(time, time + size, 0);

  timeval start, end;

  init(label, g.nodes, wl1, wl1size);

  // iterate until no more changes
  int iter = 0;
  gettimeofday(&start, NULL);
  do {
    iter++;
    wl2size = 0;

    cc_vertex_data(g, label, wl1, wl1size, wl2, wl2size, iter, time);

    std::swap(wl1, wl2);
    wl1size = wl2size;
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] wl1;
  delete [] wl2;
  delete [] time;
  return runtime;
}

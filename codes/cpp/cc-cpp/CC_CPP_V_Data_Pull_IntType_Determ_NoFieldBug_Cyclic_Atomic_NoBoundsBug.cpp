typedef int data_type;
#include "indigo_cc_vertex_cpp.h"

static void init(data_type* const label, data_type* const label_n, const int size, int* const wl1, int &wlsize)
{
  // initialize label array
  for (int v = 0; v < size; v++) {
    label_n[v] = v;
    label[v] = v;
    wl1[v] = v;
  }
  wlsize = size;
}

static void cc_vertex_data(const ECLgraph g, data_type* const label, data_type* const label_n, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time, const int threadID, const int threadCount)
{
  const int top = wl1size;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int v = wl1[idx];
    data_type d = label[v];
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    bool updated = false;

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type new_label = label[src];
      if (d > new_label) {
        d = new_label;
        updated = true;
      }
      atomicMin(&label_n[src], new_label);
    }

    if (updated) {
      label_n[v] = d;
      for (int j = beg; j < end; j++) {
        const int n = g.nlist[j];
        if (atomicMax(&time[n], iter) != iter) {
          wl2[atomicAdd(&wl2size, 1)] = n;
        }
      }
    }
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label, const int threadCount)
{
  data_type* label_new = new data_type [g.nodes];
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;
  std::thread threadHandles[threadCount];

  init(label, label_new, g.nodes, wl1, wl1size);
  std::fill(time, time + size, 0);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc_vertex_data, g, label, label_new, wl1, wl1size, wl2, std::ref(wl2size), iter, time, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::swap(wl1, wl2);
    wl1size = wl2size;
    std::swap(label, label_new);
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] (iter % 2 ? label : label_new);
  delete [] wl1;
  delete [] wl2;
  delete [] time;
  return runtime;
}

typedef int data_type;
#include "cc_vertex_cpp.h"

static void init(data_type* const label, data_type* const label_n, const int size, int* const wl1, int &wlsize)
{
  // initialize label array
  for (int v = 0; v < size; v++) {
    label_n[v] = v;
    label[v] = v;
    // initialize worklist
    wl1[v] = v;
  }
  wlsize = size;
}

static void cc_vertex_data(const ECLgraph g, data_type* const label, data_type* const label_n, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, const int threadID, const int threadCount)
{
  const int top = wl1size;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int src = wl1[idx];
    const data_type new_label = label[src];
    const int beg = g.nindex[src];
    const int end = g.nindex[src + 1];

    for (int i = beg; i < end; i++) {
      const int dst = g.nlist[i];

      if (atomicMin(&label_n[dst], new_label) > new_label) {
        wl2[atomicAdd(&wl2size, 1)] = dst;
      }
    }
    atomicMin(&label_n[src], new_label);
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label, const int threadCount)
{
  data_type* label_new = new data_type [g.nodes];
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;
  std::thread threadHandles[threadCount];

  init(label, label_new, g.nodes, wl1, wl1size);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc_vertex_data, g, label, label_new, wl1, wl1size, wl2, std::ref(wl2size), iter, i, threadCount);
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
  return runtime;
}

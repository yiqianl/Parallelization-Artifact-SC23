typedef int data_type;
#include "indigo_cc_vertex_cpp.h"

static void init(data_type* const label, const int size, int* const wl1, int &wlsize)
{
  // initialize label array
  for (int v = 0; v < size; v++) {
    label[v] = v;
    // initialize worklist
    wl1[v] = v;
  }
  wlsize = size;
}

static void cc_vertex_data(const ECLgraph g, data_type* const label, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time, const int threadID, const int threadCount)
{
  const int top = wl1size;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int src = wl1[idx];
    const data_type new_label = atomicRead(&label[src]);
    const int beg = g.nindex[src];
    const int end = g.nindex[src + 1];

    bool update = false;
    for (int i = beg; i < end; i++) {
      const int dst = g.nlist[i];

      const data_type d = atomicRead(&label[dst]);
      if (d > new_label) {
        atomicWrite(&label[dst], new_label);
        if (atomicMax(&time[dst], iter) != iter) {
          wl2[atomicAdd(&wl2size, 1)] = dst;
        }
        update = true;
      }
    }
    if (update) {
      if (atomicMax(&time[src], iter) != iter) {
        wl2[atomicAdd(&wl2size, 1)] = src;
      }
    }
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label, const int threadCount)
{
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;
  int* time = new int [size];
  std::thread threadHandles[threadCount];

  init(label, g.nodes, wl1, wl1size);
  std::fill(time, time + size, 0);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc_vertex_data, g, label, wl1, wl1size, wl2, std::ref(wl2size), iter, time, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

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

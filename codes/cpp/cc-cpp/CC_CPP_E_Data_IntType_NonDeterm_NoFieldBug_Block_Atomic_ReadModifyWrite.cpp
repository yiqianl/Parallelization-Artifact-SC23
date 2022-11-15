typedef int data_type;
#include "indigo_cc_edge_cpp.h"

static void init(data_type* const label, const int size, const ECLgraph g, int* const wl1, int &wlsize, int* const time)
{
  int idx = 0;
  // initialize label array
  for (int v = 0; v < size; v++) {
    label[v] = v;

    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      wl1[idx] = i;
      idx++;
    }
  }
  wlsize = idx;
}

static void cc_edge_data(const ECLgraph g, const int* const sp, data_type* const label, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)wl1size / threadCount;
  const int endNode = (threadID + 1) * (long)wl1size / threadCount;
  for (int idx = begNode; idx < endNode; idx++) {
    const int e = wl1[idx];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = atomicRead(&label[src]);
    if (atomicMin(&label[dst], new_label) > new_label) {
      for (int j = g.nindex[dst]; j < g.nindex[dst + 1]; j++) {
        if (atomicMax(&time[j], iter) != iter) {
          wl2[atomicAdd(&wl2size, 1)] = j;
        }
      }
    }
  }
}

static double CPUcc_edge(const ECLgraph g, data_type* label, const int* const sp, const int threadCount)
{
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;
  std::thread threadHandles[threadCount];

  init(label, g.nodes, g, wl1, wl1size, time);
  std::fill(time, time + size, 0);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc_edge_data, g, sp, label, wl1, wl1size, wl2, std::ref(wl2size), iter, time, i, threadCount);
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

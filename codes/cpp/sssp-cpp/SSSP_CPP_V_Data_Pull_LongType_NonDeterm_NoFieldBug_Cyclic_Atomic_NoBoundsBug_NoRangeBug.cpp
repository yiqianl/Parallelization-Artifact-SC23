typedef unsigned long long data_type;
#include "indigo_sssp_vertex_cpp.h"

static void init(const int src, data_type* const dist, const int size, const ECLgraph g, int* const wl1, int &wlsize, int* const time)
{
  // initialize dist array
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
  const int maxSize = std::max(g.edges, g.nodes);
  for (int v = 0; v < maxSize; v++)
  {
    time[v] = 0;
  }

  // initialize worklist
  int idx = 0;
  for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
    wl1[idx] = g.nlist[i];
    idx++;
  }
  wlsize = idx;
}

static void sssp_vertex_data(const ECLgraph g, data_type* const dist, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time, const int threadID, const int threadCount)
{
  const int top = wl1size;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int v = wl1[idx];
    data_type d = atomicRead(&dist[v]);
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    bool updated = false;

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type s = atomicRead(&dist[src]);
      if (s != maxval) {
        const data_type new_dist = s + g.eweight[i];
        if (d > new_dist) {
          d = new_dist;
          updated = true;
        }
      }
    }

    if (updated) {
      atomicWrite(&dist[v], d);
      for (int j = beg; j < end; j++) {
        const int n = g.nlist[j];
        if (atomicMax(&time[n], iter) != iter) {
          wl2[atomicAdd(&wl2size, 1)] = n;
        }
      }
    }
  }
}
static double CPUsssp_vertex(const int src, const ECLgraph& g, data_type* dist, const int threadCount)
{
  const int size = std::max(g.edges, g.nodes);
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;
  std::thread threadHandles[threadCount];

  init(src, dist, g.nodes, g, wl1, wl1size, time);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(sssp_vertex_data, g, dist, wl1, wl1size, wl2, std::ref(wl2size), iter, time, i, threadCount);
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

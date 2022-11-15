typedef int data_type;
#include "indigo_bfs_vertex_cpp.h"

static void init(const int src, data_type* const dist, const int size, int* const wl1, int* const wlsize)
{
  // initialize dist array
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
  // initialize worklist
  wl1[0] = src;
  *wlsize = 1;
}

static void bfs_vertex_data(const ECLgraph g, data_type* const dist, const int* const wl1, const int wl1size, int* const wl2, int &wl2size, const int iter, const int threadID, const int threadCount)
{
  const int top = wl1size;
  for (int idx = threadID; idx < top; idx += threadCount) {
    const int src = wl1[idx];
    const data_type s = atomicRead(&dist[src]);
    const int beg = g.nindex[src];
    const int end = g.nindex[src + 1];

    if (s != maxval) {
      for (int i = beg; i < end; i++) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + 1;
        if (atomicMin(&dist[dst], new_dist) > new_dist) {
          wl2[atomicAdd(&wl2size, 1)] = dst;
        }
      }
    }
  }
}
static double CPPbfs_vertex(const int src, const ECLgraph& g, data_type* dist, const int threadCount)
{
  const int size = std::max(g.edges, g.nodes);
  int* wl1 = new int [size];
  int wl1size = 0;
  int* wl2 = new int [size];
  int wl2size;
  std::thread threadHandles[threadCount];

  init(src, dist, g.nodes, wl1, &wl1size);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(bfs_vertex_data, g, dist, wl1, wl1size, wl2, std::ref(wl2size), iter, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::swap(wl1, wl2);
    std::swap(wl1size, wl2size);
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] wl1;
  delete [] wl2;
  return runtime;
}

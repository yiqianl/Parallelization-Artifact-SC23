typedef unsigned long long data_type;
#include "indigo_bfs_vertex_cpp.h"

static void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    dist[v] = (v == src) ? 0 : maxval;
  }
}

static void bfs(const ECLgraph g, data_type* const dist, int &goagain, const int iter, const int threadID, const int threadCount)
{
  const int top = g.nodes;
  for (int v = threadID; v < top; v += threadCount) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = atomicRead(&dist[v]);
    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type s = atomicRead(&dist[src]);
      if (s == iter) {
        const data_type new_dist = s + 1;
        if (new_dist < d) {
          d = new_dist;
          atomicWrite(&goagain, 1);
        }
      }
    }
    atomicWrite(&dist[v], d);
  }
}

static double CPPbfs_vertex(const int src, const ECLgraph& g, data_type* const dist, const int threadCount)
{
  std::thread threadHandles[threadCount];

  init(src, dist, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    goagain = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(bfs, g, dist, std::ref(goagain), iter, i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    iter++;
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

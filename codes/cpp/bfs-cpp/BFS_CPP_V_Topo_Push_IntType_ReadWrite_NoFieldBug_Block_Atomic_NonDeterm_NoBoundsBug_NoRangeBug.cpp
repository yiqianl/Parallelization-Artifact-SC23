typedef int data_type;
#include "bfs_vertex_cpp.h"

static void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    dist[v] = (v == src) ? 0 : maxval;
  }
}
static void bfs(const ECLgraph g, data_type* const dist, int &goagain, const int iter, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)g.nodes / threadCount;
  const int endNode = (threadID + 1) * (long)g.nodes / threadCount;
  for (int v = begNode; v < endNode; v++) {

    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    const data_type s = atomicRead(&dist[v]);

    if (s != maxval) {
      bool updated = false;
      for (int i = beg; i < end; i++) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + 1;
        const data_type d = atomicRead(&dist[dst]);
        if (d > new_dist) {
          atomicWrite(&dist[dst], new_dist);
          updated = true;
        }
      }
      if (updated) {
        atomicWrite(&goagain, 1);
      }
    }
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

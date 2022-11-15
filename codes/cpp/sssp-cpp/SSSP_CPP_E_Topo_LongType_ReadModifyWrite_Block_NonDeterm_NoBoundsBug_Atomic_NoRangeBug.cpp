typedef unsigned long long data_type;
#include "indigo_sssp_edge_cpp.h"
#include <atomic>

static void init(const int src, data_type* const dist, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
  }
}

static void sssp(const ECLgraph g, const int* const sp, data_type* const dist, int &goagain, const int threadID, const int threadCount)
{
  const int begEdge = threadID * (long)g.edges / threadCount;
  const int endEdge = (threadID + 1) * (long)g.edges / threadCount;
  for (int e = begEdge; e < endEdge; e++) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = atomicRead(&dist[src]);

    if (s != maxval) {
      const data_type new_dist = s + g.eweight[e];
      if (atomicMin(&dist[dst], new_dist) > new_dist) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUsssp_edge(const int src, const ECLgraph& g, data_type* dist, const int* const sp, const int threadCount)
{
  std::thread threadHandles[threadCount];

  init(src, dist, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(sssp, g, sp, dist, std::ref(goagain), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

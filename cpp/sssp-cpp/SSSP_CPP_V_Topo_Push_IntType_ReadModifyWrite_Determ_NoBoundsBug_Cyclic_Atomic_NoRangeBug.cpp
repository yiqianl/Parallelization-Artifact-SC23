typedef int data_type;
#include "indigo_sssp_vertex_cpp.h"

static void init(const int src, data_type* const dist, data_type* const dist_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v] = temp;
    dist[v] = temp;
  }
}

static void sssp(const ECLgraph g, data_type* const dist, data_type* const dist_n, int &goagain, const int threadID, const int threadCount)
{
  const int top = g.nodes;
  for (int v = threadID; v < top; v += threadCount) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    const data_type s = dist[v];

    if (s != maxval) {
      bool updated = false;
      for (int i = beg; i < end; i++) {
        const int dst = g.nlist[i];
        const data_type new_dist = s + g.eweight[i];
        if (atomicMin(&dist_n[dst], new_dist) > new_dist) {
          updated = true;
        }
      }
      if (updated) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUsssp_vertex(const int src, const ECLgraph& g, data_type* dist, const int threadCount)
{
  data_type* dist_new = new data_type [g.nodes];
  std::thread threadHandles[threadCount];

  init(src, dist, dist_new, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(sssp, g, dist, dist_new, std::ref(goagain), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::swap(dist, dist_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] (iter % 2 ? dist : dist_new);
  return runtime;
}

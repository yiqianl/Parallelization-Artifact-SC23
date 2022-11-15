typedef unsigned long long data_type;
#include "indigo_cc_edge_cpp.h"

static void init(data_type* const label, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    label[v] = v;
  }
}

static void cc(const ECLgraph g, const int* const sp, data_type* const label, int &goagain, const int threadID, const int threadCount)
{
  for (int e = threadID; e < g.edges; e += threadCount) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = atomicRead(&label[src]);

    const data_type d = atomicRead(&label[dst]);
    if (d > new_label) {
      atomicWrite(&label[dst], new_label);
      atomicWrite(&goagain, 1);
    }
  }
}

static double CPUcc_edge(const ECLgraph g, data_type* label, const int* const sp, const int threadCount)
{
  std::thread threadHandles[threadCount];

  init(label, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc, g, sp, label, std::ref(goagain), i, threadCount);
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

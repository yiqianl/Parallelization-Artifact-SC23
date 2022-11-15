typedef unsigned long long data_type;
#include "indigo_cc_edge_cpp.h"

static void init(data_type* const label, data_type* const label_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    label_n[v] = v;
    label[v] = v;
  }
}

static void cc(const ECLgraph g, const int* const sp, data_type* const label, data_type* const label_n, int &goagain, const int threadID, const int threadCount)
{
  const int begEdge = threadID * (long)g.edges / threadCount;
  const int endEdge = (threadID + 1) * (long)g.edges / threadCount;
  for (int e = begEdge; e < endEdge; e++) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type new_label = label[src];

    if (atomicMin(&label_n[dst], new_label) > new_label) {
      atomicWrite(&goagain, 1);
    }
  }
}

static double CPUcc_edge(const ECLgraph g, data_type* label, const int* const sp, const int threadCount)
{
  data_type* label_new = new data_type [g.nodes];
  std::thread threadHandles[threadCount];

  init(label, label_new, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(cc, g, sp, label, label_new, std::ref(goagain), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::swap(label, label_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] (iter % 2 ? label : label_new);
  return runtime;
}

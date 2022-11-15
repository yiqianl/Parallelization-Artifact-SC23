typedef int data_type;
#include "indigo_cc_vertex_cpp.h"

static void init(data_type* const label, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    label[v] = v;
  }
}

static void cc(const ECLgraph g, data_type* const label, int &goagain, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)g.nodes / threadCount;
  const int endNode = (threadID + 1) * (long)g.nodes / threadCount;
  for (int v = begNode; v < endNode; v++) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    data_type d = atomicRead(&label[v]);
    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type new_label = atomicRead(&label[src]);
      if (new_label < d) {
        d = new_label;
        atomicWrite(&goagain, 1);
      }
      atomicWrite(&label[v], d);
    }
  }
}

static double CPUcc_vertex(const ECLgraph g, data_type* label, const int threadCount)
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
      threadHandles[i] = std::thread(cc, g, label, std::ref(goagain), i, threadCount);
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

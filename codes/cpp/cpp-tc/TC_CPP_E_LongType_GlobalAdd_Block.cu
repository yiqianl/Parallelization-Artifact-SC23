typedef unsigned long long basic_t;
#include "indigo_tc_edge_cpp.h"

static void triCounting(basic_t& g_count, const int edges, const int* const nindex, const int* const nlist, const int* const sp, const int threadID, const int threadCount)
{
  basic_t count = 0;
  const int begEdge = threadID * (long)edges / threadCount;
  const int endEdge = (threadID + 1) * (long)edges / threadCount;
  for (int e = begEdge; e < endEdge; e++) {
    const int src = sp[e];
    const int dst = nlist[e];

    if (src > dst) {
      const int beg1 = nindex[dst];
      const int end1 = nindex[dst + 1];
      for (int i = beg1; i < end1 && nlist[i] < dst; i++) {
        const int u = nlist[i];
        int beg2 = nindex[src];
        int end2 = nindex[src + 1];
        if (find(u, beg2, end2, nlist)) count++;
      }
    }
  }
  atomicAdd(&g_count, count);
}

static double CPUtc_edge(basic_t &count, const int edges, const int* const nindex, const int* const nlist, const int* const sp, const int threadCount)
{
  std::thread threadHandles[threadCount];

  timeval start, end;
  count = 0;

  gettimeofday(&start, NULL);

  for (int i = 0; i < threadCount; ++i) {
    threadHandles[i] = std::thread(triCounting, std::ref(count), edges, nindex, nlist, sp, i, threadCount);
  }
  for (int i = 0; i < threadCount; ++i) {
    threadHandles[i].join();
  }

  gettimeofday(&end, NULL);

  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}

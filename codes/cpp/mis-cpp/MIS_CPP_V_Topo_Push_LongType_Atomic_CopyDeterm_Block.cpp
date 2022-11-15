typedef unsigned long long data_type;
#include "indigo_mis_vertex_cpp.h"

static void init(data_type* const priority, unsigned char* const status, unsigned char* const status_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++)
  {
    priority[v] = ((unsigned long)hash(v + 712313887)) | ((unsigned long)hash(v + 683067839) << (sizeof (unsigned int) * 8));
    status[v] = undecided;
    status_n[v] = undecided;
  }
}

static void mis(const ECLgraph& g, const data_type* const priority, unsigned char* const status, unsigned char* const status_n, bool &goagain, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)g.nodes / threadCount;
  const int endNode = (threadID + 1) * (long)g.nodes / threadCount;
  // go over all the nodes
  for (int v = begNode; v < endNode; v++) {
    if (atomicRead(&status[v]) == undecided) {
      int i = g.nindex[v];
      // try to find a non-excluded neighbor whose priority is higher
      while ((i < g.nindex[v + 1]) && ((atomicRead(&status[g.nlist[i]]) == excluded) || (priority[v] > priority[g.nlist[i]]) || ((priority[v] == priority[g.nlist[i]]) && (v > g.nlist[i])))) {
        i++;
      }
      if (i < g.nindex[v + 1]) {
        // found such a neighbor -> status still unknown
        atomicWrite(&goagain, true);
      } else {
        // no such neighbor -> all neighbors are "excluded" and v is "included"
        status_n[v] = included;
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          status_n[g.nlist[i]] = excluded;
        }
      }
    }
  }
}

static double CPPmis_vertex(const ECLgraph& g, data_type* const priority, unsigned char* status, const int threadCount)
{
  unsigned char* status_new = new unsigned char [g.nodes];
  std::thread threadHandles[threadCount];

  init(priority, status, status_new, g.nodes);

  timeval beg, end;
  gettimeofday(&beg, NULL);

  bool goagain;
  int iter = 0;
  do {
    iter++;
    goagain = false;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(mis, g, priority, status, status_new, std::ref(goagain), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::copy(status_new, status_new + g.nodes, status);

  } while(goagain);

  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

  // determine and print set size
  int cnt = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == included) cnt++;
  }
  printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

  delete [] status_new;
  return runtime;
}

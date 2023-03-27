typedef unsigned long long data_type;
#include "indigo_mis_vertex_omp.h"

static void init(data_type* const priority, unsigned char* const status, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++)
  {
    priority[v] = ((unsigned long)hash(v + 712313887)) | ((unsigned long)hash(v + 683067839) << (sizeof (unsigned int) * 8));
    status[v] = undecided;
  }
}

static void mis(const ECLgraph& g, const data_type* const priority, unsigned char* const status, bool &goagain)
{
  #pragma omp parallel for
  for (int v = 0; v < g.nodes; v++) {
    // go over all the nodes
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
        atomicWrite(&status[v], included);
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          atomicWrite(&status[g.nlist[i]], excluded);
        }
      }
    }
  }
}

static double OMPmis_vertex(const ECLgraph& g, data_type* const priority, unsigned char* status)
{

  init(priority, status, g.nodes);

  timeval beg, end;
  gettimeofday(&beg, NULL);

  bool goagain;
  int iter = 0;
  do {
    iter++;
    goagain = false;

    mis(g, priority, status, goagain);

  } while(goagain);

  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

  // determine and print set size
  int cnt = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == included) cnt++;
  }
  printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

  return runtime;
}
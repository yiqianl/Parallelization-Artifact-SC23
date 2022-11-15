typedef int data_type;
#include "indigo_bfs_edge_omp.h"

template <typename T>
static inline T atomicRead(T* const addr)
{
  data_type ret;
  #pragma omp atomic read
  ret = *addr;
  return ret;
}

template <typename T>
static inline void atomicWrite(T* const addr, const T val)
{
  #pragma omp atomic write
  *addr = val;
}

static inline data_type critical_min(data_type* addr, data_type val)
{
  data_type oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv > val) {
      *addr = val;
    }
  }
  return oldv;
}

static void init(const int src, data_type* const dist, data_type* const dist_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v] = temp;
    dist[v] = temp;
  }
}

static void bfs(const ECLgraph g, const int* const sp, data_type* const dist, data_type* const dist_n, int &goagain)
{
  #pragma omp parallel for
  for (int e = 0; e < g.edges; e++) {

    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = atomicRead(&dist[src]);

    if (s != maxval) {
      const data_type new_dist = s + 1;
      if (critical_min(&dist_n[dst], new_dist) > new_dist) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPUbfs_edge(const int src, const ECLgraph& g, data_type* dist, const int* const sp)
{
  data_type* dist_new = new data_type [g.nodes];
  init(src, dist, dist_new, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;
    bfs(g, sp, dist, dist_new, goagain);
    std::swap(dist, dist_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  return runtime;
}

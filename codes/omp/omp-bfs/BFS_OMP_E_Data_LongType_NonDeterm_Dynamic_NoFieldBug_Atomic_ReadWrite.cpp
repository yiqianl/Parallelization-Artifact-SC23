typedef unsigned long long data_type;
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

static inline data_type critical_max(data_type* addr, data_type val)
{
  data_type oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv < val) {
      *addr = val;
    }
  }
  return oldv;
}

static inline data_type fetch_and_add(data_type* addr)
{
  data_type old;
  #pragma omp atomic capture
  {
    old = *addr;
    (*addr)++;
  }
  return old;
}

static void init(const int src, data_type* const dist, const int size, const ECLgraph g, int* const wl1, int &wlsize, int* const time)
{
  // initialize dist array
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist[v] = temp;
    time[v] = 0;
  }
  // initialize worklist
  int idx = 0;
  for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
    wl1[idx] = i;
    idx++;
  }
  wlsize = idx;
}
static void bfs_edge_data(const ECLgraph g, const int* const sp, data_type* const dist, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time)
{
  #pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < wl1size; idx ++) {
    const int e = wl1[idx];
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = atomicRead(&dist[src]);
    if (s != maxval) {
      const data_type new_dist = s + 1;
      data_type d = atomicRead(&dist[dst]);
      if (d > new_dist) {
        atomicWrite(&dist[dst], new_dist);
        if (critical_max(&time[e], iter) != iter) {
          wl2[fetch_and_add(&wl2size)] = e;
        }
        for (int j = g.nindex[dst]; j < g.nindex[dst + 1]; j++) {
          if (critical_max(&time[j], iter) != iter) {
            wl2[fetch_and_add(&wl2size)] = j;
          }
        }
      }
    }
  }
}

static double CPUbfs_edge(const int src, const ECLgraph& g, data_type* dist, const int* const sp)
{
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;
  init(src, dist, g.nodes, g, wl1, wl1size, time);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    bfs_edge_data(g, sp, dist, wl1, wl1size, wl2, wl2size, iter, time);

    std::swap(wl1, wl2);
    wl1size = wl2size;
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] wl1;
  delete [] wl2;
  delete [] time;
  return runtime;
}

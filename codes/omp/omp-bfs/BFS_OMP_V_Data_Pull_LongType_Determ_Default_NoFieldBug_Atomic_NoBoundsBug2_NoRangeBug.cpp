typedef unsigned long long data_type;
#include "indigo_bfs_vertex_omp.h"

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

static inline data_type criticalMin(data_type* addr, data_type val)
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

static inline data_type criticalMax(data_type* addr, data_type val)
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

static void init(const int src, data_type* const dist, data_type* const dist_n, const int size, const ECLgraph g, int* const wl1, int &wlsize, int* const time)
{
  // initialize dist array
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v] = temp;
    dist[v] = temp;
    time[v] = 0;
  }

  // initialize worklist
  int idx = 0;
  for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
    wl1[idx] = g.nlist[i];
    idx++;
  }
  wlsize = idx;
}

static void bfs_vertex_data(const ECLgraph g, data_type* const dist, data_type* const dist_n, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int iter, int* const time)
{
  #pragma omp parallel for
  for (int idx = 0; idx < wl1size; idx ++) {
    const int v = wl1[idx];
    data_type d = dist[v];
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    bool updated = false;

    for (int i = beg; i < end; i++) {
      const int src = g.nlist[i];
      const data_type s = dist[src];
      if (s != maxval) {
        const data_type new_dist = s + 1;
        if (d > new_dist) {
          d = new_dist;
          updated = true;
        }
      }
      criticalMin(&dist_n[src], s);
    }

    if (updated) {
      dist_n[v] = d;
      for (int j = beg; j < end; j++) {
        const int n = g.nlist[j];
        if (criticalMax(&time[n], iter) != iter) {
          wl2[fetch_and_add(&wl2size)] = n;
        }
      }
    }
  }
}
static double CPUbfs_vertex(const int src, const ECLgraph& g, data_type* dist)
{
  data_type* dist_new = new data_type [g.nodes];
  const int size = std::max(g.edges, g.nodes);
  int wlsize;
  int* time = new int [size];
  int* wl1 = new int [size];
  int* wl2 = new int [size];
  int wl1size;
  int wl2size;

  init(src, dist, dist_new, g.nodes, g, wl1, wl1size, time);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int iter = 0;
  do {
    iter++;
    wl2size = 0;

    bfs_vertex_data(g, dist, dist_new, wl1, wl1size, wl2, wl2size, iter, time);

    std::swap(wl1, wl2);
    wl1size = wl2size;
    std::swap(dist, dist_new);
  } while (wl1size > 0);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] wl1;
  delete [] wl2;
  delete [] time;
  return runtime;
}

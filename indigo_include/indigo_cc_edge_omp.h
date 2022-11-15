#include <algorithm>
#include <set>
#include <sys/time.h>
#include "ECLgraph.h"
#include <limits>

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

template <typename T>
static inline T criticalMax(T* addr, T val)
{
  T oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv < val) {
      *addr = val;
    }
  }
  return oldv;
}

template <typename T>
static inline T criticalMin(T* addr, T val)
{
  T oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv > val) {
      *addr = val;
    }
  }
  return oldv;
}

template <typename T>
static inline T fetch_and_add(T* addr)
{
  T old;
  #pragma omp atomic capture
  {
    old = *addr;
    (*addr)++;
  }
  return old;
}

static double CPUcc_edge(const ECLgraph g, data_type* const label, const int* const sp);

static void verify(const int v, const int id, const int* const __restrict__ nidx, const int* const __restrict__ nlist, data_type* const __restrict__ nstat, const int nodes)
{
  if (nstat[v] < nodes) {
    if (nstat[v] != id) {fprintf(stderr, "ERROR: found incorrect ID value\n\n");  exit(-1);}
    nstat[v] = nodes;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, nstat, nodes);
    }
  }
}

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

int main(int argc, char* argv[])
{
  printf("cc topology-driven OMP (%s)\n", __FILE__);
  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file_name verify\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int runveri = atoi(argv[2]);
  if ((runveri != 0) && (runveri != 1)) {
    printf("has to be 0 (turn off) or 1 (turn on) verification");
  }

  // create starting point array
  int* const sp = new int [g.edges];
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  data_type* const label = new data_type [g.nodes];

  // cc
  const int runs = 9;
  double runtimes [runs];
  bool flag = true;
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUcc_edge(g, label, sp);
    if (runtimes[i] >= 30.0) {
      printf("runtime: %.6fs\n", runtimes[i]);
      printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / runtimes[i]);
      flag = false;
      break;
    }
  }
  if (flag) {
    const double med = median(runtimes, runs);
    printf("runtime: %.6fs\n", med);
    printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);
  }


  // print result
  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(label[v]);
  }
  printf("number of connected components: %d\n", s1.size());

  // compare solutions
  if (runveri) {
    /* verification code (may need extra runtime stack space due to deep recursion) */

    for (int v = 0; v < g.nodes; v++) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (label[g.nlist[i]] != label[v]) {fprintf(stderr, "ERROR: found adjacent nodes in different components\n\n");  exit(-1);}
      }
    }

    for (int v = 0; v < g.nodes; v++) {
      if (label[v] >= g.nodes) {fprintf(stderr, "ERROR: found sentinel number\n\n");  exit(-1);}
    }

    std::set<int> s2;
    int count = 0;
    for (int v = 0; v < g.nodes; v++) {
      if (label[v] < g.nodes) {
        count++;
        s2.insert(label[v]);
        verify(v, label[v], g.nindex, g.nlist, label, g.nodes);
      }
    }
    if (s1.size() != s2.size()) {fprintf(stderr, "ERROR: number of components do not match\n\n");  exit(-1);}
    if (s1.size() != count) {fprintf(stderr, "ERROR: component IDs are not unique\n\n");  exit(-1);}

    printf("verification passed\n\n");
  } else {
    printf("verification turned off\n\n");
  }

  // free memory
  delete [] label;
  freeECLgraph(g);
  return 0;
}

#include <limits>
#include <algorithm>
#include <queue>
#include <sys/time.h>
#include "ECLgraph.h"

const data_type maxval = std::numeric_limits<data_type>::max();

using pair = std::pair<int, int>;

const unsigned char undecided = 0;
const unsigned char included = 1;
const unsigned char excluded = 2;

static double OMPmis_vertex(const ECLgraph& g, data_type* const priority, unsigned char* const status);

static void verify(ECLgraph& g, unsigned char* const status)
{
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != included) && (status[v] != excluded)) {fprintf(stderr, "ERROR: found undecided node\n"); exit(-1);}
    if (status[v] == included) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == included) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == included) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
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
  printf("mis vertex-based OMP (%s)\n", __FILE__);
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

  // allocate memory
  data_type* const priority = new data_type [g.nodes];
  unsigned char* const status = new unsigned char [g.nodes];

  // launch kernel
  const int runs = 10;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = OMPmis_vertex(g, priority, status);
    // verify result
    if (runveri) {
      verify(g, status);
    } else {
      printf("turn off verification\n\n");
    }
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  delete [] priority;
  delete [] status;
  freeECLgraph(g);
  return 0;
}

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static void updateUndecided(unsigned char* const status, unsigned char* const status_n, const int size)
{
  #pragma omp parallel for
  for (int i = 0; i < size; ++i)
  {
    if (status[i] == undecided)
      status[i] = status_n[i];
  }
}

static void updateFromWorklist(unsigned char* const status, unsigned char* const status_n, const int* const worklist, const int wlsize)
{
  #pragma omp parallel for
  for (int i = 0; i < wlsize; ++i)
  {
    int v = worklist[i];
    status[v] = status_n[v];
  }
}

template <typename T>
static inline T atomicRead(T* const addr)
{
  T ret;
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
static inline T criticalAdd(T* addr, T val)
{
  T oldv;
  #pragma omp atomic capture
  {
    oldv = *addr;
    (*addr) += val;
  }
  return oldv;
}
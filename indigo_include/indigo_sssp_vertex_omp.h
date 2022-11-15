#include <algorithm>
#include <queue>
#include <sys/time.h>
#include "ECLgraph.h"
#include <limits>

const data_type maxval = std::numeric_limits<data_type>::max();
using pair = std::pair<int, int>;

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

static double CPUsssp_vertex(const int src, const ECLgraph& g, data_type* dist);

static void CPUserialDijkstra(const int src, const ECLgraph& g, data_type* const dist)
{
  // initialize dist array
  for (int i = 0; i < g.nodes; i++) dist[i] = maxval;
  dist[src] = 0;

  // set up priority queue with just source node in it
  std::priority_queue< std::pair<int, int> > pq;
  pq.push(std::make_pair(0, src));
  while (pq.size() > 0) {
    // process closest vertex
    const int v = pq.top().second;
    pq.pop();
    const data_type dv = dist[v];
    // visit outgoing neighbors
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      const int n = g.nlist[i];
      const data_type d = dv + g.eweight[i];
      // check if new lower distance found
      if (d < dist[n]) {
        dist[n] = d;
        pq.push(std::make_pair(-d, n));
      }
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
  printf("sssp topology-driven OMP (%s)\n", __FILE__);
  if (argc != 4) {fprintf(stderr, "USAGE: %s input_file_name source_node_number verify\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  if (g.eweight == NULL) {
    printf("Generating weights.\n");
    g.eweight = new int [g.edges];
    for (int i = 0; i < g.nodes; i++) {
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int nei = g.nlist[j];
        g.eweight[j] = 1 + ((i * nei) % g.nodes);
          if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
      }
    }
  }
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int source = atoi(argv[2]);
  if ((source < 0) || (source >= g.nodes)) {fprintf(stderr, "ERROR: source_node_number must be between 0 and %d\n", g.nodes); exit(-1);}
  printf("source: %d\n", source);
  const int runveri = atoi(argv[3]);
  if ((runveri != 0) && (runveri != 1)) {
    printf("has to be 0 (turn off) or 1 (turn on) verification");
  }

  // allocate memory
  data_type* const distance = new data_type [g.nodes];

  // sssp
  const int runs = 9;
  double runtimes [runs];
  bool flag = true;
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUsssp_vertex(source, g, distance);
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
  int maxnode = 0;
  for (int v = 1; v < g.nodes; v++) {
    if (distance[maxnode] < distance[v]) maxnode = v;
  }
  printf("vertex %d has maximum distance %d\n", maxnode, distance[maxnode]);

  // compare solutions
  if (runveri) {
    data_type* const verify = new data_type [g.nodes];
    CPUserialDijkstra(source, g, verify);
    for (int v = 0; v < g.nodes; v++) {
      if (distance[v] != verify[v]) {fprintf(stderr, "ERROR: verification failed for node %d: %d   instead of %d\n", v, distance[v], verify[v]); exit(-1);}
    }
    printf("verification passed\n\n");
    delete [] verify;
  } else {
    printf("turn off verification\n\n");
  }
  // free memory
  delete [] distance;
  freeECLgraph(g);
  return 0;
}
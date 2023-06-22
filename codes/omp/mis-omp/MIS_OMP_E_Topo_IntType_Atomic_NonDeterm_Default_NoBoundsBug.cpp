typedef unsigned int data_type;
#include "mis_edge_omp.h"

static void init(data_type* const priority, unsigned char* const status, bool* const lost, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    lost[v] = false;
  }
}

static void mis(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, bool* const lost)
{
  #pragma omp parallel for
  for (int e = 0; e < g.edges; e++) {
    // go over all edges

    const int src = sp[e];
    const int dst = g.nlist[e];

    // if one is included, exclude the other
    if (atomicRead(&status[src]) == included) {
      atomicWrite(&status[dst], excluded);
    }
    else if (atomicRead(&status[dst]) == included) {
      atomicWrite(&status[src], excluded);
    }
    // if neither included nor excluded -> mark lower as lost
    else if (atomicRead(&status[src]) != excluded && atomicRead(&status[dst]) != excluded) {
      if (priority[src] < priority[dst]) { //src is lower -> mark lost
      lost[src] = true;
    } else { //dst is lower  -> mark lost
    lost[dst] = true;
  }
}
}
}

static void mis_vertex_pass(unsigned char* const status, bool* const lost, const int size, bool& goagain)
{
#pragma omp parallel for
for (int v = 0; v < size; v++) {
// go over all vertexes
if (lost[v] == false) { // v didn't lose
if(status[v] == undecided) { // and is undecided -> include
status[v] = included;
}
}
else { // v lost, goagain
goagain = true;
lost[v] = false; //reset lost flag
}
}
}

static double OMPmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status)
{
bool* lost = new bool [g.nodes];

init(priority, status, lost, g.nodes);

timeval beg, end;
gettimeofday(&beg, NULL);

bool goagain;
int iter = 0;
do {
iter++;
goagain = false;

// edge pass
mis(g, sp, priority, status, lost);


// vertex pass
mis_vertex_pass(status, lost, g.nodes, goagain);

} while (goagain);
// include all remaining nodes that have no edges
#pragma omp parallel for
for (int i = 0; i < g.nodes; i++) {
if (status[i] == undecided)
status[i] = included;
}

gettimeofday(&end, NULL);
const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

// determine and print set size
int cnt = 0;
for (int v = 0; v < g.nodes; v++) {
if (status[v] == included) cnt++;
}
printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

delete [] lost;
return runtime;
}

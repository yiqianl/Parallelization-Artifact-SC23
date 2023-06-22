typedef unsigned int data_type;
#include "mis_edge_cpp.h"

static void init(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, unsigned char* const status_n, bool* const lost, int* const wl1, int &wlsize)
{
  // initialize arrays
  for (int v = 0; v < g.nodes; v++)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    status_n[v] = undecided;
    lost[v] = false;
  }
  wlsize = 0;
  for (int e = 0; e < g.edges; e++)
  {
    // initialize worklist
    if (sp[e] < g.nlist[e]) {
      wl1[wlsize++] = e;
    }
  }
}

static void mis(const ECLgraph& g, const int* const sp, const data_type* const priority, unsigned char* const status, unsigned char* const status_n, bool* const lost, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int threadID, const int threadCount)
{
  const int begEdge = threadID * (long)wl1size / threadCount;
  const int endEdge = (threadID + 1) * (long)wl1size / threadCount;

  // go over all edges in wl1
  for (int w = begEdge; w < endEdge; w++) {
    int e = wl1[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    // if one is included, exclude the other
    if (atomicRead(&status[src]) == included) {
      status_n[dst] = excluded;
    }
    else if (atomicRead(&status[dst]) == included) {
      status_n[src] = excluded;
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

static void mis_vertex_pass(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, unsigned char* const status_n, bool* const lost, const int* const wl1, const int wl1size, int* const wl2, int& wl2size, const int threadID, const int threadCount)
{
const int begEdge = threadID * (long)wl1size / threadCount;
const int endEdge = (threadID + 1) * (long)wl1size / threadCount;

// go over all vertexes
for (int w = begEdge; w < endEdge; w++) {
const int e = wl1[w];
const int src = sp[e];
const int dst = g.nlist[e];
if (lost[src] == false) { // if src won
if (status[src] == undecided) {
  // and is undecided -> include
  status_n[src] = included;
}
}
if (lost[dst] == false) { // if dst won
if (status[dst] == undecided) {
// and is undecided -> include
status_n[dst] = included;
}
}
if (status[src] == undecided || status[dst] == undecided) { // if either is still undecided, keep it in WL
wl2[atomicAdd(&wl2size, 1)] = e;
}
}
}

static void mis_last_pass(unsigned char* const status, const int size, const int threadID, const int threadCount)
{
const int begEdge = threadID * (long)size / threadCount;
const int endEdge = (threadID + 1) * (long)size / threadCount;

for (int e = begEdge; e < endEdge; e++) {
if (status[e] == undecided)
{
status[e] = included;
}
}
}

static double CPPmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, const int threadCount)
{
unsigned char* status_new = new unsigned char [g.nodes];
bool* lost = new bool [g.nodes];
std::thread threadHandles[threadCount];
const int size = std::max(g.edges, g.nodes);
int* wl1 = new int [size];
int* wl2 = new int [size];
int wl1size;
int wl2size;

init(g, sp, priority, status, status_new, lost, wl1, wl1size);

timeval beg, end;
gettimeofday(&beg, NULL);

int iter = 0;
do {
iter++;
wl2size = 0;

// edge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis, g, sp, priority, status, status_new, lost, wl1, wl1size, wl2, std::ref(wl2size), i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

// merge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(updateFromWorklist, g, sp, status, status_new, wl1, wl1size, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

// vertex pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis_vertex_pass, g, sp, priority, status, status_new, lost, wl1, wl1size, wl2, std::ref(wl2size), i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

// merge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(updateFromWorklist, g, sp, status, status_new, wl1, wl1size, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}
std::fill(lost, lost + g.nodes, false);
std::swap(wl1, wl2);
wl1size = wl2size;
} while (wl1size > 0);

for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis_last_pass, status, g.nodes, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

gettimeofday(&end, NULL);
const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;

// determine and print set size
int cnt = 0;
for (int v = 0; v < g.nodes; v++) {
if (status[v] == included) cnt++;
}
printf("iterations: %d,  elements in set: %d (%.1f%%)\n", iter, cnt, 100.0 * cnt / g.nodes);

delete [] status_new;
delete [] lost;
delete [] wl1;
delete [] wl2;
return runtime;
}

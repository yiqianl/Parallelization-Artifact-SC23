typedef unsigned int data_type;
#include "mis_edge_cpp.h"

static void init(data_type* const priority, unsigned char* const status, unsigned char* const status_n, bool* const lost, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++)
  {
    priority[v] = hash(v + 712313887);
    status[v] = undecided;
    status_n[v] = undecided;
    lost[v] = false;
  }
}

static void mis(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, unsigned char* const status_n, bool* const lost, const int threadID, const int threadCount)
{
  const int top = g.edges;
  // go over all the edges
  for (int e = threadID; e < top; e += threadCount) {
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

static void mis_vertex_pass(unsigned char* const status, unsigned char* const status_n, bool* const lost, const int size, bool& goagain, const int threadID, const int threadCount)
{
const int top = size;
// go over all vertexes
for (int v = threadID; v < top; v += threadCount) {
if (lost[v] == false) { // v didn't lose
if(status[v] == undecided) { // and is undecided -> include
status_n[v] = included;
}
}
else { // v lost, goagain
goagain = true;
lost[v] = false; //reset lost flag
}
}
}

static void mis_last_pass(unsigned char* const status, const int size, const int threadID, const int threadCount)
{
const int top = size;

for (int e = threadID; e < top; e += threadCount) {
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

init(priority, status, status_new, lost, g.nodes);

timeval beg, end;
gettimeofday(&beg, NULL);

bool goagain;
int iter = 0;
do {
iter++;
goagain = false;

// edge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis, g, sp, priority, status, status_new, lost, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

// merge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(updateUndecided, status, status_new, g.nodes, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

// vertex pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis_vertex_pass, status, status_new, lost, g.nodes, std::ref(goagain), i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

// merge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(updateUndecided, status, status_new, g.nodes, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}

} while (goagain);

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
return runtime;
}

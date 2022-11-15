typedef unsigned long long data_type;
#include "indigo_mis_edge_cpp.h"

static void init(data_type* const priority, unsigned char* const status, bool* const lost, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++)
  {
    priority[v] = ((unsigned long)hash(v + 712313887)) | ((unsigned long)hash(v + 683067839) << (sizeof (unsigned int) * 8));
    status[v] = undecided;
    lost[v] = false;
  }
}

static void mis(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, bool* const lost, const int threadID, const int threadCount)
{
  const int begEdge = threadID * (long)g.edges / threadCount;
  const int endEdge = (threadID + 1) * (long)g.edges / threadCount;
  // go over all the edges
  for (int e = begEdge; e < endEdge; e++) {
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

static void mis_vertex_pass(unsigned char* const status, bool* const lost, const int size, bool& goagain, const int threadID, const int threadCount)
{
const int begNode = threadID * (long)size / threadCount;
const int endNode = (threadID + 1) * (long)size / threadCount;
// go over all vertexes
for (int v = begNode; v < endNode; v++) {
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
bool* lost = new bool [g.nodes];
std::thread threadHandles[threadCount];

init(priority, status, lost, g.nodes);

timeval beg, end;
gettimeofday(&beg, NULL);

bool goagain;
int iter = 0;
do {
iter++;
goagain = false;

// edge pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis, g, sp, priority, status, lost, i, threadCount);
}
for (int i = 0; i < threadCount; ++i) {
threadHandles[i].join();
}


// vertex pass
for (int i = 0; i < threadCount; ++i) {
threadHandles[i] = std::thread(mis_vertex_pass, status, lost, g.nodes, std::ref(goagain), i, threadCount);
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

delete [] lost;
return runtime;
}

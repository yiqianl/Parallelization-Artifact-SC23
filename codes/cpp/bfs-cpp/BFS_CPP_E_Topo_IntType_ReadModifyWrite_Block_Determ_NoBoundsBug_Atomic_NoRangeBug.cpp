
/*
This file is part of the Indigo2 benchmark suite version 0.9.

Copyright (c) 2023, Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://cs.txstate.edu/~burtscher/research/Indigo2Suite/.

Publication: This work is described in detail in the following paper.

Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes: Lessons Learned from 1106 Programs." Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2023.
*/

typedef int data_type;
#include "bfs_edge_cpp.h"

static void init(const int src, data_type* const dist, data_type* const dist_n, const int size)
{
  // initialize arrays
  for (int v = 0; v < size; v++) {
    const data_type temp = (v == src) ? 0 : maxval;
    dist_n[v] = temp;
    dist[v] = temp;
  }
}

static void bfs(const ECLgraph g, const int* const sp, data_type* const dist, data_type* const dist_n, int &goagain, const int threadID, const int threadCount)
{
  const int begEdge = threadID * (long)g.edges / threadCount;
  const int endEdge = (threadID + 1) * (long)g.edges / threadCount;
  for (int e = begEdge; e < endEdge; e++) {
    const int src = sp[e];
    const int dst = g.nlist[e];
    const data_type s = dist[src];
    if (s != maxval) {
      const data_type new_dist = s + 1;
      if (atomicMin(&dist_n[dst], new_dist) > new_dist) {
        atomicWrite(&goagain, 1);
      }
    }
  }
}

static double CPPbfs_edge(const int src, const ECLgraph& g, data_type* dist, const int* const sp, const int threadCount)
{
  data_type* dist_new = new data_type [g.nodes];
  std::thread threadHandles[threadCount];

  init(src, dist, dist_new, g.nodes);

  timeval start, end;
  gettimeofday(&start, NULL);

  // iterate until no more changes
  int goagain;
  int iter = 0;
  do {
    iter++;
    goagain = 0;

    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i] = std::thread(bfs, g, sp, dist, dist_new, std::ref(goagain), i, threadCount);
    }
    for (int i = 0; i < threadCount; ++i) {
      threadHandles[i].join();
    }

    std::swap(dist, dist_new);
  } while (goagain);

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("iterations: %d\n", iter);

  delete [] (iter % 2 ? dist : dist_new);
  return runtime;
}

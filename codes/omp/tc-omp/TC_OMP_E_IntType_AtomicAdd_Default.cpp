
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

typedef int basic_t;
#include "tc_edge_omp.h"

static void triCounting(basic_t& g_count, const int edges, const int* const nindex, const int* const nlist, const int* const sp)
{
  #pragma omp parallel for
  for (int e = 0; e < edges; e++) {
    basic_t count = 0;
    const int src = sp[e];
    const int dst = nlist[e];
    if (src > dst) {
      const int beg1 = nindex[dst];
      const int end1 = nindex[dst + 1];
      for (int i = beg1; i < end1 && nlist[i] < dst; i++) {
        const int u = nlist[i];
        int beg2 = nindex[src];
        int end2 = nindex[src + 1];
        if (find(u, beg2, end2, nlist)) count++;
      }
      #pragma omp atomic
      g_count += count;
    }
  }
}

static double CPUtc_edge(basic_t &count, const int edges, const int* const nindex, const int* const nlist, const int* const sp)
{
  timeval start, end;
  count = 0;

  gettimeofday(&start, NULL);

  triCounting(count, edges, nindex, nlist, sp);

  gettimeofday(&end, NULL);

  return (end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
}

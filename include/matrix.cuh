#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>  // for std::memset
#include <stdio.h>
//
// Transpose a CSR‐encoded binary matrix (or directed graph).
//
// Parameters:
//   N         – number of rows (and columns) in the square matrix (i.e. number of vertices)
//   M         – number of nonzeros (i.e. number of edges)
//   h_rowptr  – CSR "row pointer" array of length N+1
//   h_colidx  – CSR "column indices" array of length M
//
// Outputs (allocated inside):
//   *h_rowptrT – CSR rowptr for the transpose (length N+1)
//   *h_colidxT – CSR colidx for the transpose (length M)
//
void transposeCSR(
    const uint32_t  N,
    const uint32_t  M,
    const uint32_t *h_rowptr,
    const uint32_t *h_colidx,
    uint32_t      **h_rowptrT,
    uint32_t      **h_colidxT
) {
    // 1) Allocate output CSR arrays
    //    rowptrT has length N+1
    //    colidxT has length M
    *h_rowptrT = (uint32_t*) std::malloc((N + 1) * sizeof(uint32_t));
    *h_colidxT = (uint32_t*) std::malloc(M * sizeof(uint32_t));
    if (*h_rowptrT == nullptr || *h_colidxT == nullptr) {
        fprintf(stderr, "Error: failed to allocate memory in transposeCSR\n");
        std::exit(EXIT_FAILURE);
    }

    // 2) Count the number of entries that will go into each row of the transpose.
    //    In other words, for each nonzero at (u → v), we will add a nonzero at (v → u) in the transpose.
    //    So first we zero out a temporary "degree" array of length N, then scan all M edges
    //    and increment deg[v] for each column index v in the original.
    uint32_t *deg = (uint32_t*) std::calloc(N, sizeof(uint32_t));
    if (deg == nullptr) {
        fprintf(stderr, "Error: failed to allocate memory for degree array\n");
        std::exit(EXIT_FAILURE);
    }

    // Walk through all edges in the original CSR:
    // for each row u, edges are h_colidx[h_rowptr[u] .. h_rowptr[u+1]-1]
    for (uint32_t u = 0; u < N; ++u) {
        uint32_t row_start = h_rowptr[u];
        uint32_t row_end   = h_rowptr[u + 1];
        for (uint32_t idx = row_start; idx < row_end; ++idx) {
            uint32_t v = h_colidx[idx];
            // increment the count of how many times v appears as a target (i.e. degree of row v in the transpose)
            ++deg[v];
        }
    }

    // 3) Build the transposed rowptrT by doing an exclusive prefix sum over deg[0..N-1].
    //    rowptrT[0] = 0; rowptrT[i] = sum_{k=0..i-1} deg[k]
    //    This makes rowptrT[i+1] = rowptrT[i] + deg[i].
    (*h_rowptrT)[0] = 0;
    for (uint32_t i = 0; i < N; ++i) {
        (*h_rowptrT)[i + 1] = (*h_rowptrT)[i] + deg[i];
    }

    // 4) At this point, (*h_rowptrT)[N] == M (total number of nonzeros).
    //    We'll reuse deg[] as a "current insertion index" array to track
    //    how many row‐entries we've already placed in each transposed row.
    //
    //    Concretely, "degT[i]" will move from rowptrT[i] up to rowptrT[i+1]-1
    //    as we insert new column‐indices. So reset deg[i] = rowptrT[i].
    for (uint32_t i = 0; i < N; ++i) {
        deg[i] = (*h_rowptrT)[i];
    }

    // 5) Fill in colidxT:
    //    For every edge u → v in the original, insert u into the
    //    list of row v in the transpose.  The place to insert is deg[v]++.
    for (uint32_t u = 0; u < N; ++u) {
        uint32_t row_start = h_rowptr[u];
        uint32_t row_end   = h_rowptr[u + 1];
        for (uint32_t idx = row_start; idx < row_end; ++idx) {
            uint32_t v = h_colidx[idx];
            uint32_t insert_pos = deg[v];
            (*h_colidxT)[insert_pos] = u;
            ++deg[v];
        }
    }

    // 6) Clean up the temporary degree buffer
    std::free(deg);
}


void CSRtoCSC(

) {

}



#endif

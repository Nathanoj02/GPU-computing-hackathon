#include "utils.cuh"

#define ROWS_PER_THREAD 2

// Kernel
__global__ void bfs_kernel_spmv(
    const uint32_t* row_offsets,      // CSR row offsets
    const uint32_t* col_indices,      // CSR column indices (neighbors)
    int* distances,                   // Output distances array
    const uint32_t* frontier,         // Current frontier
    uint32_t* next_frontier,          // Next frontier to populate
    uint32_t N,                       // Rows of the matrix (number of nodes)
    uint32_t current_level,           // BFS level (depth)
//   uint32_t* next_frontier_size      // Counter for next frontier
    bool *one_modified
) {
    // = Row index (indica il nodo della frontier su cui stiamo lavorando, cioè il potenziale neighbor)
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    // Check prima per riga (se è già stato esplorato il nodo)
    if (distances[tid] != -1) {
        return;
    }

    uint32_t row_start = row_offsets[tid];
    uint32_t row_end = row_offsets[tid + 1];

    // Controllo se row_start = row_end (sono i nodi singoli, non raggiunti da altri nodi)
    // if (row_start == row_end) {
    //     return;
    // }

    // Spmv CSR modificata (evita gli 0 nel vettore)
    bool p = false;
    for (uint32_t i = row_start; i < row_end; i++) {
        uint32_t column = col_indices[i];
        uint32_t frontier_value = frontier[column];

        if (frontier_value > 0) {
            p = true;
            break;
        }
    }

    // Assegna la profondità e aggiungi alla frontiera
    if (p) {
        distances[tid] = current_level + 1;

        // Atomically add the neighbor to the next frontier
        // uint32_t index = atomicAdd(next_frontier_size, 1);
        next_frontier[tid] = 1;

        // Set one modified true
        *one_modified = true;
    }
  
}

// Kernel
__global__ void bfs_kernel_row_spmv(
    const uint32_t* row_offsets,      // CSR row offsets
    const uint32_t* col_indices,      // CSR column indices (neighbors)
    int* distances,                   // Output distances array
    const uint32_t* frontier,         // Current frontier
    uint32_t* next_frontier,          // Next frontier to populate
    uint32_t N,                       // Rows of the matrix (number of nodes)
    uint32_t current_level,           // BFS level (depth)
    bool *one_modified
) {
    // = Row index (indica il nodo della frontier su cui stiamo lavorando, cioè il potenziale neighbor)
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    //check se il mio thread non mi serve
    if (tid >= CEILING(N, ROWS_PER_THREAD)) return;

    uint32_t computed_tid = tid * ROWS_PER_THREAD;

    // Check prima per ogni riga che esegue il mio thread, considerando l'ultimo thread che potrebbe andare out of memory
    for(uint32_t i = 0; i < ROWS_PER_THREAD && computed_tid + i <= N; i++)
        if (distances[computed_tid + i] != -1) {
            return;
    }

    uint32_t row_start = row_offsets[computed_tid];
    uint32_t row_end;
    if(computed_tid + ROWS_PER_THREAD > N)
        row_end = row_offsets[computed_tid + (N % ROWS_PER_THREAD) - 1];
    else{
        row_end = row_offsets[tid * (ROWS_PER_THREAD + 1)];
    }

    // Spmv CSR modificata (evita gli 0 nel vettore)
    bool p = false;
    bool new_row = true;
    uint32_t column;
    uint32_t frontier_value;
    // for(uint32_t offset = 0; offset < ROWS_PER_THREAD && computed_tid + offset <= N; offset++){
    for (uint32_t i = row_start; i < row_end; i++) {
        if(new_row){
            column = col_indices[i];
            frontier_value = frontier[column];
            //non posso fare break, devo gestire con una flag che finchè non cambio riga non aumento il computed_tid
            if (frontier_value > 0) {
                p = true;
            }
        }
        

        // Assegna la profondità e aggiungi alla frontiera
        if (p) {
            //set the right tid to modify, se row_offset[i] != row_offset[i + 1];
            
            distances[computed_tid] = current_level + 1;

            // Atomically add the neighbor to the next frontier
            // uint32_t index = atomicAdd(next_frontier_size, 1);
            next_frontier[computed_tid] = 1;
            new_row = false;
            p = false;
            // Set one modified true
            *one_modified = true;
        }

        if(i < row_end - 1 && row_offsets[i] != row_offsets[i + 1]){
            computed_tid++;
            new_row = true;
        }
    }
  
}

// CPU call
void gpu_bfs_spmv(
    const uint32_t N,
    const uint32_t M,
    const uint32_t *h_rowptr,
    const uint32_t *h_colidx,
    const uint32_t source,
    int *h_distances,
    bool is_placeholder
) {
    float tot_time = 0.0;
    CUDA_TIMER_INIT(H2D_copy)

    // Allocate and copy graph to device
    uint32_t* d_row_offsets; uint32_t* d_col_indices;
    CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate memory for distances and frontier queues
    int* d_distances; uint32_t* d_frontier; uint32_t* d_next_frontier; // uint32_t* d_next_frontier_size;
    bool *d_one_modified;
    CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_one_modified, sizeof(bool)));
    //   CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

    uint32_t *h_frontier = (uint32_t *) calloc (N, sizeof(uint32_t));
    h_frontier[source] = 1;

    CHECK_CUDA(cudaMemcpy(d_frontier, h_frontier, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // Initialize all distances to -1 (unvisited), and source distance to 0
    CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0

    CUDA_TIMER_STOP(H2D_copy)
    #ifdef DEBUG_PRINTS
    CUDA_TIMER_PRINT(H2D_copy)
    #endif
    tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
    CUDA_TIMER_DESTROY(H2D_copy)

    //   uint32_t current_frontier_size = 1;

    uint32_t level = 0;
    bool one_modified = true;

    // Main BFS loop
    CPU_TIMER_INIT(BFS_SPMV)
    while (one_modified) {

    #ifdef DEBUG_PRINTS
        // printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n", is_placeholder ? "" : " BASELINE", level, current_frontier_size);
    #endif
    #ifdef ENABLE_NVTX
        // Mark start of level in NVTX
        nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
    #endif

    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_one_modified, false, sizeof(bool)));

    // uint32_t block_size = 1024;
    // uint32_t num_blocks = ceil(N / (float) block_size);

    // // CUDA_TIMER_INIT(BFS_kernel)
    // bfs_kernel_spmv<<<num_blocks, block_size>>>(
    //     d_row_offsets,
    //     d_col_indices,
    //     d_distances,
    //     d_frontier,
    //     d_next_frontier,
    //     N,
    //     level,
    //     d_one_modified
    // );


    uint32_t block_size = 1024;
    uint32_t num_blocks = ceil((N / ROWS_PER_THREAD) / (float) block_size);

    // CUDA_TIMER_INIT(BFS_kernel)
    //TO DO: modify the number of threads to launch
    bfs_kernel_row_spmv<<<num_blocks, block_size>>>(
        d_row_offsets,
        d_col_indices,
        d_distances,
        d_frontier,
        d_next_frontier,
        N,
        level,
        d_one_modified
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    // CUDA_TIMER_STOP(BFS_kernel)
    // #ifdef DEBUG_PRINTS
    //   CUDA_TIMER_PRINT(BFS_kernel)
    // #endif
    // CUDA_TIMER_DESTROY(BFS_kernel)

    // Swap frontier pointers
    std::swap(d_frontier, d_next_frontier);

    // Copy size of next frontier to host
    // CHECK_CUDA(cudaMemcpy(&current_frontier_size, d_next_frontier_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // Copy one modified to host
    CHECK_CUDA(cudaMemcpy(&one_modified, d_one_modified, sizeof(bool), cudaMemcpyDeviceToHost));
    level++;

    #ifdef ENABLE_NVTX
        // End NVTX range for level
        nvtxRangePop();
    #endif
    }
    CPU_TIMER_STOP(BFS_SPMV)
    #ifdef DEBUG_PRINTS
    CPU_TIMER_PRINT(BFS_SPMV)
    #endif
    tot_time += CPU_TIMER_ELAPSED(BFS_SPMV);

    CUDA_TIMER_INIT(D2H_copy)
    CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TIMER_STOP(D2H_copy)
    #ifdef DEBUG_PRINTS
    CUDA_TIMER_PRINT(D2H_copy)
    #endif
    tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
    CUDA_TIMER_DESTROY(D2H_copy)

    printf("\n[OUT] Total%s BFS time: %f ms\n", is_placeholder ? "" : " BASELINE", tot_time);
    if (!is_placeholder) printf("[OUT] Graph diameter: %u\n", level);

    // Free device memory
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    // cudaFree(d_next_frontier_size);
    cudaFree(d_one_modified);
}
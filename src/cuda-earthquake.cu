#include "hpc.h"
#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>     /* rand() */
#include <iostream>

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
/* pre-defined seed for pseudo random initialization */
#define SEED 19
#define BLKDIM 32
#define BLKSIZE 1024

__constant__ float FDELTA = EMAX/4;


__host__ __device__ float* IDX(float *grid, int i, int j, int n)
{
    return (grid + i*n + j);
}

/**
 * Restituisce un numero reale pseudocasuale con probabilita' uniforme
 * nell'intervallo [a, b], con a < b.
 */
float randab(float a, float b)
{
    return a + (b-a) * (rand() / (float)RAND_MAX);
}

void setup(float *grid, int ext_n, float fmin, float fmax)
{
    for (int i = 1; i < ext_n-1; i++) {
        for (int j = 1; j < ext_n-1; j++) {
            *IDX(grid, i, j, ext_n) = randab(fmin, fmax);
        }
    }
}

/**
 * Somma delta a tutte le celle del dominio grid di dimensioni
 * n*n. Questa funzione realizza il passo 1 descritto nella specifica
 * del progetto.
 */
__global__ void increment_energy(float *grid, int ext_n, float delta)
{
    /* prevent the increment of halo */
    const int i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;

    if (i < ext_n-1 && j < ext_n-1) {
        *IDX(grid, i, j, ext_n) += delta;
    }
}

/**
 * Restituisce il numero di celle la cui energia e' strettamente
 * maggiore di EMAX.
 */
__global__ void count_cells(float *grid, size_t ext_n, int *res, unsigned int s)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i == 0) res[s] = 0;

    if (i < ext_n*ext_n) {
        if (grid[i] > EMAX) {
            atomicAdd(&res[s], 1);
        }
    }
}

__global__ void count_cells_shared(float *grid, size_t ext_n, int *res, unsigned int s)
{
    extern __shared__ int ldata[];
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    const int lindex = threadIdx.x;

    if (gindex == 0) res[s] = 0;
    ldata[lindex] = 0;

    if (gindex < ext_n * ext_n) {
        ldata[lindex] = (int)(grid[gindex] > EMAX);

        __syncthreads();

        if (lindex == 0) {
            for (int i = 1; i < blockDim.x; i++) {
                ldata[0] += ldata[i];
            }

            atomicAdd(&res[s], ldata[0]);
        }
    }
}

/**
 * Distribuisce l'energia di ogni cella a quelle adiacenti (se
 * presenti). cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie. Questa funzione
 * realizza il passo 2 descritto nella specifica del progetto.
 */
__global__ void propagate_energy(float* cur, float* next, size_t ext_n)
{
    const int i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;

    if (i < ext_n-1 && j < ext_n-1) {
        float F = *IDX(cur, i, j, ext_n);
        float *out = IDX(next, i, j, ext_n);

        /* Se l'energia del vicino di sinistra (se esiste) e'
           maggiore di EMAX, allora la cella (i,j) ricevera'
           energia addizionale FDELTA = EMAX/4 */

        F += (float)(*IDX(cur, i, j-1, ext_n)>EMAX) * FDELTA;
        F += (float)(*IDX(cur, i, j+1, ext_n)>EMAX) * FDELTA;
        F += (float)(*IDX(cur, i-1, j, ext_n)>EMAX) * FDELTA;
        F += (float)(*IDX(cur, i+1, j, ext_n)>EMAX) * FDELTA;

        //F += *IDX(cur, i, j-1, ext_n) > EMAX ? FDELTA : 0.0f;
        //F += *IDX(cur, i, j+1, ext_n) > EMAX ? FDELTA : 0.0f;
        //F += *IDX(cur, i-1, j, ext_n) > EMAX ? FDELTA : 0.0f;
        //F += *IDX(cur, i+1, j, ext_n) > EMAX ? FDELTA : 0.0f;

        //if (*IDX(cur, i, j-1, ext_n) > EMAX) {
        ////if (data[ti][tj-1] > EMAX) {
        //    F += FDELTA;
        //}
        ///* Idem per il vicino di destra */
        //if (*IDX(cur, i, j+1, ext_n) > EMAX) {
        ////if (data[ti][tj+1] > EMAX) {
        //    F += FDELTA;
        //}
        ///* Idem per il vicino in alto */
        //if (*IDX(cur, i-1, j, ext_n) > EMAX) {
        ////if (data[ti-1][tj] > EMAX) {
        //    F += FDELTA;
        //}
        ///* Idem per il vicino in basso */
        //if (*IDX(cur, i+1, j, ext_n) > EMAX) {
        ////if (data[ti+1][tj] > EMAX) {
        //    F += FDELTA;
        //}

        F -= (float)(F>EMAX) * EMAX;

        //if (F > EMAX) {
        //    F -= EMAX;
        //}

        /* Si noti che il valore di F potrebbe essere ancora
           maggiore di EMAX; questo non e' un problema:
           l'eventuale eccesso verra' rilasciato al termine delle
           successive iterazioni fino a riportare il valore
           dell'energia sotto la foglia EMAX. */
        *out = F;
    }
}

__global__ void propagate_energy_shared(float* cur, float* next, size_t ext_n)
{
    __shared__ float data[BLKDIM][BLKDIM];
    const int i = threadIdx.x + blockIdx.x * (blockDim.x-2);
    const int j = threadIdx.y + blockIdx.y * (blockDim.y-2);

    const int ti = threadIdx.x;
    const int tj = threadIdx.y;

    if (i < ext_n && j < ext_n) {
        data[ti][tj] = *IDX(cur, i, j, ext_n);
        __syncthreads();

        if (ti > 0 && ti < blockDim.x-1 && tj > 0 && tj < blockDim.y-1 &&
                i < ext_n-1 && j < ext_n-1) {

            //float F = *IDX(cur, i, j, ext_n);
            float F = data[ti][tj];
            float *out = IDX(next, i, j, ext_n);

            /* Se l'energia del vicino di sinistra (se esiste) e'
               maggiore di EMAX, allora la cella (i,j) ricevera'
               energia addizionale FDELTA = EMAX/4 */
            //if (*IDX(cur, i, j-1, ext_n) > EMAX) {
            F += (data[ti][tj-1] > EMAX) ? FDELTA : 0.0f;
            F += (data[ti][tj+1] > EMAX) ? FDELTA : 0.0f;
            F += (data[ti-1][tj] > EMAX) ? FDELTA : 0.0f;
            F += (data[ti+1][tj] > EMAX) ? FDELTA : 0.0f;
            //if (data[ti][tj-1] > EMAX) {
            //    F += FDELTA;
            //}
            ///* Idem per il vicino di destra */
            ////if (*IDX(cur, i, j+1, ext_n) > EMAX) {
            //if (data[ti][tj+1] > EMAX) {
            //    F += FDELTA;
            //}
            ///* Idem per il vicino in alto */
            ////if (*IDX(cur, i-1, j, ext_n) > EMAX) {
            //if (data[ti-1][tj] > EMAX) {
            //    F += FDELTA;
            //}
            ///* Idem per il vicino in basso */
            ////if (*IDX(cur, i+1, j, ext_n) > EMAX) {
            //if (data[ti+1][tj] > EMAX) {
            //    F += FDELTA;
            //}

            if (F > EMAX) {
                F -= EMAX;
            }

            /* Si noti che il valore di F potrebbe essere ancora
               maggiore di EMAX; questo non e' un problema:
               l'eventuale eccesso verra' rilasciato al termine delle
               successive iterazioni fino a riportare il valore
               dell'energia sotto la foglia EMAX. */
            *out = F;
        }
    }
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
/*__global__ void average_energy(float* grid, size_t ext_elem, float *res, unsigned int s)
{
    extern __shared__ float data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    if (i < ext_elem) {
        data[tid] = grid[i] + grid[i+blockDim.x];
    } else { //padding memory
        data[tid] = 0.0f;
    }

    if (i == 0) res[s] = 0.0f;

    __syncthreads();

    for (unsigned int r = blockDim.x/2; r > 0; r >>= 1) {
        if (tid < r) {
            data[tid] += data[tid + r];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&res[s], data[0]);
    }
}*/

/*__global__ void average_energy(float* grid, size_t ext_elem, float *res, unsigned int s) {
    extern __shared__ float local_sum[];
    const unsigned int lindex = threadIdx.x;
    const unsigned int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int bsize = blockDim.x / 2;

    if (gindex == 0)
        res[s] = 0.0f;

    if (gindex < ext_elem) {
        local_sum[lindex] = grid[gindex];
    } else {
        local_sum[lindex] = 0.0f;
    }
    __syncthreads();

    while (bsize > 0) {
        if (lindex < bsize) {
            local_sum[lindex] += local_sum[lindex + bsize];
        }
        bsize /= 2;
        __syncthreads();
    }
    if (lindex == 0) {
        atomicAdd(&res[s], local_sum[0]);
    }
}*/

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid+32];
    if (blockSize >= 32) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16) sdata[tid] += sdata[tid+ 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid+ 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid+ 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid+ 1];
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
template <unsigned int blockSize>
__global__ void average_energy(float* g_idata, float* g_odata, size_t n, unsigned int s) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0f;

    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce<BLKSIZE/2>(sdata, tid);
    if (tid == 0) atomicAdd(&g_odata[s], sdata[0]);
}

__global__ void halo_top_bottom(float* grid, int ext_n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < ext_n) {
        *IDX(grid, 0, j, ext_n) = 0.0f;
        *IDX(grid, ext_n-1, j, ext_n) = 0.0f;
    }
}

__global__ void halo_left_right(float* grid, int ext_n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ext_n) {
        *IDX(grid, i, 0, ext_n) = 0.0f;
        *IDX(grid, i, ext_n-1, ext_n) = 0.0f;
    }
}

int main(int argc, char* argv[])
{
    int n = 256, nsteps = 2048;
    srand(SEED); /* Inizializzazione del generatore pseudocasuale */

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [nsteps [n]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        char *pEnd;
        nsteps = strtol(argv[1], &pEnd, 10);
    }

    if ( argc > 2 ) {
        char *pEnd;
        n = strtol(argv[2], &pEnd, 10);
    }

    const size_t ext_n = n+2;
    const size_t ext_size = (ext_n) * (ext_n) * sizeof(float);
    const size_t ext_elem = (ext_n) * (ext_n);

    float *cur = (float *) malloc(ext_size); assert(cur);
    float *next = (float *) malloc(ext_size); assert(next);
    float *emean = (float *) malloc(sizeof(float)*nsteps); assert(emean);
    int *c = (int *) malloc(sizeof(int)*nsteps); assert(c);

    /* CUDA memory allocation */
    float *d_cur, *d_next;
    int *d_c;
    float *d_emean;

    cudaSafeCall(cudaMalloc((void **) &d_cur, ext_size));
    cudaSafeCall(cudaMalloc((void **) &d_next, ext_size));
    cudaSafeCall(cudaMalloc((void **) &d_c, sizeof(int)*nsteps));
    cudaSafeCall(cudaMalloc((void **) &d_emean, sizeof(float)*nsteps));

    dim3 block2(BLKDIM, BLKDIM);
    dim3 grid2((ext_n + BLKDIM - 1) / BLKDIM, (ext_n + BLKDIM - 1) / BLKDIM);
    dim3 block1(BLKSIZE);
    dim3 grid1((ext_elem + BLKSIZE - 1) / BLKSIZE);
    dim3 p_block(BLKDIM, BLKDIM);
    dim3 p_grid((ext_n + BLKDIM - 3) / (BLKDIM-2), (ext_n + BLKDIM - 3) / (BLKDIM-2));
    /* END CUDA memory allocation */

    setup(cur, ext_n, 0, EMAX*0.1);
    /* after setup halo cells contain garbage: cuda kernel fill halo with 0.0f */

    /* Copy to CUDA memory */
    cudaSafeCall(cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_next, cur, ext_size, cudaMemcpyHostToDevice));

    /* Fill halo cells with 0.0f */
    halo_top_bottom<<<grid1, block1>>>(d_cur, ext_n);
    halo_left_right<<<grid1, block1>>>(d_cur, ext_n);
    halo_top_bottom<<<grid1, block1>>>(d_next, ext_n);
    halo_left_right<<<grid1, block1>>>(d_next, ext_n);

    const dim3 c_block(BLKSIZE);
    int out_elem = ext_elem / (BLKSIZE/2);
    if (ext_elem % (BLKSIZE/2)) {
        out_elem++;
    }
    const dim3 c_grid(out_elem);

    const size_t sum_buff_size = sizeof(float) * BLKSIZE;

    const double tstart = hpc_gettime();
    for (unsigned int s = 0; s < nsteps; s++) {
        increment_energy<<<grid2, block2>>>(d_cur, ext_n, EDELTA);
#ifdef _SHARED
        count_cells_shared<<<grid1, block1, sizeof(int)*BLKSIZE>>>(d_cur, ext_n, d_c, s);
#else
        count_cells<<<grid1, block1>>>(d_cur, ext_n, d_c, s);
#endif

#ifdef _SHARED
        propagate_energy_shared<<<p_grid, p_block>>>(d_cur, d_next, ext_n);
#else
        propagate_energy<<<grid2, block2>>>(d_cur, d_next, ext_n);
#endif
        average_energy<BLKSIZE/2><<<c_grid, c_block, sum_buff_size>>>(d_next, d_emean, ext_elem, s);

        float *tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }
    cudaDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;

    cudaSafeCall(cudaMemcpy(c, d_c, sizeof(int)*nsteps, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(emean, d_emean, sizeof(float)*nsteps, cudaMemcpyDeviceToHost));

    for (unsigned int s = 0; s < nsteps; s++) {
        printf("%d %f\n", c[s], emean[s]/(n*n));
    }

    /* milioni di celle aggiornate per ogni secondo di wall clock time */
    double Mupdates = (((double) n) * n / 1.0e6) * nsteps;
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);

    free(cur);
    free(next);
    free(emean);
    free(c);
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaFree(d_c);
    cudaFree(d_emean);

    return EXIT_SUCCESS;
}


// vim: set nofoldenable ts=4 sw=4 sts=4 et :

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
__global__ void count_cells(float *grid, int ext_n, int *res, int s)
{
    //extern __shared__ int sdata[];

    //unsigned int tid = threadIdx.x;
    ////unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    //sdata[tid] = 0;

    //__syncthreads();

    //for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    //    if (tid % (2 * s) == 0 && *(grid+s) > EMAX) {
    //        sdata[tid]++;
    //    }

    //    __syncthreads();
    //}

    //if (tid == 0) {
    //    atomicAdd(res, sdata[0]);
    //}

    //int count = thrust::count_if(thrust::device, grid, grid+(n*n), is_emax<float>());
    ////std::cout << "Count: " << count << std::endl;
    //return count;
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i == 0) res[s] = 0;

    if (i < ext_n*ext_n) {
        if (grid[i] > EMAX) {
            atomicAdd(&res[s], 1);
        }
    }
}

/**
 * Distribuisce l'energia di ogni cella a quelle adiacenti (se
 * presenti). cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie. Questa funzione
 * realizza il passo 2 descritto nella specifica del progetto.
 */
__global__ void propagate_energy(float *cur, float *next, int ext_n)
{
    //__shared__ float data[BLKDIM][BLKDIM];
    const float FDELTA = EMAX/4;
    const int i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;

    const int ti = threadIdx.x;
    const int tj = threadIdx.y;

    if (i < ext_n-1 && j < ext_n-1) {
        //data[ti][tj] = *IDX(cur, i, j, ext_n);
        //__syncthreads();

        //if (ti > 0 && ti < blockDim.x-1 && tj > 0 && tj < blockDim.y-1) {
            float F = *IDX(cur, i, j, ext_n);
            float *out = IDX(next, i, j, ext_n);

            /* Se l'energia del vicino di sinistra (se esiste) e'
               maggiore di EMAX, allora la cella (i,j) ricevera'
               energia addizionale FDELTA = EMAX/4 */
            if (*IDX(cur, i, j-1, ext_n) > EMAX) {
            //if (data[ti][tj-1] > EMAX) {
                F += FDELTA;
            }
            /* Idem per il vicino di destra */
            if (*IDX(cur, i, j+1, ext_n) > EMAX) {
            //if (data[ti][tj+1] > EMAX) {
                F += FDELTA;
            }
            /* Idem per il vicino in alto */
            if (*IDX(cur, i-1, j, ext_n) > EMAX) {
            //if (data[ti-1][tj] > EMAX) {
                F += FDELTA;
            }
            /* Idem per il vicino in basso */
            if (*IDX(cur, i+1, j, ext_n) > EMAX) {
            //if (data[ti+1][tj] > EMAX) {
                F += FDELTA;
            }

            if (F > EMAX) {
                F -= EMAX;
            }

            /* Si noti che il valore di F potrebbe essere ancora
               maggiore di EMAX; questo non e' un problema:
               l'eventuale eccesso verra' rilasciato al termine delle
               successive iterazioni fino a riportare il valore
               dell'energia sotto la foglia EMAX. */
            *out = F;
        //}
    }
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
__global__ void average_energy(float* grid, int ext_n, float *res, int s)
{
    extern __shared__ float data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    if (i < ext_n*ext_n) {
        data[tid] = grid[i] + grid[i+blockDim.x];
    } else { //padding memory
        data[tid] = 0.0f;
    }

    if (i == 0) res[s] = 0.0f;

    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            data[tid] += data[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&res[s], data[0]);
    }
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
    dim3 grid1((ext_n*ext_n + BLKSIZE - 1) / BLKSIZE);
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

    const dim3 c_block(BLKSIZE, 1, 1);
    int out_elem = ext_n*ext_n / (BLKSIZE/2);
    if ((ext_n*ext_n) % BLKSIZE/2) {
        out_elem++;
    }
    const dim3 c_grid(out_elem, 1, 1);

    const size_t sum_buff_size = sizeof(float) * BLKSIZE;

    const double tstart = hpc_gettime();
    for (int s = 0; s < nsteps; s++) {
        /* stuff */
        increment_energy<<<grid2, block2>>>(d_cur, ext_n, EDELTA);
        count_cells<<<grid1, block1>>>(d_cur, ext_n, d_c, s);
        propagate_energy<<<grid2, block2>>>(d_cur, d_next, ext_n);
        average_energy<<<c_grid, c_block, sum_buff_size>>>(d_next, ext_n, d_emean, s);

        //printf("%d %f\n", c, emean/(n*n));

        float *tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;

    cudaSafeCall(cudaMemcpy(c, d_c, sizeof(int)*nsteps, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(emean, d_emean, sizeof(float)*nsteps, cudaMemcpyDeviceToHost));

    for (int s = 0; s < nsteps; s++) {
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

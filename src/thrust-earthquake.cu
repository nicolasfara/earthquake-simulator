#include "hpc.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/count.h>

#include <iostream>

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
/* pre-defined seed for pseudo random initialization */
#define SEED 19
#define BLKDIM 32

template <typename T>
struct is_emax
{
  __host__ __device__
  bool operator()(const T& x) const
  {
    return x > EMAX;
  }
};

__host__ __device__ int IDX(int i, int j, int n)
{
    return (i*n + j);
}

/**
 * Restituisce un numero reale pseudocasuale con probabilita' uniforme
 * nell'intervallo [a, b], con a < b.
 */
float randab(float a, float b)
{
    return a + (b-a) * (rand() / (float)RAND_MAX);
}

void setup(thrust::host_vector<float> &grid, int n, float fmin, float fmax)
{
    int ghost_n = n + 2;
    for (int i = 0; i < ghost_n; i++) {
        grid[IDX(0, i, ghost_n)] = 0;
        grid[IDX(n + 1, i, ghost_n)] = 0;
        grid[IDX(i, 0, ghost_n)] = 0;
        grid[IDX(i, n + 1, ghost_n)] = 0;
    }

    for (int i = 0; i < grid.size(); i++) {
        grid[i] = randab(fmin, fmax);
    }
}

/**
 * Somma delta a tutte le celle del dominio grid di dimensioni
 * n*n. Questa funzione realizza il passo 1 descritto nella specifica
 * del progetto.
 */
__global__ void increment_energy(float* grid, int n, float delta)
{
    const int i = threadIdx.x + blockIdx.x * (blockDim.x - 2);
    const int j = threadIdx.y + blockIdx.y * (blockDim.y - 2);

    if (i > 0 && threadIdx.x < blockDim.x - 1 && j > 0 && threadIdx.y < blockDim.y - 1) {
        grid[IDX(i, j, n)] += delta;
    }
}

/**
 * Restituisce il numero di celle la cui energia e' strettamente
 * maggiore di EMAX.
 */
int count_cells(thrust::device_vector<float> &grid)
{
    return thrust::count_if(thrust::device, grid.begin(), grid.end(), is_emax<float>());
}

/**
 * Distribuisce l'energia di ogni cella a quelle adiacenti (se
 * presenti). cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie. Questa funzione
 * realizza il passo 2 descritto nella specifica del progetto.
 */
__global__ void propagate_energy(float *cur, float *next, int n)
{
    const float FDELTA = EMAX/4;
    const int i = threadIdx.x + blockIdx.x * (blockDim.x - 2);
    const int j = threadIdx.y + blockIdx.y * (blockDim.y - 2);

    if (i > 0 && threadIdx.x < blockDim.x - 1 && j > 0 && threadIdx.y < blockDim.y - 1) {
        float F = cur[IDX(i, j, n)];
        float *out = &next[IDX(i, j, n)];

        /* Se l'energia del vicino di sinistra (se esiste) e'
           maggiore di EMAX, allora la cella (i,j) ricevera'
           energia addizionale FDELTA = EMAX/4 */
        if (cur[IDX(i, j-1, n)] > EMAX) {
            F += FDELTA;
        }
        /* Idem per il vicino di destra */
        if (cur[IDX(i, j+1, n)] > EMAX) {
            F += FDELTA;
        }
        /* Idem per il vicino in alto */
        if (cur[IDX(i-1, j, n)] > EMAX) {
            F += FDELTA;
        }
        /* Idem per il vicino in basso */
        if (cur[IDX(i+1, j, n)] > EMAX) {
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
    }
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
float average_energy(thrust::device_vector<float> &grid, int n)
{
    float sum = 0.0f;
    sum = thrust::reduce(grid.begin(), grid.end(), 0.0f, thrust::plus<float>());
    std::cout << "Sum: " << sum/(n*n) << std::endl;
    return (sum / (n*n));
}

int main(int argc, char* argv[])
{
    int n = 256, nsteps = 2048;
    int c, emean;
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

    const size_t size = (n + 2) * (n + 2) * sizeof(float);
    thrust::host_vector<float> cur(size);
    thrust::host_vector<float> next(size);

    setup(cur, n, 0, EMAX*0.1);

    thrust::device_vector<float> d_cur = cur;
    thrust::device_vector<float> d_next = next;

    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((n + BLKDIM - 3) / (BLKDIM - 2), (n + BLKDIM - 3) / (BLKDIM - 2));

    float *cur_ptr = thrust::raw_pointer_cast(&d_cur[0]);
    float *next_ptr = thrust::raw_pointer_cast(&d_next[0]);

    const double tstart = hpc_gettime();
    for (int s = 0; s < nsteps; s++) {
        /* stuff */
        std::cout << "Prev: " << d_cur[1] << std::endl;
        increment_energy<<<block, grid>>>(cur_ptr, n+2, EDELTA);
        std::cout << "Aft: " << d_cur[1] << std::endl;
        c = count_cells(d_cur);
        propagate_energy<<<block, grid>>>(cur_ptr, next_ptr, n+2);
        emean = average_energy(d_cur, n+2);

        printf("%d %f\n", c, emean);

        d_cur.swap(d_next);
        float *tmp = cur_ptr;
        cur_ptr = next_ptr;
        next_ptr = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;

    /* milioni di celle aggiornate per ogni secondo di wall clock time */
    double Mupdates = (((double) n) * n / 1.0e6) * nsteps;
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);

}


// vim: set nofoldenable ts=4 sw=4 sts=4 et :

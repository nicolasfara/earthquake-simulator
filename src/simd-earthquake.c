/****************************************************************************
 *
 * earthquake.c - Simple 2D earthquake model
 *
 * Copyright (C) 2018 Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ---------------------------------------------------------------------------
 *
 * Versione di riferimento del progetto di High Performance Computing
 * 2018/2019, corso di laurea in Ingegneria e Scienze Informatiche,
 * Universita' di Bologna. Per una descrizione del modello si vedano
 * le specifiche sulla pagina del corso:
 *
 * http://moreno.marzolla.name/teaching/HPC/
 *
 * Per compilare:
 *
 * gcc -D_XOPEN_SOURCE=600 -std=c99 -Wall -Wpedantic earthquake.c -o earthquake
 *
 * (il flag -D_XOPEN_SOURCE=600 e' superfluo perche' viene settato
 * nell'header "hpc.h", ma definirlo tramite la riga di comando fa si'
 * che il programma compili correttamente anche se inavvertitamente
 * non si include "hpc.h", o per errore non lo si include come primo
 * file come necessario).
 *
 * Per eseguire il programma si puo' usare la riga di comando seguente:
 *
 * ./earthquake 100000 256 > out
 *
 * Il primo parametro indica il numero di timestep, e il secondo la
 * dimensione (lato) del dominio. L'output consiste in coppie di
 * valori numerici (100000 in questo caso) il cui significato e'
 * spiegato nella specifica del progetto.
 *
 ****************************************************************************/
#include "hpc.h"
#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>     /* rand() */

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
/* pre-defined seed for pseudo random initialization */
#define SEED 19

#define VLEN (sizeof(vf)/sizeof(float))
typedef float vf __attribute__((vector_size(256)));


/**
 * Restituisce un puntatore all'elemento di coordinate (i,j) del
 * dominio grid con n colonne.
 */
static inline float *IDX(float *grid, int i, int j, int n)
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

/**
 * Inizializza il dominio grid di dimensioni n*n con valori di energia
 * scelti con probabilità uniforme nell'intervallo [fmin, fmax], con
 * fmin < fmax.
 *
 * NON PARALLELIZZARE QUESTA FUNZIONE: rand() non e' thread-safe,
 * qundi non va usata in blocchi paralleli OpenMP; inoltre la funzione
 * non si "comporta bene" con MPI (i dettagli non sono importanti, ma
 * posso spiegarli a chi e' interessato). Di conseguenza, questa
 * funzione va eseguita dalla CPU, e solo dal master (se si usa MPI).
 */
void setup(float* grid, int n, float fmin, float fmax)
{
    int ghost_n = n + 2;
#pragma omp parallel for default(none) shared(grid, ghost_n, n) //schedule(runtime)
    for(int j = 0; j < ghost_n; j++) {
        // righe
        *IDX(grid, 0, j, ghost_n) = 0.0f;
        *IDX(grid, n + 1, j, ghost_n) = 0.0f;
        // Colonne
        *IDX(grid, j, 0, ghost_n) = 0.0f;
        *IDX(grid, j, n + 1, ghost_n) = 0.0f;
    }
    // For non parallelizzabile (randab)
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            *IDX(grid, i, j, n) = randab(fmin, fmax);
        }
    }
}

/**
 * Somma delta a tutte le celle del dominio grid di dimensioni
 * n*n. Questa funzione realizza il passo 1 descritto nella specifica
 * del progetto.
 */
void increment_energy(float* grid, int n, float delta)
{
#pragma omp parallel for default(none) shared(grid, n, delta) //schedule(runtime)
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            *IDX(grid, i, j, n) += delta;
        }
    }
}

/**
 * Restituisce il numero di celle la cui energia e' strettamente
 * maggiore di EMAX.
 */
int count_cells(float *grid, int n)
{
    int c = 0;
    int i, j;
    __m256i vs;
#pragma omp parallel for reduction(+:c) default(none) private(j, vs) shared(grid, n) //schedule(runtime)
    for (i = 1; i < n + 1; i++) {
        for (j = 1; j < n+1; j++) {
            if (*IDX(grid, i, j, n) > EMAX) {
                c++;
            }
        }
    }
    return c;
}

/**
 * Distribuisce l'energia di ogni cella a quelle adiacenti (se
 * presenti). cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie. Questa funzione
 * realizza il passo 2 descritto nella specifica del progetto.
 */
void propagate_energy(float *cur, float *next, int n)
{
    const float FDELTA = EMAX/4;
    int i, j;
#pragma omp parallel for default(none) private(j) shared(n, cur, next) //schedule(runtime)
    for (i = 1; i < n + 1; i++) {
        for (j = 1; j < (n + 1) - 7; j += 8) {

            //float F = *IDX(cur, i, j, n);
            __m256 s_F = _mm256_loadu_ps(IDX(cur, i, j, n));
            float *out = IDX(next, i, j, n);

            /* Se l'energia del vicino di sinistra (se esiste) e'
               maggiore di EMAX, allora la cella (i,j) ricevera'
               energia addizionale FDELTA = EMAX/4 */

            __m256 ctrue = _mm256_set_ps(FDELTA, FDELTA, FDELTA, FDELTA, FDELTA, FDELTA, FDELTA, FDELTA);
            __m256 cfalse = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

            /* Nord, Sud, Est, West check */
            __m256 nord = _mm256_loadu_ps(IDX(cur, i-1, j, n));
            __m256 sud  = _mm256_loadu_ps(IDX(cur, i+1, j, n));
            __m256 est  = _mm256_loadu_ps(IDX(cur, i, j+1, n));
            __m256 west = _mm256_loadu_ps(IDX(cur, i, j-1, n));
            __m256 mask_nord = (nord > EMAX);
            __m256 mask_sud  = (sud  > EMAX);
            __m256 mask_est  = (est  > EMAX);
            __m256 mask_west = (west > EMAX);
            __m256 out_nord = _mm256_or_ps(_mm256_and_ps(mask_nord, ctrue), _mm256_andnot_ps(mask_nord, cfalse));
            __m256 out_sud  = _mm256_or_ps(_mm256_and_ps(mask_sud, ctrue), _mm256_andnot_ps(mask_sud, cfalse));
            __m256 out_est = _mm256_or_ps(_mm256_and_ps(mask_est, ctrue), _mm256_andnot_ps(mask_est, cfalse));
            __m256 out_west = _mm256_or_ps(_mm256_and_ps(mask_west, ctrue), _mm256_andnot_ps(mask_west, cfalse));
            s_F = _mm256_add_ps(s_F, out_nord);
            s_F = _mm256_add_ps(s_F, out_sud);
            s_F = _mm256_add_ps(s_F, out_est);
            s_F = _mm256_add_ps(s_F, out_west);

            /* Check if the current cell ha enegy > EMAX */
            __m256 utrue = _mm256_set_ps(EMAX, EMAX, EMAX, EMAX, EMAX, EMAX, EMAX, EMAX);
            __m256 ufalse = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            __m256 mask = (s_F > EMAX);
            __m256 sout = _mm256_or_ps(_mm256_and_ps(mask, utrue), _mm256_andnot_ps(mask, ufalse));
            __m256 sub = _mm256_sub_ps(s_F, sout);

            _mm256_storeu_ps(out, sub);


            //F += (float)(*IDX(cur, i, j-1, n) > EMAX) * FDELTA;
            //F += (float)(*IDX(cur, i, j+1, n) > EMAX) * FDELTA;
            //F += (float)(*IDX(cur, i-1, j, n) > EMAX) * FDELTA;
            //F += (float)(*IDX(cur, i+1, j, n) > EMAX) * FDELTA;
            //if (*IDX(cur, i, j-1, n) > EMAX) {
            //    F += FDELTA;
            //}
            ///* Idem per il vicino di destra */
            //if (*IDX(cur, i, j+1, n) > EMAX) {
            //    F += FDELTA;
            //}
            ///* Idem per il vicino in alto */
            //if (*IDX(cur, i-1, j, n) > EMAX) {
            //    F += FDELTA;
            //}
            ///* Idem per il vicino in basso */
            //if (*IDX(cur, i+1, j, n) > EMAX) {
            //    F += FDELTA;
            //}

            //F -= (float)(F > EMAX) * EMAX;
            //if (F > EMAX) {
            //    F -= EMAX;
            //}

            /* Si noti che il valore di F potrebbe essere ancora
               maggiore di EMAX; questo non e' un problema:
               l'eventuale eccesso verra' rilasciato al termine delle
               successive iterazioni fino a riportare il valore
               dell'energia sotto la foglia EMAX. */
            //*out = F;
        }

        for (; j < n + 1; j++) {
            float F = *IDX(cur, i, j, n);
            float *out = IDX(next, i, j, n);

            F += (float)(*IDX(cur, i, j-1, n) > EMAX) * FDELTA;
            F += (float)(*IDX(cur, i, j+1, n) > EMAX) * FDELTA;
            F += (float)(*IDX(cur, i-1, j, n) > EMAX) * FDELTA;
            F += (float)(*IDX(cur, i+1, j, n) > EMAX) * FDELTA;

            F -= (float)(F > EMAX) * EMAX;

            *out = F;
        }
    }
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
float average_energy(float *grid, int n)
{
    float sum = 0.0f;
    float p_sum = 0.0f;
    //int i, j;
    __m256 s_sum = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
#pragma omp parallel for simd reduction(+:sum) //default(none) private(j, s_sum) shared(grid, n) //schedule(runtime)
    for (int i = 1; i < n + 1; i++) {
        //for (j = 1; j < (n + 1) - 15; j+=16) {
        //    __m256 s1 = _mm256_loadu_ps(IDX(grid, i, j, n));
        //    __m256 s2 = _mm256_loadu_ps(IDX(grid, i, j+8, n));
        //    s_sum = _mm256_add_ps(s1, s_sum);
        //    s_sum = _mm256_add_ps(s2, s_sum);
        //}
        //float *s_ptr = (float *)&s_sum;
        ////if (i == 1) {
        ////    printf("%f %f %f %f\n", s_ptr[0], s_ptr[1], s_ptr[2], s_ptr[3]);
        ////}
        //p_sum += s_ptr[0] + s_ptr[1] + s_ptr[2] + s_ptr[3] + s_ptr[4] + s_ptr[5] + s_ptr[6] + s_ptr[7];
        for (int j = 1; j < n+1; j++) {
            sum += *IDX(grid, i, j, n);
        }

    }
    return (sum / (n*n));
}

int main(int argc, char* argv[])
{
    float *cur, *next;
    int s, n = 256, nsteps = 2048;
    float Emean;
    int c;

    srand(SEED); /* Inizializzazione del generatore pseudocasuale */

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [nsteps [n]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        char *pEnd;
        // 'atoi' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead [cert-err34-c] [W]
        nsteps = strtol(argv[1], &pEnd, 10);
    }

    if ( argc > 2 ) {
        char *pEnd;
        // 'atoi' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead [cert-err34-c] [W]
        n = strtol(argv[2], &pEnd, 10);
    }

    const size_t size = (n + 2) * (n + 2) * sizeof(float);

    /* Allochiamo i domini */
    cur = (float *) malloc(size); assert(cur);
    next = (float *) malloc(size); assert(next);
    //posix_memalign((void **)&cur, __BIGGEST_ALIGNMENT__, size);
    //posix_memalign((void **)&next, __BIGGEST_ALIGNMENT__, size);

    /* L'energia iniziale di ciascuna cella e' scelta
       con probabilita' uniforme nell'intervallo [0, EMAX*0.1] */
    setup(cur, n, 0, EMAX*0.1);

    const double tstart = hpc_gettime();
    for (s = 0; s < nsteps; s++) {
        /* L'ordine delle istruzioni che seguono e' importante */
        increment_energy(cur, n, EDELTA);
        c = count_cells(cur, n);
        propagate_energy(cur, next, n);
        Emean = average_energy(next, n);

        printf("%d %f\n", c, Emean);

        float *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;

    double Mupdates = (((double) n) * n / 1.0e6) * nsteps; /* milioni di celle aggiornate per ogni secondo di wall clock time */
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);

    /* Libera la memoria */
    free(cur);
    free(next);

    return EXIT_SUCCESS;
}


// vim: set nofoldenable ts=4 sw=4 sts=4 et :

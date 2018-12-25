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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>     /* rand() */
#include <x86intrin.h>

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
/* pre-defined seed for pseudo random initialization */
#define SEED 19


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
    for(int j = 0; j < ghost_n; j++) {
        // righe
        *IDX(grid, 0, j, ghost_n) = 0;
        *IDX(grid, n + 1, j, ghost_n) = 0;
        // Colonne
        *IDX(grid, j, 0, ghost_n) = 0;
        *IDX(grid, j, n + 1, ghost_n) = 0;
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
#pragma omp parallel for reduction(+:c) default(none) shared(grid, n) //schedule(runtime)
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
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
    //const __m128 fdelta = _mm_set1_ps(FDELTA);
    //const __m128 emax = _mm_set1_ps(EMAX);
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j ++) {

            float F = *IDX(cur, i, j, n);
            //float tmp_F = *IDX(cur, i, j, n);
            float *out = IDX(next, i, j, n);

            //__m128 m_F, mask, cond, res;

            //// sinistra
            //m_F = _mm_load_ps(IDX(cur, i, j-1, n));
            //mask = _mm_cmpgt_ps(m_F, fdelta); //compare m_F > FDELTA
            //cond = _mm_or_ps(_mm_and_ps(mask, fdelta), _mm_andnot_ps(mask, _mm_set1_ps(0.0f)));
            //res = _mm_add_ps(cond, _mm_set1_ps(*IDX(cur, i, j, n))); //compute the result
            //_mm_store_ps(&tmp_F, res);


            //// destra
            //m_F = _mm_load_ps(IDX(cur, i, j+1, n));
            //mask = _mm_cmpgt_ps(m_F, fdelta); //compare m_F > FDELTA
            //cond = _mm_or_ps(_mm_and_ps(mask, fdelta), _mm_andnot_ps(mask, _mm_set1_ps(0.0f)));
            //res = _mm_add_ps(cond, _mm_set1_ps(*IDX(cur, i, j, n))); //compute the result
            //_mm_store_ps(&tmp_F, res);


            //// alto
            //m_F = _mm_load_ps(IDX(cur, i-1, j, n));
            //mask = _mm_cmpgt_ps(m_F, fdelta); //compare m_F > FDELTA
            //cond = _mm_or_ps(_mm_and_ps(mask, fdelta), _mm_andnot_ps(mask, _mm_set1_ps(0.0f)));
            //res = _mm_add_ps(cond, _mm_set1_ps(*IDX(cur, i, j, n))); //compute the result
            //_mm_store_ps(&tmp_F, res);

            //// basso
            //m_F = _mm_load_ps(IDX(cur, i+1, j, n));
            //mask = _mm_cmpgt_ps(m_F, fdelta); //compare m_F > FDELTA
            //cond = _mm_or_ps(_mm_and_ps(mask, fdelta), _mm_andnot_ps(mask, _mm_set1_ps(0.0f)));
            //res = _mm_add_ps(cond, _mm_set1_ps(*IDX(cur, i, j, n))); //compute the result
            //_mm_store_ps(&tmp_F, res);

            /* Se l'energia del vicino di sinistra (se esiste) e'
               maggiore di EMAX, allora la cella (i,j) ricevera'
               energia addizionale FDELTA = EMAX/4 */
            if (*IDX(cur, i, j-1, n) > EMAX) {
                F += FDELTA;
            }
            /* Idem per il vicino di destra */
            if (*IDX(cur, i, j+1, n) > EMAX) {
                F += FDELTA;
            }
            /* Idem per il vicino in alto */
            if (*IDX(cur, i-1, j, n) > EMAX) {
                F += FDELTA;
            }
            /* Idem per il vicino in basso */
            if (*IDX(cur, i+1, j, n) > EMAX) {
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
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
float average_energy(float *grid, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
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
    //cur = (float *) malloc(size); assert(cur);
    //next = (float *) malloc(size); assert(next);
    posix_memalign((void **)&cur, __BIGGEST_ALIGNMENT__, size);
    posix_memalign((void **)&next, __BIGGEST_ALIGNMENT__, size);

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

# earthquake-simulator

Simple earthquake simulator, based on [Burridge-Knopoff's](https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/57/3/341/116471/model-and-theoretical-seismicity) cellular automata.

## Considerazioni finali

### Cuda
In CUDA si osservano comportamenti nei tempi di esecuzione contro le previsioni.
Per esempio, nell'uso della shared memory, si ottiene un leggero aumento dei
tempi di esecuzione. L'uso della shared memory viene fatto come proposto negli
esempi a lezione (Anneal), ma nonostante questo non si ottengono miglioramenti
delle prestazioni.
Inoltre l'uso dell' `atomicAdd` risulta essere maggiormente efficiente di una
implementazione con `reduction` + shared memory (vedi `count_cells`).

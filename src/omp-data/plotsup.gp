set terminal png enhanced notransparent
set output "omp-speedup.png"
set lmargin 8
set key title "Lato dominio"
set grid
set title "Speedup versione OpenMP" font ",12"
set xlabel "Numero threads"
set ylabel "Speedup"
plot [1:] \
    for [COL=2:4] 'sup.dat' using 1:COL title columnheader linewidth 2 with lines

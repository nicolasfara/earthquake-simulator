set terminal png enhanced notransparent
set output "omp-speedup.png"
set lmargin 8
set grid
set title "Speedup versione OpenMP" font ",12"
set xlabel "Numero threads"
set ylabel "Speedup"
plot [1:][0:] "sup.dat" t 'S(p)' w l lt 3 lw 2
unset multiplot

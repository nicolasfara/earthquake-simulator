set terminal png enhanced notransparent
set output "omp-sse.png"
set lmargin 8
set grid
set logscale x 2
set nokey
set title "Strong Scaling Efficiency OpenMP" font ",12"
set xlabel "Numero threads"
set ylabel "Efficienza"
plot [1:][0:] "sse.dat" t 'S(p)' w l lt 3 lw 2
unset multiplot

set terminal png enhanced notransparent
set output "omp-wse.png"
set lmargin 8
set grid
set nokey
set title "Weak Scaling Efficiency OpenMP" font ",12"
set xlabel "Numero threads"
set ylabel "Efficienza"
plot [1:][0:] "wse.dat" w l lt 3 lw 2
unset multiplot

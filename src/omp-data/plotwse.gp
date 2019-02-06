set terminal png enhanced notransparent
set output "omp-wse.png"
#set lmargin 8
set grid
set nokey
set title "Weak Scaling Efficiency OpenMP" font ",12"
set xlabel "Numero threads"
set ylabel "Efficienza"
plot \
    for [COL=2:2] 'wse.dat' using 1:COL linewidth 2 with lines

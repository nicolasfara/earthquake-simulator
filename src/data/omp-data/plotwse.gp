set terminal png enhanced notransparent
set output "omp-wse.png"
#set lmargin 8
set grid
set nokey
set title "Weak Scaling Efficiency OpenMP" font ",18"
set xlabel "Thread(s)" font ", 14"
set ylabel "Efficienza" font ", 14"
plot \
    for [COL=2:2] 'wse.dat' using 1:COL linewidth 2 with lines

set terminal png size 900,600 enhanced notransparent
set output "omp-sse.png"
set lmargin 8
set grid
set logscale x 2
set key on title "Lato dominio"
set title "Strong Scaling Efficiency OpenMP" font ",12"
set xlabel "Thread(s)"
set ylabel "Efficienza"
plot \
    for [COL=2:4] 'sse.dat' using 1:COL title columnheader linewidth 2 with lines

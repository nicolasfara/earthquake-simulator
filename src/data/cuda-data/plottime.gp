reset
set term png size 800,500 truecolor
set output 'cuda-time.png'

set title "Tempi esecuzione algoritmo Burridge-Knopoff - CUDA" font ", 18"
set grid

set xtics
set ytics

set ylabel "Secondi" font ", 14"
set xlabel "Lato dominio" font ", 14"

set style fill pattern border -1
set style data histograms
set boxwidth 1.0
set style histogram clustered gap 1

set key spacing 1
#using directly 'set key spacing 2 font fontSpec(18)' doesn't seem to work...

fn(v) = sprintf("%.1f", v)

plot \
    for [COL=2:3] 'time.dat' using COL:xticlabels(1) title columnheader fs pattern 1, \
    'time.dat' u ($0-1-1./6):2:(fn($2)) w labels offset char 0,0.5 t '', \
    'time.dat' u ($0-1+1./6):3:(fn($3)) w labels offset char 0,0.5 t ''

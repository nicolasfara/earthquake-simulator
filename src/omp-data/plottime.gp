# http://gnuplot.sourceforge.net/demo_5.0/histograms.html

set title 'Tempi di esecuzione'
set xlabel 'Thread(s)'
set ylabel 'Secondi'

set grid
set key top right vertical inside noreverse enhanced autotitle box dashtype solid title "Lato dominio"
set tics out nomirror
set border 3 front linetype black linewidth 1.0 dashtype solid

#set xrange [1:16]
set xtics 1
#set mxtics 1

#set yrange [0:250]
# set ytics 5

set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2

set style histogram clustered gap 1 title offset character 0, 0, 0
set style data histograms

set boxwidth 1.0 absolute
set style fill solid 5.0 border -1

set terminal png enhanced
set output 'omp-time.png'

plot 'time.dat' using 2:xtic(1) with histogram title '256', \
	'' using 3 title '512', \
  '' using 4 title '1024'

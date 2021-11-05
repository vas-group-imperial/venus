#!/bin/bash

radius=0.05
st_ratio=0.4
depth_power=4
splitters=1
workers=2
offline_dep=True
online_dep=True
ideal_cuts=True
timeout=3600
logfile="mnist.log"

for i in {1..100}
do
	python3 ../ --property lrob --lrob_input ../resources/mnist/evaluation_images/im${i}.pkl --lrob_radius $radius --net ../resources/mnist/mnist-net.h5  --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
done

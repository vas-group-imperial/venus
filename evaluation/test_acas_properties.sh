#!/bin/bash

st_ratio=0.7
depth_power=20.0
splitters=1
workers=4
offline_dep=False
online_dep=False
ideal_cuts=False
timeout=3600
logfile="acas.log"

# property 1
for x in {1..5}
do
    for y in {1..9}
    do
        python3 ../ --property acas --acas_prop 0 --net ../resources/acas/models/acas_${x}_${y}.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
    done
done

# property 2
for x in {2..5}
do
    for y in {1..9}
    do
        python3 ../ --property acas --acas_prop 1 --net ../resources/acas/models/acas_${x}_${y}.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
    done
done

# property 3
for x in {1..5}
do
    for y in {1..9}
    do
        if [ $x -gt 1 ] || [ $y -lt 7 ]
        then
            python3 ../ --property acas --acas_prop 2 --net ../resources/acas/models/acas_${x}_${y}.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
        fi
    done
done

# property 4
for x in {1..5}
do
    for y in {1..9}
    do
        if [ $x -gt 1 ] || [ $y -lt 7 ]
        then
            python3 ../ --property acas --acas_prop 3 --net ../resources/acas/models/acas_${x}_${y}.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
        fi
    done
done

# property 5
python3 ../ --property acas --acas_prop 4 --net ../resources/acas/models/acas_1_1.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
# property 6a
python3 ../ --property acas --acas_prop 5 --net ../resources/acas/models/acas_1_1.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
# property 6b
python3 ../ --property acas --acas_prop 6 --net ../resources/acas/models/acas_1_1.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
# property 7
python3 ../ --property acas --acas_prop 7 --net ../resources/acas/models/acas_1_9.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
# property 8
python3 ../ --property acas --acas_prop 8 --net ../resources/acas/models/acas_2_9.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
# property 9
python3 ../ --property acas --acas_prop 9 --net ../resources/acas/models/acas_3_3.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
# property 10
python3 ../ --property acas --acas_prop 10 --net ../resources/acas/models/acas_4_5.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

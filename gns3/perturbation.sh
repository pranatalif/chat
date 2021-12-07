#!/bin/sh

echo Input perturbation [1-Loss/2-Delay/3-Duplicate/4-Stress]?
read perturbation

start=$(date +'[%m-%d-%Y %H:%M:%S]')

if [ $perturbation == 1 ]
then
    echo "$start Packet loss is injected"
    pumba netem -d 10m loss -p 50 Nginx1
elif [ $perturbation == 2 ]
then
    echo "$start Packet delay is injected"
    pumba netem -d 10m delay -t 500 Nginx1
elif [ $perturbation == 3 ]
then
    echo "$start Packet duplication is injected"
    pumba netem -d 10m duplicate -p 90 Nginx1
elif [ $perturbation == 4 ]
then
    echo "$start CPU stress is injected"
    pumba stress --duration=10m --stressors="---cpu 4 --io 2" Nginx1
fi

end=$(date +'[%m-%d-%Y %H:%M:%S]')
echo "$end Perturbation ends"
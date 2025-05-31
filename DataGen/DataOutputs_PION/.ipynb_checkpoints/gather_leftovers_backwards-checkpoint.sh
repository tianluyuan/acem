#!/usr/bin/bash

for i in $(seq 1000 -1 100); do
    if [ ! -d /home/icrawsha/DataGen/DataOutputs_PION/job"$i"txts ]; then
        continue
    fi
    for file in /home/icrawsha/DataGen/DataOutputs_PION/job"$i"txts/PION*.txt; do
        if [ -f "$file" ]; then
            python /home/icrawsha/DataGen/add_to_csv.py -txt $file -csv $1
        	rm $file 
        fi
    done
    count=$(ls /home/icrawsha/DataGen/DataOutputs_PION/job"$i"txts/PION*txt 2> /dev/null | wc -l)
    if [ $count = 0 ]; then
        rm -r /home/icrawsha/DataGen/DataOutputs_PION/job"$i"txts
    fi
done
#!/usr/bin/bash
echo -n > $1

for i in $(seq 1 10000);
do
    if [ -f 'job'$i'.csv' ]; then
        cat job$i.csv >> $1
        rm job$i.csv
    fi
done
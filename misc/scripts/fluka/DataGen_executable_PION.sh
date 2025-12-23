#!/usr/bin/bash

source /home/icrawsha/.venv/bin/activate
export FLUPRO=/data/user/eyildizci/fluka
export FLUFOR=gfortran
num_runs=1
seed_num=$(($1 * $num_runs))
mkdir -p /home/icrawsha/DataGen/DataOutputs_PION/job$1inps
mkdir -p /home/icrawsha/DataGen/DataOutputs_PION/job$1txts

python /home/icrawsha/DataGen/gen_inp.py -s $num_runs -ss $seed_num -i /home/icrawsha/DataGen/inp_genPION.csv -o /home/icrawsha/DataGen/DataOutputs_PION/job$1inps -l True
for file in /home/icrawsha/DataGen/DataOutputs_PION/job$1inps/*.inp; do
    /home/icrawsha/DataGen/inp2txt.sh $file /home/icrawsha/DataGen/DataOutputs_PION/job$1txts 5
done
rm /home/icrawsha/DataGen/DataOutputs_PION/job$1txts/input.txt
rm -r /home/icrawsha/DataGen/DataOutputs_PION/job$1inps
for file in /home/icrawsha/DataGen/DataOutputs_PION/job$1txts/*.txt; do
    python /home/icrawsha/DataGen/add_to_csv.py -txt $file -csv /home/icrawsha/DataGen/DataOutputs_PION/job$1.csv
	rm $file
done

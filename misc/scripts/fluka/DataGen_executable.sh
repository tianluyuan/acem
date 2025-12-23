#!/usr/bin/bash

cd $(dirname $0)
echo $PWD
hostname
printenv
uname -a
lscpu

vtmp=${_CONDOR_SCRATCH_DIR:-`mktemp -d`}/v
echo $vtmp
python3 -m venv --copies $vtmp
source $vtmp/bin/activate || export PYTHONUSERBASE=$vtmp
pip3 install -U numpy scipy pandas -v

num_runs=1
seed_num=$(($1 * $num_runs))
dataoutputdir=DataOutputs_$(head -n1 ${2:-inp_gen.csv})
dataoutputdir=`realpath ${dataoutputdir}`
echo "saving to " ${dataoutputdir}
mkdir -p ${dataoutputdir}/job$1inps
mkdir -p ${dataoutputdir}/job$1txts

python3 gen_inp.py -s $num_runs -ss $seed_num -i ${2:-inp_gen.csv} -o ${dataoutputdir}/job$1inps -l True
ls -ltrh ${dataoutputdir}/job$1inps
for file in ${dataoutputdir}/job$1inps/*.inp; do
    ./inp2txt.sh $file ${dataoutputdir}/job$1txts 1
done
rm ${dataoutputdir}/job$1txts/input.txt
rm -r ${dataoutputdir}/job$1inps
for file in ${dataoutputdir}/job$1txts/*.txt; do
    python3 add_to_csv.py -txt $file -csv ${dataoutputdir}/job$1_`basename ${file%_[0-9]*_[0-9]*\.txt}`.csv
	rm $file
done
rm -r ${dataoutputdir}/job$1txts
rm -rf ${vtmp}

tar -cvzf job$1_`basename ${dataoutputdir}`.tar.gz `basename ${dataoutputdir}` # easy way to transfer output back with htcondor

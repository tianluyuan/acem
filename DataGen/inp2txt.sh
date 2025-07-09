#!/usr/bin/bash

# $N=number of runs
# $F=file name without ".inp"
export FLUPRO=/cvmfs/icecube.opensciencegrid.org/users/tyuan/fluka
export FLUFOR=gfortran
# Check if both parameters are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <file path of inp> <output directory> <number of runs (must be <= 999)>"
    
    exit 1
fi

# Assign parameters to variables
file_path="$1"
file_name=$(basename -- "$file_path")
F="${file_name%.*}" # isolate inp file name w/o extension
out_dir="$2"
num_runs="$3"

og_dir=$(pwd) # save original directory
cp $file_path $out_dir #copy .inp file into directory
cd $out_dir # move to output directory to run fluka in
${FLUPRO}/flutil/rfluka -e ${FLUPRO}/flukadpm3cerw -M$num_runs $F
rm ran${F}*
for ((i = 1; i <= num_runs; i++)); do
    f_i=$(printf "%03d" $i) #formatted index (three digits)
	cat <<EOF > input.txt
${F}${f_i}_fort.26

${F}${f_i}.bnn

EOF
	${FLUPRO}/flutil/usbsuw < input.txt > /dev/null
	cat <<EOF > input.txt
${F}${f_i}.bnn
${F}${f_i}.txt

EOF
	${FLUPRO}/flutil/usbrea < input.txt > /dev/null
	rm ${F}${f_i}_fort.26
	rm ${F}${f_i}.bnn
	rm ${F}${f_i}.err
	rm ${F}${f_i}.log
	rm ${F}${f_i}.out
done

rm $file_name # remove copied .inp file
cd "$og_dir"

import pandas as pd
import numpy as np
from argparse import ArgumentParser

# Function for formatting energy string
def format_energy(num):
    # Convert number to string in exponential notation with 3 decimal places
    num_str = "{:.5e}".format(num)
    # Extracting the exponent and coefficient parts
    coefficient, exponent = num_str.split('e')
    # Removing leading '+0' sign from the exponent
    exponent2 = exponent.replace('+0', '')
    if exponent2==exponent:
        raise Exception("Invalid energy for this formating. Energy must be within the interval [1,1e10)")
    # Formatting the coefficient part to have exactly 7 characters (including the '.')
    coefficient = coefficient.ljust(7, '0')
    # Return the formatted string with 9 characters
    return f"{coefficient}E{exponent2}"

#establish arguments
parser = ArgumentParser()
parser.add_argument("-s", dest="s",
                    default=None, type=int, required=True,
                    help="number of random seeds")
parser.add_argument("-ss", dest="ss",
                    default=0, type=int, required=False,
                    help="random seed number to start at")
parser.add_argument("-i", dest="inputFile",
                    default="input.csv", type=str, required=False,
                    help="file with energy values and types")
parser.add_argument("-o", dest="outputDirectory",
                    default="inps", type=str, required=False,
                    help="destination directory of generated inps")
parser.add_argument("-l", dest="useLogScale",
                    default=False, type=bool, required=False,
                    help="raise 10 to the power of each value in csv before using them as energies")
args = parser.parse_args()
s = args.s
ss = args.ss
inputFile = args.inputFile
outputDirectory = args.outputDirectory
useLogScale = args.useLogScale

fr = pd.read_csv(inputFile)#read input parameters
seeds = np.arange(ss,s+ss)#create list of random seeds
#read template .inp file and split at fill-in markers
with open('inpBase.txt','r') as f:
    base = f.read().split("*        *")

for particle in fr.columns:#for each particle type
    for energy_listing in fr[particle]:#for each energy value
        if useLogScale:
            energy = 10**energy_listing
        else:
            energy = energy_listing
        energyStr = format_energy(energy)
        for seed in seeds:#for each seed number
			#make the file
            with open(outputDirectory + '/' + (particle + '_' + energyStr + '_' + str(seed) + '_.inp').replace('+',''), 'w') as f:
                contents = ''#clear content string
                contents += base[0] + ('-' + energyStr).rjust(10," ")#add energy to inp file
                contents += base[1] + particle.ljust(10," ")#add particle type to inp file
                contents += base[2] + str(seed).rjust(10," ")#add random seed to inp file
                contents += base[3]
                f.write(contents)#write string to the file


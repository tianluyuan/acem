import pandas as pd
import numpy as np
from argparse import ArgumentParser
from scipy import signal
import scipy as sc
import csv

#Some normalization parameters to match Lief's work
L_rad = 39.7
simulation_bin_thickness = 10
# constant for smoothing in peak selection
sn_window = 8

#establish arguments
# Default gathers energy and particle type information from the file name
parser = ArgumentParser()
parser.add_argument("-txt", dest="txt",
                    default=None, type=str, required=True,
                    help="file path of txt file to add")
parser.add_argument("-csv", dest="csv",
                    default=None, type=str, required=True,
                    help="csv file to append to")
parser.add_argument("-en", dest="energy",
                    default=None, type=str, required=False,
                    help="Energy of initial particle, default uses second word in file name")
parser.add_argument("-pt", dest="ptype",
                    default=None, type=str, required=False,
                    help="Type of initial particle, default uses first word in file name")
args = parser.parse_args()
file_name = args.txt.split('/')[-1]
# Extract the energy and particle type from file name
words = file_name.split('_')
particle_type = words[0] if args.ptype == None else args.ptype
energy = words[1] if args.energy == None else args.energy


# gather bin data from the txt
with open(args.txt,'r') as in_txt:
    bin_data = np.loadtxt(in_txt, dtype=float, skiprows=9, max_rows=50)
bin_data = bin_data.reshape(bin_data.size)
ltot = np.sum(bin_data)

# calculate the number of peaks and their locations
smoothed_data = pd.Series(bin_data/ltot).rolling(sn_window).mean().dropna()
peak_locs_raw,_ = signal.find_peaks(smoothed_data,prominence=0.004,width=3)
peak_locs_raw = (peak_locs_raw + sn_window/2 - 1) * simulation_bin_thickness/L_rad
npeaks = len(peak_locs_raw)


normalized = bin_data * L_rad / (simulation_bin_thickness * ltot)
xvals = np.arange(0,500) * simulation_bin_thickness / L_rad
def myGamma(x,a,b):
    return sc.stats.gamma.pdf(x, a, loc=0, scale=1/b)
try:
    popt,pcov = sc.optimize.curve_fit(myGamma,xvals,normalized,
                                    p0=(3.5,0.3))

except:
    row = list(bin_data) + [energy,ltot,np.nan,np.nan,np.nan,np.nan,np.nan,npeaks]
else:
    row = list(bin_data) + [energy,ltot,popt[0],popt[1],pcov[0,0],pcov[1,0],pcov[1,1],npeaks]

# format peak data so that there are always 5 rows
peak_locs = [np.nan,np.nan,np.nan,np.nan,np.nan]
for i in range(min(npeaks,5)):
    peak_locs[i] = peak_locs_raw[i]

row = row + peak_locs
with open(args.csv,'a') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(row)

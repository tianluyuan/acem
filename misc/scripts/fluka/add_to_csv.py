import pandas as pd
import numpy as np
import math
from argparse import ArgumentParser
from scipy import signal
import scipy as sc
import csv

#Some normalization parameters to match Lief's work
L_rad = 36.08 / 0.9216
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
particle_type = words[0] if args.ptype is None else args.ptype
energy = words[1] if args.energy is None else args.energy


# gather bin data from the txt
with open(args.txt,'r') as in_txt:
    header = np.loadtxt(in_txt, dtype=str, skiprows=2, max_rows=3)
    nbins = np.asarray(header[:, 7], dtype=int)
    bin_up = np.asarray(header[:, 5], dtype=float)
    bin_lo = np.asarray(header[:, 3], dtype=float)
    bin_wd = (bin_up - bin_lo) / nbins
    bin_data = np.loadtxt(in_txt, dtype=float, skiprows=4, max_rows=math.ceil(nbins[2]/10))
# multiply by area to get in terms of track length [cm] per dz [cm]
bin_data = bin_data.reshape(bin_data.size) * np.prod(bin_wd[:2])
ltot = np.sum(bin_data)

# calculate the number of peaks and their locations
smoothed_data = pd.Series(bin_data/ltot).rolling(sn_window).mean().dropna()
peak_locs_raw,_ = signal.find_peaks(smoothed_data,prominence=0.004,width=3)
peak_locs_raw = (peak_locs_raw + sn_window/2 - 1) * bin_wd[2]/L_rad
npeaks = len(peak_locs_raw)

normalized = bin_data * L_rad / (bin_wd[2] * ltot)
xvals = np.arange(0,nbins[2]) * bin_wd[2] / L_rad
def myGamma(x,a,b):
    return sc.stats.gamma.pdf(x, a, loc=0, scale=1/b)
try:
    popt,pcov = sc.optimize.curve_fit(myGamma,xvals,normalized,
                                      p0=(3.5,0.3))

except: # noqa: E722
    row = list(bin_data) + [energy,ltot*bin_wd[2],np.nan,np.nan,np.nan,np.nan,np.nan,npeaks,bin_wd[2],nbins[2]]
else:
    row = list(bin_data) + [energy,ltot*bin_wd[2],popt[0],popt[1],pcov[0,0],pcov[1,0],pcov[1,1],npeaks,bin_wd[2],nbins[2]]

# format peak data so that there are always 5 rows
peak_locs = [np.nan,np.nan,np.nan,np.nan,np.nan]
for i in range(min(npeaks,5)):
    peak_locs[i] = peak_locs_raw[i]

row = row + peak_locs
with open(args.csv,'a') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(row)

#!/usr/bin/env python
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from shosim import model, util, maths, pdg

plt.style.use("paper-sans")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

DLDX_LABEL = r"\text{d}\hat{\ell}/\text{d}x"
LTOT_LABEL = r"\hat{\ell}_\text{tot}"

def fig1():
    plt.clf()
    ene = 1.0e3
    npem = util.load_npy("fluka/DataOutputs_ELECTRON/ELECTRON_1.00000E3.csv", False)

    dfem = util.load_csv("fluka/DataOutputs_ELECTRON/ELECTRON_1.00000E3.csv", False)
    dfpi = util.load_csv("fluka/DataOutputs_PION+/PION+_1.00000E3.csv", False)

    assert(dfem["Zbins"].nunique()==1)
    assert(dfem["Zwidth"].nunique()==1)
    nbins = int(dfem.iloc[0]["Zbins"])
    xs = (np.arange(nbins) + 0.5) * dfem.iloc[0]["Zwidth"]
    for _ in range(4):
        plt.plot(xs, npem[_, :nbins], color=colors[_], label=rf"Run {_ + 1} (FLUKA)")

    rwth = model.RWParametrization1D(model.Parametrization1D.FLUKA_MEDIUM)
    plt.plot(xs, rwth.mean_1d(11, ene).dldx(xs), c="k",
             label=r"1 TeV $e^-$"+"\n(RW 2013)", linestyle="--")
    plt.ylabel(rf'${DLDX_LABEL}$')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig("fig/paper/fig1a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig1a.png", bbox_inches="tight")

    plt.clf()
    bins = np.linspace(3e5, dfem["ltot"].max(), 100).tolist()
    plt.hist(
        dfem["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=r"1 TeV $e^-$ (FLUKA)",
    )
    plt.hist(
        dfpi["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=r"1 TeV $\pi^+$ (FLUKA)",
    )

    xs = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(xs, rwth.ltot_dist(11, ene).pdf(xs), "k--", label=r"1 TeV $e^-$ (RW 2013)")
    plt.plot(xs, rwth.ltot_dist(211, ene).pdf(xs), "k:", label=r"1 TeV $\pi^+$ (Rädel 2012)")

    plt.legend(loc="upper left")
    plt.xlim(xmin=bins[0])
    plt.ylim(ymin=5e-8)
    plt.yscale("log")
    plt.ylabel("Density [1/cm]")
    plt.xlabel(rf"${LTOT_LABEL}$ [cm]")
    plt.savefig("fig/paper/fig1b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig1b.png", bbox_inches="tight")


def fig2():
    plt.clf()
    ene = 1.0e3
    npem = util.load_npy(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    nppi = util.load_npy(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    dfem = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    dfpi = util.load_csv(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    assert(dfem["Zbins"].nunique()==1)
    assert(dfem["Zwidth"].nunique()==1)
    nbins = int(dfem.iloc[0]["Zbins"])
    xs = (np.arange(nbins) + 0.5) * dfem.iloc[0]["Zwidth"]
    nruns = 100
    for _ in range(nruns):
        plt.plot(xs,
                 npem[_, :nbins] / npem[_, nbins+1],
                 color=colors[0],
                 label=rf"1 TeV $e^-$ ({nruns} runs)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
    for _ in range(nruns):
        plt.plot(xs,
                 nppi[_, :nbins] / nppi[_, nbins+1],
                 color=colors[1],
                 label=rf"1 TeV $\pi^+$ ({nruns} runs)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
    plt.ylabel(rf'${LTOT_LABEL}^{{-1}}{DLDX_LABEL}$ [1/cm]')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(0, 4e-3)
    plt.legend()
    plt.savefig("fig/paper/fig2a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig2a.png", bbox_inches="tight")
    
    plt.clf()
    enes = [1e1, 1e3, 1e5]
    lsts = [':', '--', '-']
    bins = np.linspace(150, 550, 100)
    for ene, lst in zip(enes, lsts):
        estr = util.format_energy(ene)

        dfem = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{estr}.csv", False)
        dfpi = util.load_csv(f"fluka/DataOutputs_PION+/PION+_{estr}.csv", False)
        plt.hist(
            dfem["ltot"]/ene,
            bins=bins,
            density=True,
            histtype="step",
            label=rf"{ene/1000:.3g} TeV $e^-$",
            color=colors[0],
            linestyle=lst
        )
        plt.hist(
            dfpi["ltot"]/ene,
            bins=bins,
            density=True,
            histtype="step",
            label=rf"{ene/1000:.3g} TeV $\pi^+$",
            color=colors[1],
            linestyle=lst
        )

    plt.legend(loc='upper left')
    plt.xlim(bins[0], bins[-1])
    # plt.ylim(ymin=5e-8)
    plt.yscale("log")
    plt.ylabel("Density [GeV/cm]")
    plt.xlabel(rf"${LTOT_LABEL}/E$ [cm/GeV]")
    plt.savefig("fig/paper/fig2b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig2b.png", bbox_inches="tight")
        

def fig3():
    plt.clf()
    particles = ["ELECTRON", "PHOTON", "PION+", "KAON+", "KAONSHRT", "KAONLONG", "PROTON", "NEUTRON", "SIGMA+", "SIGMA-", "LAMBDA", "XSIZERO", "XSI-", "OMEGA-"]
    plabels = [r"e^-", r"\gamma", r"\pi^+", r"K^+", r"K^0_S", r"K^0_L", "p", "n", r"\Sigma^+", r"\Sigma^-", r"\Lambda^0", r"\Xi^0", r"\Xi^-", r"\Omega^-"]
    pcolors = [colors[0], colors[0], colors[1], colors[1], colors[1], colors[1], colors[2], colors[2], colors[2], colors[2], colors[3], colors[3], colors[3], colors[3]]
    plinest = ["-", "--", "-", "--", ":", "-.", "-", "--", ":", "-.", "-", "--", ":", "-."]
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=plt.gcf().get_size_inches())
    for i, particle in enumerate(particles):
        dats = util.load_batch(f"fluka/DataOutputs_{particle}/*.csv", loader=util.load_npy, clean=False)
        npeaks1 = np.asarray([(dats[_][:,507]==1).sum() for _ in dats])
        npeaks2 = np.asarray([(dats[_][:,507]==2).sum() for _ in dats])
        npeaksx = np.asarray([((dats[_][:, 507] != 1) & (dats[_][:, 507] != 2)).sum() for _ in dats])
        
        plt.figure(1)
        plt.plot(dats.keys(), npeaks1/(npeaks1+npeaks2+npeaksx), c=pcolors[i], label=rf"${plabels[i]}$", ls=plinest[i], linewidth=1.5)

        lva = []
        lvb = []
        avb = []

        for key in dats:
            data_array = dats[key]
            ltot_col = data_array[:, 501]
            apri_col = maths.aprime(data_array[:, 502])
            bpri_col = maths.aprime(data_array[:, 503])
            mask = ~(np.isnan(apri_col) & np.isnan(bpri_col))

            stacked_data = np.stack((ltot_col[mask], apri_col[mask], bpri_col[mask]), axis=1)
            rho_matrix = stats.spearmanr(stacked_data).statistic

            lva.append(rho_matrix[0, 1]) 
            lvb.append(rho_matrix[0, 2])
            avb.append(rho_matrix[1, 2])

        ax[0].plot(dats.keys(), lva, c=pcolors[i], label=rf"${plabels[i]}$", ls=plinest[i], linewidth=1.5)
        ax[1].plot(dats.keys(), lvb, c=pcolors[i], label=rf"${plabels[i]}$", ls=plinest[i], linewidth=1.5)
        ax[2].plot(dats.keys(), avb, c=pcolors[i], label=rf"${plabels[i]}$", ls=plinest[i], linewidth=1.5)

    plt.figure(1)
    plt.xscale("log")
    plt.xlabel("$E$ [GeV]")
    plt.ylabel("Proportion of showers with 1 peak")
    plt.ylim(ymax=1)
    plt.legend(ncol=2)
    plt.savefig("fig/paper/fig3a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig3a.png", bbox_inches="tight")

    ax[0].set_ylabel(rf"$\varrho({LTOT_LABEL}, a')$")
    ax[1].set_ylabel(rf"$\varrho({LTOT_LABEL}, b')$")
    ax[2].set_ylabel(r"$\varrho(a', b')$")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("$E$ [GeV]")
    [ax[_].set_ylim(-1, 1) for _ in range(3)]
    fig.savefig("fig/paper/fig3b.pdf", bbox_inches="tight")
    fig.savefig("fig/paper/fig3b.png", bbox_inches="tight")


def fig4():
    plt.clf()
    ene = 1.0e3
    npem = util.load_npy(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    nppi = util.load_npy(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    dfem = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)

    assert(dfem["Zbins"].nunique()==1)
    assert(dfem["Zwidth"].nunique()==1)
    nbins = int(dfem.iloc[0]["Zbins"])
    zwdth = float(dfem.iloc[0]["Zwidth"])
    xs = np.arange(0, nbins * zwdth)
    nruns = 100

    a_ems = []
    b_ems = []
    for _ in range(nruns):
        a_em = npem[_, nbins+2]
        b_em = npem[_, nbins+3]
        plt.plot(xs,
                 stats.gamma(
                     a_em,
                     scale=model.Parametrization1D.FLUKA_MEDIUM.lrad/b_em).pdf(xs),
                 color=colors[0],
                 label=rf"1 TeV $e^-$ ({nruns} fits)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
        a_ems.append(a_em)
        b_ems.append(b_em)
    a_pis = []
    b_pis = []
    for _ in range(nruns):
        a_pi = nppi[_, nbins+2]
        b_pi = nppi[_, nbins+3]
        plt.plot(xs,
                 stats.gamma(
                     a_pi,
                     scale=model.Parametrization1D.FLUKA_MEDIUM.lrad/b_pi).pdf(xs),
                 color=colors[1],
                 label=rf"1 TeV $\pi^+$ ({nruns} fits)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
        a_pis.append(a_pi)
        b_pis.append(b_pi)
    plt.ylabel(rf'${LTOT_LABEL}^{{-1}}{DLDX_LABEL}$ [1/cm]')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(0, 4e-3)
    plt.legend()
    plt.savefig("fig/paper/fig4a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig4a.png", bbox_inches="tight")

    plt.clf()
    plt.plot(maths.aprime(np.asarray(a_ems)), maths.bprime(np.asarray(b_ems)), '.', color=colors[0], label=rf"1 TeV $e^-$ ({nruns} fits)", markersize=1.5)
    plt.plot(maths.aprime(np.asarray(a_pis)), maths.bprime(np.asarray(b_pis)), '.', color=colors[1], label=rf"1 TeV $\pi^+$ ({nruns} fits)", markersize=1.5)
    plt.xlabel(r"$a'$")
    plt.ylabel(r"$b'$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("fig/paper/fig4b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig4b.png", bbox_inches="tight")


def fig5():
    from spline import bins_a, bins_b, ga, gb

    _part = "PION+"
    enes = [1e1, 1e3, 1e6]
    labs = [r"10 GeV $\pi^+$", r"1 TeV $\pi^+$", r"1 PeV $\pi^+$"]
    bspl = model.Parametrization1D.THETAS[pdg.FLUKA2PDG[_part]]
    vmax = 50 #np.exp(np.median([bspl(*bspl.mode(np.log10(ene)), np.log10(ene)) for ene in enes]))*1.5

    _wide, _height = plt.gcf().get_size_inches()
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(_wide*3.1, _height*2))
    for i, (ene, lab) in enumerate(zip(enes, labs)):
        df = util.load_csv(f"fluka/DataOutputs_{_part}/{_part}_{util.format_energy(ene)}.csv", False)
        avals = ga(df.gammaA)
        bvals = gb(df.gammaB)
        ax[0][i].hist2d(avals,bvals,bins=(bins_a,bins_b),cmap="plasma", density=True,
                        vmin=0, vmax=vmax)
                     # norm=colors.LogNorm())
        ax[0][i].text(
            0.96,  # Slight offset from the right edge (1.0 is the exact edge)
            0.04,  # Slight offset from the bottom edge (0.0 is the exact bottom)
            f"{lab} (MC)",
            transform=ax[0][i].transAxes,  # Use Axes coordinates (0 to 1)
            fontsize=18,
            verticalalignment='bottom', # Text goes up from the baseline
            horizontalalignment='right', # Text flows left from the point
            color='white'
        )
    X, Y = np.meshgrid(bins_a, bins_b)
    for i, (ene, lab) in enumerate(zip(enes, labs)):
        Z= bspl(X, Y, np.log10(ene))
        im = ax[1][i].pcolormesh(X, Y, np.exp(Z), cmap='plasma', shading='auto', vmin=0, vmax=vmax)
        ax[1][i].text(
            0.96,  # Slight offset from the right edge (1.0 is the exact edge)
            0.04,  # Slight offset from the bottom edge (0.0 is the exact bottom)
            f"{lab} (model)",
            transform=ax[1][i].transAxes,  # Use Axes coordinates (0 to 1)
            fontsize=18,
            verticalalignment='bottom', # Text goes up from the baseline
            horizontalalignment='right', # Text flows left from the point
            color='white'
        )
    fig.supylabel(r"$b'$", x=0.065)
    fig.supxlabel(r"$a'$", y=0.02)
    cbar_ax_position = [0.92, 0.15, 0.01, 0.7] # 0.92 is far right, 0.01 is thin width
    cax = fig.add_axes(cbar_ax_position)
    cbar = fig.colorbar(im,
                        cax=cax, 
                        # orientation='horizontal', # <-- Set orientation to horizontal
                        label=r"$f(a', b' ; E)$"
                        )
    plt.savefig("fig/paper/fig5.png", bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    fig5()
    fig4()
    fig3()
    fig2()
    fig1()
    plt.close("all")

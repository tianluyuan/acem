import numpy as np
from matplotlib import pyplot as plt

from shosim import model, media, util

plt.style.use("paper-sans")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

DLDX_LABEL = r"\text{d}\hat{\ell}/\text{d}x"
LTOT_LABEL = r"\hat{\ell}_\text{tot}"

MEDIUM_FLUKA = media.Medium(0.9216, 1.33)


def fig2():
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
        _row = dfem.iloc[_]
        plt.plot(xs, npem[_, :nbins], color=colors[_], label=rf"Run {_ + 1} (FLUKA)")

    rwth = model.RWShower(MEDIUM_FLUKA)
    plt.plot(xs, rwth.dldx(11, ene)(xs), c="k",
             label=r"1 TeV $e^-$ (RW 2013)", linestyle="--")
    plt.ylabel(rf'${DLDX_LABEL}$')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig("fig/paper/fig2a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig2a.png", bbox_inches="tight")

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
    plt.plot(xs, rwth.ltot(11, ene).pdf(xs), "k--", label=r"1 TeV $e^-$ (RW 2013)")
    plt.plot(xs, rwth.ltot(211, ene).pdf(xs), "k:", label=r"1 TeV $\pi^+$ (Rädel 2012)")

    plt.legend(loc="upper left")
    plt.xlim(xmin=bins[0])
    plt.ylim(ymin=5e-8)
    plt.yscale("log")
    plt.ylabel("Density [1/cm]")
    plt.xlabel(rf"${LTOT_LABEL}$ [cm]")
    plt.savefig("fig/paper/fig2b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig2b.png", bbox_inches="tight")


def fig3():
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
        _row = dfem.iloc[_]
        plt.plot(xs,
                 npem[_, :nbins],
                 color=colors[0],
                 label=rf"1 TeV $e^-$ ({nruns} runs)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
    for _ in range(nruns):
        _row = dfpi.iloc[_]
        plt.plot(xs,
                 nppi[_, :nbins],
                 color=colors[1],
                 label=rf"1 TeV $\pi^+$ ({nruns} runs)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
    plt.ylabel(rf'${DLDX_LABEL}$')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig("fig/paper/fig3a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig3a.png", bbox_inches="tight")
    
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

    plt.legend()
    plt.xlim(bins[0], bins[-1])
    # plt.ylim(ymin=5e-8)
    # plt.yscale("log")
    plt.ylabel("Density [GeV/cm]")
    plt.xlabel(rf"${LTOT_LABEL}/E$ [cm/GeV]")
    plt.savefig("fig/paper/fig3b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig3b.png", bbox_inches="tight")
        

if __name__ == "__main__":
    fig2()
    fig3()

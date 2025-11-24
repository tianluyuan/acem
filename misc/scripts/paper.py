import numpy as np
from matplotlib import pyplot as plt

from shosim.media import IC3
from shosim import model, media
from shosim import util

plt.style.use("paper-sans")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

DLDX_LABEL = r"$\text{d}\hat{\ell}/\text{d}x$"
LTOT_LABEL = r"$\hat{\ell}_\text{tot}$"

MEDIUM_FLUKA = media.Medium(0.9216, 1.33)


def fig2():
    ene = 1.0e3
    npem = util.load_npy("fluka/DataOutputs_ELECTRON/ELECTRON_1.00000E3.csv", False)
    dfem = util.load_csv("fluka/DataOutputs_ELECTRON/ELECTRON_1.00000E3.csv", False)

    dfpi = util.load_csv("fluka/DataOutputs_PION+/PION+_1.00000E3.csv", False)

    rwth = model.RWShower(MEDIUM_FLUKA)
    assert(dfem["Zbins"].nunique()==1)
    assert(dfem["Zwidth"].nunique()==1)
    nbins = int(dfem.iloc[0]["Zbins"])
    xs = np.arange(nbins) * dfem.iloc[0]["Zwidth"]
    for _ in range(4):
        _row = dfem.iloc[_]
        plt.plot(xs, npem[_, :nbins], color=colors[_], label=rf"Run {_ + 1} (FLUKA)")
    plt.plot(xs, rwth.dldx(11, ene)(xs), c="k",
             label=r"1 TeV $e^-$ (RW 2013)", linestyle="--")
    plt.ylabel(DLDX_LABEL)
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig("fig/paper/fig2a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig2a.png", bbox_inches="tight")

    plt.clf()
    bins = np.linspace(3e5, dfem["ltot"].max(), 100)
    xs = np.linspace(bins[0], bins[-1], 1000)
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
    plt.plot(xs, rwth.ltot(11, ene).pdf(xs), "k--", label=r"1 TeV $e^-$ (RW 2013)")
    plt.plot(xs, rwth.ltot(211, ene).pdf(xs), "k:", label=r"1 TeV $\pi^+$ (Rädel 2012)")
    plt.legend(loc="upper left")
    plt.xlim(xmin=bins[0])
    plt.ylim(ymin=5e-8)
    plt.yscale("log")
    plt.ylabel("PDF")
    plt.xlabel(f"{LTOT_LABEL} [cm]")
    plt.savefig("fig/paper/fig2b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig2b.png", bbox_inches="tight")


if __name__ == "__main__":
    fig2()

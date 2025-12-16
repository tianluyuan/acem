#!/usr/bin/env python
import numpy as np
from scipy import stats, optimize
from matplotlib import pyplot as plt

from shosim import model, util, maths, pdg

plt.style.use("paper-sans")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

DLDX_LABEL = r"\text{d}\hat{\ell}/\text{d}x"
LTOT_LABEL = r"\hat{\ell}_\text{tot}"
SPGP_LABEL = r"s_p g_p"

PARTICLES = ["ELECTRON", "PHOTON", "PION+", "KAON+", "KAONSHRT", "KAONLONG", "PROTON", "NEUTRON", "SIGMA+", "SIGMA-", "LAMBDA", "XSIZERO", "XSI-", "OMEGA-"]
PLABELS = [r"e^-", r"\gamma", r"\pi^+", r"K^+", r"K^0_S", r"K^0_L", "p", "n", r"\Sigma^+", r"\Sigma^-", r"\Lambda^0", r"\Xi^0", r"\Xi^-", r"\Omega^-"]
PCOLORS = [colors[0], colors[0], colors[1], colors[1], colors[1], colors[1], colors[2], colors[2], colors[2], colors[2], colors[3], colors[3], colors[3], colors[3]]
PLINEST = ["-", "--", "-", "--", ":", "-.", "-", "--", ":", "-.", "-", "--", ":", "-."]


def fig1():
    plt.clf()
    ene = 1.0e3
    np1 = util.load_npy(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)

    df1 = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    df2 = util.load_csv(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    assert(df1["Zbins"].nunique()==1)
    assert(df1["Zwidth"].nunique()==1)
    nbins = int(df1.iloc[0]["Zbins"])
    xs = (np.arange(nbins) + 0.5) * df1.iloc[0]["Zwidth"]
    for _ in range(4):
        plt.plot(xs, np1[_, :nbins], color=colors[_], label=rf"Run {_ + 1} (FLUKA)")

    rwth = model.RWParametrization1D(model.Parametrization1D.FLUKA_MEDIUM)
    plt.plot(xs, rwth.mean_1d(11, ene).dldx(xs), c="k",
             label=r"1 TeV $e^-$"+"\n(RW 2013) 2013)", linestyle="--")
    plt.ylabel(rf'${DLDX_LABEL}$')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 1500)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig("fig/paper/fig1a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig1a.png", bbox_inches="tight")

    plt.clf()
    bins = np.linspace(3e5, max(df1["ltot"].max(), df2["ltot"].max()), 100).tolist()
    plt.hist(
        df1["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=r"1 TeV $e^-$ (FLUKA)",
    )
    plt.hist(
        df2["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=r"1 TeV $\pi^+$ (FLUKA)",
    )

    ul = bins[-1]*1.02
    xs = np.linspace(bins[0], ul, 1000)
    plt.plot(xs, rwth.ltot_dist(11, ene).pdf(xs), "k--", label=r"1 TeV $e^-$ (RW 2013) 2013)")
    plt.plot(xs, rwth.ltot_dist(211, ene).pdf(xs), "k:", label=r"1 TeV $\pi^+$ (Rädel 2012)")

    plt.legend(loc="upper left")
    plt.xlim(bins[0], ul)
    plt.ylim(ymin=5e-8)
    plt.yscale("log")
    plt.ylabel("Density [1/cm]")
    plt.xlabel(rf"${LTOT_LABEL}$ [cm]")
    plt.savefig("fig/paper/fig1b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig1b.png", bbox_inches="tight")
    plt.close("all")


def fig2():
    plt.clf()
    ene = 1.0e3
    np1 = util.load_npy(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    np2 = util.load_npy(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    df1 = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    df2 = util.load_csv(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    assert(df1["Zbins"].nunique()==1)
    assert(df1["Zwidth"].nunique()==1)
    nbins = int(df1.iloc[0]["Zbins"])
    xs = (np.arange(nbins) + 0.5) * df1.iloc[0]["Zwidth"]
    nruns = 100
    for _ in range(nruns):
        plt.plot(xs,
                 np1[_, :nbins] / np1[_, nbins+1],
                 color=colors[0],
                 label=rf"1 TeV $e^-$ ({nruns} runs)" if _==0 else None,
                 linewidth=0.5,
                 alpha=0.5)
    for _ in range(nruns):
        plt.plot(xs,
                 np2[_, :nbins] / np2[_, nbins+1],
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

        df1 = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{estr}.csv", False)
        df2 = util.load_csv(f"fluka/DataOutputs_PION+/PION+_{estr}.csv", False)
        plt.hist(
            df1["ltot"]/ene,
            bins=bins,
            density=True,
            histtype="step",
            label=rf"{ene/1000:.3g} TeV $e^-$",
            color=colors[0],
            linestyle=lst
        )
        plt.hist(
            df2["ltot"]/ene,
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
    plt.close("all")
        

def fig3():
    plt.clf()
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=plt.gcf().get_size_inches())
    for i, particle in enumerate(PARTICLES):
        dats = util.load_batch(f"fluka/DataOutputs_{particle}/*.csv", loader=util.load_npy, clean=False)
        npeaks1 = np.asarray([(dats[_][:,507]==1).sum() for _ in dats])
        npeaks2 = np.asarray([(dats[_][:,507]==2).sum() for _ in dats])
        npeaksx = np.asarray([((dats[_][:, 507] != 1) & (dats[_][:, 507] != 2)).sum() for _ in dats])
        
        plt.figure(1)
        plt.plot(dats.keys(), npeaks1/(npeaks1+npeaks2+npeaksx), c=PCOLORS[i], label=rf"${PLABELS[i]}$", ls=PLINEST[i], linewidth=1.5)

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

        ax[0].plot(dats.keys(), lva, c=PCOLORS[i], label=rf"${PLABELS[i]}$", ls=PLINEST[i], linewidth=1.5)
        ax[1].plot(dats.keys(), lvb, c=PCOLORS[i], label=rf"${PLABELS[i]}$", ls=PLINEST[i], linewidth=1.5)
        ax[2].plot(dats.keys(), avb, c=PCOLORS[i], label=rf"${PLABELS[i]}$", ls=PLINEST[i], linewidth=1.5)

    plt.figure(1)
    plt.xscale("log")
    plt.xlabel("$E$ [GeV]")
    plt.ylabel("Proportion of showers with 1 peak")
    plt.ylim(ymax=1)
    plt.xlim(1., 1e6)
    plt.legend(ncol=2)
    plt.savefig("fig/paper/fig3a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig3a.png", bbox_inches="tight")

    ax[0].set_ylabel(rf"$\varrho({LTOT_LABEL}, a')$")
    ax[1].set_ylabel(rf"$\varrho({LTOT_LABEL}, b')$")
    ax[2].set_ylabel(r"$\varrho(a', b')$")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("$E$ [GeV]")
    ax[2].set_xlim(1., 1e6)
    [ax[_].set_ylim(-1, 1) for _ in range(3)]
    fig.savefig("fig/paper/fig3b.pdf", bbox_inches="tight")
    fig.savefig("fig/paper/fig3b.png", bbox_inches="tight")
    plt.close("all")


def fig4():
    plt.clf()
    ene = 1.0e3
    np1 = util.load_npy(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    np2 = util.load_npy(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)

    df1 = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)

    assert(df1["Zbins"].nunique()==1)
    assert(df1["Zwidth"].nunique()==1)
    nbins = int(df1.iloc[0]["Zbins"])
    zwdth = float(df1.iloc[0]["Zwidth"])
    xs = np.arange(0, nbins * zwdth)
    nruns = 100

    a_ems = []
    b_ems = []
    for _ in range(nruns):
        a_em = np1[_, nbins+2]
        b_em = np1[_, nbins+3]
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
        a_pi = np2[_, nbins+2]
        b_pi = np2[_, nbins+3]
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
    plt.close("all")


def fig5():
    from spline import bins_a, bins_b, ga, gb

    _part = "PION+"
    enes = [1e1, 1e3, 1e5]
    labs = [r"10 GeV $\pi^+$", r"1 TeV $\pi^+$", r"100 TeV $\pi^+$"]
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
        ax[0][i].text(
            0.96,  # Slight offset from the right edge (1.0 is the exact edge)
            0.04,  # Slight offset from the bottom edge (0.0 is the exact bottom)
            f"{lab} (MC)",
            transform=ax[0][i].transAxes,
            fontsize=18,
            verticalalignment='bottom',
            horizontalalignment='right',
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
            transform=ax[1][i].transAxes,
            fontsize=18,
            verticalalignment='bottom',
            horizontalalignment='right',
            color='white'
        )
    fig.supylabel(r"$b'$", x=0.065)
    fig.supxlabel(r"$a'$", y=0.02)
    # 0.92 is far right, 0.01 is thin width
    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    _ = fig.colorbar(im,
                     cax=cax, 
                     label=r"$f(a', b' ; E)$"
                     )
    plt.savefig("fig/paper/fig5.png", bbox_inches="tight")
    plt.close("all")


def fig6():
    plt.clf()
    # copied over from ltot.py
    for particle, _c in zip(["ELECTRON", "PION+"], colors[:2]):
        if particle in ['ELECTRON', 'PHOTON']:
            form = stats.norminvgauss
            # 2x shape, loc, scale
            p_fn = [maths.sxt, maths.sxt, maths.sxt, maths.sxt]
            sgns = [1, -1, 1, 1]
            clean = False
            markers = ['<', '>', 'o', 's']
            labels = [r"\alpha", r"-\beta", r"\xi", r"\omega" ]
            # lsts = [':', '-.', '-', '--']
        else:
            form = stats.skewnorm
            # lin for loc (mean), maths.sxt for scale (omega)
            p_fn = [maths.sxt, maths.sxt, maths.sxt]
            sgns = [1, 1, 1]
            clean = True  # mask tricky decays
            markers = ['<', 'o', 's']
            labels = [r"\alpha", r"\xi", r"\omega"]
            # lsts = [':', '-', '--']
        Dat = util.load_batch(f'fluka/DataOutputs_{particle}/*.csv', clean=clean)
        ens = list(Dat.keys())
        log_ens = np.log10(ens)
        n_E = len(log_ens)
        results = []

        for i in range(n_E):
            df = Dat[ens[i]]
            ltots = df['ltot']
            _res = form.fit(ltots, method='MLE')
            results.append(_res)
        results = np.asarray(results) * sgns
        _sel = results[:, 0] > 0
        par_fits = [optimize.curve_fit(_f, log_ens[_sel], np.log(_y[_sel]))[0]
                    for _f, _y in zip(p_fn, results.T)]
        for i, (_f, _y, _p, _m, _l) in enumerate(zip(p_fn, results.T, par_fits, markers, labels)):
            plt.plot(10**log_ens[_sel], _y[_sel], _m, color=_c, markersize=2, label=rf'${_l}$')
            plt.plot(10**log_ens[~_sel], _y[~_sel], 'x', color=_c, markersize=2)
            plt.plot(10**log_ens, np.exp(_f(log_ens, *_p)), color=_c, linewidth=1,
                     label=None)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(ncol=2)
    plt.xlabel(r'$E$ [GeV]')
    plt.ylabel("Parameter values")
    plt.xlim(1., 1.e6)
    plt.savefig("fig/paper/fig6a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig6a.png", bbox_inches="tight")
    # end copy from ltot.py

    ene = 1.0e3
    df1 = util.load_csv(f"fluka/DataOutputs_ELECTRON/ELECTRON_{util.format_energy(ene)}.csv", False)
    df2 = util.load_csv(f"fluka/DataOutputs_PION+/PION+_{util.format_energy(ene)}.csv", False)
    bins = np.linspace(3e5, max(df1["ltot"].max(), df2["ltot"].max()), 100).tolist()
    plt.clf()
    plt.hist(
        df1["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=r"1 TeV $e^-$ (FLUKA)",
    )
    plt.hist(
        df2["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=r"1 TeV $\pi^+$ (FLUKA)",
    )

    ul = 1.02*bins[-1]
    xs = np.linspace(bins[0], ul, 1000)
    par = model.Parametrization1D(model.Parametrization1D.FLUKA_MEDIUM)
    plt.plot(xs, par.ltot_dist(11, ene).pdf(xs), "--", color=colors[0], label=r"1 TeV $e^-$ (model)")
    plt.plot(xs, par.ltot_dist(211, ene).pdf(xs), ":", color=colors[1], label=r"1 TeV $\pi^+$ (model)")

    plt.legend(loc="upper left")
    plt.xlim(bins[0], ul)
    plt.ylim(ymin=5e-8)
    plt.yscale("log")
    plt.ylabel("Density [1/cm]")
    plt.xlabel(rf"${LTOT_LABEL}$ [cm]")
    plt.savefig("fig/paper/fig6b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig6b.png", bbox_inches="tight")
    plt.close("all")


def fig7():
    ene = 1e3

    fkp = "KAON+"
    fks = "KAONSHRT"
    tex1 = pdg.PDG2LATEX[pdg.FLUKA2PDG[fkp]]
    tex2 = pdg.PDG2LATEX[pdg.FLUKA2PDG[fks]]
    np1 = util.load_npy(f"fluka/DataOutputs_{fkp}/{fkp}_{util.format_energy(ene)}.csv", False)
    np2 = util.load_npy(f"fluka/DataOutputs_{fks}/{fks}_{util.format_energy(ene)}.csv", False)

    # sorted by numpeaks
    np1 = np1[np.argsort(np1[:,507])]
    np2 = np2[np.argsort(np2[:,507])]
    nbins = int(np1[0,509])

    xs = (np.arange(nbins) + 0.5) * np1[0,508]
    for i, (npx, tex) in enumerate(zip([np1, np2], [tex1, tex2])):
        plt.plot(xs,
                 npx[-1, :nbins]/npx[-1, nbins+1],
                 label=rf"1 TeV ${tex}$ (FLUKA)")

    for i, (npx, tex, lst) in enumerate(zip([np1, np2], [tex1, tex2], ['--', ':'])):
        _a = npx[-1, nbins+2]
        _b = npx[-1, nbins+3]
        plt.plot(xs,
                 stats.gamma(
                     _a,
                     scale=model.Parametrization1D.FLUKA_MEDIUM.lrad/_b).pdf(xs),
                 color=colors[i],
                 label=rf"1 TeV ${tex}$ (fit)",
                 linestyle=lst)
    
    plt.ylabel(rf'${LTOT_LABEL}^{{-1}}{DLDX_LABEL}$ [1/cm]')
    plt.xlabel(r"$x$ [cm]")
    plt.xlim(0, 3000)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig("fig/paper/fig7a.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig7a.png", bbox_inches="tight")
    
    ene = 1.e1
    df1 = util.load_csv(f"fluka/DataOutputs_{fkp}/{fkp}_{util.format_energy(ene)}.csv", False)
    df2 = util.load_csv(f"fluka/DataOutputs_{fks}/{fks}_{util.format_energy(ene)}.csv", False)
    bins = np.linspace(0, max(df1["ltot"].max(), df2["ltot"].max()), 100).tolist()
    plt.clf()
    plt.hist(
        df1["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=rf"10 GeV ${tex1}$ (FLUKA)",
    )
    plt.hist(
        df2["ltot"],
        bins=bins,
        density=True,
        histtype="step",
        label=rf"10 GeV ${tex2}$ (FLUKA)",
    )

    ul = 1.03*bins[-1]
    xs = np.linspace(bins[0], ul, 1000)
    par = model.Parametrization1D(model.Parametrization1D.FLUKA_MEDIUM)
    plt.plot(xs, par.ltot_dist(pdg.FLUKA2PDG[fkp], ene).pdf(xs), "--", color=colors[0], label=rf"10 GeV ${tex1}$ (model)")
    plt.plot(xs, par.ltot_dist(pdg.FLUKA2PDG[fks], ene).pdf(xs), ":", color=colors[1], label=rf"10 GeV ${tex2}+$ (model)")

    plt.legend(loc="upper left")
    plt.xlim(bins[0], ul)
    plt.yscale('log')
    plt.ylim(1e-6, 4e-2)
    plt.ylabel("Density [1/cm]")
    plt.xlabel(rf"${LTOT_LABEL}$ [cm]")
    plt.savefig("fig/paper/fig7b.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig7b.png", bbox_inches="tight")
    plt.close("all")


def fig8():
    def get_ssr(Dat, _pdg):
        ssrs = []
        ssrw = []
        for ene in Dat.keys():
            darr = Dat[ene]
            nbins = int(darr[0,509])
            bwidt = darr[0,508]

            Xa, Ya = np.meshgrid(np.arange(0, nbins)*bwidt, darr[:,nbins+2])
            _, Yb = np.meshgrid(np.arange(0, nbins)*bwidt, darr[:,nbins+3])
            parr = stats.gamma(
                Ya, scale=model.Parametrization1D.FLUKA_MEDIUM.lrad/Yb).pdf(Xa)
            ssrs.append(((darr[:,:500]/darr[:,501][:,None] - parr)**2).sum(axis=1))
            if particle in ["ELECTRON", "PION+", "PROTON"]:
                rwth = model.RWParametrization1D(model.Parametrization1D.FLUKA_MEDIUM)
                gamm = rwth._shape(_pdg, ene)
                Xa, Ya = np.meshgrid(np.arange(0, nbins)*bwidt, gamm.args[0])
                _, Yb = np.meshgrid(np.arange(0, nbins)*bwidt, gamm.kwds['scale'])
                parr = stats.gamma(
                    Ya, scale=model.Parametrization1D.FLUKA_MEDIUM.lrad/Yb).pdf(Xa)
                ssrw.append(((darr[:,:500]/darr[:,501][:,None] - parr)**2).sum(axis=1))

        return ssrs, ssrw

    def icolumn(particle):
        if particle in ["ELECTRON", "PHOTON"]:
            return 0
        if particle in ["PION+", "KAON+", "KAONSHRT", "KAONLONG"]:
            return 1
        return 2

    def get_ks(Dat, _pdg):
        par = model.Parametrization1D(model.Parametrization1D.FLUKA_MEDIUM,
                                      random_state=np.random.default_rng(1))
        ks_l = []
        ks_a = []
        ks_b = []
        for ene in Dat.keys():
            shos = par.sample(_pdg, ene, 1000)
            ltots = [_.ltot for _ in shos]
            aprim = [_.shape.args[0] for _ in shos]
            bprim = [par.medium.lrad / _.shape.kwds['scale'] for _ in shos]
            ks_l.append(stats.ks_2samp(Dat[ene][:, 501], ltots))
            ks_a.append(stats.ks_2samp(Dat[ene][:, 502], aprim))
            ks_b.append(stats.ks_2samp(Dat[ene][:, 503], bprim))
        return ks_l, ks_a, ks_b
    
    plt.clf()
    _wide, _height = plt.gcf().get_size_inches()
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row', sharex='col', figsize=(_wide*3.1, _height*2.))
    handles = []
    labels = []
    for i, particle in enumerate(PARTICLES):
        j = icolumn(particle)
        Dat = util.load_batch(f'fluka/DataOutputs_{particle}/*.csv',
                              loader=util.load_npy,
                              clean=True)
        _pdg = pdg.FLUKA2PDG[particle]
        ssrs, ssrw = get_ssr(Dat, _pdg)

        if ssrw:
            extra_label = "(RW 2013)" if particle in ["ELECTRON", "PHOTON"] else "(Rädel 2012)"
            ax[0][j].plot(Dat.keys(),
                          [np.median(_) for _ in ssrw],
                          c='gray',
                          label=rf"${PLABELS[i]}$ {extra_label}",
                          ls=PLINEST[i],
                          linewidth=1.5)
            ax[0][j].legend()
        line, = ax[0][j].plot(Dat.keys(),
                              [np.median(_) for _ in ssrs],
                              c=PCOLORS[i],
                              ls=PLINEST[i],
                              linewidth=1.5)
        handles.append(line)
        labels.append(rf"${PLABELS[i]}$")
        ax[0][j].set_xscale('log')
        ax[0][j].set_xlim(1., 1.e6)

        ks_l, ks_a, ks_b = get_ks(Dat, _pdg)
        ax[1][0].plot(Dat.keys(), [_.statistic for _ in ks_a],
                      c=PCOLORS[i],
                      ls=PLINEST[i],
                      linewidth=1.5)
        ax[1][1].plot(Dat.keys(), [_.statistic for _ in ks_b],
                      c=PCOLORS[i],
                      ls=PLINEST[i],
                      linewidth=1.5)
        ax[1][2].plot(Dat.keys(), [_.statistic for _ in ks_l],
                      c=PCOLORS[i],
                      ls=PLINEST[i],
                      linewidth=1.5)

    _d = dict(fontsize=18,
              verticalalignment='bottom',
              horizontalalignment='right',
              color='black')
    ax[1][0].set_title(
        r"$a'$",
    )
    ax[1][1].set_title(
        r"$b'$",
    )
    ax[1][2].set_title(
        rf"${LTOT_LABEL}$",
    )

    ax[0][0].set_yscale('log')
    ax[0][0].set_ylabel('SSR')
    ax[1][0].set_ylim(ymin=0.)
    ax[1][0].set_ylabel('KS statistic')
    [ax[1][_].set_xscale('log') for _ in range(3)]
    [ax[1][_].set_xlim(1., 1.e6) for _ in range(3)]
    fig.supxlabel(r"$E$ [GeV]", y=0.02)
    fig.legend(handles=handles,
               labels=labels,
               bbox_to_anchor=(0.5, 0.98),
               loc='upper center', ncol=(len(labels) + 1)//2)
    plt.savefig("fig/paper/fig8.pdf", bbox_inches="tight")
    plt.savefig("fig/paper/fig8.png", bbox_inches="tight")
    plt.close("all")
    
    
if __name__ == "__main__":
    fig8()
    # fig7()
    # fig6()
    # fig5()
    # fig4()
    # fig3()
    # fig2()
    # fig1()

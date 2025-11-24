from collections import namedtuple
from functools import partial
import pythia8mc
import numpy as np
from scipy import stats
from ian import Sample
from ian.ltot import efn, qrt, cbc, lin
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

LRAD = 0.3608 / 0.9216 * 100  # cm


def simulate_neutrino_dis(num_events=10, init_energy_gev=100000.0, init_pdg=12, target_pdg=2212):
    """
    Simulates electron neutrino - proton Deep Inelastic Scattering (DIS)
    using Pythia8 via the pythia8mc package.

    Args:
        num_events (int): The number of events to generate.
        init_energy_gev (float): The energy of the incoming electron neutrino in GeV.
                                     Default is 100 TeV (100,000 GeV).
    """

    # 1. Initialize Pythia
    # You can pass the path to the Pythia8 XML documentation if needed,
    # but for basic runs, it might find it automatically or use internal settings.
    # pythia = pythia8mc.Pythia(xml_dir='/path/to/pythia8/xmldoc')
    try:
        pythia = pythia8mc.Pythia()
        print("Pythia8 initialized successfully.")
    except Exception as e:
        print(f"Error initializing Pythia8: {e}")
        print("Please ensure Pythia8 and pythia8mc are correctly installed and configured.")
        print("You might need to set the PYTHIA8DATA environment variable to point to the xmldoc directory.")
        return

    # 2. Set up beam parameters
    pythia.readString("Beams:frameType = 2")  # Fixed target frame
    pythia.readString(f"Beams:idA = {init_pdg}")      # Beam A: pdg of incoming particle A
    pythia.readString(f"Beams:eA = {init_energy_gev}") # Beam A energy in GeV
    pythia.readString(f"Beams:idB = {target_pdg}")    # Beam B: proton
    pythia.readString(f"Beams:eB = 0")  # target at rest

    # 3. Configure physics processes for CC+NC DIS
    # Turn off other weak processes if you want to be very specific, then enable only W exchange.
    pythia.readString("WeakBosonExchange:all = off")
    pythia.readString("WeakBosonExchange:ff2ff(t:W) = on") # Enable f f' -> f'' f''' via t-channel W
    pythia.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    # pythia.readString("WeakDoubleBoson:all = on")
    # pythia.readString("WeakSingleBoson:all = on")
    pythia.readString("ParticleDecays:limitTau0 = on")
    # pythia.readString("ParticleDecays:tau0Max = 2.6e-5") # decay up to pi0s
    pythia.readString("ParticleDecays:tau0Max = 0.5") # let heavy quark particles decay
    # Ensure parton distribution functions (PDFs) are used for the proton.
    # Pythia8 has internal defaults. For more specific studies, you might use LHAPDF.
    # Example for LHAPDF (if Pythia8 is compiled with LHAPDF support):
    # pythia.readString("PDF:pSet = LHAPDF6:NNPDF31_nnlo_as_0118")
    # If not, Pythia uses its default, which is usually fine for examples.
    pythia.readString("PDF:useHard = on") # Use PDFs for hard process (should be default for DIS)
    pythia.readString("PDF:lepton = on")
    pythia.readString("HardQCD:all = on")

    # Settings for DIS phase space (optional, Pythia has defaults)
    # pythia.readString("SigmaDIS:Q2min = 1.0") # Minimum Q^2 in GeV^2
    # pythia.readString("SigmaDIS:Wmin = 2.0")  # Minimum W (hadronic mass) in GeV

    # Ensure hadronization and multi-parton interactions (MPI) are on for a full event.
    pythia.readString("PartonLevel:MPI = on")    # Multi-Parton Interactions (recommended)
    pythia.readString("HadronLevel:all = on")    # Hadronization

    # 4. Set random seed for reproducibility (optional)
    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 12345") # You can change this seed

    # 5. Initialize Pythia for the run with the above settings
    # This checks all settings and prepares for event generation.
    try:
        if not pythia.init():
            print("Error: Pythia initialization failed!")
            pythia.stat() # Print status, may contain error messages
            return
        print("Pythia initialized for event generation.")
    except RuntimeError as e:
        print(f"RuntimeError during Pythia initialization: {e}")
        print("This can happen if XML files are not found or settings are inconsistent.")
        return

    # 6. Event loop
    print(f"\nStarting event generation for {num_events} events...")
    events = []
    Event = namedtuple('Event', 'is_cc hadron_pids hadron_energies hadron_vprods electron_energies electron_vprods gamma_energies gamma_vprods'.split())
    for iEvent in range(num_events):
        if not pythia.next():
            # If Pythia.next() returns false, it means the end of the run,
            # often due to reaching the maximum number of errors.
            print(f"Warning: pythia.next() failed at event {iEvent}, possibly due to errors.")
            break

        print(f"\n----- Event {iEvent} -----")

        num_final_particles = 0
        num_final_hadrons = 0
        num_final_leptons = 0
        primary_lepton = None

        hadron_pids = []
        hadron_energies = []
        electron_energies = []
        gamma_energies = []
        hadron_vprods = []
        electron_vprods = []
        gamma_vprods = []
        is_cc = True
        for i in range(pythia.event.size()):
            particle = pythia.event[i] # Get particle from event record

            if particle.isFinal():
                num_final_particles += 1
                if particle.isHadron():
                    num_final_hadrons += 1
                    hadron_energies.append(particle.e())
                    hadron_pids.append(particle.id())
                    hadron_vprods.append(particle.vProd())
                if particle.isLepton():
                    num_final_leptons += 1
                    if abs(particle.id()) == 11:
                        electron_energies.append(particle.e())
                        electron_vprods.append(particle.vProd())
                    if primary_lepton is None:
                        # take the first lepton
                        primary_lepton = particle
                        is_cc = primary_lepton.id() != init_pdg
                if particle.id() == 22:
                    gamma_energies.append(particle.e())
                    gamma_vprods.append(particle.vProd())

        print(f"Total final particles: {num_final_particles}")
        print(f"Final hadrons: {num_final_hadrons}")
        print(f"Final leptons: {num_final_leptons}")

        if primary_lepton:
            print(f"Primary outgoing lepton: id={primary_lepton.id()}, "
                  f"e={primary_lepton.e():.2f} GeV, "
                  f"pT={primary_lepton.pT():.2f} GeV, "
                  f"eta={primary_lepton.eta():.2f}, phi={primary_lepton.phi():.2f}")
        else:
            if (10 < init_pdg <= 18):
                raise RuntimeError("No final state lepton found when initial particle was a lepton.")
        events.append(Event(np.asarray(is_cc),
                            np.asarray(hadron_pids),
                            np.asarray(hadron_energies),
                            np.asarray(hadron_vprods),
                            np.asarray(electron_energies),
                            np.asarray(electron_vprods),
                            np.asarray(gamma_energies),
                            np.asarray(gamma_vprods)))

        # You can add more detailed analysis here, e.g., listing particles:
        if iEvent < np.inf: # Print details for first few events
            print("Final state particles (PDG ID, e, pT, eta, phi, status):")
            for i in range(pythia.event.size()):
                p = pythia.event[i]
                if p.isFinal():
                    print(f"  {p.id()} {p.e():.2f} {p.pT():.2f} {p.eta():.2f} {p.phi():.2f} {p.status()}")

    # 7. Print statistics
    # This includes cross sections, number of accepted/rejected events, etc.
    print("\n----- Pythia Statistics -----")
    pythia.stat()
    print("---------------------------")
    return events


def plot_subparticles(xs,
                      coeffs,
                      ltpars,
                      energies,
                      vprods,
                      rng_instance,
                      cmap=plt.cm.jet,
                      minimum=10):
    colors = cmap(np.linspace(0.25, 0.75, len(energies)+1))
    ys = np.zeros_like(xs)
    emiss = 0
    ltfns = []

    # since the fit is performed in log-space, distribution parameters with all-negative values are abs'd
    # the stored 's' keeps track of the final sign to apply
    sgns = ltpars['s']
    if len(sgns) == 3:
        sdist = stats.skewnorm
    elif len(sgns) == 4:
        sdist = stats.norminvgauss
    else:
        raise RuntimeError('Unable to match distributions')

    for i, sgn in enumerate(sgns):
        _p = ltpars[f'p{i}']
        if len(_p) == 2:
            ltfns.append((lin, sgn, _p))
        elif len(_p) == 4:
            ltfns.append((cbc, sgn, _p))
        elif len(_p) == 5:
            ltfns.append((qrt, sgn, _p))
        else:
            raise RuntimeError('Unable to match parameters to function')

    for i, (energy, vprod) in enumerate(zip(energies, vprods)):
        if energy < minimum:
            emiss += energy
            continue
        a, b = Sample.sample_ab(coeffs, energy, 1, rng=rng_instance)
        ltot = sdist.rvs(*[_sgn * efn(energy, _fn, *_p) for _fn, _sgn, _p in ltfns],
                         random_state=rng_instance)
        y = ltot * stats.gamma.pdf(xs, a, loc=vprod.pz()/10, scale=LRAD/b)
        ys += y
        plt.plot(xs, y, linestyle=':', color=colors[i])

    # catch the remainder
    if emiss > minimum:
        a, b = Sample.sample_ab(coeffs, emiss, 1, rng=rng_instance)
        ltot = sdist.rvs(*[_sgn * efn(emiss, _fn, *_p) for _fn, _sgn, _p in ltfns],
                         random_state=rng_instance)
        y = ltot * stats.gamma.pdf(xs, a, scale=LRAD/b)
        ys += y
        plt.plot(xs, y, linestyle=':', color=colors[i+1])
    return ys


def pid_with_fallback(pid, available_pids):
    if pid < 100:
        return pid

    available_pids = np.asarray(available_pids)
    if 100 <= pid < 1000:
        sel = (100 <= available_pids) & (available_pids < 1000)
    else:
        sel = available_pids >= 1000

    diff = np.inf
    spid = None
    for _pid in available_pids[sel]:
        this_diff = np.abs(_pid - pid)
        if this_diff < diff:
            diff = this_diff
            spid = _pid
    return spid


if __name__ == '__main__':
    plt.style.use('present')
    enu = 80000
    pnu = 12
    events = simulate_neutrino_dis(num_events=200,
                                   init_energy_gev=enu,
                                   init_pdg=pnu)

    pdg2fluka = {11:'ELECTRON',
                 22:'PHOTON',
                 211:'PION+',
                 130:'KAONLONG',
                 310:'KAONSHRT',
                 321:'KAON+',
                 # 411:'D+',
                 2212:'PROTON',
                 2112:'NEUTRON',
                 3122:'LAMBDA',
                 3222:'SIGMA+',
                 3112:'SIGMA-',
                 }
    coefs_all = {}
    ltots_all = {}
    for k in pdg2fluka:
        coefs_all[k] = np.load(f"ian/Coeffs_{pdg2fluka[k]}.npy")
        ltots_all[k] = np.load(f"ian/ltot_{pdg2fluka[k]}.npz")

    xs = np.arange(0, 3000.1, 10)
    rng = np.random.default_rng(1234)
    print(f"Mean inelasticity: {np.mean([sum(_.hadron_energies) for _ in events])/enu}")
    for i, event in enumerate(events):
        ys = np.zeros_like(xs)
        for upid in np.unique(np.abs(event.hadron_pids)):
            # _c = str(upid)[0] == '4'
            sel = np.abs(event.hadron_pids) == upid
            _pid = pid_with_fallback(upid, list(pdg2fluka.keys()))
            ys += plot_subparticles(xs,
                                    coefs_all.get(upid, coefs_all[_pid]), # if not _c else coefs_all[411]),
                                    ltots_all.get(upid, ltots_all[_pid]),
                                    event.hadron_energies[sel],
                                    event.hadron_vprods[sel],
                                    rng,
                                    plt.cm.Blues_r, # if not _c else plt.cm.Greens_r,
                                    minimum=10)
        plt.plot(xs, ys, c='b', label=r'$\sum E_{had}$')

        # EM showers
        yg = plot_subparticles(xs,
                               coefs_all[22],
                               ltots_all[22],
                               event.gamma_energies,
                               event.gamma_vprods,
                               rng,
                               plt.cm.Oranges_r,
                               minimum=10)
        ye = plot_subparticles(xs,
                               coefs_all[11],
                               ltots_all[11],
                               event.electron_energies,
                               event.electron_vprods,
                               rng,
                               plt.cm.Purples_r,
                               minimum=10)
        if np.any(ye+yg):
            plt.plot(xs, ye+yg, c='r', label=r'$\sum E_{em}$')
        plt.plot(xs, ys + yg + ye, c='grey', label=r'Total')

        # CMC
        # nugen lumps into a single hadron so sum over these for e_hd
        e_hd = sum(event.hadron_energies) + sum(event.gamma_energies)
        a_hd = 1.58357292+0.41886807 * np.log(e_hd)
        b_hd = 0.33833116
        y_hd = e_hd * stats.gamma.pdf(xs, a_hd, scale=LRAD/b_hd)
        # plt.plot(xs, y_hd, c='b', linestyle='--', label='CMC (had)')

        y_em = np.zeros_like(xs)
        if event.is_cc:
            e_em = event.electron_energies[0]
            a_em = 2.01849 + 0.63176 * np.log(e_em)
            b_em = 0.63207
            y_em = e_em * stats.gamma.pdf(xs, a_em, scale=LRAD/b_em)
            # plt.plot(xs, y_em, c='r', label='CMC (EM)', linestyle='--')
        # plt.plot(xs, y_hd + y_em, c='grey', linestyle='--', label=r'CMC (total)')

        plt.legend()
        plt.title(rf'$E_\nu = {enu / 1e3}$ TeV, PDG={pnu}, $E_{{had}} = {sum(event.hadron_energies)/1e3:.2f}$ TeV')
        plt.xlim(xmin=0)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/100:.2g}'))
        plt.xlabel('[m]')
        plt.ylabel('dl/dx')
        plt.ylim(ymin=0)
        plt.savefig(f'figs/{i}.png', bbox_inches='tight')
        plt.clf()

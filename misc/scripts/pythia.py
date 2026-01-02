#!/usr/bin/env python
from collections import namedtuple
import pythia8mc
import numpy as np
from acem import model, media
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def simulate_neutrino_dis(num_events=10, init_energy_gev=100000.0, init_pdg=12, target_pdg=2212, seed=12345):
    """
    Simulates collisions using Pythia8 via the pythia8mc package.

    Parameters
    ----------
        num_events (int): Number of collision events to generate.
        init_energy_gev (float): Energy of the incoming beam particle (Beam A) in GeV.
            Default is 100 TeV (1e5 GeV).
        init_pdg (int): PDG ID of the incoming particle (e.g., 12 for nu_e, 14 for nu_mu).
        target_pdg (int): PDG ID of the target nucleon or nucleus. 
            Use 2212 for Proton, 2112 for Neutron
        seed (int): Random seed for reproducibility.

    Returns
    -------
        events: List of Events which is a collection of the properties of the interaction
    outgoing hadrons, electrons and gammas
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
    pythia.readString("Beams:eB = 0")  # target at rest

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
    pythia.readString(f"Random:seed = {seed}") # You can change this seed

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
                      par,
                      pids,
                      enes,
                      vprods,
                      cmap=plt.cm.jet,
                      minimum=10):
    colors = cmap(np.linspace(0.25, 0.75, len(enes)+1))
    ys = np.zeros_like(xs)
    emiss = 0

    for i, (pid, energy, vprod) in enumerate(zip(pids, enes, vprods)):
        if energy < minimum:
            emiss += energy
            continue
        sho = par.sample(pid, energy)
        y = sho.dldx(xs - vprod.pz()/10.)
        ys += y
        plt.plot(xs, y, linestyle=':', color=colors[i])

    # catch the remainder
    if emiss > minimum:
        sho = par.sample(pid, emiss)
        y = sho.dldx(xs)
        ys += y
        plt.plot(xs, y, linestyle=':', color=colors[i+1])
    return ys


if __name__ == '__main__':
    plt.style.use('present')
    enu = 10000
    pnu = 12
    seed = 1234
    events = simulate_neutrino_dis(num_events=200,
                                   init_energy_gev=enu,
                                   init_pdg=pnu,
                                   seed=seed)

    rng = np.random.default_rng(seed)
    parm = model.Parametrization1D(media.IC3, random_state=rng)
    parw = model.RWParametrization1D(media.IC3, random_state=rng)

    xs = np.arange(0, 3000.1, 10)
    print(f"Mean inelasticity: {np.mean([sum(_.hadron_energies) for _ in events])/enu}")
    for i, event in enumerate(events):
        ys = plot_subparticles(xs,
                               parm,
                               event.hadron_pids,
                               event.hadron_energies,
                               event.hadron_vprods,
                               plt.cm.Blues_r, # if not _c else plt.cm.Greens_r,
                               minimum=10)

        plt.plot(xs, ys, c='b', label=r'$\sum E_{had}$')

        # EM showers
        yg = plot_subparticles(xs,
                               parm,
                               [22]*len(event.gamma_energies),
                               event.gamma_energies,
                               event.gamma_vprods,
                               plt.cm.Oranges_r,
                               minimum=1.)
        ye = plot_subparticles(xs,
                               parm,
                               [11]*len(event.electron_energies),
                               event.electron_energies,
                               event.electron_vprods,
                               plt.cm.Purples_r,
                               minimum=1.)
        if np.any(ye+yg):
            plt.plot(xs, ye+yg, c='r', label=r'$\sum E_{em}$')
        plt.plot(xs, ys + yg + ye, c='k', label=r'Total')

        # CMC
        # nugen lumps into a single hadron so sum over these for e_hd
        e_hd = sum(event.hadron_energies) + sum(event.gamma_energies)
        show = parw.sample(211, e_hd)
        y_hd = show.dldx(xs)
        plt.plot(xs, y_hd, c='b', linestyle='--', label='CMC (had)')

        y_em = np.zeros_like(xs)
        if event.is_cc:
            e_em = event.electron_energies[0]
            show = parw.sample(11, e_em)
            y_em = show.dldx(xs)
            plt.plot(xs, y_em, c='r', label='CMC (EM)', linestyle='--')
        plt.plot(xs, y_hd + y_em, c='grey', linestyle='--', label=r'CMC (total)')

        plt.legend()
        plt.title(rf'$E_\nu = {enu / 1e3}$ TeV, PDG={pnu}, $E_{{had}} = {sum(event.hadron_energies)/1e3:.2f}$ TeV')
        plt.xlim(xmin=0)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/100:.2g}'))
        plt.xlabel('[m]')
        plt.ylabel('dl/dx')
        plt.ylim(ymin=0)
        plt.savefig(f'fig/pythia/{i}.png', bbox_inches='tight')
        plt.clf()

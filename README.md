[![PyPI version](https://img.shields.io/pypi/v/acem)](https://pypi.org/project/acem) [![Status](https://github.com/tianluyuan/acem/actions/workflows/checks.yml/badge.svg)](https://github.com/tianluyuan/acem/actions) [![Python versions](https://img.shields.io/pypi/pyversions/acem)](https://pypi.org/project/acem)

# ACEM: Approximate Cherenkov Emission Model

`acem` is a Python package designed for generation of Cherenkov light yield profiles from particle showers. One of the main goals is to more accurately include shower-to-shower fluctuations while preserving model simplicity.

## Installation

You can install `acem` directly from [PyPI](https://pypi.org/project/acem/).

```bash
pip install acem
```

Alternatively, clone this repository and from the directory run `pip install .` or to work with scripts in `misc/` use `pip install .[misc]`. The latter will allow you to run checks including plots like this

```bash
cd misc/scripts/
./check.py 3 42 (energy at 1TeV, initial seed of 42)
```

It's also possible to install directly from github with

```bash
pip install git+https://github.com/pathtorepo/acem
```

## Usage

Here is an example of how to import the models and use the package:

```python
from acem import model, media

par = model.Parametrization1D(media.ICE)

# Sample 100 showers profiles initiated by a 1 TeV pi+
shos = par.sample(211, 1e3, 100)
```

`shos` will be a list of `Shower1D` objects, which is a simple container that consists of the amplitude and 1D shape of the Cherenkov-weighted track length (units are in cm). Its `.dldx(xs)` method can be used to evaluate the shower profile at some given distance(s) `xs` from the start position.

## Example
The 1D models are based on [FLUKA](https://www.fluka.eu/Fluka/www/html/fluka.php?) simulations. As an example, when used in combination with a hadronization library such as [PYTHIA8](https://pythia.org/), it can be used to generate neutrino DIS shower profiles.

<img width="100%" alt="fig9" src="https://github.com/user-attachments/assets/db34e002-9217-4bd0-a655-db20028a9747">

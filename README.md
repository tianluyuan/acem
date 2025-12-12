TODO: fill in and add LICENSE
[![Status](https://github.com/tianluyuan/actions/workflows/checks.yml/badge.svg)](https://github.com/tianluyuan/shosim/actions)

# shosim

`shosim` is a Python package designed for generation of Cherenkov light yield profiles from particle showers. One of the main goals is to more accurately include shower-to-shower fluctuations while preserving model simplicity.

## Installation

You can install `shosim` directly from PyPI: [TBD]

```bash
pip install shosim
```

Alternatively, clone this repository and from the directory run `pip install .` or to work with scripts in `misc/` use `pip install .[misc]`. The latter will allow you to run checks including plots like this

```bash
cd misc/scripts/
./check.py 3 42 (energy at 1TeV, initial seed of 42)
```

If you prefer to not clone the repo it's possible to install directly from github with

```bash
pip install git+https://github.com/pathtorepo/shosim
```

## Usage

Here is an example of how to import the models and use the package:

```python
from shosim import model, media

par = model.Parametrization1D(media.ICE)

# Sample 100 showers profiles initiated by a 1 TeV pi+
shos = par.sample(211, 1e3, 100)
```

`shos` will be a list of `Shower1D` objects, which is a simple container that consists of the amplitude and 1D shape of the Cherenkov-weighted track length. Its `.dldx(xs)` method can be used to evaluate the shower profile at some given distance(s) `xs` from the start position.

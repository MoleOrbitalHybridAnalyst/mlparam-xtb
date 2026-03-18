# mlparam-xtb

`mlparam-xtb` is a framework for machine learning-parameterized semi-empirical quantum chemistry calculations. It leverages the MACE (Message Passing Neural Network) architecture to dynamically predict atom-wise parameters for the GFN1-xTB Hamiltonian.

The entire package is implemented in **JAX** using PySCFAD and MACE-JAX, making it fully, end-to-end differentiable.

## Features

- **End-to-End Differentiable:** Exact analytical gradients for atomic forces are obtained effortlessly via JAX's automatic differentiation (`jax.value_and_grad`).
- **ML-Parameterized GFN1-xTB:** A MACE backend processes local chemical environments to predict scalings for standard xTB parameters (e.g., electronegativity, hardness, effective nuclear charge, repulsions).
- **Advanced QM/MM Capabilities:** 
  - Native support for explicit QM/MM embedding.
  - Ewald summation integration for handling periodic boundary conditions (PBC).


## Dependencies

This package relies on JAX-based MLIP and quantum chemistry libraries:

- `mace-jax` (Tested with `github.com:ACEsuit/mace-jax`, `main` branch, commit `9fe59d9d0f953c4a052b522a386d5b855bea248f`)
- `pyscfad` (Tested with `https://github.com/MoleOrbitalHybridAnalyst/pyscfad.git`, branch `fishjojo-mlxtb`, commit `7f99f546ebe3f0e2369591785e79a419e42cc60e`)

## Basic Usage

The xTB model can be instantiated and run inside a standard JAX/Flax environment:

```python
from flax import nnx
from mlparam_xtb.models import XTBModel
from mlparam_xtb.data import QMMMData, _collate, pad_batch
import jax

# 1. Initialize MACE backend and XTB parameters
# ... (load mace_module, basis, and param arrays) ...

# 2. Build the XTBModel
xtb_model = XTBModel(
    mace_model=mace_module,
    xtb_param=param,
    basis=basis,
    max_qm=100,
    max_mm=5000,
    # ... extra arguments
)

# 3. Run predictions
@nnx.jit
def eval_step(model, batch):
    return model(batch)

energy = eval_step(xtb_model, padded_batch)
```

Checkout examples/ for more details

## References

https://arxiv.org/abs/2509.10967
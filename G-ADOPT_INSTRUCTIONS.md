# Using Mass-Conservative Time Discretisation in G-ADOPT

This document explains how to use the mass-conservative time discretisation for nonlinear mass terms (like `Dt(theta(h))` in Richards equation) in your G-ADOPT simulations. It is written for someone building Richards equation or similar variably-saturated flow models.

## Quick start

The key is to use the **stage value stepper** (`stage_type="value"`) with a **stiffly accurate Butcher tableau** when you have nonlinear mass terms.

```python
from firedrake import *
from irksome import TimeStepper, BackwardEuler, Dt, MeshConstant

# Your mesh, function spaces
mesh = ...
V = FunctionSpace(mesh, "CG", 1)
h = Function(V)  # pressure head

# Define theta(h) nonlinearly
def theta(h):
    return theta_r + (theta_s - theta_r) * exp(alpha * h)

# Your form with nonlinear mass term
F = inner(Dt(theta(h)), v) * dx + inner(K(h)*grad(h), grad(v)) * dx + ...

# Create time stepper with MeshConstant for G-ADOPT's block-varying timesteps
MC = MeshConstant(mesh)
t = MC.Constant(0.0)
dt = MC.Constant(dt_value)

stepper = TimeStepper(
    F, BackwardEuler(), t, dt, h,
    stage_type="value",  # <-- This is crucial for conservation
    solver_parameters={...}
)

# Advance in time
for step in range(nsteps):
    stepper.advance()
    t.assign(float(t) + float(dt))
```

That's it. With `stage_type="value"`, the time discretisation of `Dt(theta(h))` becomes an exact finite difference: `(theta(h_new) - theta(h_old)) / dt`. Mass is conserved to solver tolerance.

## Why this matters

Consider the Richards equation for variably saturated soil:

```
∂θ/∂t + ∇·(K(h) ∇h) = 0
```

where `θ(h)` is moisture content (a nonlinear function of pressure head h) and K(h) is hydraulic conductivity.

The **conserved quantity is θ, not h**. For exact mass balance, the time derivative of θ must be discretised as a finite difference of θ values, not via the capacity form:

```
Exact:     (θ(h_new) - θ(h_old)) / dt
Capacity:  C(h) dh/dt,  where C = dθ/dh
```

These are mathematically equivalent in continuous time, but with finite dt the linearisation error is systematic. Over 100+ timesteps, mass drifts noticeably.

The stage value stepper with `Dt(theta(h))` computes the two-evaluation finite difference directly, avoiding the linearisation error entirely.

## Butcher tableaux selection

You can use **any stiffly accurate Runge-Kutta method**. For G-ADOPT, the recommended choices are:

| Method | Accuracy | Implicit? | Cost |
|--------|----------|-----------|------|
| `BackwardEuler()` | 1st order | Fully implicit | 1 solve per step |
| `RadauIIA(2)` | 3rd order | Fully implicit | 2 solves per step |
| `RadauIIA(3)` | 5th order | Fully implicit | 3 solves per step |

All are stiffly accurate (u_new = U_s, the last stage equals the new solution) so mass is conserved exactly.

**Do NOT use** non-stiffly-accurate methods like:
- `GaussLegendre(1)` (implicit midpoint) -- will raise NotImplementedError
- `GaussLegendre(2)` or higher
- Explicit methods

These methods are incompatible with the conservative finite difference formulation. If you try to use them, you'll get a clear error message.

```python
# This will raise NotImplementedError
stepper = TimeStepper(F, GaussLegendre(1), t, dt, h, stage_type="value")
# Error: Composite time derivatives require stiffly accurate methods
```

## MeshConstant for block-varying timesteps

G-ADOPT uses block-varying timesteps (different dt for different blocks). The irksome TimeStepper requires `MeshConstant` for time and timestep, not plain `Constant`:

```python
from irksome import MeshConstant

MC = MeshConstant(mesh)
t = MC.Constant(0.0)      # Current time
dt = MC.Constant(dt_val)  # Current timestep

stepper = TimeStepper(F, ..., t, dt, h, ...)
```

The reason: `MeshConstant.Constant` is tied to the mesh topology and supports block-varying values in G-ADOPT's infrastructure. Plain `Constant` (from Firedrake) does not.

Between timesteps, update both:

```python
for step in range(nsteps):
    stepper.advance()
    
    # Update time and dt for next step
    t.assign(float(t) + float(dt))
    if step < nsteps - 1:
        dt.assign(next_dt_value)
```

## Theta model example

For an exponential soil model (e.g., van Genuchten):

```python
# Parameters
theta_r = 0.15  # residual
theta_s = 0.45  # saturated
alpha = 0.328   # shape parameter
Ks = 1e-5       # saturated conductivity

def theta(h):
    """Moisture content as function of pressure head."""
    return theta_r + (theta_s - theta_r) * exp(alpha * h)

def conductivity(h):
    """Hydraulic conductivity as function of pressure head."""
    return Ks * exp(alpha * h)
```

Define these as Python functions and use them in your form:

```python
F = inner(Dt(theta(h)), v) * dx + inner(conductivity(h)*grad(h), grad(v)) * dx - ...
```

Irksome will handle the nonlinear composition correctly.

## Verifying mass conservation

To check that your simulation conserves mass:

```python
from firedrake import assemble, dx

# Compute initial moisture
initial_mass = assemble(theta(h) * dx)

# Run timesteps
for step in range(nsteps):
    stepper.advance()
    t.assign(float(t) + float(dt))
    
    # Compute cumulative boundary flux (example: constant flux on boundary 4)
    flux = 1e-6  # [units]
    expected_change = flux * float(dt)
    
    # Check moisture change
    current_mass = assemble(theta(h) * dx)
    actual_change = current_mass - initial_mass
    
    # This should match to solver tolerance (typically 1e-12)
    error = abs(actual_change - expected_change)
    if error > 1e-10:
        print(f"Step {step}: mass error = {error:.2e} (too large!)")
    
    initial_mass = current_mass
```

For a well-posed problem with appropriate solver tolerance, mass balance error should be **< 1e-10** (limited only by Newton solver tolerances, not discretisation).

## Solver parameters for performance

The stage value stepper solves a coupled system of all stages simultaneously. For large spatial meshes, use a block-structured preconditioner to exploit the Butcher matrix structure:

```python
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "pc_type": "bjacobi",
    "sub_ksp_type": "preonly",
    "sub_pc_type": "lu",
    "sub_pc_factor_mat_solver_type": "mumps",
    "ksp_max_it": 100,
    "ksp_rtol": 1e-6,
    "snes_type": "newtonls",
    "snes_atol": 1e-14,
    "snes_rtol": 1e-8,
    "snes_max_it": 20,
}
```

For a DIRK tableau (lower-triangular Butcher matrix), you can use a multiplicative fieldsplit to solve stage-by-stage:

```python
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    # Then configure each field (each stage)
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    # ... repeat for other stages
}
```

This requires experimentation for your specific mesh and problem size.

## Complete example: Richards equation on a unit square

```python
from firedrake import *
from irksome import TimeStepper, BackwardEuler, Dt, MeshConstant
import numpy as np

# Mesh and function space
N = 20
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "CG", 1)

# Initial condition
h = Function(V, name="h").assign(-1.0)
v = TestFunction(V)

# Soil parameters
theta_r = 0.15
theta_s = 0.45
alpha = 0.328
Ks = 1e-5

def theta(h):
    return theta_r + (theta_s - theta_r) * exp(alpha * h)

def conductivity(h):
    return Ks * exp(alpha * h)

# Time setup
MC = MeshConstant(mesh)
t = MC.Constant(0.0)
dt_val = 100.0
dt = MC.Constant(dt_val)

# Form: Richards diffusion with constant flux boundary condition
F = inner(Dt(theta(h)), v) * dx
F += inner(conductivity(h) * grad(h), grad(v)) * dx
F -= 1e-6 * v * ds(4)  # flux on boundary 4

# Create stepper with conservative mass discretisation
stepper = TimeStepper(
    F, BackwardEuler(), t, dt, h,
    stage_type="value",  # <-- Conservative
    solver_parameters={
        "mat_type": "aij",
        "snes_type": "newtonls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_atol": 1e-14,
    }
)

# Run simulation
nsteps = 50
for step in range(nsteps):
    print(f"Step {step+1}/{nsteps}, t={float(t):.1f}")
    stepper.advance()
    t.assign(float(t) + float(dt))
    
    # Optional: check mass
    mass = assemble(theta(h) * dx)
    print(f"  Total moisture: {mass:.6e}")
```

## Comparison: stage_type="value" vs "dirk"

If you accidentally use `stage_type="dirk"` with `Dt(theta(h))`:

```python
# WRONG: will apply chain rule, lose conservation
stepper = TimeStepper(F, BackwardEuler(), t, dt, h, stage_type="dirk")
```

What happens:
- The chain rule is applied: `Dt(theta(h))` → `theta'(h) * Dt(h)`
- The capacity form is used: `C(h) * dh/dt`
- Over 50 timesteps, cumulative mass error grows to ~1e-6
- This is systematic: θ increases/decreases more than boundary flux predicts

With `stage_type="value"`:
- No chain rule for the time-derivative term
- Exact finite difference: `(theta(h_new) - theta(h_old)) / dt`
- Cumulative mass error < 1e-10 (limited by solver tolerance)

The stage value stepper is slightly more expensive (solves all stages together rather than one-by-one) but the conservation gain is worth it for Richards equation and other flows where θ or other conserved quantities have nonlinear dependence on the primary variable.

## Troubleshooting

### "NotImplementedError: Composite time derivatives require stiffly accurate methods"

You used a non-stiffly-accurate tableau with `Dt(g(u))`. Switch to a stiffly accurate method:

```python
# Wrong
stepper = TimeStepper(F, GaussLegendre(1), t, dt, h, stage_type="value")

# Right
stepper = TimeStepper(F, BackwardEuler(), t, dt, h, stage_type="value")
# or
stepper = TimeStepper(F, RadauIIA(2), t, dt, h, stage_type="value")
```

### Large mass balance errors (> 1e-8)

Check your solver tolerance. If `snes_atol` is too loose (> 1e-12), mass error will be dominated by the nonlinear solver:

```python
# Tighten Newton solver tolerance
solver_parameters = {
    "snes_atol": 1e-14,  # <-- Make this tighter
    "snes_rtol": 1e-8,
}
```

Also check your boundary condition types. Pure Neumann problems (Dirichlet everywhere would over-constrain moisture flow) give the clearest mass balance.

### Timestep too large, solver not converging

The stage value stepper solves a larger coupled system (all stages simultaneously). For stiff problems, you may need smaller dt or better preconditioner. Try:

```python
# Smaller timestep
dt.assign(dt_val / 2)

# Or better preconditioner
solver_parameters = {
    "ksp_type": "gmres",
    "pc_type": "asm",  # or "fieldsplit" for DIRK tableaux
    "pc_asm_overlap": 2,
}
```

### "Irksome was not compiled with the stage value stepper"

This shouldn't happen with recent Irksome. Make sure you have the merged branch with Pablo's changes. Check:

```bash
git log --oneline | grep -i "conservative\|stage"
```

You should see commits about conservative mass terms and stage value improvements.

## Further reading

- **CONSERVATIVE_DT_DESIGN.md** in this repository: full technical context for engineers implementing the feature
- **ROB_KIRBY_SUGGESTIONS.md**: Rob Kirby's (Irksome maintainer) design notes
- **tests/test_mass_conservation.py**: working example with exponential soil model
- Irksome documentation: https://irksome.readthedocs.io/

## Questions?

If you encounter issues or have questions:

1. Check that `stage_type="value"` is set (easy to forget!)
2. Verify your Butcher tableau is stiffly accurate (`tableau.is_stiffly_accurate` should be True)
3. Ensure you're using `MeshConstant` for t and dt (not plain `Constant`)
4. Tighten solver tolerances (`snes_atol`, `ksp_rtol`) if you see large mass errors

For implementation details or modifications, refer to the technical design documents (CONSERVATIVE_DT_DESIGN.md, ROB_KIRBY_SUGGESTIONS.md) in the Irksome repository root.

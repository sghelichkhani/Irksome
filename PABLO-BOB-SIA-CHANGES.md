# Conservative Mass Terms: Contributions from Pablo, Rob, and Sia

This document summarizes the collaborative work on mass-conservative time discretisation for nonlinear mass terms (`Dt(g(u))`) in Irksome. It explains who contributed what and how the pieces fit together.

## The problem (Rob's diagnosis)

Rob Kirby identified (GitHub issue #203, Slack discussion) that `expand_time_derivatives` was applying the chain rule to composite time derivatives, breaking mass conservation for nonlinear functions like `Dt(theta(h))` in Richards equation.

His insight: the **stage value stepper can handle nonlinear mass terms natively without the chain rule**, using a two-evaluation substitution instead of finite differences.

## Contributions

### Pablo Brubeck (PR #204: `pbrubeck/conservative-stage-value`)

**UFL and form manipulation infrastructure:**
- Redesigned `irksome/ufl/manipulation.py` with cleaner DAGTraverser-based approach
- Implemented `TimeDerivativeChecker` to validate that time derivatives are applied linearly
- Created `split_time_derivative_terms()` function to cleanly separate time-derivative terms from spatial terms
- Created `remove_time_derivatives()` to strip `Dt` wrappers from expressions
- These allow stage value stepper to work on composite `Dt(g(u))` without the chain rule

**Updated `is_ode()` check in `tools.py`:**
- For `Dt(g(u))` where g is nonlinear, `is_ode()` correctly returns False
- This is fine because stage value stepper handles it differently than DIRK stepper
- Assertion `assert is_ode(F, u0) or butch.is_stiffly_accurate` passes because stage value + stiffly accurate is allowed

**Stage derivative stepper update:**
- Minor changes to work with the new form manipulation pipeline

**Discontinuous Galerkin stepper update:**
- Adapted to use the new `split_time_derivative_terms()` API

### Rob Kirby (design direction)

**Design decision:**
- Recommended fixing the stage value and DG steppers rather than making DIRK more complex
- Suggested two-evaluation substitution: evaluate `g(u)` at both stage value and initial state, then subtract
- Recognized that stiffly accurate methods are required for exact mass conservation

### Sia Ghelichkhani (PR #207: `conservative-mass-test`)

**Two-evaluation substitution implementation in stage value stepper:**
- Modified `irksome/stage_value.py` `getFormStage()` function (lines 109-129)
- Replaced subtraction trick `u0: w_np[i] - u0` with:
  ```python
  repl_new = {t: t + c[i] * dt, v: A2invTv[i], u0: w_np[i]}
  repl_old = {v: A2invTv[i]}
  Fnew += replace(F_dtless, repl_new) - replace(F_dtless, repl_old)
  ```
- This gives `g(w[i]) - g(u0)` (conservative) instead of `g(w[i] - u0)` (linearised)

**Test case: Richards equation mass conservation**
- File: `tests/test_mass_conservation.py`
- Self-contained exponential soil model (`theta(h) = theta_r + (theta_s - theta_r) * exp(alpha * h)`)
- CG1 finite elements on unit square with pure Neumann boundary conditions
- Parametrised tests: BackwardEuler and RadauIIA(2) (both stiffly accurate)
- `test_mass_conservation_stage_value`: cumulative mass error < 1e-10 ✓
- `test_mass_conservation_fails_stage_derivative`: confirms stage derivative form does NOT conserve mass (error > 1e-8)
- Validates that the conservative form works and the non-conservative form fails

**MeshConstant integration:**
- Used `MeshConstant` for time stepping (required for G-ADOPT's block-varying timesteps)
- Addressed review feedback on solver parameters and test design

**Documentation for end-users:**
- File: `G-ADOPT_INSTRUCTIONS.md`
- Complete guide for G-ADOPT developers on how to use conservative mass discretisation
- Quick-start example, solver parameters, troubleshooting
- Theta model examples, mass verification code

## How the pieces fit together

```
        Pablo's Infrastructure
        (split_time_derivative_terms, remove_time_derivatives)
                    |
                    v
        Stage Value Stepper (stage_value.py)
                    |
              +-----+-----+
              |           |
              v           v
      Time-derivative   Spatial terms
      terms (F_dtless)  (F_remainder)
              |           |
        Sia's two-eval  Standard
        substitution    substitution
              |           |
              +-----+-----+
                    |
                    v
            Nonlinear mass
            conservation
            (g(U_i) - g(u0))/dt
```

1. **Pablo** provides clean separation of time terms using DAGTraverser
2. **Sia** applies two-evaluation substitution to time terms in the stage value stepper
3. **Rob's design** guides the choice of stiffly accurate methods and stage value solver
4. Together: `Dt(theta(h))` becomes exact finite difference `(theta(U_i) - theta(u0))/dt`

## Key files and their roles

| File | Contributor | Role |
|------|-------------|------|
| `irksome/ufl/manipulation.py` | Pablo | Form splitting infrastructure |
| `irksome/stage_value.py` | Sia (building on Pablo) | Two-evaluation substitution |
| `irksome/tools.py` | Pablo | Updated `is_ode` check |
| `tests/test_mass_conservation.py` | Sia | Validation test |
| `G-ADOPT_INSTRUCTIONS.md` | Sia | End-user guide |

## Design decisions explained

**Q: Why not fix DIRK stepper?**
- DIRK solves for stage derivatives k_i one-by-one
- Can't easily substitute a derivative into composite `TimeDerivative(g(u))` without re-applying chain rule
- Stage value stepper is more natural fit (Rob's insight)

**Q: Why stiffly accurate methods only?**
- For non-stiffly-accurate methods, u_new ≠ U_s
- The conservation telescoping only works if final stage equals new solution
- Backward Euler, RadauIIA, Lobatto methods are stiffly accurate
- Gauss-Legendre methods are NOT and will raise NotImplementedError

**Q: Why two evaluations instead of linearity check?**
- Could optimize by detecting linear time-derivative terms and using single substitution
- Current design uses two evaluations unconditionally for simplicity
- Cost is acceptable (builds 2x UFL for time terms, not spatial terms)

**Q: Why MeshConstant?**
- G-ADOPT uses block-varying timesteps (different dt for different spatial blocks)
- `MeshConstant` is tied to mesh topology and supports block-varying values
- Plain `Constant` does not

## Testing the unified branch

The test suite validates:

1. **Mass conservation with stage_type="value"** 
   - Test: `test_mass_conservation_stage_value`
   - Result: error < 1e-10 ✓

2. **Non-conservation with stage_type="dirk"**
   - Test: `test_mass_conservation_fails_stage_derivative`
   - Result: error > 1e-8 ✓
   - Confirms that stage derivative form (linearised) does NOT conserve mass

3. **Regression tests**
   - Existing Irksome test suite should pass unchanged
   - DIRK, CPG, stage derivative steppers unaffected (use chain rule as before)
   - Only stage value stepper behavior changes (for better mass conservation)

## Usage summary for G-ADOPT

```python
from irksome import TimeStepper, BackwardEuler, Dt, MeshConstant

MC = MeshConstant(mesh)
t = MC.Constant(0.0)
dt = MC.Constant(dt_val)

F = inner(Dt(theta(h)), v) * dx + ...  # nonlinear mass term

stepper = TimeStepper(
    F, BackwardEuler(), t, dt, h,
    stage_type="value",  # <-- This is the key
)

for step in range(nsteps):
    stepper.advance()
    t.assign(float(t) + float(dt))
```

See `G-ADOPT_INSTRUCTIONS.md` for complete guide.

## Reference documents

- **CONSERVATIVE_DT_DESIGN.md** - Sia's initial design (before discovering Pablo's parallel work)
- **ROB_KIRBY_SUGGESTIONS.md** - Rob's design notes from issue #203
- **G-ADOPT_INSTRUCTIONS.md** - End-user guide for G-ADOPT developers
- **tests/test_mass_conservation.py** - Working example with Richards equation

## Timeline

1. Rob identifies problem (issue #203, early Apr 2026)
2. Sia implements initial version on `conservative-dt` branch
3. Sia discovers Pablo's parallel work on PR #204 (`pbrubeck/conservative-stage-value`)
4. Sia creates PR #207 on top of Pablo's branch with test case and two-evaluation substitution
5. Review feedback (Rob, Pablo) → refine test, update solver parameters
6. This merged branch combines all contributions with comprehensive G-ADOPT documentation

## Contributors

- **Pablo Brubeck**: UFL infrastructure, form manipulation
- **Rob Kirby** (Irksome maintainer): Design direction, diagnosis
- **Sia Ghelichkhani**: Two-evaluation implementation, test case, G-ADOPT documentation

All working toward the same goal: exact mass conservation for Richards equation and similar problems with nonlinear mass terms.

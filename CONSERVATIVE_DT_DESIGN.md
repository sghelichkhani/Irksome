# Conservative time discretisation for nonlinear mass terms

This document captures the full technical context for implementing mass-conservative
time discretisation in Irksome. It is written so that an engineer who hasn't been
part of the discussion can pick it up and do the work.

Tracking issue: https://github.com/firedrakeproject/Irksome/issues/203
Working branch: https://github.com/sghelichkhani/Irksome/tree/conservative-dt-stage-value


## The problem

We are solving the Richards equation for variably saturated flow:

    d(theta(h))/dt + div(K(h) (grad(h) + e_z)) = 0

where h is pressure head and theta(h) is moisture content (a nonlinear function of h).
The conserved quantity is theta, not h. For correct mass balance, the time derivative
must be discretised as a finite difference of theta:

    (theta(h_new) - theta(h_old)) / dt

and not via the linearised capacity form C(h) dh/dt, where C = d(theta)/dh.

Currently, writing `Dt(theta(h))` in Irksome causes `expand_time_derivatives` to
apply the UFL chain rule, producing `C(h) * Dt(h)`. This is a correct differential
identity, but once discretised with finite dt, `C(h) * delta_h` only approximates
`delta_theta`. The error is small but systematic and accumulates over time.


## The test

File: `tests/test_mass_conservation.py`

A self-contained Richards diffusion on a 10x10 unit square with DQ1 elements and
an exponential soil model (no G-ADOPT dependency). Constant flux on the top
boundary, no-flux elsewhere. 50 timesteps of implicit midpoint (GaussLegendre(1))
using the stage value stepper.

The test accumulates per-step mass balance error: the absolute difference between
the actual change in total moisture (integral of theta over the domain) and the
expected change (boundary flux times dt). With the chain-rule form this error is
O(1e-3) after 50 steps. With a conservative discretisation it should be near
machine precision (< 1e-10).

The test currently fails on master because the stage value stepper applies the
chain rule to `Dt(theta(h))` via `expand_time_derivatives`, destroying conservation.


## What expand_time_derivatives does

File: `irksome/ufl/deriv.py`

`expand_time_derivatives` calls two things in sequence:

1. `apply_algebra_lowering` (from UFL) -- expands compound operators like `inner`,
   `div`, `trace`, `det` into basic algebra (indexed products, gradients, etc).
   This runs first so the derivative rules see a normalised expression tree.

2. `apply_time_derivatives` -- traverses the UFL DAG using `TimeDerivativeRuleDispatcher`,
   which delegates to `TimeDerivativeRuleset`.

`TimeDerivativeRuleset` inherits from UFL's `GenericDerivativeRuleset`. This
inheritance gives it two capabilities that are currently entangled:

### Capability A: pushing Dt through linear spatial operators

The `terminal_modifier` method (line 81) is registered for `Indexed`, `Grad`, `Div`,
`Curl`, `ReferenceGrad`, `ReferenceValue`, `Variable`, and `Conj`. It is decorated
with `@DAGTraverser.postorder`, meaning operands are processed first (recursively),
then the node is reconstructed with the processed operands:

    Dt(Grad(u))  -->  Grad(Dt(u))
    Dt(u[i])     -->  Dt(u)[i]

This is linearity of differentiation. It is needed for mixed function spaces where u
gets decomposed into indexed components. Without it, the DIRK stepper's substitution
rule `{TimeDerivative(u0): k}` can't reach the leaf-level Dt(u0) nodes.

### Capability B: applying the chain rule through nonlinear compositions

`GenericDerivativeRuleset` (in UFL at `ufl/algorithms/apply_derivatives.py`) provides
handlers for all arithmetic and transcendental operations: `Product`, `Division`,
`Power`, `Sin`, `Cos`, `Exp`, `Ln`, etc. These implement the standard derivative rules:

    Dt(a * b)    -->  Dt(a)*b + a*Dt(b)     (product rule)
    Dt(sin(u))   -->  cos(u) * Dt(u)        (chain rule)
    Dt(u**n)     -->  n * u**(n-1) * Dt(u)  (power rule)

These two capabilities are inherited together through `GenericDerivativeRuleset`.
The chain rule expansion is what breaks mass conservation. It was not a deliberate
design choice but an unintentional side effect of introducing
`expand_time_derivatives` for the indexing support (per Pablo Brubeck, issue #203).


## How the different steppers handle time derivatives

### DIRK stepper (dirk_stepper.py)

After `expand_time_derivatives` runs, all `TimeDerivative` nodes have been
reduced to `TimeDerivative(u0)`. The stepper substitutes
`{TimeDerivative(u0): k}` to build the stage residual for each stage.

This formulation is stage-by-stage and in derivative form. It fundamentally
requires the chain rule to have already flattened everything to
`TimeDerivative(u0)`. If the chain rule is not applied, `Dt(theta(u0))`
stays as `TimeDerivative(theta(u0))` and the substitution misses it.

### Stage derivative stepper (stage_derivative.py)

Same issue as DIRK. It substitutes `{TimeDerivative(u0): dtusub}` directly.
Composite `TimeDerivative` nodes cannot be handled this way.

### CPG stepper (galerkin_stepper.py)

Same issue. Substitutes `{TimeDerivative(u0): dtu0sub / dt}` at quadrature
points.

### Stage value stepper (stage_value.py)

This stepper works differently. It solves for stage *values* (the solution at
collocation points) rather than stage derivatives. The time derivative handling
goes through `extract_terms` + `strip_dt_form`:

1. `extract_terms(F)` splits the form into integrals containing `TimeDerivative`
   and those that don't.
2. `strip_dt_form(split_form.time)` removes the `TimeDerivative` wrapper,
   giving the naked expression inside.
3. The stripped expression is substituted with stage values.

Currently (line 118), the substitution for the time terms is:

    repl = {u0: as_tensor(w_np[i]) - u0}

This gives `g(w[i] - u0)`, which for linear `g(u) = u` yields the finite
difference `w[i] - u0`. But for nonlinear g, `g(w[i] - u0)` is NOT the same
as `g(w[i]) - g(u0)`. The subtraction trick relies on linearity.

### DG stepper (discontinuous_galerkin_stepper.py)

Similar to stage value: uses `extract_terms` + `strip_dt_form`. Has the same
subtraction-trick issue in both the jump terms (line 78: `u0: u_at_0 - u0`)
and interior quadrature terms (line 86: `u0: dtu0sub[q] / dt`).


## The approach: fix the stage value stepper

This is based on Rob Kirby's suggestion (issue #203 and Slack discussion).

The stage value stepper is the right place to solve this. It already separates
time-derivative terms from spatial terms. It just needs two changes:

1. Skip the chain rule for time-derivative terms.
2. Use a two-evaluation substitution instead of the subtraction trick.

After these changes, `Dt(theta(h))` with `stage_type="value"` will give
exact mass conservation, because the finite difference `theta(U_i) - theta(u0)`
evaluates the nonlinear function at the stage values directly.

Users who want conservation with a DIRK tableau should use
`stage_type="value"` instead of `stage_type="dirk"`. With PETSc configured
for a block lower-triangular solve, the stage value solver can still exploit
the triangular structure of the DIRK Butcher matrix to solve stage-by-stage.


## What needs to change

### Change 1: Fix terminal check in extract_terms (manipulation.py)

File: `irksome/ufl/manipulation.py`, function `_check_timederiv`, line 82.

`extract_terms` calls `check_integrals`, which calls `_check_timederiv` for
each `TimeDerivative` node. This function traverses the terminals of the
operand and requires exactly one non-MultiIndex terminal that is a
`Coefficient`:

    terminals = tuple(set([x for x in traverse_unique_terminals(op)
                           if not isinstance(x, MultiIndex)]))
    if len(terminals) != 1 or not isinstance(terminals[0], Coefficient):
        raise ValueError("Time derivative must apply to a single coefficient")

For `Dt(h)` (after chain rule), the only terminal is `h`. This passes.

For `Dt(theta(h))` (without chain rule), the terminals include `h` PLUS
the float constants in the theta expression (THETA_R, THETA_S, ALPHA).
The check fails because `len(terminals) != 1`.

Fix: filter to `Coefficient` terminals only:

    coefficients = tuple(set([x for x in traverse_unique_terminals(op)
                              if isinstance(x, Coefficient)]))
    if len(coefficients) != 1:
        raise ValueError("Time derivative must apply to a single coefficient")
    return coefficients

This preserves the intent (one prognostic variable under each Dt) while
allowing constants in the expression. It is backward compatible: after
chain rule expansion, `Dt(h)` has one Coefficient terminal `h`, same as
before.


### Change 2: Skip chain rule in stage value stepper (stage_value.py)

File: `irksome/stage_value.py`, function `getFormStage`, line 79.

Currently:

    F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))

Replace with:

    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
    F = apply_algebra_lowering(F)

This keeps the algebra lowering (expanding `inner`, `div`, etc.) but skips
the chain rule. The rest of the pipeline -- `extract_terms`, `strip_dt_form`,
and the substitutions -- works on the lowered form without needing the chain
rule.

Why this is safe:
- `extract_terms` separates integrals containing `TimeDerivative` from those
  that don't. `TimeDerivative` is not touched by `apply_algebra_lowering`.
- `strip_dt_form` removes `TimeDerivative` wrappers regardless of what is
  inside them.
- The remainder terms (spatial) contain no `TimeDerivative` nodes, so
  `apply_time_derivatives` on them would be a no-op anyway.
- For mixed function spaces, `Dt(u[i])` does NOT need the Dt-through-Indexed
  push because the stage value stepper substitutes `u0` (the full Function),
  and UFL propagates the substitution through all indexed accesses.


### Change 3: Two-evaluation substitution for time terms (stage_value.py)

File: `irksome/stage_value.py`, function `getFormStage`, lines 115-119.

Currently:

    for i in range(num_stages):
        repl = {t: t + c[i] * dt,
                v: A2invTv[i],
                u0: as_tensor(w_np[i]) - u0}
        Fnew += replace(F_dtless, repl)

This gives `g(w[i] - u0)`. For nonlinear g, we need `g(w[i]) - g(u0)`.

Replace with:

    for i in range(num_stages):
        repl_new = {t: t + c[i] * dt,
                    v: A2invTv[i],
                    u0: as_tensor(w_np[i])}
        repl_old = {v: A2invTv[i]}
        Fnew += replace(F_dtless, repl_new) - replace(F_dtless, repl_old)

This evaluates the stripped expression at U_i (the stage value) and at u0
(the initial state), then takes the difference. For linear g(u) = u, this
gives `w[i] - u0`, same as before. For nonlinear g, this gives the
conservative form `g(w[i]) - g(u0)`.

The `repl_old` substitution does not set `u0` (it stays as the initial-state
Function) or `t` (it stays at the beginning of the step). It only sets `v`
so that the test function weighting is consistent between the two evaluations.

Cost: this builds twice as many UFL sub-expressions for the time terms. For
forms where the time derivative is linear (the common case), this is
unnecessary overhead. A linearity check could be added to use the single-
substitution path when possible, but that is an optimisation for later.


### Change 4: Fallback for other steppers

The DIRK, stage derivative, and CPG steppers continue to call
`expand_time_derivatives` and use the chain rule. No changes needed: they
work as before for `Dt(u)`, and they produce the linearised (non-conservative)
form for `Dt(g(u))`.

If a user writes `Dt(theta(h))` and uses `stage_type="dirk"`, the chain rule
expands it to `theta'(h) * Dt(h)`. The stepper works, but mass is not
conserved exactly. This matches the current behaviour.

A future improvement would be to detect composite `TimeDerivative` nodes in
these steppers and emit a warning suggesting `stage_type="value"` for mass
conservation. But this is not required for the initial implementation.


## Jacobian computation

The conservative mass term `(theta(U_i) - theta(u0)) / dt` depends on U_i
(the stage value being solved for) nonlinearly through theta. Firedrake
computes the Jacobian automatically via UFL differentiation:

    d/dU_i [theta(U_i) - theta(u0)] = theta'(U_i)

This is computed correctly by UFL's AD. No special handling needed.


## Verification: why the two-evaluation form conserves mass

For a stiffly accurate method (where the last stage value equals the
solution at the next time step), the update is simply `u_new = U_s`.
The total time-derivative contribution over all stages is:

    sum_i (A2inv)_{ij} * [g(U_i) - g(u0)]

summed with the appropriate test function weights. The A2inv weighting
ensures this is consistent with the collocation conditions.

For backward Euler (single stage, A = [1], A2inv = [1]):

    g(U_1) - g(u0) = g(u_new) - g(u0)

This is the exact finite difference. No linearisation error. The change
in total g (moisture content) exactly equals what the spatial operator
prescribed. Hence exact mass conservation (to solver tolerance).


## Using DIRK tableaux with stage_type="value"

The stage value solver accepts any Butcher tableau. When a DIRK tableau
(lower-triangular A) is used, the coupled stage system has a block
lower-triangular structure. With PETSc configured for a block
lower-triangular preconditioner, the solve can proceed stage-by-stage,
recovering the efficiency of the DIRK stepper:

    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
    }

This needs testing to find the right PETSc configuration.


## Summary of changes

1. `irksome/ufl/manipulation.py`:
   - Fix `_check_timederiv` to filter terminals to `Coefficient` only.

2. `irksome/stage_value.py`:
   - Replace `expand_time_derivatives` with `apply_algebra_lowering`.
   - Change time-term substitution from single (subtraction trick) to
     two evaluations (conservative finite difference).

3. `tests/test_mass_conservation.py`:
   - Self-contained Richards test with exponential soil model.
   - Checks cumulative mass balance error < 1e-10 with stage_type="value".


## Future work

- DG stepper: similar two-evaluation fix for jump terms and interior
  quadrature terms. The DG formulation can also use integration by parts
  in time to naturally produce conservative boundary terms.

- Stage derivative / CPG fallback: detect composite `TimeDerivative` nodes,
  apply the chain rule as a fallback, emit a warning recommending
  `stage_type="value"`.

- Mixed function space test: verify that `Dt(theta(u[0]))` works correctly
  when u is a mixed Function.

- Mixed Dt forms: test that a form with both `Dt(u)` (linear mass) and
  `Dt(theta(u))` (nonlinear mass) works correctly.

- Linearity optimisation: detect when the time-derivative expression is
  linear in u0 and use the cheaper single-substitution path.

# Conservative Dt via stage value and DG steppers

This document captures the alternative implementation strategy suggested by
Rob Kirby (Irksome maintainer) in GitHub issue #203 and Slack discussion
(2026-04-01). It is written for an engineer picking up the work cold.

Previous design doc: CONSERVATIVE_DT_DESIGN.md (describes the current
`conservative-dt` branch approach using b-weighted finite differences inside
the DIRK stepper).

Tracking issue: https://github.com/firedrakeproject/Irksome/issues/203


## The short version

Rob's suggestion is: don't fix the DIRK stepper.  Instead, stop applying the
chain rule in `expand_time_derivatives` and modify the stage value and DG
steppers to handle `Dt(g(u))` natively.  Users who need conservation with a
DIRK tableau should feed it to the stage value solver (`stage_type="value"`)
rather than the DIRK solver (`stage_type="dirk"`).


## Why not the DIRK stepper

The DIRK stepper solves for stage *derivatives* k_i one at a time.  After
`expand_time_derivatives` applies the chain rule, every `TimeDerivative` node
has been reduced to `TimeDerivative(u0)`.  The stepper then substitutes
`{TimeDerivative(u0): k}` to build the stage residual.

If we stop applying the chain rule, `Dt(g(u0))` stays as
`TimeDerivative(g(u0))` and that substitution misses it.  There is no clean
way to substitute a stage derivative into a composite time derivative without
either (a) re-applying the chain rule or (b) constructing finite differences
with b-weighted intermediate states (what the `conservative-dt` branch does).
Option (b) works but adds significant complexity to a stepper that was not
designed for it, and it forces order reduction because the mass and spatial
terms end up evaluated at different states (b-weighted vs a-weighted).

Rob says the DIRK stepper is fundamentally the wrong place to solve this
problem.  The stage value and DG formulations handle it more naturally.


## Why stage value and DG work

Both the stage value stepper (`stage_value.py`) and the DG stepper
(`discontinuous_galerkin_stepper.py`) solve for stage *values* (the solution
at collocation/quadrature points) rather than stage derivatives.  They use
`extract_terms` + `strip_dt_form` to separate out the time-derivative terms
and handle them differently from the spatial terms.

The key difference from DIRK: these steppers never need to substitute a stage
derivative k into a `TimeDerivative` node.  Instead, they strip the `Dt`
wrapper and perform a substitution on the naked expression.  This means if we
can get the substitution right for `g(u0)` (the stripped form of `Dt(g(u0))`),
the stepper naturally evaluates g at the stage values without needing the
chain rule.


## What the current substitution does (and why it breaks for nonlinear g)

Both steppers use a subtraction trick for the time-derivative terms.  After
`strip_dt_form` removes the `Dt` wrapper, the substitution replaces `u0` with
a difference:

Stage value (`stage_value.py` line 118):

    repl = {u0: as_tensor(w_np[i]) - u0}

DG jump term (`discontinuous_galerkin_stepper.py` line 78):

    repl = {u0: u_at_0 - u0}

For `Dt(u)` (linear mass), stripping Dt gives `u`.  Substituting `u0` with
`w[i] - u0` gives `w[i] - u0`, representing the finite difference
`(U_i - u0) / dt`.  Correct.

For `Dt(g(u))` without chain rule expansion, stripping Dt gives `g(u)`.
Substituting `u0` with `w[i] - u0` gives `g(w[i] - u0)`.  But the
conservative form requires `g(w[i]) - g(u0)`.  These are NOT the same
for nonlinear g.  The substitution trick relies on linearity.

The DG interior terms have the same problem.  The substitution
`u0: dtu0sub[q] / dt` (line 86) puts the polynomial time derivative into
the argument of g, giving `g(u'(t_q) / dt)` instead of
`d/dt g(u(t_q))`.


## What needs to change

### Part 1: Stop the chain rule in expand_time_derivatives

This is the UFL-layer change that Pablo originally suggested.  Override the
nonlinear composition handlers inherited from `GenericDerivativeRuleset` in
`TimeDerivativeRuleset` so that `Dt(g(u))` is NOT expanded to `g'(u)*Dt(u)`.

The `TimeDerivativeRuleset` class (`irksome/ufl/deriv.py` line 42) inherits
from `GenericDerivativeRuleset` which provides handlers for `Product`,
`Division`, `Power`, `Sin`, `Cos`, `Exp`, `Ln`, and all other nonlinear
operations.  These handlers implement the product rule, chain rule, etc.
They need to be overridden to NOT differentiate through nonlinear
compositions.

Rules to KEEP (linearity):
- `Sum`: `Dt(a + b) = Dt(a) + Dt(b)`
- Scalar multiplication by constants: `Dt(c * u) = c * Dt(u)`

Rules to KEEP (spatial operator pass-through, line 81-92):
- `Grad`, `Div`, `Curl`, `Indexed`, `ReferenceGrad`, `ReferenceValue`,
  `Variable`, `Conj`: `Dt(Grad(u)) = Grad(Dt(u))`

Rules to OVERRIDE (stop the chain rule):
- `Product` (when both operands depend on time)
- `Division`, `Power`
- All transcendentals: `Sin`, `Cos`, `Tan`, `Exp`, `Ln`, `Sqrt`, etc.

For the overridden rules, the handler should reconstruct the original
(undifferentiated) expression and wrap it in a `TimeDerivative`.

Subtlety with `Product`: `Dt(c * u)` where c is a constant should still
give `c * Dt(u)`.  Only `Dt(a * b)` where BOTH a and b are time-dependent
should be left unexpanded.  The `GenericDerivativeRuleset` already tracks
which operands are "zero derivatives" (constants); the override should
check this and only block expansion when both operands carry nonzero
derivatives.

Subtlety with postorder traversal: `GenericDerivativeRuleset` handlers use
`@DAGTraverser.postorder`, meaning operands are recursively differentiated
before the node handler runs.  When overriding to NOT apply the chain rule,
the handler must NOT use `@DAGTraverser.postorder`.  Instead, it should
reconstruct the original node from its original (unprocessed) operands and
wrap the result in `TimeDerivative`.  This prevents the recursion from
differentiating sub-expressions that we want to keep intact.

A cleaner alternative (suggested by Pablo in issue #203): write a separate
`DAGTraverser` that only pushes `Dt` through spatial/indexing operators
(`Grad`, `Indexed`, etc.) and leaves everything else alone.  This avoids
the complexity of selectively overriding inherited handlers and gives a
clear two-pass design:
  Pass 1 (new traverser): `Dt(Grad(g(u[0])))` becomes `Grad(Dt(g(u[0])))`
  Pass 2 (existing strip_dt + substitution): handled by each stepper

The `expand_time_derivatives` function (`deriv.py` line 126) currently calls
`apply_algebra_lowering` then `apply_time_derivatives`.  The modified version
would call `apply_algebra_lowering` then the new spatial-only Dt pusher.


### Part 2: Modify stage value stepper substitution

File: `irksome/stage_value.py`, function `getFormStage`, lines 107-119.

Currently the time-derivative terms are handled with a single substitution
that replaces `u0` with a difference:

    repl = {u0: w_np[i] - u0}  -->  gives g(w[i] - u0)  (WRONG for nonlinear g)

The fix: detect whether the stripped time-derivative expression is nonlinear
in u0, and if so, use TWO evaluations instead of the difference trick.

For nonlinear `Dt(g(u))`, after stripping Dt we have `g(u)`.  The
conservative finite difference is:

    (g(w[i]) - g(u0)) / dt

This requires two substitutions of `F_dtless`:
1. `repl_new = {u0: w_np[i], v: A2invTv[i]}`  -->  gives `g(w[i])`
2. `repl_old = {u0: u0, v: A2invTv[i]}`        -->  gives `g(u0)`
3. Contribution = replace(F_dtless, repl_new) - replace(F_dtless, repl_old)

For linear `Dt(u)` this gives `w[i] - u0`, same as before.  For nonlinear
`Dt(g(u))` this gives `g(w[i]) - g(u0)`, the conservative form.  So we can
use the two-evaluation approach unconditionally without needing to detect
linearity.

But there is a cost: we now build twice as many UFL expressions for the time
terms.  For forms where the time derivative is linear (the common case), this
is unnecessary overhead.  If that matters, add a linearity check: inspect
whether `F_dtless` is linear in `u0` (for instance, traverse the expression
tree and check if u0 appears inside any nonlinear operator).  If linear, use
the existing single-substitution trick.  If nonlinear, use the
two-evaluation approach.

Rob's comment on the issue hints at this: "We need to split that term off
and just replace it with something like inner(g(u), v) * dx evaluated at
the endpoints of the time interval."


### Part 3: Modify DG stepper substitution

File: `irksome/discontinuous_galerkin_stepper.py`, function
`getTermDiscGalerkin`, lines 69-87.

The DG stepper has TWO places that handle time-derivative terms:

(a) Jump terms (lines 74-80): The jump at the left boundary of the time
    step is `[g(u)]_{jump} = g(u(0+)) - g(u_old)`.  Currently the
    substitution `u0: u_at_0 - u0` gives `g(u_at_0 - u0)`.  Fix: use
    two evaluations, `g(u_at_0) - g(u0)`.

(b) Interior quadrature terms (lines 82-87): Currently substitutes
    `u0: dtu0sub[q] / dt`, which puts the polynomial time derivative
    into g.  For nonlinear g this is nonsensical.

    For the DG formulation, the correct treatment of `Dt(g(u))` in the
    interior integral depends on whether we use integration by parts in
    time.  Without IBP, the interior integral is:

        sum_q w_q * inner(d/dt g(u(t_q)), v(t_q)) * dt * dx

    For nonlinear g, `d/dt g(u(t_q)) = g'(u(t_q)) * u'(t_q)`, which IS
    the chain rule.  If we want to avoid the chain rule entirely, we need
    IBP in time:

        int inner(Dt(g(u)), v) dt = [inner(g(u), v)] - int inner(g(u), Dt(v)) dt

    The boundary term gives `g(u(1)) * v(1) - g(u(0)) * v(0)` (conservative).
    The interior integral involves `g(u(t_q))` weighted by `v'(t_q)` (no
    chain rule needed).

    This is a more significant restructuring of the DG stepper's time
    derivative handling.  It may be easier to start with the stage value
    stepper and handle DG as a follow-up.


### Part 4: Stage derivative and CPG steppers

These steppers substitute `{TimeDerivative(u0): stage_derivative}` directly,
similar to DIRK.  They cannot handle composite `TimeDerivative(g(u0))` nodes
without the chain rule.

Rob's suggestion (from the issue): for these steppers, if the form contains
composite `TimeDerivative` nodes after the modified `expand_time_derivatives`,
apply the chain rule to those nodes as a fallback and issue a warning.

    "For stage derivative, I think we might try to do the substitution
    rule and if compilation fails (Dt still present) we splat out
    derivatives and issue some kind of a warning that we're rewriting
    a time derivative and if this breaks mass conservation or gives
    other undesirable numerical properties that the user switch to
    stage_type='value'."

Implementation: after calling the modified `expand_time_derivatives` (which
no longer chain-rules), scan the form for `TimeDerivative` nodes whose
operand is not simply `u0`.  If found, apply the chain rule to just those
nodes (call the old `GenericDerivativeRuleset` on them) and emit a warning
recommending `stage_type="value"` for mass conservation.


### Part 5: Using DIRK tableaux with the stage value solver

Users who want both a DIRK tableau (for efficiency) and conservative mass
terms should use `stage_type="value"` instead of `stage_type="dirk"`.  The
stage value solver accepts any Butcher tableau, not just fully implicit ones.

When a DIRK tableau (lower-triangular A) is fed to the stage value solver,
the resulting coupled system has a block lower-triangular structure.  With
PETSc configured for a block lower-triangular preconditioner (fieldsplit
with multiplicative composition), the solve can still proceed
stage-by-stage, recovering the efficiency of the DIRK stepper.

This is what Rob means by "Unless you feed a DIRK tableau to stage value
form, don't use 'dirk' stage type, and tweak PETSc."

The relevant PETSc options would look something like:

    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
    }

This needs experimentation to find the right configuration.


### Part 6: Clean up the conservative-dt branch

Once the stage value path is working, the `ConservativeDt` /
`ConservativeTimeDerivative` machinery on the current branch becomes
unnecessary.  Remove:

- `ConservativeTimeDerivative` class from `deriv.py`
- `ConservativeDt` function from `deriv.py`
- `check_no_conservative_dt` function from `deriv.py`
- `_replace_conservative_dts` function from `dirk_stepper.py`
- All `W_prev` / `b_val` accumulation logic from `dirk_stepper.py`
- `ConservativeDt` export from `__init__.py`
- `check_no_conservative_dt` calls from `stage_derivative.py` and
  `stage_value.py`


## Suggested implementation order

1. Modify `expand_time_derivatives` to stop the chain rule (Part 1).
   This is the foundation everything else depends on.

2. Modify the stage value stepper substitution (Part 2).  This is the
   most important stepper to get right and the simplest change.

3. Add chain-rule fallback to stage derivative and CPG (Part 4).
   This preserves backward compatibility for those steppers.

4. Write tests: a Richards-like problem with nonlinear mass term,
   comparing mass conservation with `stage_type="value"` vs
   `stage_type="dirk"`.  Include convergence rate tests.

5. DG stepper changes (Part 3) can be done as a follow-up.

6. Clean up the conservative-dt branch (Part 6).

7. Document the DIRK+stage_value workflow with PETSc options (Part 5).


## Key test: Richards equation mass balance

The definitive test is the same one from CONSERVATIVE_DT_DESIGN.md.  On a
unit square with DQ1 elements, exponential soil model theta(h) = exp(alpha*h),
200 timesteps of backward Euler or implicit midpoint:

1. Compute the total moisture: integral of theta(h) over the domain.
2. Compare final_moisture - initial_moisture against the cumulative flux
   through the boundaries.
3. The difference is the mass balance error.

With the chain-rule form (current default), mass balance error is O(1e-10).
With the conservative form, mass balance error should be O(1e-12) or better
(limited only by solver tolerance).

Run this test with `stage_type="value"` and a DIRK tableau (e.g.
BackwardEuler, Alexander(2)).  Verify that mass balance is at machine
precision.


## What Rob said, verbatim

From the GitHub issue:

> "My understanding of the maths is that stage value and DG-in-time
> formulation should either work or could be extended relatively easily if
> not to support Dt(g(u)) but that stage derivative formulation and CPG
> wouldn't."

> "For stage-value form, the mathematics can handle inner(Dt(g(u), v) * dx
> if it's by itself.  We need to split that term off and just replace it
> with something like inner(g(u), v) * dx evaluated at the endpoints of the
> time interval.  I think what we've got might actually do the right thing
> if we avoided splatting out the Dt with a u in it?"

> "For stage derivative, I think we might try to do the substitution rule
> and if compilation fails (Dt still present) we splat out derivatives and
> issue some kind of a warning that we're rewriting a time derivative and
> if this breaks mass conservation or gives other undesirable numerical
> properties that the user switch to stage_type='value'."

From Slack:

> "if you're ok with fully implicit schemes, I think the maths work out
> fine (code to be tested) if we can just avoid expanding time derivatives
> and use stage value or DG steppers."

> "CPG and stage derivative won't work.  It looks like our DIRK
> implementation is stage-by-stage but in derivative form, so I expect DIRK
> to be broken as well.  (Unless you feed a DIRK tableau to stage value
> form, don't use 'dirk' stage type, and tweak PETSc)."

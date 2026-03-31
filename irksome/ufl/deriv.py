from functools import singledispatchmethod

from ufl.constantvalue import as_ufl
from ufl.core.ufl_type import ufl_type
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.apply_derivatives import GenericDerivativeRuleset
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.form import BaseForm
from ufl.classes import (Coefficient, Conj, Curl, ConstantValue, Derivative,
                         Div, Expr, Grad, Indexed, ReferenceGrad,
                         ReferenceValue, SpatialCoordinate, Variable)


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class TimeDerivative(Derivative):
    """UFL node representing a time derivative of some quantity/field.
    Note: Currently form compilers do not understand how to process
    these nodes.  Instead, Irksome pre-processes forms containing
    `TimeDerivative` nodes."""
    __slots__ = ()

    def __new__(cls, f):
        return Derivative.__new__(cls)

    def __init__(self, f):
        Derivative.__init__(self, (f,))

    def __str__(self):
        return "d{%s}/dt" % (self.ufl_operands[0],)


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class ConservativeTimeDerivative(Derivative):
    """UFL node representing a conservative (finite-difference) time derivative.

    Unlike :class:`TimeDerivative`, this node is *not* expanded via the chain
    rule during :func:`expand_time_derivatives`.  Instead, DIRK steppers
    discretise it as a finite difference of the operand expression:

        ConservativeDt(f(u))  -->  (f(u_new) - f(u_old)) / dt

    This preserves the nonlinear structure of ``f`` and is essential for
    mass-conservative discretisations of equations like Richards', where the
    accumulated quantity (e.g. moisture content theta) is a nonlinear function
    of the prognostic variable (e.g. pressure head h).

    For multi-stage DIRK methods, the finite difference is constructed using
    cumulative b-weighted update points W_i = u0 + dt * sum_{j<=i} b_j * K_j,
    giving a per-stage term (f(W_i) - f(W_{i-1})) / (b_i * dt) whose
    telescoping sum yields exact global conservation:
    f(u_new) - f(u_old) = sum_i (f(W_i) - f(W_{i-1})).
    """
    __slots__ = ()

    def __new__(cls, f):
        return Derivative.__new__(cls)

    def __init__(self, f):
        Derivative.__init__(self, (f,))

    def __str__(self):
        return "d_conservative{%s}/dt" % (self.ufl_operands[0],)


def Dt(f, order=1):
    """Short-hand function to produce a :class:`TimeDerivative` of a given order."""
    for k in range(order):
        f = TimeDerivative(f)
    return f


def ConservativeDt(f):
    """Short-hand for a :class:`ConservativeTimeDerivative`.

    Use this instead of :func:`Dt` when the time derivative wraps a nonlinear
    function of the prognostic variable and mass conservation is required.

    Example::

        # Standard (chain rule): Dt(theta(h)) expands to C(h)*Dt(h)
        # Conservative (finite diff): ConservativeDt(theta(h)) stays intact
        F = inner(v, ConservativeDt(theta(h))) * dx + spatial_terms
    """
    return ConservativeTimeDerivative(f)


class TimeDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AD rules to time derivative expressions."""
    def __init__(self, t=None, timedep_coeffs=None):
        GenericDerivativeRuleset.__init__(self, ())
        self.t = t
        self._Id = as_ufl(1.0)
        self.timedep_coeffs = timedep_coeffs

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(ConstantValue)
    def constant(self, o):
        if self.t is not None and o is self.t:
            return self._Id
        else:
            return self.independent_terminal(o)

    @process.register(Coefficient)
    @process.register(SpatialCoordinate)
    def terminal(self, o):
        if self.t is not None and o is self.t:
            return self._Id
        elif self.timedep_coeffs is None or o in self.timedep_coeffs:
            return TimeDerivative(o)
        else:
            return self.independent_terminal(o)

    @process.register(TimeDerivative)
    @DAGTraverser.postorder
    def time_derivative(self, o, f):
        if isinstance(f, TimeDerivative):
            return TimeDerivative(f)
        else:
            return self(f)

    @process.register(Conj)
    @process.register(Curl)
    @process.register(Derivative)
    @process.register(Div)
    @process.register(Grad)
    @process.register(Indexed)
    @process.register(ReferenceGrad)
    @process.register(ReferenceValue)
    @process.register(Variable)
    @DAGTraverser.postorder
    def terminal_modifier(self, o, *operands):
        return o._ufl_expr_reconstruct_(*operands)


class TimeDerivativeRuleDispatcher(DAGTraverser):
    '''
    Mapping rules to splat out time derivatives so that replacement should
    work on more complex problems.
    '''
    def __init__(self, t=None, timedep_coeffs=None, **kwargs):
        super().__init__(**kwargs)
        self.rules = TimeDerivativeRuleset(t=t, timedep_coeffs=timedep_coeffs)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(TimeDerivative)
    def time_derivative(self, o):
        f, = o.ufl_operands
        return self.rules(f)

    @process.register(ConservativeTimeDerivative)
    def conservative_time_derivative(self, o):
        # Do NOT apply the chain rule.  Leave intact for the DIRK stepper
        # to replace with a finite difference.
        return o

    @process.register(Expr)
    @process.register(BaseForm)
    def _generic(self, o):
        return self.reuse_if_untouched(o)


def apply_time_derivatives(expression, t=None, timedep_coeffs=None):
    rules = TimeDerivativeRuleDispatcher(t=t, timedep_coeffs=timedep_coeffs)
    return map_integrands(rules, expression)


def expand_time_derivatives(expression, t=None, timedep_coeffs=None):
    expression = apply_algebra_lowering(expression)
    expression = apply_time_derivatives(expression, t=t, timedep_coeffs=timedep_coeffs)
    return expression


def check_no_conservative_dt(F):
    """Raise NotImplementedError if the form contains ConservativeTimeDerivative.

    Call this from non-DIRK steppers that do not support ConservativeDt."""
    from ufl.algorithms.analysis import extract_type
    if extract_type(F, ConservativeTimeDerivative):
        raise NotImplementedError(
            "ConservativeDt is only supported with DIRK time steppers "
            "(DIRKTimeStepper).  Use standard Dt for stage-coupled, "
            "Galerkin, or explicit methods."
        )

# Mixed sample-weight curvature

`weight_curvature` computes the mixed parameter and query-loss derivatives when two supplied directions change the training-row weights. It refits the original positive-ridge logistic objective once, then solves three systems with that fit's Hessian. It does not build a third-order tensor or use finite differences in production.

This is numerical implicit differentiation at a converged fit, not exact arithmetic. The full fit, per-row contractions, three right-hand sides, parameter derivatives, query-loss decomposition and solve residuals are returned.

## Equations

Keep the normalizer `Z` and ridge `lambda` fixed. For supplied weights `w` and directions `u,v`,

```text
F(theta,s,t) = sum_i (w_i+s*u_i+t*v_i) loss_i(theta)/Z
               + lambda ||theta||^2 / 2.
```

All features, including an explicit constant column, are regularized. Let `theta` be the converged stationary point at `s=t=0`, `H` its full ridge Hessian, `g_i` a row loss gradient and `C_i` its Hessian. Differentiating the stationary equation gives

```text
H a = -sum_i u_i g_i / Z       a = dtheta/ds
H b = -sum_i v_i g_i / Z       b = dtheta/dt.
```

The mixed parameter derivative `c=d²theta/(ds dt)` satisfies

```text
H c = -sum_i [w_i T_i[a,b] + u_i C_i b + v_i C_i a] / Z.
```

Both weight-Hessian terms are needed. This is not obtained by applying the original inverse Hessian twice while freezing its derivatives. Ridge contributes to `H` but has no third derivative and no explicit weight derivative.

For one logistic row, with feature vector `x_i` and `p_i=sigmoid(theta.x_i)`,

```text
k_i = p_i(1-p_i)
C_i b = k_i x_i (x_i.b)
T_i[a,b] = k_i(1-2p_i) x_i (x_i.a)(x_i.b).
```

The implementation computes `1-2p_i` as `-tanh((theta.x_i)/2)` to avoid subtracting a rounded probability from one at extreme logits. The existing stable logistic routine supplies loss, gradient and `k_i`.

For the supplied unregularized query loss `q(theta)`,

```text
d²q/(ds dt) = a^T Hessian(q) b + grad(q).c.
```

The report separates these two terms. Dropping the parameter-refit curvature term is not the mixed derivative of the retrained query loss.

Positive ridge makes `H` positive definite for the supported nonnegative base weights. Thus the stationary equation has a local smooth implicit branch. For a base weight on a boundary, the report is that branch's derivative; actual nonnegative-weight changes may only be feasible on one side. No negative-weight fit is performed or promised. The base fit remains the existing bounded nonnegative-weight convex fit. No claim is made about a large signed perturbation remaining feasible or convex.

## A negative-curvature control

The caller fixture reuses the original seven-row developed example, query `(1,2)` with label one, all-one weights, normalizer seven and ridge `0.1`. Its left direction deletes row 5 and its right direction deletes row 6, locally. The mixed query derivative is about `-0.063551309`:

| Contribution | Value |
|---|---:|
| Query Hessian paired with both parameter directions | -0.084606281 |
| Query gradient paired with mixed parameter derivative | +0.021054972 |
| Sum | -0.063551309 |

A separate directional control uses row 6 deletion for both directions:

| Contribution | Value |
|---|---:|
| Query Hessian term | +0.146986244 |
| Parameter-refit curvature term | -0.348215167 |
| Full second directional query derivative | -0.201228923 |

The query loss is convex in the model parameters, but its composition with retraining weights need not be convex. The negative second derivative above is local and numerical. It is not the effect of deleting a full row, a proof about arbitrary data, a causal interaction or an unlearning result.

## Run

```python
from influencelab import weight_curvature

report = weight_curvature(
    [[1], [2]], [0, 1], [1], 1,
    left=[1, 0], right=[0, 1], ridge=.1,
)
assert report['fit']['converged']
```

```sh
python -m influencelab curvature examples/curvature-input.json
python examples/verify_wheel.py
```

The JSON request has exactly `version`, `X`, `y`, `query`, `label`, `left`, `right`, `ridge`, `weights` and `normalizer`. Version is integer one. The CLI requires explicit weights and normalizer; the Python defaults are all-one weights and the original row count. Existing feature, row-count, ridge, weight and normalizer bounds apply. Each direction is a finite numeric vector of row count, with entries in `[-10,10]`. Booleans, nonfinite values, duplicate or unknown JSON fields and input above 100,000 bytes are rejected. A JSON parser-recursion failure is a controlled input error.

The normalizer is not recomputed from perturbed weights. Neither ridge nor query parameters vary along the weight directions. No damping approximation is introduced.

## Checks and limits

Twelve added tests include:

- An independent 65-digit Decimal scalar optimizer using 240 monotone bisection steps, followed by four-corner mixed differences at step `1e-8`. Eighteen models check the mixed parameter and query-loss derivatives without using production Newton, Hessian, third-contraction or implicit-derivative code.
- Twelve seeded models of width one through three check every mixed parameter coordinate and the query derivative against two-sided converged refits. These are numerical comparisons with stated test tolerances, not symbolic proofs or certified error bounds.
- Mixed symmetry, bilinearity, polarization, weight/normalizer scaling, duplicate rows, label complements, row permutations and orthogonal coordinates.
- Every per-row contraction sum and linear-solve residual, both query terms, negative directional curvature, a local second-order prediction check, zero directions/features/base weights, maximum supported dimensions, extreme logits, strict CLI validation, input isolation and unchanged global RNG state.

The full suite has 55 tests. The wheel verifier runs every test outside the checkout, checks all six package origins before and after, compares all five complete module CLI reports and regenerates both original figures. The earlier four reports and two SVG files are unchanged.

The chain rule and implicit derivatives here are standard, not a new influence algorithm. [DESIGN.md](DESIGN.md) and the influence-function references in the README provide the original first-order context. This extension adds an inspectable second-order calculation for the same finite convex model. It does not add statistical confidence intervals, guaranteed finite-deletion accuracy, scalable tensor methods, neural-network influence, causal attribution, privacy deletion or machine unlearning. Floating-point solve residuals measure numerical consistency, not statistical uncertainty.

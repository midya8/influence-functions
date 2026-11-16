"""Mixed sample-weight derivatives at a converged ridge-logistic fit.

Contract third derivatives with two directions; no third-order tensor or
finite differences are used in production. The normalizer and ridge stay fixed.
"""
import math
from .core import dot, fit, number, point, rows, solve


def weight_curvature(x, y, query, label, left, right, ridge=0.1,
                     weights=None, normalizer=None):
    x, y = rows(x, y)
    q, labels = rows([query], [label])
    if len(q[0]) != len(x[0]):
        raise ValueError('query dimensions must match')
    n, d = len(x), len(x[0])

    def direction(values):
        if not isinstance(values, (list, tuple)) or len(values) != n:
            raise ValueError('direction must align with training rows')
        return [number(v, 'weight direction', -10, 10) for v in values]

    u, v = direction(left), direction(right)
    model = fit(x, y, ridge=ridge, weights=weights, normalizer=normalizer)
    theta, h, z = model['theta'], model['hessian'], model['normalizer']
    pieces = [point(theta, row, yi) for row, yi in zip(x, y)]
    rhs_u = [-math.fsum(a * p[1][j] / z for a, p in zip(u, pieces)) for j in range(d)]
    rhs_v = [-math.fsum(a * p[1][j] / z for a, p in zip(v, pieces)) for j in range(d)]
    du, _ = solve(h, rhs_u)
    dv, _ = solve(h, rhs_v)
    witnesses = []
    for i, (row, p) in enumerate(zip(x, pieces)):
        su, sv = dot(row, du), dot(row, dv)
        # Stable tanh avoids computing 1-2*p from a rounded probability.
        third = -p[2] * math.tanh(dot(theta, row) / 2)
        tensor = [model['weights'][i] * third * a * su * sv / z for a in row]
        left_hessian = [u[i] * p[2] * a * sv / z for a in row]
        right_hessian = [v[i] * p[2] * a * su / z for a in row]
        witnesses.append({'row': i, 'sample_gradient': p[1],
                          'sample_curvature': p[2], 'sample_third_logit': third,
                          'left_logit_derivative': su, 'right_logit_derivative': sv,
                          'weighted_third_contraction': tensor,
                          'left_weight_hessian_contraction': left_hessian,
                          'right_weight_hessian_contraction': right_hessian})
    terms = ('weighted_third_contraction', 'left_weight_hessian_contraction',
             'right_weight_hessian_contraction')
    sums = {term: [math.fsum(w[term][j] for w in witnesses) for j in range(d)]
            for term in terms}
    rhs_uv = [-math.fsum(sums[term][j] for term in terms) for j in range(d)]
    mixed, _ = solve(h, rhs_uv)
    loss, gradient, curvature, probability = point(theta, q[0], labels[0])
    direct = curvature * dot(q[0], du) * dot(q[0], dv)
    refit_term = dot(gradient, mixed)

    def residual(direction, rhs):
        return [dot(row, direction) - value for row, value in zip(h, rhs)]

    residuals = {'left': residual(du, rhs_u), 'right': residual(dv, rhs_v),
                 'mixed': residual(mixed, rhs_uv)}
    return {'version': 1, 'fit': model, 'left': u, 'right': v,
            'query': q[0], 'query_label': labels[0], 'query_loss': loss,
            'query_probability': probability, 'query_gradient': gradient,
            'left_rhs': rhs_u, 'right_rhs': rhs_v, 'mixed_rhs': rhs_uv,
            'left_parameter_derivative': du, 'right_parameter_derivative': dv,
            'mixed_parameter_derivative': mixed,
            'left_query_derivative': dot(gradient, du),
            'right_query_derivative': dot(gradient, dv),
            'query_hessian_term': direct, 'query_refit_curvature_term': refit_term,
            'mixed_query_derivative': direct + refit_term,
            'sample_witnesses': witnesses, 'summed_contractions': sums,
            'solve_residuals': residuals,
            'max_abs_solve_residual': max(abs(a) for values in residuals.values() for a in values),
            'objective': 'sum((weights_i+s*left_i+t*right_i)*loss_i)/fixed_normalizer + ridge*||theta||^2/2',
            'scope': 'numerical local mixed implicit derivative at a converged positive-ridge fit; fixed normalizer and ridge; no dense third-order tensor, finite-deletion guarantee or unlearning claim'}

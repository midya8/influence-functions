import copy
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from influencelab import fit
from influencelab.core import point
from influencelab.curvature import weight_curvature

X = [[1, -2], [1, -1], [1, -.5], [1, .5], [1, 1], [1, 2], [1, 3]]
Y = [0, 0, 1, 0, 1, 1, 0]
U = [0, 0, 0, 0, 0, -1, 0]
V = [0, 0, 0, 0, 0, 0, -1]


def scalar_reference(xs, ys, q, label, weights, normalizer, ridge):
    """Independent 65-digit monotone root, not Newton or production derivatives."""
    def derivative(theta):
        total = ridge * theta
        for x, y, w in zip(xs, ys, weights):
            p = 1 / (1 + (-theta * x).exp())
            total += w * (p - y) * x / normalizer
        return total
    lo, hi = Decimal(-100), Decimal(100)
    assert derivative(lo) < 0 < derivative(hi)
    for _ in range(240):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    theta = (lo + hi) / 2
    z = q * theta
    loss = (1 + z.exp()).ln() - label * z
    return theta, loss


class CurvatureTests(unittest.TestCase):
    def report(self, u=U, v=V, **kw):
        return weight_curvature(X, Y, [1, 2], 1, u, v, **kw)

    def test_decimal_scalar_bisection_mixed_partials(self):
        with localcontext() as context:
            context.prec = 65
            h = Decimal('0.00000001')
            for case in range(18):
                xs = [Decimal(-2), Decimal('0.5'), Decimal(1 + case % 3)]
                ys = [Decimal(case % 2), Decimal(1), Decimal(0)]
                weights = [Decimal('0.7'), Decimal('1.2'), Decimal('1.8')]
                u = [Decimal(1), Decimal('-0.3'), Decimal('0.2')]
                v = [Decimal('0.1'), Decimal('-0.6'), Decimal(1)]
                if case % 3 == 0:
                    v = u[:]
                q, label = Decimal('1.3'), Decimal(case % 2)
                ridge, norm = Decimal('0.15') + Decimal(case) / 20, Decimal(4)
                values = {}
                for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    ws = [w + a * h * du + b * h * dv for w, du, dv in zip(weights, u, v)]
                    values[a, b] = scalar_reference(xs, ys, q, label, ws, norm, ridge)
                reference = [(values[1, 1][j] - values[1, -1][j] - values[-1, 1][j] + values[-1, -1][j]) / (4*h*h) for j in range(2)]
                report = weight_curvature([[float(x)] for x in xs], [int(y) for y in ys], [float(q)], int(label), list(map(float, u)), list(map(float, v)), float(ridge), list(map(float, weights)), float(norm))
                self.assertAlmostEqual(report['mixed_parameter_derivative'][0], float(reference[0]), delta=3e-10)
                self.assertAlmostEqual(report['mixed_query_derivative'], float(reference[1]), delta=3e-10)

    def test_all_vector_coordinates_against_two_sided_refits(self):
        random_source = random.Random(79)
        for case in range(12):
            n, d = 3 + case % 4, 1 + case % 3
            x = [[random_source.uniform(-2, 2) for _ in range(d)] for _ in range(n)]
            y = [random_source.randrange(2) for _ in range(n)]
            q = [random_source.uniform(-1, 1) for _ in range(d)]
            u = [random_source.uniform(-1, 1) for _ in range(n)]
            v = [random_source.uniform(-1, 1) for _ in range(n)]
            r = weight_curvature(x, y, q, 1, u, v, ridge=.4)
            h = .002
            values = {}
            for a, b in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                m = fit(x, y, ridge=.4, weights=[1+h*(a*du+b*dv) for du, dv in zip(u,v)], tolerance=1e-12)
                values[a,b] = m['theta'] + [point(m['theta'],q,1)[0]]
            refs = [(values[1,1][j]-values[1,-1][j]-values[-1,1][j]+values[-1,-1][j])/(4*h*h) for j in range(d+1)]
            for actual, expected in zip(r['mixed_parameter_derivative']+[r['mixed_query_derivative']],refs):
                self.assertAlmostEqual(actual,expected,delta=1.5e-6)

    def test_symmetry_bilinearity_and_polarization(self):
        r, swapped = self.report(), self.report(V,U)
        for key in ['mixed_rhs','mixed_parameter_derivative']:
            for a,b in zip(r[key],swapped[key]):self.assertAlmostEqual(a,b,delta=1e-15)
        self.assertAlmostEqual(r['mixed_query_derivative'],swapped['mixed_query_derivative'],delta=1e-15)
        total = self.report([a+b for a,b in zip(U,V)],[a+b for a,b in zip(U,V)])
        uu,vv=self.report(U,U),self.report(V,V)
        self.assertAlmostEqual(total['mixed_query_derivative'],uu['mixed_query_derivative']+2*r['mixed_query_derivative']+vv['mixed_query_derivative'],delta=1e-14)
        scaled=self.report([2*a for a in U],[-3*b for b in V])
        self.assertAlmostEqual(scaled['mixed_query_derivative'],-6*r['mixed_query_derivative'],delta=1e-14)

    def test_all_witness_sums_and_linear_solve_residuals(self):
        r=self.report()
        for term,summed in r['summed_contractions'].items():
            for j,value in enumerate(summed):self.assertEqual(value,math.fsum(w[term][j] for w in r['sample_witnesses']))
        for j in range(2):
            self.assertEqual(r['mixed_rhs'][j],-math.fsum(values[j] for values in r['summed_contractions'].values()))
        self.assertLess(r['max_abs_solve_residual'],1e-15)
        self.assertLess(r['fit']['max_abs_gradient'],1e-10)
        self.assertEqual(r['mixed_query_derivative'],r['query_hessian_term']+r['query_refit_curvature_term'])
        self.assertGreater(abs(r['query_refit_curvature_term']),.001)
        self.assertGreater(abs(r['query_hessian_term']),.001)
        for values in r['summed_contractions'].values():self.assertGreater(max(map(abs,values)),.001)

    def test_zero_direction_zero_features_and_zero_base_weights(self):
        r=self.report([0]*7,V)
        self.assertEqual(r['mixed_parameter_derivative'],[0,0]);self.assertEqual(r['mixed_query_derivative'],0)
        r=weight_curvature([[0],[0]],[0,1],[1],1,[1,0],[0,1])
        self.assertEqual(r['mixed_parameter_derivative'],[0]);self.assertEqual(r['mixed_query_derivative'],0)
        r=weight_curvature([[1]],[1],[1],1,[1],[1],weights=[0])
        self.assertEqual(r['fit']['theta'],[0]);self.assertAlmostEqual(r['left_parameter_derivative'][0],5)
        self.assertAlmostEqual(r['mixed_parameter_derivative'][0],-25)
        self.assertAlmostEqual(r['mixed_query_derivative'],18.75)

    def test_permutation_orthogonal_coordinates_and_label_complement(self):
        r=self.report();perm=[6,1,3,0,5,2,4]
        p=weight_curvature([X[i]for i in perm],[Y[i]for i in perm],[1,2],1,[U[i]for i in perm],[V[i]for i in perm])
        c=weight_curvature(X,[1-y for y in Y],[1,2],0,U,V)
        rotated=weight_curvature([[-x[1],x[0]]for x in X],Y,[-2,1],1,U,V)
        for other in [p,c,rotated]:self.assertAlmostEqual(other['mixed_query_derivative'],r['mixed_query_derivative'],delta=2e-12)
        for a,b in zip(c['mixed_parameter_derivative'],r['mixed_parameter_derivative']):self.assertAlmostEqual(a,-b,delta=2e-12)
        self.assertAlmostEqual(rotated['mixed_parameter_derivative'][0],-r['mixed_parameter_derivative'][1],delta=2e-12)
        self.assertAlmostEqual(rotated['mixed_parameter_derivative'][1],r['mixed_parameter_derivative'][0],delta=2e-12)

    def test_fixed_normalizer_scaling_and_duplicate_rows(self):
        r=self.report(weights=[.8]*7,normalizer=5,ridge=.2)
        s=self.report([2*a for a in U],[2*b for b in V],weights=[1.6]*7,normalizer=10,ridge=.2)
        self.assertAlmostEqual(r['mixed_query_derivative'],s['mixed_query_derivative'],delta=1e-13)
        dup=weight_curvature([x for x in X for _ in range(2)],[y for y in Y for _ in range(2)],[1,2],1,[u for u in U for _ in range(2)],[v for v in V for _ in range(2)])
        self.assertAlmostEqual(dup['mixed_query_derivative'],self.report()['mixed_query_derivative'],delta=1e-13)

    def test_strict_contracts_and_isolation(self):
        x,y,u,v=copy.deepcopy(X),Y[:],U[:],V[:];before=copy.deepcopy((x,y,u,v));state=random.getstate()
        a=weight_curvature(x,y,[1,2],1,u,v);self.assertEqual(before,(x,y,u,v));self.assertEqual(state,random.getstate())
        a['sample_witnesses'][0]['sample_gradient'][0]=100
        self.assertNotEqual(a,self.report())
        for bad in [None,{},[0], [True]*7, [float('nan')]*7,[float('inf')]*7,[11]*7]:
            with self.assertRaises(ValueError):self.report(bad,V)
            with self.assertRaises(ValueError):self.report(U,bad)
        for kw in [dict(weights=[-1]*7),dict(weights=[11]*7),dict(normalizer=0),dict(ridge=0)]:
            with self.assertRaises(ValueError):self.report(**kw)
        with self.assertRaises(ValueError):weight_curvature(X,Y,[1],1,U,V)

    def test_cli_complete_report_and_strict_fields(self):
        request={'version':1,'X':X,'y':Y,'query':[1,2],'label':1,'left':U,'right':V,'ridge':.1,'weights':[1]*7,'normalizer':7}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json'
            def run(raw):
                path.write_text(raw)
                return subprocess.run([sys.executable,'-m','influencelab','curvature',str(path)],capture_output=True,text=True)
            p=run(json.dumps(request));self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),self.report())
            for raw in [json.dumps({**request,'extra':1}),json.dumps({**request,'version':True}),json.dumps({**request,'left':[True]*7}),json.dumps(request)[:-1]+',"version":1}', '{"x":NaN}', '['*2000,' '*100001]:
                p=run(raw);self.assertEqual(p.returncode,2,p.stderr);self.assertEqual(p.stdout,'')

    def test_negative_directional_curvature_not_query_hessian_alone(self):
        r=self.report(V,V)
        self.assertGreater(r['query_hessian_term'],.14)
        self.assertLess(r['mixed_query_derivative'],-.20)
        self.assertAlmostEqual(r['mixed_query_derivative'],-.2012289225823458,delta=1e-12)
        self.assertAlmostEqual(r['query_refit_curvature_term'],-.34821516668204777,delta=1e-12)

    def test_second_order_local_prediction_against_converged_refits(self):
        r=self.report(V,V)
        errors=[]
        for h in [.04,.02,.01]:
            weights=[1+h*v for v in V]
            m=fit(X,Y,weights=weights,tolerance=1e-12)
            actual=point(m['theta'],[1,2],1)[0]-r['query_loss']
            first=h*r['left_query_derivative']
            second=first+h*h*r['mixed_query_derivative']/2
            self.assertLess(abs(second-actual),abs(first-actual))
            errors.append(abs(second-actual))
        self.assertLess(errors[1],errors[0]/7)
        self.assertLess(errors[2],errors[1]/7)

    def test_maximum_supported_dimensions_and_stable_extreme_logits(self):
        r=weight_curvature([[0]*12 for _ in range(128)],[i%2 for i in range(128)],[1]*12,1,[10]*128,[-10]*128)
        self.assertEqual(r['mixed_parameter_derivative'],[0]*12)
        self.assertEqual(r['mixed_query_derivative'],0)
        # The explicit constant feature is regularized, even at large logits.
        r=weight_curvature([[100]],[1],[100],1,[1],[1],ridge=1e-6)
        for w in r['sample_witnesses']:
            self.assertTrue(math.isfinite(w['sample_third_logit']))
            self.assertLess(w['sample_third_logit'],0)
        self.assertTrue(math.isfinite(r['mixed_query_derivative']))

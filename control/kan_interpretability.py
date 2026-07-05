"""KAN Interpretability: Layers 1, 4, 5 for trained KAN policies.

Layer 1 (Causal Graph): Prune edges → sparse connectivity → which inputs matter?
Layer 4 (Attribution):  Decompose output action into per-input contributions
Layer 5 (Symbolic):     Fit edge functions to elementary functions (sin, linear, etc.)

Usage:
  from control.kan_interpretability import KANInterpreter
  interp = KANInterpreter(trained_policy)
  interp.analyze()  # run all three layers
"""
import torch
import numpy as np
from kanrf import bspline_basis


class KANInterpreter:
    """Extract interpretable structure from a trained KAN policy network."""

    def __init__(self, policy, input_names=None, grid_points=100):
        self.policy = policy
        self.policy.eval()
        self.input_names = input_names or [f'x{i}' for i in range(policy.state_dim)]
        self.grid_points = grid_points

        # Cached analysis results
        self.causal_graph = None       # Layer 1
        self.attributions = None       # Layer 4 (example states)
        self.symbolic_edges = None     # Layer 5

    # ══════════════════════════════════════════════════════════════════════
    # Layer 1: Causal Connectivity Graph
    # ══════════════════════════════════════════════════════════════════════

    def prune_and_analyze(self, threshold=0.05):
        """Prune edges and build causal connectivity graph.

        For each layer, identifies which input→output connections are
        active (spline magnitude > threshold).

        Returns:
            dict with per-layer connectivity matrices and statistics.
        """
        edges = self.policy.get_edge_functions()
        graph = {'layers': [], 'input_names': self.input_names}

        for layer_idx, e in enumerate(edges):
            sw = e['spline_weight']  # (out_dim, in_dim, n_basis)
            out_dim, in_dim, n_basis = sw.shape
            mean_mag = np.abs(sw).mean(axis=-1)  # (out_dim, in_dim)
            active = mean_mag > threshold

            # Per-input statistics
            input_activity = active.sum(axis=0)  # (in_dim,) how many outputs each input connects to
            output_activity = active.sum(axis=1)  # (out_dim,) how many inputs each output uses

            layer_info = {
                'layer_idx': layer_idx,
                'in_dim': in_dim,
                'out_dim': out_dim,
                'active_edges': int(active.sum()),
                'total_edges': in_dim * out_dim,
                'sparsity': 1.0 - active.sum() / (in_dim * out_dim),
                'connectivity': active.tolist(),
                'input_activity': input_activity.tolist(),
                'output_activity': output_activity.tolist(),
            }
            graph['layers'].append(layer_info)

        self.causal_graph = graph
        return graph

    def print_causal_graph(self):
        """Pretty-print the causal connectivity graph."""
        if self.causal_graph is None:
            self.prune_and_analyze()

        print("\n" + "=" * 60)
        print("Layer 1: Causal Connectivity Graph")
        print("=" * 60)

        for layer in self.causal_graph['layers']:
            print(f"\n  Layer {layer['layer_idx']}: "
                  f"{layer['in_dim']} → {layer['out_dim']}")
            print(f"    Active edges: {layer['active_edges']}/{layer['total_edges']} "
                  f"(sparsity: {layer['sparsity']*100:.0f}%)")

            # Per-input analysis
            in_names = self.input_names[:layer['in_dim']]
            print(f"    Input importance (# of active outputs):")
            for i, name in enumerate(in_names):
                bar = '█' * int(layer['input_activity'][i])
                print(f"      {name:8s}: {int(layer['input_activity'][i]):2d}/{layer['out_dim']} {bar}")

            # Per-output analysis
            print(f"    Output node fan-in (# of active inputs):")
            for o in range(min(6, layer['out_dim'])):
                bar = '█' * int(layer['output_activity'][o])
                print(f"      h{o:3d}: {int(layer['output_activity'][o]):2d}/{layer['in_dim']} {bar}")

    # ══════════════════════════════════════════════════════════════════════
    # Layer 4: Additive Attribution
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def attribute(self, s, return_details=False):
        """Decompose the output action into per-input contributions."""
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        x = s
        edge_energies = []
        for layer in self.policy.layers:
            x, B_mat, E_mat = layer(x, return_activations=True)
            edge_energies.append(E_mat)

        a = torch.tanh(self.policy.output_layer(x))

        # ── Per-input contributions through the additive structure ──
        # First layer: E0[out_dim, in_dim] = energy of φ_{in→out}
        E0 = edge_energies[0][0]  # (hidden_dim, in_dim)

        # Output layer weights
        W = self.policy.output_layer.weight.data.squeeze()  # (hidden_dim,)

        # Each hidden node's contribution = |W_j * h_j|
        h_final = x.squeeze()  # (hidden_dim,) — output of last KAN layer
        hidden_contrib = (W * h_final).abs()
        hidden_contrib = hidden_contrib / (hidden_contrib.sum() + 1e-8)

        # Distribute to inputs: input i's contribution = Σ_j hidden_contrib[j] * |E0[j,i]|
        in_dim = s.shape[1]
        input_contrib = torch.zeros(in_dim)
        for i in range(in_dim):
            input_contrib[i] = (hidden_contrib * E0[:, i].abs()).sum()
        input_contrib = input_contrib / (input_contrib.sum() + 1e-8)

        result = input_contrib.detach().cpu().numpy()
        if return_details:
            return result, {
                'hidden_contributions': hidden_contrib.cpu().numpy(),
                'action': a.item(),
            }
        return result

    def attribute_batch(self, states, labels=None):
        """Compute attributions for a batch of states.

        Args:
            states: list of (state_dim,) arrays
            labels: optional list of state labels

        Returns:
            dict mapping label → (contributions_array, action_value)
        """
        results = {}
        for i, s in enumerate(states):
            label = labels[i] if labels else f's{i}'
            contrib, details = self.attribute(s, return_details=True)
            results[label] = {
                'contributions': contrib,
                'action': details['action'],
            }
        self.attributions = results
        return results

    def print_attributions(self):
        """Pretty-print per-input attributions for example states."""
        if self.attributions is None:
            return

        print("\n" + "=" * 60)
        print("Layer 4: Additive Attribution (per-input contributions to action)")
        print("=" * 60)

        for label, data in self.attributions.items():
            contrib = data['contributions']
            action = data['action']
            parts = ' + '.join(
                f'{contrib[i]*100:.0f}%×{self.input_names[i]}'
                for i in range(len(contrib))
            )
            print(f"\n  {label:12s} → a={action:+.3f}")
            print(f"    = {parts}")

    # ══════════════════════════════════════════════════════════════════════
    # Layer 5: Symbolic Formula Extraction
    # ══════════════════════════════════════════════════════════════════════

    def fit_symbolic(self, min_r2=0.90):
        """Fit each edge function to elementary symbolic forms.

        For each edge (out_dim, in_dim) in each layer, evaluate the
        B-spline function at grid_points, then try to fit:
          - linear:    f(x) = a*x + b
          - quadratic: f(x) = a*x² + b*x + c
          - cubic:     f(x) = a*x³ + b*x² + c*x + d
          - sin:       f(x) = a*sin(b*x + c) + d
          - sigmoid:   f(x) = a/(1+exp(-b*(x-c))) + d

        Reports the best fit and R² for each edge.

        Returns:
            list of dicts: per-edge symbolic fit results
        """
        edges = self.policy.get_edge_functions()
        x_eval = np.linspace(-1, 1, self.grid_points)
        results = []

        for layer_idx, e in enumerate(edges):
            grid = e['grid']
            sw = e['spline_weight']  # (out_dim, in_dim, n_basis)
            bw = e['base_weight']    # (out_dim, in_dim)
            out_dim, in_dim, n_basis = sw.shape

            # Compute B-spline basis once
            x_t = torch.tensor(x_eval, dtype=torch.float32)
            B = bspline_basis(x_t, torch.tensor(grid), 3).numpy()  # (grid_pts, n_basis)

            for out_i in range(out_dim):
                for in_j in range(in_dim):
                    c = sw[out_i, in_j, :]  # (n_basis,)
                    w = bw[out_i, in_j].item() if isinstance(bw, torch.Tensor) else bw[out_i][in_j]

                    # Evaluate edge function
                    # φ(x) = w * silu(x) + Σ c_k B_k(x)
                    silu = x_eval / (1 + np.exp(-x_eval))  # approximate SiLU
                    y = w * silu + B @ c

                    # Try fits
                    fits = []
                    try:
                        fits.append(_fit_linear(x_eval, y))
                        fits.append(_fit_quadratic(x_eval, y))
                        fits.append(_fit_sin(x_eval, y))
                        fits.append(_fit_sigmoid(x_eval, y))
                    except Exception:
                        pass

                    if fits:
                        best = max(fits, key=lambda f: f['r2'])
                        if best['r2'] >= min_r2:
                            results.append({
                                'layer': layer_idx,
                                'edge': f'in{in_j}→out{out_i}',
                                'in_dim': in_j,
                                'out_dim': out_i,
                                'form': best['form'],
                                'params': best['params'],
                                'r2': best['r2'],
                            })

        self.symbolic_edges = results
        return results

    def print_symbolic(self):
        """Pretty-print symbolic formula extraction results."""
        if self.symbolic_edges is None:
            return

        print("\n" + "=" * 60)
        print("Layer 5: Symbolic Formula Extraction")
        print("=" * 60)

        if len(self.symbolic_edges) == 0:
            print("  No edges could be fitted to elementary functions (R² < 0.90)")
            return

        # Summary
        forms = {}
        for r in self.symbolic_edges:
            f = r['form']; forms[f] = forms.get(f, 0) + 1
        print(f"\n  Summary: {len(self.symbolic_edges)} edges fitted")
        for f, n in sorted(forms.items()):
            print(f"    {f:12s}: {n} edges")

        # Top fits
        print(f"\n  Top-10 best fits:")
        sorted_edges = sorted(self.symbolic_edges, key=lambda r: -r['r2'])[:10]
        for r in sorted_edges:
            in_name = self.input_names[r['in_dim']] if r['in_dim'] < len(self.input_names) else f"x{r['in_dim']}"
            print(f"    L{r['layer']} {in_name}→h{r['out_dim']}: "
                  f"{r['form']:10s}  R²={r['r2']:.4f}  params={r['params']}")

    # ══════════════════════════════════════════════════════════════════════
    # Full Analysis
    # ══════════════════════════════════════════════════════════════════════

    def analyze(self, example_states=None, example_labels=None):
        """Run full three-layer analysis."""
        print("\n" + "=" * 60)
        print(f"KAN Policy Interpretability Analysis")
        print(f"  Architecture: {self._describe_architecture()}")
        print("=" * 60)

        # Layer 1
        self.prune_and_analyze()
        self.print_causal_graph()

        # Layer 4
        if example_states is None:
            example_states = [
                np.array([-1.0, 0.0, 0.0]),   # bottom
                np.array([0.0, 0.0, 0.5]),     # mid
                np.array([0.0, 1.0, 0.0]),     # upright
            ]
            example_labels = ['bottom', 'mid', 'upright']
        self.attribute_batch(example_states, example_labels)
        self.print_attributions()

        # Layer 5
        self.fit_symbolic()
        self.print_symbolic()

    def _describe_architecture(self):
        dims = [self.policy.state_dim]
        for layer in self.policy.layers:
            dims.append(layer.out_dim)
        dims.append(self.policy.action_dim)
        return '→'.join(str(d) for d in dims)


# ═══════════════════════════════════════════════════════════════════════════════
# Symbolic fitting functions
# ═══════════════════════════════════════════════════════════════════════════════

def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _fit_linear(x, y):
    p = np.polyfit(x, y, 1)
    y_pred = np.polyval(p, x)
    return {'form': 'linear', 'params': [float(p[0]), float(p[1])],
            'r2': float(_r2_score(y, y_pred))}


def _fit_quadratic(x, y):
    p = np.polyfit(x, y, 2)
    y_pred = np.polyval(p, x)
    return {'form': 'quadratic', 'params': [float(p[0]), float(p[1]), float(p[2])],
            'r2': float(_r2_score(y, y_pred))}


def _fit_sin(x, y):
    """Fit a*sin(b*x + c) + d using simple grid search + linear fit."""
    # Rough estimate: f(x) ≈ a*sin(b*x + c) + d
    # Use FFT to estimate b
    from numpy.fft import rfft, rfftfreq
    n = len(x)
    yf = rfft(y - np.mean(y))
    xf = rfftfreq(n, (x[1] - x[0]))
    idx = np.argmax(np.abs(yf))
    b_est = 2 * np.pi * xf[idx] if idx > 0 else 1.0

    # Grid search around b_est for better fit
    best_r2 = -np.inf; best_params = None
    for b in np.linspace(b_est * 0.5, b_est * 2.0, 20):
        # Fit a*sin(b*x) + c*cos(b*x) + d via linear regression
        A = np.column_stack([np.sin(b * x), np.cos(b * x), np.ones_like(x)])
        try:
            coeff = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = A @ coeff
            r2 = _r2_score(y, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                # Convert to a*sin(b*x + c) + d
                a = np.sqrt(coeff[0]**2 + coeff[1]**2)
                c = np.arctan2(coeff[1], coeff[0])
                best_params = [float(a), float(b), float(c), float(coeff[2])]
        except np.linalg.LinAlgError:
            continue

    return {'form': 'sin', 'params': best_params or [0, 1, 0, 0],
            'r2': float(best_r2)}


def _fit_sigmoid(x, y):
    """Fit a/(1+exp(-b*(x-c))) + d using curve_fit."""
    from scipy.optimize import curve_fit as _curve_fit
    try:
        def sigmoid(x, a, b, c, d):
            return a / (1 + np.exp(-b * (x - c))) + d
        y_range = np.max(y) - np.min(y)
        p0 = [y_range, 5.0, 0.0, np.mean(y)]
        popt, _ = _curve_fit(sigmoid, x, y, p0=p0, maxfev=2000)
        y_pred = sigmoid(x, *popt)
        r2 = _r2_score(y, y_pred)
        return {'form': 'sigmoid', 'params': [float(p) for p in popt],
                'r2': float(r2)}
    except Exception:
        return {'form': 'sigmoid', 'params': [0, 1, 0, 0], 'r2': -1.0}

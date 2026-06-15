"""
Major notebook update:
- Insert: md-main-result, cell-leading-fig, cell-network-ceiling (after cell-sec2)
- Modify: cell-sec3, cell-sec4, cell-sec5, cell-sec10, cell-dashboard,
          cell-build-html, cell-validate
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'multiseed_results_inspection.ipynb'
nb = json.load(open(path, encoding='utf-8'))

# ── helpers ───────────────────────────────────────────────────────────────────

def get_src(cell): return ''.join(cell['source'])

def set_src(cell, s): cell['source'] = [s]

def make_code_cell(cid, source):
    return {'cell_type': 'code', 'id': cid, 'metadata': {},
            'source': [source], 'outputs': [], 'execution_count': None}

def make_md_cell(cid, source):
    return {'cell_type': 'markdown', 'id': cid, 'metadata': {},
            'source': [source]}

def find_idx(cells, cid):
    for i, c in enumerate(cells):
        if c.get('id') == cid:
            return i
    raise KeyError(cid)

changes = []

# ═══════════════════════════════════════════════════════════════════════════════
# NEW CELL SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

SRC_MD_MAIN = """\
## Main Result — Readout Performance vs CKA Stabilization

The left column compares network accuracy with frozen-feature probe accuracy (logistic
regression, linear SVM, RBF SVM). The right column shows CKA-based representation
similarities. Across all three learning rates, probes approach final-network-level
performance before CKA-to-final, mean future CKA, and local CKA indicate full
representational stabilization. This is consistent with the interpretation that
readout-accessible class information becomes available before the full representation
geometry stabilizes.

Local CKA is reported as CKA(Φₜ, Φₜ₋₁) = 1 − ΔCKA(t), so all three CKA curves
are oriented such that larger values indicate greater similarity. Local CKA measures
adjacent-checkpoint stability and should not be confused with closeness to the final
representation.

Random forest probes were also evaluated as an auxiliary nonparametric baseline. They
underperformed the logistic regression, linear SVM, and RBF SVM probes and did not
alter the qualitative timing pattern, so the main figures focus on the latter three probes."""

SRC_LEADING_FIG = """\
# ── Leading figure: readout performance vs CKA stabilization ─────────────────
# 3-row × 2-col: row = LR, col1 = accuracy, col2 = CKA similarities.

_sample_traj = next(
    (dfs[(lr, s)].get('master_trajectory.csv')
     for lr in LR_LIST for s in SEED_LIST
     if dfs.get((lr, s), {}).get('master_trajectory.csv') is not None),
    None
)
_avail_cols = set(_sample_traj.columns) if _sample_traj is not None else set()

MAIN_PROBES = [
    ('logistic_acc',   'Logistic',   '#1565c0'),
    ('linear_svm_acc', 'Linear SVM', '#558b2f'),
    ('rbf_svm_acc',    'RBF SVM',    '#6a1b9a'),
]
MAIN_PROBES_AVAIL = [(col, lbl, clr) for col, lbl, clr in MAIN_PROBES if col in _avail_cols]
NET_COLOR = '#c62828'

MAIN_DIR = AGG_DIR / 'plots' / 'main'
MAIN_DIR.mkdir(parents=True, exist_ok=True)

print(f'Main probes available: {[lbl for _, lbl, _ in MAIN_PROBES_AVAIL]}')

fig, axes = plt.subplots(3, 2, figsize=(14, 12),
                         gridspec_kw={'wspace': 0.28, 'hspace': 0.38})

for row_i, lr in enumerate(LR_LIST):
    ax_acc = axes[row_i][0]
    ax_cka = axes[row_i][1]

    # ── Network accuracy
    net = _agg_over_seeds(lr, 'network_test_acc')
    final_net = float(net['mean'].iloc[-1]) if not net.empty else np.nan
    if not net.empty:
        ax_acc.plot(net['epoch'], net['mean'], color=NET_COLOR, lw=2.2, zorder=5, label='Network')
        ax_acc.fill_between(net['epoch'], net['mean'] - net['std'],
                            net['mean'] + net['std'], alpha=0.12, color=NET_COLOR)
        ax_acc.axhline(final_net, color=NET_COLOR, ls=':', lw=1.0, alpha=0.7)
        for delta, alpha in [(0.05, 0.20), (0.02, 0.32), (0.01, 0.44)]:
            ax_acc.axhline(final_net - delta, color=NET_COLOR, ls='--', lw=0.8, alpha=alpha)

    # ── Probe accuracies (logistic, linear SVM, RBF SVM — no RF)
    for col, lbl, clr in MAIN_PROBES_AVAIL:
        agg_d = _agg_over_seeds(lr, col)
        if agg_d.empty:
            continue
        ax_acc.plot(agg_d['epoch'], agg_d['mean'], color=clr, lw=1.6, label=lbl)
        ax_acc.fill_between(agg_d['epoch'], agg_d['mean'] - agg_d['std'],
                            agg_d['mean'] + agg_d['std'], alpha=0.10, color=clr)

    ax_acc.set_ylim(0.35, 1.0)
    ax_acc.set_ylabel('Test accuracy', fontsize=9)
    ax_acc.set_xlabel('Epoch', fontsize=9)
    ax_acc.set_title(f'LR={lr} — accuracy', fontsize=10, fontweight='bold')
    ax_acc.legend(fontsize=8, loc='lower right')

    # ── CKA similarities (all oriented higher = more similar)
    CKA_PLOT = [
        ('cka_to_final',     'CKA-to-final',   '#0277bd'),
        ('mean_future_cka',  'Mean future CKA', '#00695c'),
        ('local_cka_change', 'Local CKA',       '#e65100'),
    ]
    for col, lbl, clr in CKA_PLOT:
        agg_d = _agg_over_seeds(lr, col)
        if agg_d.empty:
            continue
        if col == 'local_cka_change':
            agg_d = agg_d.copy()
            agg_d['mean'] = 1.0 - agg_d['mean']
        ax_cka.plot(agg_d['epoch'], agg_d['mean'], color=clr, lw=1.6, label=lbl)
        ax_cka.fill_between(agg_d['epoch'], agg_d['mean'] - agg_d['std'],
                            agg_d['mean'] + agg_d['std'], alpha=0.10, color=clr)

    ax_cka.axhline(0.98, color='#b71c1c', ls='--', lw=1.0, label='τ=0.98 (local CKA)')
    ax_cka.set_ylim(0.0, 1.02)
    ax_cka.set_ylabel('CKA similarity', fontsize=9)
    ax_cka.set_xlabel('Epoch', fontsize=9)
    ax_cka.set_title(f'LR={lr} — CKA similarity', fontsize=10, fontweight='bold')
    ax_cka.legend(fontsize=8, loc='lower right')

fig.suptitle(
    'Main result: readout performance vs CKA stabilization\\n'
    'Left: network and frozen-probe test accuracy '
    '(dotted = final network acc; dashed lines = 1/2/5 pp below).\\n'
    'Right: CKA-based similarities (higher = more similar). '
    'Local CKA = 1 − ΔCKA, adjacent checkpoints only.',
    fontsize=10, fontweight='bold'
)

for ext in ['png', 'pdf']:
    out = MAIN_DIR / f'leading_probe_vs_cka_by_lr.{ext}'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Saved: {out}')

plt.show()
print('Leading figure done.')"""

SRC_NETWORK_CEILING = """\
# ── Network-ceiling timing ────────────────────────────────────────────────────
# First epoch t where probe_test_acc(t) >= final_network_acc - delta.
# Uses pre-computed <probe>_gap_to_final = final_network_acc - probe_acc(t).

CEILING_DELTAS = [0.05, 0.02, 0.01]

CEILING_PROBES = [
    ('logistic_acc',   'Logistic',   'logistic_gap_to_final'),
    ('linear_svm_acc', 'Linear SVM', 'linear_svm_gap_to_final'),
    ('rbf_svm_acc',    'RBF SVM',    'rbf_svm_gap_to_final'),
]
CEILING_PROBES = [(col, lbl, gcol) for col, lbl, gcol in CEILING_PROBES if col in _avail_cols]

nc_seed_rows = []
for lr in LR_LIST:
    for seed in SEED_LIST:
        df = _get_traj(lr, seed)
        if df is None or 'network_test_acc' not in df.columns:
            continue
        rname = run_name(lr, seed)
        df_sorted = df.sort_values('epoch')
        final_ep = int(df_sorted['epoch'].iloc[-1])
        final_net = float(df_sorted.loc[df_sorted['epoch'] == final_ep, 'network_test_acc'].iloc[0])

        for col, probe_lbl, gap_col in CEILING_PROBES:
            if col not in df.columns:
                continue
            final_probe_rows = df_sorted.loc[df_sorted['epoch'] == final_ep, col]
            final_probe = float(final_probe_rows.iloc[0]) if not final_probe_rows.empty else np.nan

            for delta in CEILING_DELTAS:
                thr = final_net - delta
                hit_epoch, hit_acc = None, None
                for _, row in df_sorted.iterrows():
                    v = row.get(col)
                    if pd.isna(v):
                        continue
                    if float(v) >= thr:
                        hit_epoch = int(row['epoch'])
                        hit_acc = float(v)
                        break
                nc_seed_rows.append({
                    'lr': lr, 'seed': seed, 'run_name': rname,
                    'probe': probe_lbl, 'delta': delta,
                    'threshold_acc': round(thr, 6),
                    'final_network_acc': round(final_net, 6),
                    'final_probe_acc': round(final_probe, 6) if not np.isnan(final_probe) else np.nan,
                    'epoch': hit_epoch,
                    'probe_acc_at_epoch': round(hit_acc, 6) if hit_acc is not None else np.nan,
                })

nc_seed_df = pd.DataFrame(nc_seed_rows) if nc_seed_rows else pd.DataFrame()
out_seed = AGG_DIR / 'network_ceiling_timing_by_seed.csv'
if not nc_seed_df.empty:
    nc_seed_df.to_csv(out_seed, index=False)
    print(f'Saved: {out_seed}  shape={nc_seed_df.shape}')
else:
    print('WARNING: no network-ceiling timing data computed')

# ── Aggregate over seeds
nc_lr_rows = []
for lr in LR_LIST:
    sub_lr = nc_seed_df[nc_seed_df['lr'] == lr] if not nc_seed_df.empty else pd.DataFrame()
    for _, probe_lbl, _ in CEILING_PROBES:
        sub_p = sub_lr[sub_lr['probe'] == probe_lbl] if not sub_lr.empty else pd.DataFrame()
        for delta in CEILING_DELTAS:
            sub_d = sub_p[sub_p['delta'] == delta].dropna(subset=['epoch']) if not sub_p.empty else pd.DataFrame()
            epochs = sub_d['epoch'].tolist() if not sub_d.empty else []
            nc_lr_rows.append({
                'lr': lr, 'probe': probe_lbl, 'delta': delta,
                'mean_epoch': round(float(np.mean(epochs)), 1) if epochs else np.nan,
                'std_epoch':  round(float(np.std(epochs)), 1)  if len(epochs) > 1 else np.nan,
                'n': len(epochs),
                'mean_final_network_acc': round(float(sub_p['final_network_acc'].mean()), 4) if not sub_p.empty else np.nan,
                'mean_final_probe_acc':   round(float(sub_p['final_probe_acc'].mean()), 4)   if not sub_p.empty else np.nan,
            })

nc_lr_df = pd.DataFrame(nc_lr_rows) if nc_lr_rows else pd.DataFrame()
out_lr = AGG_DIR / 'network_ceiling_timing_by_lr.csv'
if not nc_lr_df.empty:
    nc_lr_df.to_csv(out_lr, index=False)
    print(f'Saved: {out_lr}  shape={nc_lr_df.shape}')

print()
print('=== Network-ceiling timing: first epoch reaching within delta of final network accuracy ===')
print('(Because final frozen-probe and final network accuracies are nearly identical,')
print(' this criterion asks when the frozen representation supports near-final network-level')
print(' readout performance.)')
if not nc_lr_df.empty:
    display(nc_lr_df)"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INSERT NEW CELLS after cell-sec2
# ═══════════════════════════════════════════════════════════════════════════════

insert_after_id = 'cell-sec2'
idx = find_idx(nb['cells'], insert_after_id)
new_cells = [
    make_md_cell('md-main-result',        SRC_MD_MAIN),
    make_code_cell('cell-leading-fig',    SRC_LEADING_FIG),
    make_code_cell('cell-network-ceiling', SRC_NETWORK_CEILING),
]
for cell in reversed(new_cells):
    nb['cells'].insert(idx + 1, cell)
changes.append('Inserted md-main-result, cell-leading-fig, cell-network-ceiling after cell-sec2')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODIFY cell-sec3
#    - Add linear SVM probe row; fix y-limits; add RF note
# ═══════════════════════════════════════════════════════════════════════════════

for cell in nb['cells']:
    if cell.get('id') == 'cell-sec3':
        old = get_src(cell)
        new = old

        # Extend PROBE_COLS to include linear SVM (checked against _avail_cols)
        new = new.replace(
            "PROBE_COLS = [('logistic_acc','Logistic'), ('rbf_svm_acc','RBF SVM')]",
            "PROBE_COLS = [(col, lbl) for col, lbl in [\n"
            "    ('logistic_acc','Logistic'), ('linear_svm_acc','Linear SVM'), ('rbf_svm_acc','RBF SVM')\n"
            "] if col in _avail_cols]"
        )

        # Add ylim on probe plots
        new = new.replace(
            "        ax.set_title(f'{lbl} — LR={lr}', fontsize=10, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8); ax.set_ylabel('Test accuracy', fontsize=8)\n"
            "        ax.legend(fontsize=7)",
            "        ax.set_ylim(0.35, 1.0)\n"
            "        ax.set_title(f'{lbl} — LR={lr}', fontsize=10, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8); ax.set_ylabel('Test accuracy', fontsize=8)\n"
            "        ax.legend(fontsize=7)"
        )

        # Update figure size for 3 rows (if linear SVM available)
        new = new.replace(
            "fig, axes = plt.subplots(2, 3, figsize=(15, 8))",
            "fig, axes = plt.subplots(len(PROBE_COLS), 3, figsize=(15, 4*len(PROBE_COLS)))"
        )

        # Fix axes indexing for dynamic row count
        new = new.replace(
            "for row_i, (col, lbl) in enumerate(PROBE_COLS[:2]):",
            "for row_i, (col, lbl) in enumerate(PROBE_COLS):"
        )

        # Add ylim to CKA subplot
        new = new.replace(
            "    ax.set_ylabel('CKA similarity', fontsize=8)\n"
            "    ax.set_title(f'CKA Metrics — LR={lr}', fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",
            "    ax.set_ylim(0.0, 1.02)\n"
            "    ax.set_ylabel('CKA similarity', fontsize=8)\n"
            "    ax.set_title(f'CKA Metrics — LR={lr}', fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        # Replace suptitle + show block to add RF note
        new = new.replace(
            "fig.suptitle('Probe Trajectories: mean ± std across 5 seeds', fontsize=13, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()",
            "print('Note: random forest probes omitted from main figures (underperform; do not alter qualitative pattern).')\n"
            "fig.suptitle('Probe Trajectories: mean ± std across 5 seeds', fontsize=13, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-sec3: linear SVM + ylim + RF note')
        else:
            changes.append('WARN cell-sec3: no changes matched')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODIFY cell-sec4 — add ylim for accuracy panels
# ═══════════════════════════════════════════════════════════════════════════════

for cell in nb['cells']:
    if cell.get('id') == 'cell-sec4':
        old = get_src(cell)
        new = old.replace(
            "    ax.set_title(lbl, fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8)\n"
            "    ax.legend(fontsize=7)",
            "    if col in ('network_test_acc','logistic_acc','linear_svm_acc','rbf_svm_acc'):\n"
            "        ax.set_ylim(0.35, 1.0)\n"
            "    elif col in ('cka_to_final','mean_future_cka','local_cka_change'):\n"
            "        ax.set_ylim(0.0, 1.02)\n"
            "    ax.set_title(lbl, fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8)\n"
            "    ax.legend(fontsize=7)"
        )
        # Also add linear SVM to COMPARE_COLS
        new = new.replace(
            "    ('rbf_svm_acc',       'RBF SVM probe acc'),",
            "    ('rbf_svm_acc',       'RBF SVM probe acc'),\n"
            "    ('linear_svm_acc',    'Linear SVM probe acc'),"
        )
        if new != old:
            set_src(cell, new)
            changes.append('cell-sec4: ylim + linear SVM added to COMPARE_COLS')
        else:
            changes.append('WARN cell-sec4: no changes matched')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODIFY cell-sec5 — add ylim for logistic acc probe panel
# ═══════════════════════════════════════════════════════════════════════════════

for cell in nb['cells']:
    if cell.get('id') == 'cell-sec5':
        old = get_src(cell)
        new = old.replace(
            "        ax.set_title(f'{lbl} — LR={lr}', fontsize=10, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8)\n"
            "        ax.legend(fontsize=7)",
            "        if col in ('logistic_acc','linear_svm_acc','rbf_svm_acc','network_test_acc'):\n"
            "            ax.set_ylim(0.35, 1.0)\n"
            "        ax.set_title(f'{lbl} — LR={lr}', fontsize=10, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8)\n"
            "        ax.legend(fontsize=7)"
        )
        if new != old:
            set_src(cell, new)
            changes.append('cell-sec5: ylim added')
        else:
            changes.append('WARN cell-sec5: no changes matched')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODIFY cell-sec10 — add network-ceiling as primary, keep legacy timing
# ═══════════════════════════════════════════════════════════════════════════════

NEW_SEC10 = """\
# ── Section 10: Operational Timing Summaries ─────────────────────────────────

# ── Part A: Network-ceiling timing (primary) ──────────────────────────────────
print('=== A. Network-ceiling timing (PRIMARY) ===')
print()
print('First epoch where probe_test_acc(t) >= final_network_acc - delta.')
print('Because final frozen-probe and final network accuracies are nearly identical,')
print('the network-ceiling criterion asks when the frozen representation already supports')
print('near-final network-level readout performance.')
print()

if 'nc_lr_df' in dir() and not nc_lr_df.empty:
    # Compact pivot: rows = probe x delta, cols = LR
    _nc_disp = nc_lr_df.copy()
    _nc_disp['criterion'] = _nc_disp['probe'] + ' (within ' + (_nc_disp['delta'] * 100).astype(int).astype(str) + 'pp)'
    try:
        _nc_pivot = _nc_disp.pivot(index='criterion', columns='lr', values='mean_epoch')
        print(_nc_pivot.to_string())
        display(_nc_pivot.reset_index())
    except Exception:
        display(_nc_disp[['lr', 'probe', 'delta', 'mean_epoch', 'std_epoch', 'n']])
else:
    print('WARNING: network-ceiling timing (nc_lr_df) not available; run cell-network-ceiling first.')
print()

# ── Part B: Legacy timing events (secondary) ─────────────────────────────────
print('=== B. Additional timing events (secondary: includes CKA / NC / eNTK milestones) ===')

TIMING_EVENTS = [
    ('logistic_acc',            None, 0.95, 'above', 'Logistic probe: 95% of final probe value'),
    ('logistic_acc',            None, 0.99, 'above', 'Logistic probe: 99% of final probe value'),
    ('rbf_svm_acc',             None, 0.95, 'above', 'RBF SVM: 95% of final probe value'),
    ('cka_to_final',            0.95, None, 'above', 'CKA-to-final ≥ 0.95'),
    ('cka_to_final',            0.99, None, 'above', 'CKA-to-final ≥ 0.99'),
    ('local_cka_change',        None, 0.02, 'below', 'Local stabilization: local ΔCKA ≤ 0.02 / local CKA ≥ 0.98'),
    ('entk_distance_final',     None, 0.05, 'below', 'eNTK dist-to-final ≤ 0.05'),
    ('entk_distance_final',     None, 0.01, 'below', 'eNTK dist-to-final ≤ 0.01'),
    ('log10_nc1',               None, -1.0, 'below', 'log10(NC1) ≤ -1'),
]

timing_rows = []
for lr in LR_LIST:
    for col, abs_thr, frac_thr, direction, label in TIMING_EVENTS:
        epochs_hit = []
        for seed in SEED_LIST:
            df = _get_traj(lr, seed)
            if df is None or col not in df.columns: continue
            series = df[col].dropna()
            if series.empty: continue
            if frac_thr is not None:
                thr = float(series.iloc[-1]) * frac_thr
            else:
                thr = abs_thr
            ep = _first_epoch(df, col, thr, direction)
            if ep is not None:
                epochs_hit.append(ep)
        timing_rows.append({
            'LR': lr,
            'Event': label,
            'Mean epoch': round(np.mean(epochs_hit), 1) if epochs_hit else np.nan,
            'Std epoch':  round(np.std(epochs_hit),  1) if len(epochs_hit) > 1 else np.nan,
            'n': len(epochs_hit),
        })

timing_df = pd.DataFrame(timing_rows)
print('(Probe thresholds are relative to each probe\\'s own final value;')
print(' local stabilization uses local ΔCKA ≤ 0.02, equivalently local CKA ≥ 0.98)')
try:
    pivot = timing_df.pivot(index='Event', columns='LR', values='Mean epoch').reset_index()
    display(pivot)
except Exception:
    display(timing_df)"""

for cell in nb['cells']:
    if cell.get('id') == 'cell-sec10':
        set_src(cell, NEW_SEC10)
        changes.append('cell-sec10: full rewrite with network-ceiling as primary')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODIFY cell-dashboard
#    - Add _make_leading_fig_b64()
#    - Fix ylim in _make_probe_cka_fig_b64 and _make_lr_comparison_fig_b64
#    - Add linear SVM to probe figures
# ═══════════════════════════════════════════════════════════════════════════════

for cell in nb['cells']:
    if cell.get('id') == 'cell-dashboard':
        old = get_src(cell)
        new = old

        # ── Add _make_leading_fig_b64 before _make_nc_fig_b64 ──────────────
        leading_fn = (
            "\ndef _make_leading_fig_b64():\n"
            "    fig, axes = plt.subplots(3, 2, figsize=(14, 12),\n"
            "                             gridspec_kw={'wspace': 0.28, 'hspace': 0.38})\n"
            "    for row_i, lr in enumerate(LR_LIST):\n"
            "        ax_acc = axes[row_i][0]\n"
            "        ax_cka = axes[row_i][1]\n"
            "        net = _agg_over_seeds(lr, 'network_test_acc')\n"
            "        final_net = float(net['mean'].iloc[-1]) if not net.empty else np.nan\n"
            "        if not net.empty:\n"
            "            ax_acc.plot(net['epoch'], net['mean'], color='#c62828', lw=2.2, zorder=5, label='Network')\n"
            "            ax_acc.fill_between(net['epoch'], net['mean']-net['std'], net['mean']+net['std'], alpha=0.12, color='#c62828')\n"
            "            ax_acc.axhline(final_net, color='#c62828', ls=':', lw=1.0, alpha=0.7)\n"
            "            for delta, alpha in [(0.05,0.20),(0.02,0.32),(0.01,0.44)]:\n"
            "                ax_acc.axhline(final_net - delta, color='#c62828', ls='--', lw=0.8, alpha=alpha)\n"
            "        _lp = [('logistic_acc','Logistic','#1565c0'),('linear_svm_acc','Linear SVM','#558b2f'),('rbf_svm_acc','RBF SVM','#6a1b9a')]\n"
            "        _lp = [(c,l,clr) for c,l,clr in _lp if c in (_avail_cols if '_avail_cols' in dir() else set())]\n"
            "        for col, lbl, clr in _lp:\n"
            "            agg_d = _agg_over_seeds(lr, col)\n"
            "            if agg_d.empty: continue\n"
            "            ax_acc.plot(agg_d['epoch'], agg_d['mean'], color=clr, lw=1.6, label=lbl)\n"
            "            ax_acc.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.10, color=clr)\n"
            "        ax_acc.set_ylim(0.35, 1.0)\n"
            "        ax_acc.set_ylabel('Test accuracy', fontsize=9); ax_acc.set_xlabel('Epoch', fontsize=9)\n"
            "        ax_acc.set_title(f'LR={lr} — accuracy', fontsize=10, fontweight='bold'); ax_acc.legend(fontsize=8, loc='lower right')\n"
            "        for col, lbl, clr in [('cka_to_final','CKA-to-final','#0277bd'),('mean_future_cka','Mean future CKA','#00695c'),('local_cka_change','Local CKA','#e65100')]:\n"
            "            agg_d = _agg_over_seeds(lr, col)\n"
            "            if agg_d.empty: continue\n"
            "            if col == 'local_cka_change': agg_d = agg_d.copy(); agg_d['mean'] = 1.0 - agg_d['mean']\n"
            "            ax_cka.plot(agg_d['epoch'], agg_d['mean'], color=clr, lw=1.6, label=lbl)\n"
            "            ax_cka.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.10, color=clr)\n"
            "        ax_cka.axhline(0.98, color='#b71c1c', ls='--', lw=1.0, label='τ=0.98 (local CKA)')\n"
            "        ax_cka.set_ylim(0.0, 1.02)\n"
            "        ax_cka.set_ylabel('CKA similarity', fontsize=9); ax_cka.set_xlabel('Epoch', fontsize=9)\n"
            "        ax_cka.set_title(f'LR={lr} — CKA similarity', fontsize=10, fontweight='bold'); ax_cka.legend(fontsize=8, loc='lower right')\n"
            "    fig.suptitle('Main result: readout performance vs CKA stabilization', fontsize=12, fontweight='bold')\n"
            "    plt.tight_layout()\n"
            "    return _fig_to_b64(fig)\n"
        )
        new = new.replace(
            "\ndef _make_nc_fig_b64():",
            leading_fn + "\ndef _make_nc_fig_b64():"
        )

        # ── Fix ylim in _make_probe_cka_fig_b64 — probe rows ───────────────
        new = new.replace(
            "            ax.axhline(fv*0.95, color='#e65100', ls='--', lw=1.1, label='95% of final probe')\n"
            "            ax.set_title(f'{lbl} — LR={lr}', fontsize=9, fontweight='bold')\n"
            "            ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",
            "            ax.axhline(fv*0.95, color='#e65100', ls='--', lw=1.1, label='95% of final probe')\n"
            "            ax.set_ylim(0.35, 1.0)\n"
            "            ax.set_title(f'{lbl} — LR={lr}', fontsize=9, fontweight='bold')\n"
            "            ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        # ── Fix ylim in _make_probe_cka_fig_b64 — CKA row ─────────────────
        new = new.replace(
            "        ax.axhline(0.98, color='#b71c1c', ls='--', lw=1, label='τ=0.98 (local CKA)')\n"
            "        ax.set_ylabel('CKA similarity', fontsize=8)\n"
            "        ax.set_title(f'CKA — LR={lr}', fontsize=9, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",
            "        ax.axhline(0.98, color='#b71c1c', ls='--', lw=1, label='τ=0.98 (local CKA)')\n"
            "        ax.set_ylim(0.0, 1.02)\n"
            "        ax.set_ylabel('CKA similarity', fontsize=8)\n"
            "        ax.set_title(f'CKA — LR={lr}', fontsize=9, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        # ── Fix ylim in _make_lr_comparison_fig_b64 ─────────────────────────
        new = new.replace(
            "        ax.set_title(lbl, fontsize=10, fontweight='bold'); ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",
            "        if col in ('logistic_acc','linear_svm_acc','rbf_svm_acc','network_test_acc'): ax.set_ylim(0.35, 1.0)\n"
            "        elif col in ('cka_to_final','mean_future_cka','local_cka_change'): ax.set_ylim(0.0, 1.02)\n"
            "        ax.set_title(lbl, fontsize=10, fontweight='bold'); ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-dashboard: added _make_leading_fig_b64 + ylim fixes')
        else:
            changes.append('WARN cell-dashboard: no changes matched')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MODIFY cell-build-html
#    - Add leading figure section (before section 3)
#    - Add network-ceiling table in section 3
#    - Update section 10 note
# ═══════════════════════════════════════════════════════════════════════════════

for cell in nb['cells']:
    if cell.get('id') == 'cell-build-html':
        old = get_src(cell)
        new = old

        # ── Add leading figure section before section 3 ─────────────────────
        lead_section = (
            "\n    # ── Main result figure ────────────────────────────────────────────────────\n"
            "    p.append(_section('MAIN RESULT: READOUT PERFORMANCE vs CKA STABILIZATION'))\n"
            "    p.append(_prose(\n"
            "        'The left column compares network accuracy with frozen-feature probe accuracy '\n"
            "        '(logistic regression, linear SVM, RBF SVM). '\n"
            "        'The right column shows CKA-based representation similarities. '\n"
            "        'Across all three learning rates, probes approach final-network-level performance '\n"
            "        'before CKA-to-final, mean future CKA, and local CKA indicate full representational stabilization. '\n"
            "        'Local CKA = 1 &minus; &Delta;CKA(t) compares adjacent checkpoints; '\n"
            "        'higher values indicate greater similarity. '\n"
            "        'Random forest probes are omitted from the main figure; they underperform '\n"
            "        'and do not alter the qualitative timing pattern.'\n"
            "    ))\n"
            "    _lead_png = AGG_DIR / 'plots' / 'main' / 'leading_probe_vs_cka_by_lr.png'\n"
            "    _lead_b64 = _img_b64(_lead_png) if _lead_png.exists() else _make_leading_fig_b64()\n"
            "    if _lead_b64: p.append(f\"<img src='{_lead_b64}' style='max-width:100%;border:1px solid #ddd;border-radius:4px'>\")\n"
            "    else: p.append('<p style=\\'color:gray\\'>Leading figure not available.</p>')\n"
        )
        new = new.replace(
            "\n    # ── 3. Primary evidence: probes vs CKA ───────────────────────────────────\n",
            lead_section + "\n    # ── 3. Primary evidence: probes vs CKA ───────────────────────────────────\n"
        )

        # ── Add network-ceiling table in section 3 ──────────────────────────
        nc_table_block = (
            "\n    p.append(_sub('Network-ceiling timing: first checkpoint at near-final network-level readout performance'))\n"
            "    p.append(_note(\n"
            "        'Because final frozen-probe and final network accuracies are nearly identical, '\n"
            "        'the network-ceiling criterion asks when the frozen representation already supports '\n"
            "        'near-final network-level readout performance (within 1/2/5 percentage points of final network accuracy).'\n"
            "    ))\n"
            "    if 'nc_lr_df' in dir() and not nc_lr_df.empty:\n"
            "        _nc_d = nc_lr_df.copy()\n"
            "        _nc_d['criterion'] = _nc_d['probe'] + ' (within ' + (_nc_d['delta']*100).astype(int).astype(str) + 'pp)'\n"
            "        try:\n"
            "            _nc_pivot = _nc_d.pivot(index='criterion', columns='lr', values='mean_epoch').reset_index()\n"
            "            p.append(_html_table(_nc_pivot, 'Mean epoch (across 5 seeds) reaching within delta of final network accuracy'))\n"
            "        except Exception:\n"
            "            p.append(_html_table(_nc_d[['lr','probe','delta','mean_epoch','std_epoch','n']],\n"
            "                                 'Network-ceiling timing'))\n"
            "    else:\n"
            "        p.append('<p style=\\'color:gray\\'>Network-ceiling timing data not available.</p>')\n"
        )
        new = new.replace(
            "\n    b64 = _make_probe_cka_fig_b64()\n",
            nc_table_block + "\n    b64 = _make_probe_cka_fig_b64()\n"
        )

        # ── Update section 10 note ───────────────────────────────────────────
        new = new.replace(
            "        'Probe thresholds (95%/99%) are relative to each probe\\'s own final-epoch accuracy. '\n"
            "        'Local stabilization uses local &Delta;CKA &le; 0.02 (equivalently local CKA &ge; 0.98). '\n"
            "        'These epoch numbers describe when each metric operationally matures '\n"
            "        'and do not constitute a definition of sufficiency.'",
            "        'Primary criterion (Part A): network-ceiling — first epoch where probe accuracy '\n"
            "        'is within 1/2/5 percentage points of final network accuracy. '\n"
            "        'Secondary events (Part B): probe thresholds are relative to each probe\\'s own final value; '\n"
            "        'local stabilization uses local &Delta;CKA &le; 0.02 (equivalently local CKA &ge; 0.98). '\n"
            "        'None of these events constitute a definition of representational sufficiency.'"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-build-html: leading section + network-ceiling table + sec10 note')
        else:
            changes.append('WARN cell-build-html: no changes matched')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# 8. MODIFY cell-validate — add new file checks
# ═══════════════════════════════════════════════════════════════════════════════

for cell in nb['cells']:
    if cell.get('id') == 'cell-validate':
        old = get_src(cell)
        new = old

        # Add leading figure and network ceiling to validation checks
        new = new.replace(
            "html_path = GEN_DIR / 'dashboard.html'\n"
            "html_ok = html_path.exists()\n"
            "print(f'  HTML         [{\"OK\" if html_ok else \"MISSING\"}]  {html_path}')\n"
            "if not html_ok: all_ok = False",
            "html_path = GEN_DIR / 'dashboard.html'\n"
            "html_ok = html_path.exists()\n"
            "print(f'  HTML         [{\"OK\" if html_ok else \"MISSING\"}]  {html_path}')\n"
            "if not html_ok: all_ok = False\n"
            "\n"
            "REQUIRED_MAIN = [\n"
            "    AGG_DIR / 'plots' / 'main' / 'leading_probe_vs_cka_by_lr.png',\n"
            "    AGG_DIR / 'plots' / 'main' / 'leading_probe_vs_cka_by_lr.pdf',\n"
            "    AGG_DIR / 'network_ceiling_timing_by_seed.csv',\n"
            "    AGG_DIR / 'network_ceiling_timing_by_lr.csv',\n"
            "]\n"
            "for p_check in REQUIRED_MAIN:\n"
            "    ok = p_check.exists()\n"
            "    if not ok: all_ok = False\n"
            "    print(f'  Main output  [{\"OK\" if ok else \"MISSING\"}]  {p_check.name}')"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-validate: added leading figure and network-ceiling checks')
        else:
            changes.append('WARN cell-validate: no changes matched')
        break

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('=== Changes applied ===')
for c in changes:
    print(' ', c)

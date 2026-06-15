"""
Patch multiseed_results_inspection.ipynb to convert local_cka_change
into local_cka (1 - change) in all trajectory plots.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'multiseed_results_inspection.ipynb'
nb = json.load(open(path, encoding='utf-8'))

def get_src(cell): return ''.join(cell['source'])

def set_src(cell, s):
    # Store as a single string in source list (valid nbformat)
    cell['source'] = [s] if s else []

changes = []

# ── md-sec3 ───────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'md-sec3':
        old = get_src(cell)
        insert = (
            '\n\nFor visual comparability, local stabilization is shown as **local CKA similarity**,'
            ' defined as CKA(Φₜ, Φₜ₋₁) = 1 − ΔCKA(t).'
            ' All three CKA curves are therefore oriented so that higher values indicate greater similarity.'
            ' Note, however, that the three quantities compare different checkpoint pairs:'
            ' adjacent checkpoints (local CKA), the final checkpoint (CKA-to-final),'
            ' and all future checkpoints on average (mean future CKA).'
            ' Local CKA therefore measures short-term stability, not necessarily closeness to the final representation.'
        )
        target = 'The remaining sections test two complementary explanations.'
        if target in old:
            new = old.replace(target, insert.strip() + '\n\n' + target)
            set_src(cell, new)
            changes.append('md-sec3: added local CKA explanation')

# ── cell-sec3 ─────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'cell-sec3':
        old = get_src(cell)
        new = old

        # Update CKA_COLS definition
        new = new.replace(
            "CKA_COLS   = [('cka_to_final','CKA-to-final'), ('local_cka_change','Local CKA Δ'), ('mean_future_cka','Mean future CKA')]",
            "CKA_COLS   = [('cka_to_final','CKA-to-final'), ('local_cka_change','Local CKA'), ('mean_future_cka','Mean future CKA')]"
        )

        # Add inversion in the loop + update threshold
        new = new.replace(
            "    for col, lbl in CKA_COLS:\n"
            "        agg_d = _agg_over_seeds(lr, col)\n"
            "        if agg_d.empty: continue\n"
            "        ax.plot(agg_d['epoch'], agg_d['mean'], lw=1.8, label=lbl)\n"
            "        ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.15)\n"
            "    ax.axhline(0.02, color='#b71c1c', ls='--', lw=1, label='τ=0.02 (local Δ)')\n"
            "    ax.set_title(f'CKA Metrics — LR={lr}', fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",

            "    for col, lbl in CKA_COLS:\n"
            "        agg_d = _agg_over_seeds(lr, col)\n"
            "        if agg_d.empty: continue\n"
            "        if col == 'local_cka_change':\n"
            "            agg_d = agg_d.copy(); agg_d['mean'] = 1.0 - agg_d['mean']\n"
            "        ax.plot(agg_d['epoch'], agg_d['mean'], lw=1.8, label=lbl)\n"
            "        ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.15)\n"
            "    ax.axhline(0.98, color='#b71c1c', ls='--', lw=1, label='τ=0.98 (local CKA)')\n"
            "    ax.set_ylabel('CKA similarity', fontsize=8)\n"
            "    ax.set_title(f'CKA Metrics — LR={lr}', fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-sec3: inversion + tau 0.98 + ylabel')
        else:
            changes.append('WARN cell-sec3: no change made - check strings')

# ── cell-sec4 ─────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'cell-sec4':
        old = get_src(cell)
        new = old

        # Update COMPARE_COLS label
        new = new.replace(
            "    ('local_cka_change',  'Local CKA Δ'),",
            "    ('local_cka_change',  'Local CKA'),"
        )

        # Add inversion inside per-LR loop
        new = new.replace(
            "for i, (col, lbl) in enumerate(COMPARE_COLS):\n"
            "    ax = axes_flat[i]\n"
            "    for lr in LR_LIST:\n"
            "        agg_d = _agg_over_seeds(lr, col)\n"
            "        if agg_d.empty: continue\n"
            "        ax.plot(agg_d['epoch'], agg_d['mean'], color=LR_COLORS[lr], lw=1.8, label=f'LR={lr}')\n"
            "        ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.15, color=LR_COLORS[lr])\n"
            "    ax.set_title(lbl, fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8)\n"
            "    ax.legend(fontsize=7)",

            "for i, (col, lbl) in enumerate(COMPARE_COLS):\n"
            "    ax = axes_flat[i]\n"
            "    for lr in LR_LIST:\n"
            "        agg_d = _agg_over_seeds(lr, col)\n"
            "        if agg_d.empty: continue\n"
            "        if col == 'local_cka_change':\n"
            "            agg_d = agg_d.copy(); agg_d['mean'] = 1.0 - agg_d['mean']\n"
            "        ax.plot(agg_d['epoch'], agg_d['mean'], color=LR_COLORS[lr], lw=1.8, label=f'LR={lr}')\n"
            "        ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.15, color=LR_COLORS[lr])\n"
            "    ax.set_title(lbl, fontsize=10, fontweight='bold')\n"
            "    ax.set_xlabel('Epoch', fontsize=8)\n"
            "    ax.legend(fontsize=7)"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-sec4: Local CKA label + inversion')
        else:
            changes.append('WARN cell-sec4: no change made - check strings')

# ── cell-sec10 ────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'cell-sec10':
        old = get_src(cell)
        new = old.replace(
            "    ('local_cka_change',        None, 0.02, 'below', 'Local CKA Δ ≤ 0.02'),",
            "    ('local_cka_change',        None, 0.02, 'below', 'Local stabilization: local ΔCKA ≤ 0.02 / local CKA ≥ 0.98'),"
        )
        if new != old:
            set_src(cell, new)
            changes.append('cell-sec10: timing label updated')
        else:
            changes.append('WARN cell-sec10: no change - check string')

# ── cell-dashboard ────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'cell-dashboard':
        old = get_src(cell)
        new = old

        # _make_probe_cka_fig_b64: add mean_future_cka and update label
        new = new.replace(
            "    CKA_COLS_D   = [('cka_to_final','CKA-to-final'), ('local_cka_change','Local CKA Δ')]",
            "    CKA_COLS_D   = [('cka_to_final','CKA-to-final'), ('local_cka_change','Local CKA'), ('mean_future_cka','Mean future CKA')]"
        )

        # _make_probe_cka_fig_b64: add inversion + update threshold in CKA row
        new = new.replace(
            "        for col2, lbl2 in CKA_COLS_D:\n"
            "            agg_d = _agg_over_seeds(lr, col2)\n"
            "            if agg_d.empty: continue\n"
            "            ax.plot(agg_d['epoch'], agg_d['mean'], lw=1.8, label=lbl2)\n"
            "            ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.13)\n"
            "        ax.axhline(0.02, color='#b71c1c', ls='--', lw=1, label='τ=0.02')\n"
            "        ax.set_title(f'CKA — LR={lr}', fontsize=9, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",

            "        for col2, lbl2 in CKA_COLS_D:\n"
            "            agg_d = _agg_over_seeds(lr, col2)\n"
            "            if agg_d.empty: continue\n"
            "            if col2 == 'local_cka_change':\n"
            "                agg_d = agg_d.copy(); agg_d['mean'] = 1.0 - agg_d['mean']\n"
            "            ax.plot(agg_d['epoch'], agg_d['mean'], lw=1.8, label=lbl2)\n"
            "            ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.13)\n"
            "        ax.axhline(0.98, color='#b71c1c', ls='--', lw=1, label='τ=0.98 (local CKA)')\n"
            "        ax.set_ylabel('CKA similarity', fontsize=8)\n"
            "        ax.set_title(f'CKA — LR={lr}', fontsize=9, fontweight='bold')\n"
            "        ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        # _make_lr_comparison_fig_b64: update label
        new = new.replace(
            "            ('cka_to_final','CKA-to-final'),('local_cka_change','Local CKA Δ'),",
            "            ('cka_to_final','CKA-to-final'),('local_cka_change','Local CKA'),"
        )

        # _make_lr_comparison_fig_b64: add inversion in loop
        new = new.replace(
            "    for ax, (col, lbl) in zip(axes.flatten(), cols):\n"
            "        for lr in LR_LIST:\n"
            "            agg_d = _agg_over_seeds(lr, col)\n"
            "            if agg_d.empty: continue\n"
            "            ax.plot(agg_d['epoch'], agg_d['mean'], color=LR_COLORS[lr], lw=1.8, label=f'LR={lr}')\n"
            "            ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.15, color=LR_COLORS[lr])\n"
            "        ax.set_title(lbl, fontsize=10, fontweight='bold'); ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)",

            "    for ax, (col, lbl) in zip(axes.flatten(), cols):\n"
            "        for lr in LR_LIST:\n"
            "            agg_d = _agg_over_seeds(lr, col)\n"
            "            if agg_d.empty: continue\n"
            "            if col == 'local_cka_change':\n"
            "                agg_d = agg_d.copy(); agg_d['mean'] = 1.0 - agg_d['mean']\n"
            "            ax.plot(agg_d['epoch'], agg_d['mean'], color=LR_COLORS[lr], lw=1.8, label=f'LR={lr}')\n"
            "            ax.fill_between(agg_d['epoch'], agg_d['mean']-agg_d['std'], agg_d['mean']+agg_d['std'], alpha=0.15, color=LR_COLORS[lr])\n"
            "        ax.set_title(lbl, fontsize=10, fontweight='bold'); ax.set_xlabel('Epoch', fontsize=8); ax.legend(fontsize=7)"
        )

        # prose in build_dashboard section 3
        new = new.replace(
            "        'Near-final probe performance is reached substantially earlier than CKA-based stabilization '\n"
            "        'across all three learning rates and all five seeds. The logistic probe in particular '\n"
            "        'reaches 95% of its final-epoch accuracy well before the local CKA change falls below '\n"
            "        'the &tau;=0.02 threshold, consistent with the interpretation that readout-accessible '\n"
            "        'class information matures before representation geometry stabilizes. '\n"
            "        'The remaining sections test two complementary explanations. '",

            "        'Near-final probe performance is reached substantially earlier than CKA-based stabilization '\n"
            "        'across all three learning rates and all five seeds. '\n"
            "        'Local CKA is shown as a similarity metric (= 1 − ΔCKA) so all three CKA curves '\n"
            "        'share the same orientation: higher = more similar. '\n"
            "        'The logistic probe reaches 95% of its final-epoch accuracy well before local CKA rises '\n"
            "        'above the τ=0.98 threshold (equivalently, local ΔCKA falls below 0.02), '\n"
            "        'consistent with the interpretation that readout-accessible class information matures '\n"
            "        'before representation geometry stabilizes. '\n"
            "        'Note that local CKA measures short-term stability (adjacent checkpoints), '\n"
            "        'not closeness to the final representation. '\n"
            "        'The remaining sections test two complementary explanations. '"
        )

        # section 10 note in build_dashboard
        new = new.replace(
            "        'Probe thresholds (95%/99%) are relative to each probe\\'s own final-epoch accuracy, '\n"
            "        'not absolute thresholds. CKA/eNTK thresholds are absolute. '\n"
            "        'These epoch numbers describe when each metric operationally matures '\n"
            "        'and do not constitute a definition of sufficiency.'",

            "        'Probe thresholds (95%/99%) are relative to each probe\\'s own final-epoch accuracy. '\n"
            "        'Local stabilization uses local ΔCKA ≤ 0.02 (equivalently local CKA ≥ 0.98). '\n"
            "        'These epoch numbers describe when each metric operationally matures '\n"
            "        'and do not constitute a definition of sufficiency.'"
        )

        if new != old:
            set_src(cell, new)
            changes.append('cell-dashboard: all local_cka fixes applied')
        else:
            changes.append('WARN cell-dashboard: no change made')

json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('=== Changes applied ===')
for c in changes:
    print(' ', c)

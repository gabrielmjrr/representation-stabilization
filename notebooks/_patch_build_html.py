"""
Patch cell-build-html: update prose text in section 3 and section 10 note
to use local CKA similarity language (tau=0.98, not 0.02).
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'multiseed_results_inspection.ipynb'
nb = json.load(open(path, encoding='utf-8'))

def get_src(cell): return ''.join(cell['source'])
def set_src(cell, s): cell['source'] = [s]

changes = []

for cell in nb['cells']:
    if cell.get('id') == 'cell-build-html':
        old = get_src(cell)
        new = old

        # ── Section 3 prose ──────────────────────────────────────────────────
        old_prose = (
            "        'Near-final probe performance is reached substantially earlier than CKA-based stabilization '\n"
            "        'across all three learning rates and all five seeds. The logistic probe in particular '\n"
            "        'reaches 95% of its final-epoch accuracy well before the local CKA change falls below '\n"
            "        'the &tau;=0.02 threshold, consistent with the interpretation that readout-accessible '\n"
            "        'class information matures before representation geometry stabilizes. '\n"
            "        'The remaining sections test two complementary explanations. '\n"
            "        'Cross-learning-rate CKA asks whether different learning rates converge to similar '\n"
            "        'representation geometries. Spectral component analysis asks whether early probe '\n"
            "        'performance is concentrated in the dominant singular directions of the representation.'"
        )
        new_prose = (
            "        'Near-final probe performance is reached substantially earlier than CKA-based stabilization '\n"
            "        'across all three learning rates and all five seeds. '\n"
            "        'Local CKA is shown as a similarity metric (local CKA = 1 &minus; &Delta;CKA) '\n"
            "        'so that all three CKA curves share the same orientation: higher = more similar. '\n"
            "        'The logistic probe reaches 95% of its final-epoch accuracy well before '\n"
            "        'local CKA rises above the &tau;=0.98 threshold (equivalently, local &Delta;CKA falls below 0.02), '\n"
            "        'consistent with the interpretation that readout-accessible class information '\n"
            "        'matures before representation geometry stabilizes. '\n"
            "        'Note that local CKA compares adjacent checkpoints, not closeness to the final representation. '\n"
            "        'The remaining sections test two complementary explanations. '\n"
            "        'Cross-learning-rate CKA asks whether different learning rates converge to similar '\n"
            "        'representation geometries. Spectral component analysis asks whether early probe '\n"
            "        'performance is concentrated in the dominant singular directions of the representation.'"
        )
        if old_prose in new:
            new = new.replace(old_prose, new_prose)
            changes.append('cell-build-html section-3 prose: updated')
        else:
            changes.append('WARN cell-build-html section-3 prose: NOT MATCHED')

        # ── Section 10 note ──────────────────────────────────────────────────
        old_note = (
            "        'Probe thresholds (95%/99%) are relative to each probe\\'s own final-epoch accuracy, '\n"
            "        'not absolute thresholds. CKA/eNTK thresholds are absolute. '\n"
            "        'These epoch numbers describe when each metric operationally matures '\n"
            "        'and do not constitute a definition of sufficiency.'"
        )
        new_note = (
            "        'Probe thresholds (95%/99%) are relative to each probe\\'s own final-epoch accuracy. '\n"
            "        'Local stabilization event: local &Delta;CKA &le; 0.02 (equivalently local CKA &ge; 0.98). '\n"
            "        'These epoch numbers describe when each metric operationally matures '\n"
            "        'and do not constitute a definition of sufficiency.'"
        )
        if old_note in new:
            new = new.replace(old_note, new_note)
            changes.append('cell-build-html section-10 note: updated')
        else:
            changes.append('WARN cell-build-html section-10 note: NOT MATCHED')

        if new != old:
            set_src(cell, new)

json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('=== Changes ===')
for c in changes: print(' ', c)

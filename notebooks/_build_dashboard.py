"""
Standalone dashboard builder — runs all notebook logic non-interactively
and writes notebooks/.generated/dashboard.html.
"""
import sys, os, builtins

# ── Mock IPython so imports don't fail ────────────────────────────────────────
import types

_ipython_mod = types.ModuleType('IPython')
_display_mod  = types.ModuleType('IPython.display')

_html_class = type('HTML', (), {'__init__': lambda self, s: None})
_display_fn  = lambda *args, **kwargs: None   # no-op

_display_mod.HTML    = _html_class
_display_mod.display = _display_fn
_ipython_mod.display = _display_mod
_ipython_mod.get_ipython = lambda: None
_ipython_mod.version_info = (99, 0, 0, '', 0)  # report high version so mpl skips backend switch
_ipython_mod.__version__ = '99.0.0'

sys.modules.setdefault('IPython', _ipython_mod)
sys.modules.setdefault('IPython.display', _display_mod)

# Also override builtins.display so any bare display() call works
builtins.display = _display_fn

# ── Force non-interactive matplotlib backend ───────────────────────────────────
import matplotlib
matplotlib.use('Agg')

# ── Now execute the extracted notebook code ────────────────────────────────────
_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_run_dashboard.py')
print(f'Running: {_script}')
with open(_script, encoding='utf-8') as _fh:
    _code = _fh.read()

exec(compile(_code, _script, 'exec'), {'__file__': _script, '__name__': '__main__'})

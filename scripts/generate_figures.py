import json
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)
NOTEBOOK_PATH = ROOT / 'notebook' / 'aimo-solver-v58.ipynb'
if not NOTEBOOK_PATH.exists():
    NOTEBOOK_PATH = ROOT / 'notebook' / 'aimo-solver-58.ipynb'
LOG_PATH = ROOT / 'log.md'

# -------------------------
# Load notebook and parse tables
# -------------------------
nb = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))
cell12 = nb['cells'][11]  # UI Cell 12
outs = cell12.get('outputs', [])

problem_names = []
for out in outs:
    if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
        txt = ''.join(out.get('text', []))
        m = re.search(r'Problem:\s*(.+?)\.\.\.', txt)
        if m:
            problem_names.append(m.group(1).strip())

attempt_tables = []
vote_tables = []
for out in outs:
    if out.get('output_type') not in ('display_data', 'execute_result'):
        continue
    data = out.get('data', {})
    if 'text/plain' not in data:
        continue
    txt = ''.join(data['text/plain']) if isinstance(data['text/plain'], list) else str(data['text/plain'])
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        continue
    header = lines[0]
    if 'Attempt' in header and 'Entropy' in header:
        df = pd.read_fwf(StringIO(txt))
        # Drop accidental unnamed index column if present
        for c in list(df.columns):
            if str(c).startswith('Unnamed'):
                df = df.drop(columns=[c])
        attempt_tables.append(df)
    elif 'Answer' in header and 'Votes' in header and 'Score' in header:
        df = pd.read_fwf(StringIO(txt))
        for c in list(df.columns):
            if str(c).startswith('Unnamed'):
                df = df.drop(columns=[c])
        vote_tables.append(df)

# Map attempts to problems by order
attempt_frames = []
for i, df in enumerate(attempt_tables):
    p = problem_names[i] if i < len(problem_names) else f'Problem {i+1}'
    cur = df.copy()
    cur['Problem'] = p
    cur['VoteWeight'] = 1.0 / cur['Entropy'].clip(lower=1e-9)
    attempt_frames.append(cur)

attempt_all = pd.concat(attempt_frames, ignore_index=True) if attempt_frames else pd.DataFrame()
attempt_csv = FIG_DIR / 'data_attempt_metrics.csv'
attempt_all.to_csv(attempt_csv, index=False)

# -------------------------
# Parse runtime metrics from Aimo Solver-log
# -------------------------
log_text = LOG_PATH.read_text(encoding='utf-8', errors='ignore')

runtime = []

def find_float(pattern):
    m = re.search(pattern, log_text)
    return float(m.group(1)) if m else None

preload = find_float(r'Processed\s+\d+\s+files\s+\([\d.]+\s+GB\)\s+in\s+([\d.]+)\s+seconds')
server = find_float(r'Server is ready \(took\s+([\d.]+)\s+seconds\)')
kernels = find_float(r'Kernels initialized in\s+([\d.]+)\s+seconds')

if preload is not None:
    runtime.append(('Model weight preload', preload))
if server is not None:
    runtime.append(('vLLM server startup', server))
if kernels is not None:
    runtime.append(('Kernel pool init', kernels))

# local sample problem runtimes from timestamped lines in log
problem_blocks = re.findall(
    r'(\d+\.\d+)s\s+\d+\s+\s*Problem:\s*(.+?)\.\.\.[\s\S]*?(\d+\.\d+)s\s+\d+\s+\s*Final Answer:',
    log_text
)
for idx, (t0, pname, t1) in enumerate(problem_blocks, start=1):
    dt = max(0.0, float(t1) - float(t0))
    runtime.append((f'Local sample problem {idx}', dt))

runtime_df = pd.DataFrame(runtime, columns=['Stage', 'Seconds'])
runtime_csv = FIG_DIR / 'data_runtime_breakdown.csv'
runtime_df.to_csv(runtime_csv, index=False)

# -------------------------
# Figure 1: architecture diagram (matplotlib)
# -------------------------
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')

boxes = {
    'A': (0.05, 0.75, 0.2, 0.12, 'Kaggle API\\nProblem Input'),
    'B': (0.32, 0.75, 0.24, 0.12, 'Harmony Prompt Builder'),
    'C': (0.64, 0.75, 0.28, 0.12, 'Parallel Attempts (N=8)'),
    'D': (0.64, 0.53, 0.13, 0.12, 'LLM Stream'),
    'E': (0.79, 0.53, 0.13, 0.12, 'Python Tool'),
    'F': (0.64, 0.31, 0.28, 0.12, 'Answer + Entropy Extraction'),
    'G': (0.64, 0.10, 0.28, 0.12, 'Entropy-Weighted Voting\\nFinal Integer')
}

for _, (x, y, w, h, text) in boxes.items():
    rect = plt.Rectangle((x, y), w, h, ec='black', fc='#e9f2ff', lw=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

arrows = [
    ((0.25, 0.81), (0.32, 0.81)),
    ((0.56, 0.81), (0.64, 0.81)),
    ((0.78, 0.75), (0.70, 0.65)),
    ((0.84, 0.75), (0.85, 0.65)),
    ((0.705, 0.53), (0.705, 0.43)),
    ((0.855, 0.53), (0.855, 0.43)),
    ((0.78, 0.31), (0.78, 0.22)),
]
for (x0, y0), (x1, y1) in arrows:
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

ax.set_title('Figure 1. AIMO Solver (v58) Inference Architecture', fontsize=14, pad=18)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig01_architecture.png', dpi=220)
plt.close(fig)

# -------------------------
# Figure 2: attempt diagnostics from notebook evidence
# -------------------------
fig, ax = plt.subplots(figsize=(10.5, 5.5))
if attempt_all.empty:
    ax.text(0.5, 0.5, 'No attempt-level table parsed from AIMO Solver (v58) notebook outputs',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    diag = attempt_all.groupby('Problem', as_index=False).agg(
        mean_entropy=('Entropy', 'mean'),
        mean_length=('Length', 'mean'),
        attempts=('Attempt', 'count')
    )
    x = np.arange(len(diag))
    bars = ax.bar(x, diag['mean_entropy'], color='#59a14f', width=0.55, label='Mean entropy')
    ax.set_ylabel('Mean entropy')
    ax.set_xticks(x)
    ax.set_xticklabels(diag['Problem'], rotation=12, ha='right')
    ax2 = ax.twinx()
    ax2.plot(x, diag['mean_length'], color='#e15759', marker='o', linewidth=2.0, label='Mean response length')
    ax2.set_ylabel('Mean response length (tokens/chars as parsed)')
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"n={int(diag['attempts'].iloc[i])}",
                ha='center', va='bottom', fontsize=8)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    ax.set_title('Figure 2. AIMO Solver (v58) Attempt Diagnostics from Notebook Evidence')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig02_attempt_diagnostics.png', dpi=220)
plt.close(fig)

# -------------------------
# Figure 3: runtime breakdown
# -------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
if runtime_df.empty:
    ax.text(0.5, 0.5, 'No runtime metrics parsed from Aimo Solver-log', ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    sdf = runtime_df.sort_values('Seconds', ascending=False)
    ax.barh(sdf['Stage'], sdf['Seconds'], color='#4e79a7')
    for y, s in enumerate(sdf['Seconds']):
        ax.text(s + max(sdf['Seconds']) * 0.01, y, f'{s:.2f}s', va='center', fontsize=9)
    ax.set_xlabel('Seconds')
    ax.set_title('Figure 3. AIMO Solver (v58) Runtime Breakdown from Aimo Solver-log')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig03_runtime_breakdown.png', dpi=220)
plt.close(fig)

# -------------------------
# Figure 4: entropy vs vote-weight scatter
# -------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))
if attempt_all.empty:
    ax.text(0.5, 0.5, 'No attempt-level table parsed from notebook outputs',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    for pname, grp in attempt_all.groupby('Problem'):
        ax.scatter(grp['Entropy'], grp['VoteWeight'], label=pname, alpha=0.8)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Vote Weight = 1/Entropy')
    ax.set_title('Figure 4. AIMO Solver (v58) Entropy vs Vote Weight')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig04_entropy_vs_vote_weight.png', dpi=220)
plt.close(fig)

# -------------------------
# Figure 5: execution timeline from log events
# -------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
event_specs = [
    ('Model preload finished', r'^(\d+\.\d+)s\s+\d+\s+Processed\s+\d+\s+files\s+\([\d.]+\s+GB\)\s+in\s+[\d.]+\s+seconds\.$'),
    ('vLLM server ready', r'^(\d+\.\d+)s\s+\d+\s+Server is ready \(took\s+[\d.]+\s+seconds\)\.$'),
    ('Kernel pool ready', r'^(\d+\.\d+)s\s+\d+\s+Kernels initialized in\s+[\d.]+\s+seconds\.$'),
    ('Problem 1 answered', r'^(\d+\.\d+)s\s+\d+\s+Final Answer:\s+0$'),
]

events = []
for name, pattern in event_specs:
    m = re.search(pattern, log_text, flags=re.MULTILINE)
    if m:
        events.append((name, float(m.group(1))))

# Add all final answer events to capture sequence length in v58 run.
for i, m in enumerate(re.finditer(r'^(\d+\.\d+)s\s+\d+\s+Final Answer:\s+.*$', log_text, flags=re.MULTILINE), start=1):
    events.append((f'Final answer {i}', float(m.group(1))))

if not events:
    ax.text(0.5, 0.5, 'No execution events parsed from Aimo Solver-log', ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    # Keep earliest timestamp per label, then sort by time.
    dedup = {}
    for n, t in events:
        dedup[n] = min(t, dedup.get(n, t))
    ev = pd.DataFrame([(k, v) for k, v in dedup.items()], columns=['Event', 'Time'])
    ev = ev.sort_values('Time').reset_index(drop=True)
    t0 = ev['Time'].iloc[0]
    ev['Elapsed'] = ev['Time'] - t0
    ax.plot(ev['Elapsed'], np.arange(len(ev)), marker='o', linestyle='-', color='#f28e2b')
    ax.set_yticks(np.arange(len(ev)))
    ax.set_yticklabels(ev['Event'])
    ax.set_xlabel('Elapsed seconds from first event')
    ax.set_title('Figure 5. AIMO Solver (v58) Execution Timeline from Aimo Solver-log')
    for idx, row in ev.iterrows():
        ax.text(row['Elapsed'] + 0.3, idx, f"t={row['Time']:.1f}s", va='center', fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig05_execution_timeline.png', dpi=220)
plt.close(fig)

# -------------------------
# Figure 6: per-problem entropy distribution
# -------------------------
fig, ax = plt.subplots(figsize=(11, 5.8))
if attempt_all.empty:
    ax.text(0.5, 0.5, 'No attempt-level table parsed from notebook outputs',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    summary = attempt_all.groupby('Problem', as_index=False)['Entropy'].mean()
    ordered_problems = summary.sort_values('Entropy', ascending=False)['Problem'].tolist()
    entropy_groups = [attempt_all.loc[attempt_all['Problem'] == p, 'Entropy'].values for p in ordered_problems]

    bp = ax.boxplot(
        entropy_groups,
        tick_labels=ordered_problems,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=5),
    )
    for patch in bp['boxes']:
        patch.set_facecolor('#cfe8ff')
        patch.set_edgecolor('#2f4b7c')
        patch.set_linewidth(1.2)

    rng = np.random.default_rng(7)
    for i, values in enumerate(entropy_groups, start=1):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=16, alpha=0.55, color='#1f77b4')

    ax.set_ylabel('Attempt entropy')
    ax.set_xticklabels(ordered_problems, rotation=10, ha='right')
    ax.set_title('Figure 6. AIMO Solver (v58) Per-Problem Entropy Distribution')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig06_entropy_distribution.png', dpi=220)
plt.close(fig)

# -------------------------
# Save short manifest
# -------------------------
manifest = (
    'Generated figures:\n'
    '- fig01_architecture.png\n'
    '- fig02_attempt_diagnostics.png\n'
    '- fig03_runtime_breakdown.png\n'
    '- fig04_entropy_vs_vote_weight.png\n'
    '- fig05_execution_timeline.png\n'
    '- fig06_entropy_distribution.png\n\n'
    'Generated data:\n'
    '- data_attempt_metrics.csv\n'
    '- data_runtime_breakdown.csv\n'
)
(FIG_DIR / 'README.txt').write_text(manifest, encoding='utf-8')
print(manifest)

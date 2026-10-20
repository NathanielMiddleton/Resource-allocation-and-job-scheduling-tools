# slurmfit — "what's free, and when will my job fit?"

Two user-facing tools for a Slurm cluster, sharing one library. Python 3.8+, standard library only,
no root, no slurmrestd. Copy the three `.py` files to a directory on `$PATH` (or `module`-load them).

| file | what it is |
|---|---|
| `cluster_view.py` | **Tool 1.** Totals / used / available, current allocations, scheduled future allocations, and gaps (free space per node + cluster-wide free-capacity timeline). |
| `fit_tool.py` | **Tool 2.** Tune a job's requirements (nodes, CPUs, memory, GPUs + type, features, partition, walltime) and see the earliest start change live. Terminal UI, local web UI, or one-shot CLI. |
| `slurmfit.py` | Shared library: parses `scontrol show node -o` and `scontrol show job -o`, builds per-node resource timelines, runs the fit estimator. |

Everything accepts `--demo` to run against a synthetic 44-node cluster when Slurm isn't reachable — useful for trying it out and for testing changes.

## Tool 1: cluster_view.py

```
cluster_view.py                 # whole cluster
cluster_view.py -p gpu          # one partition
cluster_view.py -a              # job tables for all users (default: only yours; resource/gap views are always cluster-wide)
cluster_view.py -u alice        # job tables for one other user
cluster_view.py --gaps-only --min-gpus 1   # "where is at least one GPU free right now, and for how long?"
cluster_view.py --hours 72      # longer free-capacity timeline
cluster_view.py --json          # full snapshot for other scripts
```

Sections, in order:

1. **Resources** — total / used / available / utilisation for nodes, CPUs, memory and each GPU type.
2. **Current allocations** — running jobs sorted by end time.
3. **Future allocations** — pending jobs that Slurm's backfill scheduler has already given a `StartTime` and `SchedNodeList`, sorted by planned start. Pending jobs with no estimate yet are counted at the bottom with their `Reason`.
4. **Gaps: free space on nodes right now** — per node: free CPU / memory / GPUs, whether the node is fully idle or partially used, how long that space is *guaranteed* to stay free (until the next scheduled job starts on it), and what that next change is.
5. **Gaps: cluster-wide free capacity** — a step table of free CPUs / memory / GPUs / idle nodes at every change point over the horizon, so you can see "20 H200s free from 21:30 until 23:54".

## Tool 2: fit_tool.py

### Terminal UI (default)

```
fit_tool.py
fit_tool.py --gpus 2 --gpu-type a100 --time 8:00:00 --mem 64G -c 16   # start from these values
```

Arrow keys: up/down pick a field, left/right change it, space toggles a feature,
`1`–`8` applies one of the "start sooner if you…" suggestions, `r` re-reads Slurm,
`s` prints an `#SBATCH` header for the current settings and quits, `q` quits.

### Web UI

```
fit_tool.py --web            # serves http://localhost:8765 on the login node (localhost only)
```

From a laptop: `ssh -L 8765:localhost:8765 login01` then open <http://localhost:8765>.
Use `--host 0.0.0.0` only if your login node's firewall policy allows it (there is no authentication).

The page shows sliders on the left; on the right the earliest start, which nodes it would land on,
a 48-hour chart of free capacity for the resource you're constrained by with your job's window drawn on it,
clickable suggestions that would start the job earlier, and the matching `#SBATCH` lines.
Below: the pending queue ("planned work") and the running jobs finishing soonest, your own by default,
with a checkbox to show every user's.

### One shot

```
fit_tool.py --once -N 2 -G 4 --gpu-type h200 --mem 500G -c 48 -t 1-00:00:00 -C nvlink -p gpu
```

## How the estimate works

1. Each node's free CPUs / memory / GPUs *right now* come straight from `scontrol show node`
   (`CPUAlloc`, `AllocMem`, `GresUsed`), so the present is exact.
2. Running jobs release their share of each node at their `EndTime` (`TimeLimit` for unlimited jobs = never).
3. Pending jobs that already have a Slurm `StartTime` and `SchedNodeList` reserve their share on those nodes
   from that start for their time limit.
4. That gives a piecewise-constant "free resources" timeline per node. For your request the tool finds, per
   eligible node (partition, features, GPU type, capacity), every window long enough for your walltime, then
   the earliest instant at which `--nodes` of them overlap.
5. Suggestions are single-dimension tweaks (shorter walltime, one GPU fewer, other GPU type, half the memory,
   one node fewer, drop constraints…) re-evaluated the same way; only ones that gain more than a minute are shown.

### What it does not know (tell your users)

* **Priority / fairshare / QOS / account limits.** The estimate is "resources are physically free", i.e. a best
  case. Slurm may still start a higher-priority job first. The *relative* effect of changing requirements is reliable;
  the absolute time is optimistic.
* **Pending jobs with no Slurm estimate yet** are not projected. Both tools say how many there are.
* **Reservations and licenses** are ignored (`scontrol show res` is an easy extension if you use them heavily).
* **Per-node split of multi-node jobs** is approximated as total TRES ÷ node count (present-time usage is still exact).
* Jobs that finish early (most do) make everything better than predicted, not worse.

## Adapting to your site

* Node states treated as unusable: `BAD_NODE_STATES` in `slurmfit.py`.
* Walltime slider steps: `WALL_STEPS` in `fit_tool.py`.
* Snapshot refresh interval for the web/TUI: `SNAPSHOT_TTL` (default 30 s) — raise it on very busy controllers,
  since each refresh is one `scontrol show node` and one `scontrol show job`.
* If your cluster uses MPS or sharded GPUs (`gres/shard`), extend `parse_gres`; currently only `gpu:*` is counted.

## Tests without a cluster

```
python3 cluster_view.py --demo
python3 fit_tool.py --once --demo -G 4 --gpu-type a100 -t 12:00:00
python3 fit_tool.py --web --demo
```

## Tool 3: pending_jobs.py

```
pending_jobs.py                  # your pending jobs: requirements + expected start, soonest first
pending_jobs.py -a               # all users
pending_jobs.py -u alice -p gpu  # another user / one partition
pending_jobs.py --estimate       # add a best-case start (~) for jobs Slurm hasn't scheduled yet
pending_jobs.py --json
```

#!/usr/bin/env python3
"""
pending_jobs.py - pending jobs, what they asked for, and when they are expected to start.

  pending_jobs.py                  your pending jobs, soonest expected start first
  pending_jobs.py -a               all users
  pending_jobs.py -u alice         one other user
  pending_jobs.py -p gpu           one partition
  pending_jobs.py --estimate       for jobs Slurm has not given a start time yet, add this tool's
                                   own best-case estimate (marked with ~)
  pending_jobs.py --json           machine-readable
  pending_jobs.py --demo           synthetic data, no Slurm needed

"start" is Slurm's own backfill estimate (squeue --start). Jobs still waiting on priority usually
have none; --estimate fills those in from free-resource timelines, ignoring priority, so it is a
best case.
"""

import argparse
import getpass
import json
import sys
from slurmfit import (collect, build_timelines, estimate, Request, fmt_mem, fmt_gpus, fmt_dur,
                      fmt_time, INF)


def table(headers, rows):
    if not rows:
        return "  (none)\n"
    widths = [len(h) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  " + "  ".join("{:<%d}" % w for w in widths)
    out = [fmt.format(*headers), "  " + "  ".join("-" * w for w in widths)]
    out += [fmt.format(*r) for r in srows]
    return "\n".join(out) + "\n"


def own_estimate(snap, tl, j):
    """Best-case start for a pending job that Slurm has not scheduled yet."""
    c, m, g = j.per_node()
    gtype, gcount = "any", 0
    if g:
        gtype, gcount = max(g.items(), key=lambda kv: kv[1])
        if gtype == "gpu":
            gtype = "any"
    feats = [f for f in j.features.replace("&", ",").split(",") if f and "|" not in f] if j.features else []
    req = Request(nodes=max(1, j.num_nodes), cpus=max(1, c), mem=m, gpus=gcount, gpu_type=gtype,
                  features=feats, partition=j.partition, wall=j.time_limit or 3600)
    return estimate(snap, req, tl)["start"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-u", "--user", help="show this user's jobs (default: you)")
    ap.add_argument("-a", "--all", action="store_true", help="all users")
    ap.add_argument("-p", "--partition")
    ap.add_argument("--estimate", action="store_true", help="add best-case estimate where Slurm has none")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    user = None if args.all else (args.user or getpass.getuser())
    if args.demo and not args.all and not args.user:
        user = "alice"   # demo data has no real users

    try:
        snap = collect(demo=args.demo)
    except Exception as e:  # noqa
        sys.exit(f"could not read Slurm state: {e}\n(try --demo to test without Slurm)")
    now = snap.now
    jobs = [j for j in snap.pending
            if (not user or j.user == user) and (not args.partition or j.partition == args.partition)]

    tl = build_timelines(snap) if args.estimate else None
    rows = []
    for j in jobs:
        start, kind = j.start, "slurm"
        if start is None and args.estimate:
            start, kind = own_estimate(snap, tl, j), "estimate"
        rows.append((j, start, kind))
    rows.sort(key=lambda r: (r[1] is None, r[1] or 0, r[0].id))

    if args.json:
        print(json.dumps([{
            "job": j.id, "user": j.user, "name": j.name, "partition": j.partition,
            "nodes": j.num_nodes, "cpus": j.cpus, "mem_mb": j.mem, "gpus": j.gpus,
            "features": j.features, "time_limit_s": j.time_limit, "reason": j.reason,
            "start": None if start in (None, INF) else start, "start_source": kind if start else None,
            "planned_nodes": j.nodes,
        } for j, start, kind in rows], indent=1))
        return

    for w in snap.warnings:
        print(f"note: {w}")
    print(f"snapshot: {fmt_time(now)}   {len(rows)} pending job(s)"
          + (f"   user: {user}" if user else "   all users") + (f"   partition: {args.partition}" if args.partition else ""))
    out = []
    for j, start, kind in rows:
        if start is None:
            when, wait = "no estimate", ""
        elif start == INF:
            when, wait = "never fits", ""
        else:
            when = ("~" if kind == "estimate" else "") + fmt_time(start, now)
            wait = fmt_dur(start - now)
        out.append([j.id, j.user, j.name[:20], j.partition, j.num_nodes, j.cpus, fmt_mem(j.mem),
                    fmt_gpus(j.gpus), j.features or "-", fmt_dur(j.time_limit), when, wait,
                    j.reason or "-", ",".join(j.nodes[:3]) + (f",+{len(j.nodes) - 3}" if len(j.nodes) > 3 else "")])
    print(table(["job", "user", "name", "part", "N", "cpus", "mem", "gpus", "features", "limit",
                 "start", "in", "reason", "planned nodes"], out))
    n_none = sum(1 for _, s, _ in rows if s is None)
    if n_none:
        print(f"  {n_none} job(s) have no Slurm start estimate" +
              ("" if args.estimate else "; run with --estimate for a best-case guess"))
    if user:
        print("  showing your jobs only; -a for all users")
    if args.estimate:
        print("  ~ = this tool's estimate (resources free, priority ignored); others are Slurm's own.")


if __name__ == "__main__":
    main()

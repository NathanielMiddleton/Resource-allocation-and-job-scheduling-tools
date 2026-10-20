#!/usr/bin/env python3
"""
fit_tool.py - tune a job's requirements until it fits sooner.

Three ways to use it:

  fit_tool.py                      terminal UI (arrow keys adjust sliders, estimate updates live)
  fit_tool.py --web [--port 8765]  local web page with sliders; on a login node run it, then
                                   from your laptop: ssh -L 8765:localhost:8765 login.cluster
                                   and open http://localhost:8765
  fit_tool.py --once --gpus 2 --gpu-type a100 --time 4:00:00
                                   print the estimate for one set of requirements and exit

Add --demo to any of these to use a synthetic cluster (no Slurm needed).

The estimate is "earliest moment the resources are physically free", based on running jobs'
end times and pending jobs Slurm has already scheduled. Priority, fairshare and QOS limits are
not modelled, so real start times can be later; the *relative* effect of changing requirements
is what the tool is for.
"""

import argparse
import getpass
import json
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from slurmfit import (Request, collect, build_timelines, estimate, suggestions, sbatch_snippet,
                      capacity_curve, parse_duration, parse_mem_mb, fmt_mem, fmt_dur, fmt_time,
                      fmt_eta, fmt_gpus, INF)

WALL_STEPS = [15 * 60, 30 * 60, 3600, 2 * 3600, 4 * 3600, 6 * 3600, 8 * 3600, 12 * 3600,
              24 * 3600, 36 * 3600, 48 * 3600, 72 * 3600, 5 * 86400, 7 * 86400]
SNAPSHOT_TTL = 30  # seconds


# ----------------------------------------------------------------------------
# shared state
# ----------------------------------------------------------------------------

class Cluster:
    """Snapshot + timelines, refreshed lazily."""

    def __init__(self, demo):
        self.demo = demo
        self.lock = threading.Lock()
        self.snap = None
        self.tl = None
        self.loaded_at = 0
        self.refresh(force=True)

    def refresh(self, force=False):
        with self.lock:
            if not force and time.time() - self.loaded_at < SNAPSHOT_TTL:
                return
            self.snap = collect(demo=self.demo)
            self.tl = build_timelines(self.snap)
            self.loaded_at = time.time()

    def limits(self):
        s = self.snap
        usable = [n for n in s.nodes if n.usable] or s.nodes
        return {
            "max_nodes": max(1, min(64, len(usable))),
            "max_cpus": max(n.cpus for n in usable) if usable else 1,
            "max_mem_gb": max(1, max(n.mem for n in usable) // 1024) if usable else 1,
            "max_gpus": max([max(n.gpus.values()) for n in usable if n.gpus] or [0]),
            "gpu_types": s.gpu_types,
            "features": s.feature_list,
            "partitions": s.partition_list,
        }

    def evaluate(self, req, horizon_h=48):
        self.refresh()
        with self.lock:
            snap, tl = self.snap, self.tl
        base = estimate(snap, req, tl)
        sugg = suggestions(snap, req, base, tl)
        curve = capacity_curve(snap, tl, horizon_s=horizon_h * 3600,
                               partition=req.partition or None)
        return {
            "now": snap.now,
            "request": asdict(req),
            "describe": req.describe(),
            "start": base["start"],
            "eta": fmt_eta(base["start"], snap.now),
            "wait_s": None if base["start"] is None else base["start"] - snap.now,
            "nodes": base["nodes"],
            "eligible": base["eligible"],
            "reason": base["reason"],
            "suggestions": [{**x, "eta": fmt_eta(x["start"], snap.now), "gain": fmt_dur(x["gain"])}
                            for x in sugg],
            "sbatch": sbatch_snippet(req),
            "curve": [{**r, "mem_gb": r["mem"] // 1024} for r in curve],
            "warnings": snap.warnings,
        }

    def planned_work(self, user=None):
        with self.lock:
            snap = self.snap
        now = snap.now
        pend = sorted([j for j in snap.pending if not user or j.user == user],
                      key=lambda j: (j.start is None, j.start or 0))
        run = sorted([j for j in snap.running if j.end not in (None, INF) and (not user or j.user == user)],
                     key=lambda j: j.end)
        return {
            "pending": [{"id": j.id, "user": j.user, "name": j.name, "partition": j.partition,
                         "nodes": j.num_nodes, "cpus": j.cpus, "mem": fmt_mem(j.mem),
                         "gpus": fmt_gpus(j.gpus), "limit": fmt_dur(j.time_limit),
                         "start": None if j.start is None else fmt_time(j.start, now),
                         "start_in": None if j.start is None else fmt_dur(j.start - now),
                         "reason": j.reason, "planned": ",".join(j.nodes)} for j in pend],
            "ending": [{"id": j.id, "user": j.user, "name": j.name, "partition": j.partition,
                        "nodes": ",".join(j.nodes[:4]) + ("…" if len(j.nodes) > 4 else ""),
                        "cpus": j.cpus, "mem": fmt_mem(j.mem), "gpus": fmt_gpus(j.gpus),
                        "end": fmt_time(j.end, now), "end_in": fmt_dur(j.end - now)} for j in run[:25]],
            "totals": snap.totals(),
            "user": user,
        }


def request_from_args(a, lim):
    gpu_type = a.gpu_type or ("any" if a.gpus else "any")
    return Request(nodes=a.nodes, cpus=a.cpus, mem=parse_mem_mb(a.mem), gpus=a.gpus, gpu_type=gpu_type,
                   features=[f for f in (a.features or "").split(",") if f],
                   partition=a.partition or "", wall=parse_duration(a.time) or 3600)


def print_result(res):
    print(f"request : {res['describe']}")
    print(f"start   : {res['eta']}")
    if res["nodes"]:
        print(f"on      : {','.join(res['nodes'])}")
    if res["reason"]:
        print(f"why not : {res['reason']}  ({res['eligible']} eligible node(s))")
    if res["suggestions"]:
        print("\nstart sooner if you:")
        for s in res["suggestions"]:
            print(f"  - {s['label']:<42} -> {s['eta']}  (saves {s['gain']})")
    print("\n" + res["sbatch"])
    for w in res["warnings"]:
        print(f"\nnote: {w}")


# ----------------------------------------------------------------------------
# terminal UI
# ----------------------------------------------------------------------------

def run_tui(cluster, req):
    import curses

    lim = cluster.limits()
    gpu_types = ["any"] + lim["gpu_types"]
    partitions = [""] + lim["partitions"]
    feats = lim["features"]
    wall_idx = min(range(len(WALL_STEPS)), key=lambda i: abs(WALL_STEPS[i] - req.wall))
    req.wall = WALL_STEPS[wall_idx]

    fields = ["nodes", "cpus", "mem", "gpus", "gpu_type", "features", "partition", "wall"]
    labels = {"nodes": "Nodes", "cpus": "CPUs / node", "mem": "Memory / node", "gpus": "GPUs / node",
              "gpu_type": "GPU type", "features": "Features", "partition": "Partition", "wall": "Walltime"}
    state = {"field": 0, "feat_idx": 0, "wall_idx": wall_idx, "result": None, "dirty": True, "msg": ""}

    def value_str(f):
        if f == "mem":
            return fmt_mem(req.mem)
        if f == "wall":
            return fmt_dur(req.wall)
        if f == "gpu_type":
            return req.gpu_type
        if f == "partition":
            return req.partition or "any"
        if f == "features":
            if not feats:
                return "(cluster has none)"
            parts = []
            for i, ft in enumerate(feats):
                on = "x" if ft in req.features else " "
                cur = ">" if (fields[state["field"]] == "features" and i == state["feat_idx"]) else " "
                parts.append(f"{cur}[{on}]{ft}")
            return " ".join(parts)
        return str(getattr(req, f))

    def adjust(f, d):
        if f == "nodes":
            req.nodes = max(1, min(lim["max_nodes"], req.nodes + d))
        elif f == "cpus":
            step = 1 if req.cpus < 8 or (d < 0 and req.cpus <= 8) else 4
            req.cpus = max(1, min(lim["max_cpus"], req.cpus + d * step))
        elif f == "mem":
            gb = req.mem // 1024
            step = 1 if gb < 16 or (d < 0 and gb <= 16) else (8 if gb < 128 else 64)
            req.mem = max(1024, min(lim["max_mem_gb"] * 1024, (gb + d * step) * 1024))
        elif f == "gpus":
            req.gpus = max(0, min(lim["max_gpus"], req.gpus + d))
        elif f == "gpu_type":
            i = (gpu_types.index(req.gpu_type) if req.gpu_type in gpu_types else 0) + d
            req.gpu_type = gpu_types[i % len(gpu_types)]
        elif f == "partition":
            i = (partitions.index(req.partition) if req.partition in partitions else 0) + d
            req.partition = partitions[i % len(partitions)]
        elif f == "wall":
            state["wall_idx"] = max(0, min(len(WALL_STEPS) - 1, state["wall_idx"] + d))
            req.wall = WALL_STEPS[state["wall_idx"]]
        elif f == "features" and feats:
            state["feat_idx"] = (state["feat_idx"] + d) % len(feats)
            return
        state["dirty"] = True

    def toggle_feature():
        if not feats:
            return
        ft = feats[state["feat_idx"]]
        if ft in req.features:
            req.features.remove(ft)
        else:
            req.features.append(ft)
        state["dirty"] = True

    def apply_suggestion(k):
        res = state["result"]
        if res and k < len(res["suggestions"]):
            r = res["suggestions"][k]["request"]
            for key, val in r.items():
                setattr(req, key, val)
            state["wall_idx"] = min(range(len(WALL_STEPS)), key=lambda i: abs(WALL_STEPS[i] - req.wall))
            state["dirty"] = True

    def draw(scr):
        scr.erase()
        h, w = scr.getmaxyx()
        y = 0

        def put(line, attr=0):
            nonlocal y
            if y < h - 1:
                scr.addnstr(y, 0, line, w - 1, attr)
                y += 1

        src = "DEMO data" if cluster.demo else "Slurm"
        put(f" fit_tool  [{src}, snapshot {fmt_time(cluster.snap.now)}]   "
            "up/down: field   left/right: change   space: toggle feature   1-8: apply suggestion   "
            "r: refresh   s: print sbatch & quit   q: quit", curses.A_DIM)
        put("")
        for i, f in enumerate(fields):
            sel = i == state["field"]
            put(f"  {'>' if sel else ' '} {labels[f]:<14} {value_str(f)}",
                curses.A_REVERSE if sel else 0)
        put("")
        res = state["result"]
        if res:
            put(f"  Estimated start: {res['eta']}", curses.A_BOLD)
            if res["nodes"]:
                put(f"  would land on:   {', '.join(res['nodes'][:8])}{' …' if len(res['nodes']) > 8 else ''}")
            if res["reason"]:
                put(f"  cannot fit:      {res['reason']}  ({res['eligible']} eligible nodes)")
            put("")
            if res["suggestions"]:
                put("  Start sooner if you (press number to apply):")
                for k, s in enumerate(res["suggestions"], 1):
                    put(f"    {k}. {s['label']:<42} {s['eta']:<28} saves {s['gain']}")
            else:
                put("  No single change to these requirements starts the job earlier." if res["start"]
                    else "  No simple change makes this request fit.")
            put("")
            # mini timeline of the relevant resource over the next 24h
            key = None
            if req.gpus:
                key = ("gpus", req.gpu_type)
            put("  Free " + (f"{req.gpu_type} GPUs" if req.gpus else "CPUs") + " over the next 48h (need "
                + str(req.nodes * (req.gpus or req.cpus)) + "):")
            width = max(20, min(w - 12, 96))
            now = res["now"]
            horizon = 48 * 3600
            curve = res["curve"]
            cells = []
            for i in range(width):
                t = now + horizon * i / width
                val = 0
                for row in curve:
                    if row["t"] <= t:
                        val = (sum(row["gpus"].values()) if key[1] == "any" else row["gpus"].get(key[1], 0)) \
                            if key else row["cpus"]
                cells.append(val)
            mx = max(cells + [1])
            bars = " ▁▂▃▄▅▆▇█"
            line = "".join(bars[min(8, int(8 * v / mx))] for v in cells)
            put("  " + line)
            if res["start"] is not None and res["start"] - now < horizon:
                a = int((res["start"] - now) / horizon * width)
                b = min(width, max(a + 1, int((res["start"] + req.wall - now) / horizon * width)))
                put("  " + " " * a + "^" * (b - a) + "  your job")
            put("  now" + " " * (width - 10) + "+48h")
            for wmsg in res["warnings"]:
                put("  note: " + wmsg, curses.A_DIM)
        if state["msg"]:
            put("  " + state["msg"], curses.A_DIM)
        scr.refresh()

    def loop(scr):
        curses.curs_set(0)
        scr.timeout(200)
        while True:
            if state["dirty"]:
                state["result"] = cluster.evaluate(req)
                state["dirty"] = False
            draw(scr)
            k = scr.getch()
            if k == -1:
                continue
            f = fields[state["field"]]
            if k in (ord("q"), 27):
                return None
            if k == ord("s"):
                return sbatch_snippet(req)
            if k == ord("r"):
                cluster.refresh(force=True)
                state["dirty"] = True
                state["msg"] = "refreshed"
            elif k in (curses.KEY_UP, ord("k")):
                state["field"] = (state["field"] - 1) % len(fields)
            elif k in (curses.KEY_DOWN, ord("j"), 9):
                state["field"] = (state["field"] + 1) % len(fields)
            elif k in (curses.KEY_LEFT, ord("h")):
                adjust(f, -1)
            elif k in (curses.KEY_RIGHT, ord("l")):
                adjust(f, +1)
            elif k in (ord(" "), 10, 13) and f == "features":
                toggle_feature()
            elif ord("1") <= k <= ord("8"):
                apply_suggestion(k - ord("1"))

    out = curses.wrapper(loop)
    if out:
        print(out)


# ----------------------------------------------------------------------------
# web UI
# ----------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Will my job fit?</title>
<style>
:root{color-scheme:light dark;
  --bg:#f1f3f6;--panel:#ffffff;--ink:#1b2430;--muted:#66717f;--line:#d9dee6;
  --free:#1f7a74;--free-soft:#d6ecea;--job:#c9742a;--job-soft:#f8e3cf;--focus:#2b5cb8;
  --font:"Avenir Next","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{
  --bg:#12171e;--panel:#1a2029;--ink:#e4e9f0;--muted:#8d98a6;--line:#2c3540;
  --free:#5fcfc6;--free-soft:#163a38;--job:#f0a35a;--job-soft:#4a2f16;--focus:#7fa6f0}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--font)}
main{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 4px}
h2{font-size:17px;font-weight:600;margin:0 0 12px}
.sub{color:var(--muted);margin:0 0 24px;max-width:70ch}
.grid{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:20px;align-items:start}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.ctl{display:grid;grid-template-columns:1fr auto;gap:2px 12px;align-items:baseline;padding:10px 0;border-top:1px solid var(--line)}
.ctl:first-of-type{border-top:0;padding-top:0}
.ctl label{font-weight:500}
.ctl output{font-variant-numeric:tabular-nums;color:var(--muted)}
.ctl input[type=range]{grid-column:1/-1;width:100%;accent-color:var(--free);margin:4px 0 0}
.ctl select{grid-column:1/-1;margin-top:6px;font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.chips{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chips label{font-weight:400;border:1px solid var(--line);border-radius:999px;padding:2px 10px;cursor:pointer;user-select:none}
.chips input{display:none}
.chips input:checked+span{color:var(--free);font-weight:600}
.chips label:has(input:checked){border-color:var(--free);background:var(--free-soft)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.eta{font-size:44px;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:2px 0 2px}
.eta.never{color:var(--job)}
.when{color:var(--muted)}
.on{margin-top:6px;font-family:var(--mono);font-size:13px;color:var(--muted);word-break:break-all}
svg{width:100%;height:auto;display:block;margin-top:14px}
.legend{display:flex;gap:18px;color:var(--muted);font-size:13px;margin-top:4px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
ol.sugg{margin:8px 0 0;padding:0;list-style:none}
ol.sugg li{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-top:1px solid var(--line)}
ol.sugg li:first-child{border-top:0}
ol.sugg button{font:inherit;background:none;border:0;color:var(--focus);cursor:pointer;padding:0;text-align:left}
ol.sugg button:hover{text-decoration:underline}
.gain{white-space:nowrap;color:var(--free);font-weight:600;font-variant-numeric:tabular-nums}
.gain small{display:block;font-weight:400;color:var(--muted)}
pre{font:13px/1.45 var(--mono);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto;margin:10px 0 0}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
@media (max-width:820px){.row2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:500;color:var(--muted);padding:4px 8px 6px 0;border-bottom:1px solid var(--line)}
td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.scroll{overflow-x:auto}
.note{color:var(--muted);font-size:13px;margin-top:14px}
.stale{color:var(--job)}
@media (prefers-reduced-motion:no-preference){.eta{transition:color .2s}}
</style>
</head>
<body>
<main>
<h1>Will my job fit?</h1>
<p class="sub">Move the sliders and watch the earliest start time. The estimate counts free nodes, running jobs' end times and already-scheduled pending jobs; it does not know your priority, so it is a best case, but the <em>difference</em> between two settings is what matters.</p>

<div class="grid">
  <section class="panel" aria-label="Job requirements">
    <h2>Your job</h2>
    <div class="ctl"><label for="nodes">Nodes</label><output id="nodes-o"></output>
      <input type="range" id="nodes" min="1" value="1"></div>
    <div class="ctl"><label for="cpus">CPUs per node</label><output id="cpus-o"></output>
      <input type="range" id="cpus" min="1" value="4"></div>
    <div class="ctl"><label for="mem">Memory per node</label><output id="mem-o"></output>
      <input type="range" id="mem" min="1" value="16"></div>
    <div class="ctl"><label for="gpus">GPUs per node</label><output id="gpus-o"></output>
      <input type="range" id="gpus" min="0" value="0"></div>
    <div class="ctl"><label for="gpu_type">GPU type</label><span></span>
      <select id="gpu_type"></select></div>
    <div class="ctl"><label>Node features</label><span></span>
      <div class="chips" id="features"></div></div>
    <div class="ctl"><label for="partition">Partition</label><span></span>
      <select id="partition"></select></div>
    <div class="ctl"><label for="wall">Walltime</label><output id="wall-o"></output>
      <input type="range" id="wall" min="0" value="2"></div>
  </section>

  <section class="panel" aria-live="polite">
    <h2>Earliest start</h2>
    <div class="eta" id="eta">…</div>
    <div class="when" id="when"></div>
    <div class="on" id="on"></div>
    <svg id="chart" viewBox="0 0 760 190" role="img" aria-label="Free capacity over the next 48 hours"></svg>
    <div class="legend"><span><i style="background:var(--free-soft);border:1px solid var(--free)"></i>free <b id="legend-res"></b></span>
      <span><i style="background:var(--job-soft);border:1px solid var(--job)"></i>your job</span>
      <span id="need"></span></div>
    <h2 style="margin-top:22px">Start sooner if you…</h2>
    <ol class="sugg" id="sugg"></ol>
    <h2 style="margin-top:22px">Submit it like this</h2>
    <pre id="sbatch"></pre>
    <p class="note" id="warn"></p>
  </section>
</div>

<p class="note" style="margin:22px 0 -8px"><label><input type="checkbox" id="allusers"> Show every user's jobs in the two tables below (default: only yours)</label></p>
<div class="row2">
  <section class="panel">
    <h2>Planned work (pending queue)</h2>
    <div class="scroll"><table id="pending"></table></div>
  </section>
  <section class="panel">
    <h2>Running jobs finishing soonest</h2>
    <div class="scroll"><table id="ending"></table></div>
  </section>
</div>
<p class="note" id="status"></p>
</main>

<script>
const WALL=[900,1800,3600,7200,14400,21600,28800,43200,86400,129600,172800,259200,432000,604800];
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function dur(s){s=Math.max(0,Math.round(s));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);
  if(d)return `${d}d${String(h).padStart(2,'0')}h`;if(h)return `${h}h${String(m).padStart(2,'0')}m`;return `${m}m`;}
function mem(gb){return gb>=1024?(gb/1024).toFixed(1)+'T':gb+'G'}
let lim=null,timer=null,ctrl=null;

function params(){
  const feats=[...document.querySelectorAll('#features input:checked')].map(i=>i.value);
  return {nodes:+$('nodes').value,cpus:+$('cpus').value,mem:(+$('mem').value)*1024,gpus:+$('gpus').value,
    gpu_type:$('gpu_type').value,features:feats.join(','),partition:$('partition').value,wall:WALL[+$('wall').value]};
}
function showValues(){
  const p=params();
  $('nodes-o').textContent=p.nodes;$('cpus-o').textContent=p.cpus;$('mem-o').textContent=mem(p.mem/1024);
  $('gpus-o').textContent=p.gpus||'none';$('wall-o').textContent=dur(p.wall);
  $('gpu_type').disabled=!p.gpus;
}
function schedule(){showValues();clearTimeout(timer);timer=setTimeout(update,120);}
async function update(){
  if(ctrl)ctrl.abort();ctrl=new AbortController();
  const q=new URLSearchParams(params());
  let r;try{r=await(await fetch('/api/estimate?'+q,{signal:ctrl.signal})).json();}catch(e){if(e.name!=='AbortError')$('status').textContent='Lost contact with fit_tool.py — is it still running?';return;}
  render(r);
}
function render(r){
  const e=$('eta');
  if(r.start===null){e.textContent='does not fit';e.classList.add('never');$('when').textContent=r.reason+(r.eligible?` (${r.eligible} eligible nodes)`:'');$('on').textContent='';}
  else{e.classList.remove('never');e.textContent=r.wait_s<=1?'now':'in '+dur(r.wait_s);
    $('when').textContent=new Date(r.start*1000).toLocaleString([], {weekday:'short',hour:'2-digit',minute:'2-digit',month:'short',day:'numeric'});
    $('on').textContent='on '+r.nodes.join(', ');}
  const s=$('sugg');s.innerHTML='';
  if(!r.suggestions.length){s.innerHTML=`<li><span class="when">${r.start===null?'No single change makes this request fit — try several at once.':'Nothing simple starts it sooner; these requirements already fit as early as the cluster allows.'}</span></li>`;}
  for(const x of r.suggestions){const li=document.createElement('li');
    li.innerHTML=`<button type="button">${esc(x.label)}</button><span class="gain">${esc(x.eta.replace(/\s*\(.*\)/,''))}<small>saves ${esc(x.gain)}</small></span>`;
    li.querySelector('button').onclick=()=>applyRequest(x.request);s.appendChild(li);}
  $('sbatch').textContent=r.sbatch;
  $('warn').textContent=r.warnings.join(' ');
  drawChart(r);
  $('status').textContent='Snapshot '+new Date(r.now*1000).toLocaleTimeString()+' — refreshes every 30 s while you move a slider.';
}
function applyRequest(q){
  $('nodes').value=q.nodes;$('cpus').value=q.cpus;$('mem').value=Math.max(1,Math.round(q.mem/1024));$('gpus').value=q.gpus;
  $('gpu_type').value=q.gpu_type;$('partition').value=q.partition;
  let best=0;WALL.forEach((w,i)=>{if(Math.abs(w-q.wall)<Math.abs(WALL[best]-q.wall))best=i});$('wall').value=best;
  document.querySelectorAll('#features input').forEach(i=>i.checked=q.features.includes(i.value));
  schedule();
}
function drawChart(r){
  const p=r.request,W=760,H=190,L=44,R=10,T=14,B=28,hor=48*3600,now=r.now;
  const pick=row=>p.gpus?(p.gpu_type==='any'?Object.values(row.gpus).reduce((a,b)=>a+b,0):(row.gpus[p.gpu_type]||0)):row.cpus;
  const need=p.nodes*(p.gpus||p.cpus);
  const pts=r.curve.map(row=>[row.t,pick(row)]);
  const mx=Math.max(need,...pts.map(q=>q[1]),1);
  const x=t=>L+(W-L-R)*Math.min(1,Math.max(0,(t-now)/hor)),y=v=>T+(H-T-B)*(1-v/mx);
  let d=`M${x(now)},${y(pts[0]?pts[0][1]:0)}`;
  for(let i=0;i<pts.length;i++){const v=pts[i][1],t2=i+1<pts.length?pts[i+1][0]:now+hor;d+=` H${x(t2)}`;if(i+1<pts.length)d+=` V${y(pts[i+1][1])}`;}
  const area=d+` V${y(0)} H${x(now)} Z`;
  let job='';
  if(r.start!==null&&r.start-now<hor){const a=x(r.start),b=x(Math.min(now+hor,r.start+p.wall));
    job=`<rect x="${a}" y="${T}" width="${Math.max(2,b-a)}" height="${H-T-B}" fill="var(--job-soft)" stroke="var(--job)" stroke-width="1.2" rx="2"/>`;}
  let ticks='';
  for(const h of [0,6,12,18,24,30,36,42,48]){const xx=x(now+h*3600);ticks+=`<line x1="${xx}" y1="${T}" x2="${xx}" y2="${H-B}" stroke="var(--line)" stroke-dasharray="2 4"/><text x="${xx}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${h?'+'+h+'h':'now'}</text>`;}
  const yt=[0,Math.round(mx/2),mx].map(v=>`<text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${v}</text>`).join('');
  $('chart').innerHTML=`${ticks}${yt}<path d="${area}" fill="var(--free-soft)"/><path d="${d}" fill="none" stroke="var(--free)" stroke-width="1.8"/>
    <line x1="${L}" y1="${y(need)}" x2="${W-R}" y2="${y(need)}" stroke="var(--job)" stroke-width="1" stroke-dasharray="5 4"/>${job}`;
  $('legend-res').textContent=p.gpus?(p.gpu_type==='any'?'GPUs (any type)':p.gpu_type+' GPUs'):'CPUs';
  $('need').textContent=`dashed line: ${need} ${p.gpus?'GPUs':'CPUs'} you need in total`;
}
function tableRows(id,cols,rows,empty){
  const t=$(id);if(!rows.length){t.innerHTML=`<tr><td class="when">${empty}</td></tr>`;return;}
  t.innerHTML='<tr>'+cols.map(c=>`<th>${c[0]}</th>`).join('')+'</tr>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${esc(c[1](r))}</td>`).join('')+'</tr>').join('');
}
async function loadPlanned(){
  const d=await(await fetch('/api/planned'+($('allusers').checked?'?all=1':''))).json();
  const empty=d.user?`none of yours (${d.user})`:'none';
  tableRows('pending',[['job',r=>r.id],['user',r=>r.user],['part',r=>r.partition],['N',r=>r.nodes],['cpu',r=>r.cpus],['mem',r=>r.mem],['gpu',r=>r.gpus],['limit',r=>r.limit],
    ['expected start',r=>r.start?`${r.start} (${r.start_in})`:'no estimate ('+r.reason+')']],d.pending,'Pending: '+empty+'.');
  tableRows('ending',[['job',r=>r.id],['user',r=>r.user],['nodes',r=>r.nodes],['cpu',r=>r.cpus],['mem',r=>r.mem],['gpu',r=>r.gpus],['ends',r=>`${r.end} (${r.end_in})`]],d.ending,'Running: '+empty+'.');
}
async function init(){
  lim=await(await fetch('/api/limits')).json();
  $('nodes').max=lim.max_nodes;$('cpus').max=lim.max_cpus;$('mem').max=lim.max_mem_gb;$('gpus').max=lim.max_gpus;$('wall').max=WALL.length-1;
  $('gpu_type').innerHTML=['any',...lim.gpu_types].map(t=>`<option>${esc(t)}</option>`).join('');
  $('partition').innerHTML=['',...lim.partitions].map(p=>`<option value="${esc(p)}">${p||'any'}</option>`).join('');
  $('features').innerHTML=lim.features.length?lim.features.map(f=>`<label><input type="checkbox" value="${esc(f)}"><span>${esc(f)}</span></label>`).join(''):'<span class="when">none defined</span>';
  const init=lim.initial;if(init)applyRequest(init);
  document.querySelectorAll('input,select').forEach(el=>{if(el.id!=='allusers')el.addEventListener('input',schedule)});
  $('allusers').addEventListener('change',loadPlanned);
  showValues();update();loadPlanned();setInterval(loadPlanned,60000);
}
init();
</script>
</body>
</html>
"""


def run_web(cluster, req, port, host):
    initial = asdict(req)
    current_user = "alice" if cluster.demo else getpass.getuser()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self._send(HTML, "text/html")
                elif u.path == "/api/limits":
                    self._send(json.dumps({**cluster.limits(), "initial": initial}))
                elif u.path == "/api/planned":
                    cluster.refresh()
                    self._send(json.dumps(cluster.planned_work(None if q.get("all") else current_user)))
                elif u.path == "/api/estimate":
                    r = Request(nodes=int(q.get("nodes", 1)), cpus=int(q.get("cpus", 1)),
                                mem=int(float(q.get("mem", 1024))), gpus=int(q.get("gpus", 0)),
                                gpu_type=q.get("gpu_type", "any") or "any",
                                features=[f for f in q.get("features", "").split(",") if f],
                                partition=q.get("partition", ""), wall=int(float(q.get("wall", 3600))))
                    res = cluster.evaluate(r)
                    self._send(json.dumps(res, default=lambda o: None if o == INF else o))
                else:
                    self.send_error(404)
            except Exception as e:  # noqa
                self.send_error(500, str(e))

    srv = ThreadingHTTPServer((host, port), H)
    print(f"fit_tool web UI: http://{'localhost' if host in ('127.0.0.1', 'localhost') else host}:{port}")
    if host in ("127.0.0.1", "localhost"):
        print(f"  remote? from your machine:  ssh -L {port}:localhost:{port} <this-login-node>")
    print("  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--web", action="store_true", help="serve the browser UI")
    mode.add_argument("--once", action="store_true", help="print one estimate and exit")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address for --web (default localhost only)")
    ap.add_argument("--demo", action="store_true", help="synthetic cluster, no Slurm needed")
    ap.add_argument("-N", "--nodes", type=int, default=1)
    ap.add_argument("-c", "--cpus", type=int, default=4, help="CPUs per node")
    ap.add_argument("--mem", default="16G", help="memory per node, e.g. 64G")
    ap.add_argument("-G", "--gpus", type=int, default=0, help="GPUs per node")
    ap.add_argument("--gpu-type", default=None, help="e.g. a100 (default: any)")
    ap.add_argument("-C", "--features", default="", help="comma-separated node features")
    ap.add_argument("-p", "--partition", default="")
    ap.add_argument("-t", "--time", default="1:00:00", help="walltime, Slurm format")
    args = ap.parse_args()

    try:
        cluster = Cluster(demo=args.demo)
    except Exception as e:  # noqa
        sys.exit(f"could not read Slurm state: {e}\n(try --demo to test without Slurm)")
    req = request_from_args(args, cluster.limits())

    if args.once:
        print_result(cluster.evaluate(req))
    elif args.web:
        run_web(cluster, req, args.port, args.host)
    else:
        try:
            import curses  # noqa
        except ImportError:
            sys.exit("curses is not available in this Python; use --web or --once")
        if not sys.stdout.isatty():
            sys.exit("not a terminal; use --web or --once")
        run_tui(cluster, req)


if __name__ == "__main__":
    main()

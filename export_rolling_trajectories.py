"""Export executed rolling trajectories to one Excel workbook."""
from __future__ import annotations
import argparse, ast, json, random
from pathlib import Path
import pandas as pd
from simulate_continuous import load_full_flight_schedule
from env.taxi_graph_env import TaxiGraphEnv

METHODS = ("MAPPO", "FCFS")
SCENARIOS = ("baseline", "delay", "closure", "priority")

def compact_nodes(values):
    out=[]
    for value in values:
        value=int(value)
        if not out or out[-1] != value:
            out.append(value)
    return out

def graph_lengths():
    env=TaxiGraphEnv(num_agents=1,max_steps=1,close_prob=0.0,delay_prob=0.0,priority_prob=0.0,layout="F",auto_speed=True,random_flow_mix=False)
    return {(int(e.start),int(e.end)): float(e.length) for e in env.edges}

def schedule(meta):
    random.seed(int(meta.get("seed",42)))
    return load_full_flight_schedule(
        meta["excel"], time_scale=float(meta["time_scale"]),
        transit_ratio=float(meta.get("transit_ratio",0.35)),
        start_hour=float(meta.get("start_hour",8.0)), end_hour=float(meta.get("end_hour",24.0)),
    )

def build_for_method(root, scenario, method, flights, lengths, scale):
    prefix=method
    agents=pd.read_csv(root/scenario/f"{prefix}_per_agent.csv")
    raw=pd.read_csv(root/scenario/f"{prefix}_raw_actions.csv")
    node_rows=[]; edge_rows=[]; summary_rows=[]
    for _, rec in agents.iterrows():
        fid=int(rec.flight_index)
        fr=raw[raw.flight_index==fid].sort_values("step")
        events=[]
        if not fr.empty:
            first=fr.iloc[0]
            if int(first.on_edge_before)==0:
                events.append({"sequence":0,"node":int(first.position_before),"arrival_step":int(rec.spawn_step if pd.notna(rec.spawn_step) else first.step),"arrival_time_min":float((rec.spawn_step if pd.notna(rec.spawn_step) else first.step)*scale/60.0)})
            for _, row in fr.iterrows():
                if int(row.on_edge_after)==0:
                    node=int(row.position_after); step=int(row.step)+1
                    if not events or events[-1]["node"] != node:
                        events.append({"sequence":len(events),"node":node,"arrival_step":step,"arrival_time_min":step*scale/60.0})
        nodes=compact_nodes([e["node"] for e in events])
        # Reindex after de-duplication while retaining arrival timestamps.
        events=[e for i,e in enumerate(events) if i==0 or e["node"] != events[i-1]["node"]]
        for e in events:
            node_rows.append({"scenario":scenario,"method":method,"flight_index":fid,**e})
        for idx in range(len(events)-1):
            a,b=events[idx],events[idx+1]
            edge_rows.append({"scenario":scenario,"method":method,"flight_index":fid,"sequence":idx,"edge_start":a["node"],"edge_end":b["node"],"depart_step":a["arrival_step"],"arrive_step":b["arrival_step"],"depart_time_min":a["arrival_time_min"],"arrive_time_min":b["arrival_time_min"],"edge_time_min":b["arrival_time_min"]-a["arrival_time_min"],"edge_length_m":lengths.get((a["node"],b["node"]),lengths.get((b["node"],a["node"]),0.0))})
        info=flights[fid] if fid < len(flights) else {}
        spawn=float(rec.spawn_step)*scale/60.0 if pd.notna(rec.spawn_step) else None
        finish=float(rec.finish_step)*scale/60.0 if pd.notna(rec.finish_step) else None
        elapsed=finish-spawn if spawn is not None and finish is not None else None
        taxi=float(rec.taxi_sec)/60.0 if pd.notna(rec.taxi_sec) else None
        summary_rows.append({"scenario":scenario,"method":method,"flight_index":fid,"agent_role":info.get("agent_role"),"type":info.get("type"),"gate":info.get("gate"),"schedule_time":info.get("time_str"),"start":info.get("start"),"goal":info.get("goal"),"status":rec.status,"priority":rec.priority,"spawn_step":rec.spawn_step,"finish_step":rec.finish_step,"activation_time_min":spawn,"finish_time_min":finish,"elapsed_service_time_min":elapsed,"executed_taxi_time_min":taxi,"moving_time_min":float(rec.moving_sec)/60.0 if pd.notna(rec.moving_sec) else None,"onroute_wait_min":float(rec.onroute_wait_sec)/60.0 if pd.notna(rec.onroute_wait_sec) else None,"queue_wait_min":float(rec.wait_time_min) if pd.notna(rec.wait_time_min) else None,"path_distance_m":float(rec.path_dist) if pd.notna(rec.path_dist) else None,"node_count":len(nodes),"node_sequence":" -> ".join(map(str,nodes))})
    return pd.DataFrame(summary_rows),pd.DataFrame(node_rows),pd.DataFrame(edge_rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--result-root",required=True); ap.add_argument("--out",default=None); args=ap.parse_args()
    root=Path(args.result_root); out=Path(args.out) if args.out else root/"rolling_trajectories.xlsx"
    lengths=graph_lengths(); all_summary=[]; all_nodes=[]; all_edges=[]
    for scenario in SCENARIOS:
        meta=json.loads((root/scenario/"experiment_meta.json").read_text(encoding="utf-8")); flights=schedule(meta); scale=float(meta["time_scale"])
        for method in METHODS:
            s,n,e=build_for_method(root,scenario,method,flights,lengths,scale); all_summary.append(s); all_nodes.append(n); all_edges.append(e)
    summary=pd.concat(all_summary,ignore_index=True); nodes=pd.concat(all_nodes,ignore_index=True); edges=pd.concat(all_edges,ignore_index=True)
    with pd.ExcelWriter(out,engine="openpyxl") as writer:
        summary.to_excel(writer,index=False,sheet_name="trajectory_summary")
        nodes.to_excel(writer,index=False,sheet_name="node_arrivals")
        edges.to_excel(writer,index=False,sheet_name="edge_segments")
        for scenario in SCENARIOS:
            summary[summary.scenario==scenario].to_excel(writer,index=False,sheet_name=f"{scenario}_summary")
    print(f"[OK] trajectory workbook written to {out}")

if __name__=="__main__": main()

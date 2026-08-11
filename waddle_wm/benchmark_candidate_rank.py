"""Paired post-failure candidate ranking benchmark."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from waddle_wm.agent import SkillAgent
from waddle_wm.perception import landing_pad
from waddle_wm.planner import SkillStep
from waddle_wm.sim.env import pick_place_trace

def step(block_xy, target_xy):
    trace = pick_place_trace(block_xy, target_xy, (0, 0))
    for entry in trace:
        if 'target' in entry:
            entry['target'] = [np.float32(v) for v in entry['target']]
    return SkillStep('red block', 'green pad', trace)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--checkpoint', type=Path, default=Path('models/multiblock_world_model.pt'))
    ap.add_argument('--out', type=Path, default=Path('results/benchmark_candidate_rank_50.json'))
    args = ap.parse_args()
    rows=[]
    offsets_cm=(-6,-3,0,3,6)
    for seed in range(args.episodes):
        rng = np.random.default_rng(seed + 10_000)
        source=SkillAgent(args.checkpoint, seed, verifier_mode='none', repairs=0)
        source.observe()
        red=source.env.block_positions()['red_block'][:2]; pad,_=landing_pad(source.env.model)
        perceived_pad = np.asarray(pad) + rng.normal(0, 0.02, 2)
        bad=step(red, [pad[0]+0.15, pad[1]])
        for mode in ('none','world-model'):
            agent=SkillAgent(args.checkpoint, seed, verifier_mode=mode, repairs=0)
            agent.observe(block_xy=red)
            failed, _=agent.execute(bad)
            if failed.get('success') is not False:
                raise RuntimeError(f'seed {seed}: injected 15 cm placement did not fail')
            _, current_latent=agent.observe_current()
            detections = agent.perceive()
            current_red=next(d.point_base[:2] for d in detections if d.label=='red block')
            positions = {d.label.replace(' ', '_'): d.point_base for d in detections}
            positions['green_pad'] = [*perceived_pad, 0.0]
            candidates=[]
            for offset in offsets_cm:
                candidate=step(current_red, [perceived_pad[0]+offset/100, perceived_pad[1]])
                if mode=='world-model':
                    result=agent.verifier.verify(current_latent, candidate.trace, 'red_block', 'green_pad', positions)
                    score=result.success_probability if result else -1
                else:
                    score=-abs(offset)
                candidates.append((score,offset,candidate))
            _, chosen_offset, chosen=max(candidates,key=lambda x:x[0])
            started=time.time(); outcome, _=agent.execute(chosen)
            rows.append({'seed':seed,'mode':mode,'initial_failure':failed.get('failure_mode'),
                         'post_failure_block_xy_cm':[round(100*float(v),2) for v in current_red],
                         'pad_detection_error_cm':round(100*float(np.linalg.norm(perceived_pad-pad)),2),
                         'chosen_offset_cm':chosen_offset,'success':outcome.get('success',True),
                         'failure_mode':outcome.get('failure_mode'),'target_distance_cm':round(100*outcome.get('target_distance',0),2),
                         'latency_s':round(time.time()-started,3)})
    summary={}
    for mode in ('none','world-model'):
        x=[r for r in rows if r['mode']==mode]
        summary[mode]={'n':len(x),'success':sum(r['success'] for r in x)/len(x),
                       'mean_target_distance_cm':round(sum(r['target_distance_cm'] for r in x)/len(x),2),
                       'chosen_offsets_cm':{str(o):sum(r['chosen_offset_cm']==o for r in x) for o in offsets_cm}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'summary':summary,'rows':rows},indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Rescore saved judge outputs offline; never calls model APIs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sarcbench.config import RunConfig
from sarcbench.io import load_jsonl,write_json
from sarcbench.judges import validate_judgment_schema,write_sample_scores
from sarcbench.metrics import filter_judgment_rows,summarize_judgment_rows,judgment_instance_key


def rescore(config,candidate,task):
    folder=config.output_dir/'judgments'/candidate.alias
    raw=load_jsonl(folder/f'{task}_judgments.jsonl')
    valid=[]; malformed=[]
    judge_families={m.alias:m.family for m in config.judge_models}
    seen=set()
    for row in raw:
        try:
            if row.get('candidate_model')!=candidate.alias or row.get('task',task)!=task:
                raise ValueError('candidate/task mismatch in saved judgment')
            validate_judgment_schema(task,row['judgment'])
            key=(judgment_instance_key(row,task),row['judge_model'],str(row['judge_repeat']))
            if key in seen: raise ValueError('duplicate judgment identity')
            seen.add(key)
            row['candidate_family']=candidate.family
            row['judge_family']=judge_families.get(row['judge_model'],row.get('judge_family'))
            valid.append(row)
        except (KeyError,TypeError,ValueError) as error:
            malformed.append({'sample_id':row.get('sample_id'),'error':str(error)})
    metrics=summarize_judgment_rows(valid,task,exclude_self_family=config.protocol.exclude_self_family)
    retained=filter_judgment_rows(valid,exclude_self_family=config.protocol.exclude_self_family)
    predictions=load_jsonl(config.output_dir/'candidates'/candidate.alias/f'{task}_predictions.jsonl')
    expected={judgment_instance_key(row,task) for row in predictions}
    observed={judgment_instance_key(row,task) for row in retained}
    metrics.update({'candidate_model':candidate.alias,'num_malformed_or_duplicate_rows':len(malformed),
                    'num_candidate_predictions':len(predictions),'num_predictions_without_valid_judgments':len(expected-observed),
                    'candidate_denominator_available':bool(predictions),'source':'saved judgments only; no inference rerun'})
    if malformed: write_json(folder/f'{task}_reaggregation_errors.json',{'errors':malformed})
    write_sample_scores(folder/f'{task}_scores_by_sample.jsonl',retained,task,exclude_self_family=config.protocol.exclude_self_family)
    write_json(folder/f'{task}_judge_metrics.json',metrics)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/paper_models.json')); p.add_argument('--models',default='all'); p.add_argument('--tasks',default='task2,task3'); args=p.parse_args()
    config=RunConfig.load(args.config)
    for candidate in config.select_candidates(args.models):
        for task in args.tasks.split(','):
            if task not in {'task2','task3'}: raise ValueError(f'Unsupported task {task}')
            file=config.output_dir/'judgments'/candidate.alias/f'{task}_judgments.jsonl'
            if not file.exists(): raise FileNotFoundError(f'Missing original judgment records: {file}; aggregate tables cannot be used to reconstruct per-sample judgments.')
            m=rescore(config,candidate,task); print(json.dumps({'candidate':candidate.alias,'task':task,'mean_score_pct':m['mean_score_pct'],'valid_instances':m['num_samples'],'retained_judgments':m['num_judgments'],'self_excluded':m['num_self_judgments_excluded']}))

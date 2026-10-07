#!/usr/bin/env python3
"""Recompute descriptive dataset statistics; no model requests or imputed values."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def audit(data):
    task1=list(rows(data/'task1_benchmark.jsonl'))
    master=list(rows(data/'benchmark_master.jsonl'))
    n=len(task1)
    labels=Counter(); full_cardinality=Counter(); canonical_cardinality=Counter()
    canonical=('polarity_reversal','exaggeration','contrast_with_context','rhetorical_question','understatement','sarcasm_marker')
    for row in task1:
        label_set=set(row['gold']['Mechanism'])
        labels.update(label_set); full_cardinality[len(label_set)]+=1
        canonical_cardinality[len(label_set & set(canonical))]+=1
    total_canonical=sum(labels[k] for k in canonical)
    gold_shares={k:labels[k]/total_canonical for k in canonical}
    gold_entropy=-sum(v*math.log2(v) for v in gold_shares.values() if v)
    context=Counter(); context_by_language={}; intent=Counter(); strengths=Counter(); models=Counter(); reconstructed=Counter()
    for row in master:
        core=row['core_annotation']; language=row['prompt_language']
        context_by_language.setdefault(language,Counter())
        span=core['context_dependency_span'] if core['requires_context'] else 'independent'
        context.update([span]); context_by_language[language].update([span])
        intent.update([core['intent_label']]); strengths.update([core['sarcasm_strength']])
        models.update([row['annotation_metadata']['annotator_model']]); reconstructed.update([row['reconstruction_action']])
    language_by_id={str(r['sample_id']):r['prompt_language'] for r in master}
    context_lengths={}
    for task in ['task1','task3']:
        totals=Counter(); counts=Counter()
        for row in rows(data/f'{task}_benchmark.jsonl'):
            language=language_by_id[str(row['sample_id'])]
            for group in ['all',language]:
                totals[group]+=len(row['comments']); counts[group]+=1
        context_lengths[task]={key:totals[key]/counts[key] for key in counts}
    task_ids={str(r['sample_id']) for r in task1}
    views={}
    for task in ['task1','task2','task3']:
        records=list(rows(data/f'{task}_benchmark.jsonl')); ids=[str(r['sample_id']) for r in records]
        views[task]={'n':len(ids),'unique_ids':len(set(ids)),'numeric_ids':sum(i.isdigit() for i in ids),'aligned_ids':set(ids)==task_ids}
        if task=='task2': views[task]['three_style_references']=sum(set(r['gold'].get('reference_replies',{}))=={'teasing','pushback','neutral'} for r in records)
    dependent=n-context['independent']
    return {'events':n,'base_task_records':3*n,'full_context_queries_per_model':5*n,
            'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob('*.jsonl'))},
            'comment_character_means':context_lengths,'task_views':views,'label_counts_including_other':dict(labels),'full_label_cardinality':dict(full_cardinality),
            'canonical_label_cardinality':dict(canonical_cardinality),'canonical_annotation_total':total_canonical,
            'canonical_prevalence_pct':{k:100*labels[k]/n for k in canonical},
            'gold_canonical_share_pct':{k:100*v for k,v in gold_shares.items()},'gold_canonical_entropy_bits':gold_entropy,
            'other_label_instances':labels['other'],'mean_full_label_cardinality':sum(k*v for k,v in full_cardinality.items())/n,
            'context_counts':dict(context),'context_dependent_n':dependent,'context_dependent_pct':100*dependent/n,
            'conditional_context_pct':{k:100*v/dependent for k,v in context.items() if k!='independent'},
            'context_by_language':{k:dict(v) for k,v in context_by_language.items()},
            'intent_counts':dict(intent),'intent_prevalence_pct':{k:100*v/n for k,v in intent.items()},
            'strength_counts':dict(strengths),'annotation_models':dict(models),'reconstruction_counts':dict(reconstructed)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',type=Path,default=Path('data')); parser.add_argument('--output',type=Path,default=Path('docs/dataset_statistics.json')); args=parser.parse_args()
    result=audit(args.data_dir); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['events','context_counts','full_label_cardinality','canonical_label_cardinality','gold_canonical_entropy_bits','other_label_instances','task_views']},indent=2))

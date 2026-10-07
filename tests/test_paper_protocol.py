import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sarcbench.config import ModelSpec, RunConfig
from sarcbench.metrics import SCORED_CRITERIA, summarize_judgment_rows, aggregate_task1
from sarcbench.http_client import call_openai_chat
from sarcbench.judges import run_judge_task, write_sample_scores
from sarcbench.candidates import run_candidate_task
from sarcbench.prompts import render_task2_judge_prompt


def judgment(task, truths=6, overall=False):
    criteria = SCORED_CRITERIA[task]
    return {**{key: i < truths for i, key in enumerate(criteria)}, 'overall_pass': overall,
            'explanation': {**{key: 'Evidence.' for key in criteria}, 'overall': 'Evidence.'}}


def row(sample, judge, truths=6, candidate='gpt-5.4', style=None):
    result = {'sample_id':sample,'candidate_model':candidate,'judge_model':judge,
              'judge_repeat':'1','judgment':judgment('task2' if style else 'task3',truths)}
    if style: result['requested_style']=style
    return result


class ScoringTests(unittest.TestCase):
    def test_empty_understanding_does_not_report_zero_accuracy(self):
        self.assertIsNone(aggregate_task1([])["task1_macro_accuracy_pct"])

    def test_overall_pass_does_not_change_six_criterion_score(self):
        r = row('1','gemini-3.1-pro',5)
        first = summarize_judgment_rows([r],'task3')
        self.assertAlmostEqual(first['mean_score_pct'],100*5/6)
        r['judgment']['overall_pass']=True
        self.assertEqual(first['mean_score'],summarize_judgment_rows([r],'task3')['mean_score'])

    def test_other_version_of_candidate_family_excluded(self):
        rows=[row('1','gpt-5.5',6),row('1','gemini-3.1-pro',0),row('1','claude-opus-4.6',3)]
        result=summarize_judgment_rows(rows,'task3')
        self.assertEqual(result['num_self_judgments_excluded'],1)
        self.assertEqual(result['num_judgments'],2)
        self.assertAlmostEqual(result['mean_score_pct'],25)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scores.jsonl'; write_sample_scores(path,rows,'task3')
            score=json.loads(path.read_text())
            self.assertEqual(len(score['explanations']),2)
            self.assertAlmostEqual(score['score_pct'],25)

    def test_equal_instance_weighting_with_failed_judge_calls(self):
        rows=[row('1','gemini-3.1-pro',6),row('1','claude-opus-4.6',6),row('2','gemini-3.1-pro',0)]
        result=summarize_judgment_rows(rows,'task3')
        self.assertEqual(result['mean_score_pct'],50)
        self.assertEqual(result['criteria_pct']['meaning_preserved'],50)

    def test_style_instances_are_separate(self):
        rows=[row('1','gemini-3.1-pro',6,style='neutral'),row('1','gemini-3.1-pro',0,style='teasing')]
        result=summarize_judgment_rows(rows,'task2')
        self.assertEqual(result['num_samples'],2)
        self.assertEqual(result['by_style']['neutral']['mean_score_pct'],100)
        self.assertEqual(result['by_style']['teasing']['mean_score_pct'],0)

    def test_no_retained_judgments_is_missing_not_zero(self):
        result=summarize_judgment_rows([row('1','gpt-5.5')],'task3')
        self.assertIsNone(result['mean_score'])
        self.assertIsNone(result['mean_score_pct'])

    def test_malformed_criteria_rejected(self):
        r=row('1','gemini-3.1-pro'); r['judgment']['natural_style']='true'
        with self.assertRaises(ValueError): summarize_judgment_rows([r],'task3')


class RequestAndReferenceTests(unittest.TestCase):
    def test_openai_compatible_request_omits_unset_output_limit(self):
        spec=ModelSpec('gpt-5.4','GPT-5.4','openai_compatible','provider_api','gpt-5.4',base_url='https://example.test/v1',api_key='test')
        with patch('sarcbench.http_client.request_json',return_value={'choices':[{'message':{'content':'reply'}}]}) as http:
            call_openai_chat(spec,'prompt',temperature=0,timeout=120,max_attempts=1)
        self.assertEqual(http.call_args.args[0],'https://example.test/v1/chat/completions')
        self.assertNotIn('max_tokens',http.call_args.args[1])
        self.assertEqual(http.call_args.args[1]['temperature'],0)

    def test_style_specific_reference_used(self):
        record={'comments':'thread','gold':{'reference_replies':{'neutral':'calm','teasing':'joke','pushback':'disagree'}}}
        text=render_task2_judge_prompt('__REFERENCE_REPLY__ __REFERENCE_STYLE__ __RESPONSE_STYLE__',record,'candidate',requested_style='neutral')
        self.assertEqual(text,'calm neutral neutral')
        with self.assertRaises(ValueError): render_task2_judge_prompt('__REFERENCE_REPLY__',record,'candidate',requested_style='unknown')

    def test_legacy_reference_style_preserved(self):
        record={'comments':'thread','gold':{'reference_reply':'joke','reply_style_label':'teasing'}}
        text=render_task2_judge_prompt('__REFERENCE_REPLY__ __REFERENCE_STYLE__ __RESPONSE_STYLE__',record,'candidate',requested_style='neutral')
        self.assertEqual(text,'joke teasing neutral')

    def test_paper_roster_and_decoding(self):
        config=RunConfig.load(Path('configs/paper_models.json'))
        aliases={m.alias for m in config.candidate_models}
        self.assertEqual(len(aliases),12)
        self.assertTrue({'gpt-5.4','claude-opus-4.6','deepseek-v3.2','qwen-3.6-plus','minimax-2.7'} <= aliases)
        for model in config.candidate_models+config.judge_models:
            self.assertIsNone(model.max_tokens)
            self.assertEqual(model.provider,'openai_compatible')
        for model in config.candidate_models:
            self.assertEqual(model.temperature,0.7 if model.family=='qwen' else 0.0)

    def test_resumed_prefix_does_not_count_other_cached_instances(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'data').mkdir(); output=root/'out'; output.mkdir()
            records=[{'sample_id':str(i),'comments':'context','gold':{}} for i in [1,2]]
            (root/'data/task1_benchmark.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            metric={'intent_exact':True,'target_exact':True,'mechanism_exact':True,'mechanism_f1':1,'joint_exact':True,'task1_macro_accuracy':1}
            (output/'task1_predictions.jsonl').write_text(''.join(json.dumps({'sample_id':str(i),'metrics':metric})+'\n' for i in [1,2]))
            payload={'paths':{'data_dir':str(root/'data'),'output_dir':str(output)},'candidate_models':[{'alias':'gpt-5.4','provider':'openai','model':'gpt-5.4'}],'judge_models':[]}
            (root/'config.json').write_text(json.dumps(payload)); config=RunConfig.load(root/'config.json')
            with patch('sarcbench.candidates.call_candidate_once') as call:
                result=run_candidate_task(config,config.candidate_models[0],'task1','',output,task2_styles=['neutral'],limit=1,reset=False,verbose=False)
            call.assert_not_called(); self.assertEqual(result['n'],1); self.assertEqual(result['num_valid'],1); self.assertEqual(result['num_attempted'],1)

    def test_self_family_calls_not_dispatched(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'data').mkdir(); (root/'out/candidates/gpt-5.4').mkdir(parents=True)
            record={'sample_id':'1','comments':'context','intended_meaning':'meaning','gold':{'target':'target','preferred_mechanisms':['exaggeration'],'reference_sarcastic_comment':'reference'}}
            (root/'data/task3_benchmark.jsonl').write_text(json.dumps(record)+'\n')
            (root/'out/candidates/gpt-5.4/task3_predictions.jsonl').write_text(json.dumps({'sample_id':'1','candidate_model':'gpt-5.4','prediction':'reply'})+'\n')
            payload={'paths':{'data_dir':str(root/'data'),'output_dir':str(root/'out')},'protocol':{'judge_repeats':3},'candidate_models':[{'alias':'gpt-5.4','provider':'openai','model':'gpt-5.4'}],'judge_models':[{'alias':'gpt-5.5','provider':'openai','model':'gpt-5.5'},{'alias':'gemini-3.1-pro','provider':'gemini','model':'gemini-3.1-pro'}]}
            (root/'config.json').write_text(json.dumps(payload)); config=RunConfig.load(root/'config.json'); dest=root/'out/judgments/gpt-5.4'; dest.mkdir(parents=True)
            def fake(config,task,prompts,benchmark,prediction,judge,repeat):
                return {**row('1',judge.alias,5),'judge_repeat':str(repeat)}
            with patch('sarcbench.judges.call_judge_once',side_effect=fake) as call:
                result=run_judge_task(config,config.candidate_models[0],config.judge_models,'task3',{},dest,limit=None,reset=False,verbose=False)
            self.assertEqual(call.call_count,3)
            self.assertEqual(result['expected_judgments_per_sample'],3)
            self.assertEqual(result['num_predictions_without_valid_judgments'],0)


if __name__=='__main__': unittest.main()

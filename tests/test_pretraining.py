from dataclasses import replace
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
import zipfile
import json

from thoughtformula import (BPETokenizer, ModernMicroGPT, TrainingConfig, TrainingState, fit, evaluate,
                           save_training_checkpoint, load_training_checkpoint, reset_optimizer, token_sequences)
from thoughtformula.pretraining import parameters


def initialized(seed=7):
    random.seed(seed)
    tokenizer=BPETokenizer();tokenizer.train('hello hello world',max_merges=0)
    model=ModernMicroGPT(len(tokenizer.vocab),n_embd=8,n_heads=2,n_kv_heads=1,n_layers=1,n_experts=1,moe_top_k=1,expert_hidden=8,lora_rank=2)
    ids=tokenizer.encode('hello world')
    pairs=[(ids[:4],ids[1:5]),(ids[2:6],ids[3:7])]
    return model,tokenizer,pairs


class PretrainingTests(unittest.TestCase):
    def test_resume_preserves_float_weights_adam_and_order_exactly(self):
        model,tokenizer,pairs=initialized()
        config=TrainingConfig(max_steps=5,block_size=4,max_seconds=10)
        continuous=fit(model,pairs,config)
        expected={key:(list(t.data),list(t.m) if t.m is not None else None,list(t.v) if t.v is not None else None) for key,t in parameters(model).items()}
        partial,tokenizer,pairs=initialized()
        result=fit(partial,pairs,config,stop=lambda:partial.total_steps>=2)
        self.assertEqual(result.status,'paused')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.tft'
            save_training_checkpoint(path,partial,tokenizer,config,result.state,metadata={'corpus':'fixture-v1'})
            restored,restored_tokenizer,restored_config,state,metadata=load_training_checkpoint(path)
            self.assertEqual(metadata,{'corpus':'fixture-v1'})
            self.assertEqual(restored_tokenizer.to_dict(),tokenizer.to_dict())
            completed=fit(restored,pairs,restored_config,state=state)
            self.assertEqual((completed.state.step,completed.state.tokens),(continuous.state.step,continuous.state.tokens))
            for key,tensor in parameters(restored).items():
                actual=(list(tensor.data),list(tensor.m) if tensor.m is not None else None,list(tensor.v) if tensor.v is not None else None)
                self.assertEqual(actual,expected[key],key)
            with self.assertRaises(FileExistsError): save_training_checkpoint(path,restored,tokenizer,config,completed.state)

    def test_loss_and_cooperative_limits_are_observable(self):
        model,tokenizer,pairs=initialized()
        before=evaluate(model,pairs)
        metrics=[]
        result=fit(model,pairs,TrainingConfig(max_steps=20,block_size=4,max_seconds=10),observer=metrics.append)
        after=evaluate(model,pairs)
        self.assertEqual(result.status,'completed')
        self.assertLess(after['loss'],before['loss'])
        self.assertEqual(after['tokens'],8)
        self.assertEqual(metrics[-1].tokens,80)
        paused=fit(model,pairs,TrainingConfig(max_steps=21,block_size=4),state=result.state,stop=lambda:True)
        self.assertEqual(paused.status,'paused')
        limited=fit(model,pairs,TrainingConfig(max_steps=21,block_size=4,max_seconds=0.001),state=result.state)
        self.assertEqual(limited.status,'limited')

    def test_new_specialization_resets_optimizer_without_changing_weights(self):
        model,tokenizer,pairs=initialized()
        fit(model,pairs,TrainingConfig(max_steps=2,block_size=4))
        previous={key:list(tensor.data) for key,tensor in parameters(model).items()}
        reset_optimizer(model)
        self.assertEqual(model.total_steps,0)
        for key,tensor in parameters(model).items():
            self.assertEqual(list(tensor.data),previous[key])
            self.assertIsNone(tensor.m)
        self.assertEqual(fit(model,pairs,TrainingConfig(max_steps=1,block_size=4)).state.step,1)

    def test_token_windows_preserve_documents_and_refuse_unknown_text(self):
        model,tokenizer,pairs=initialized()
        result=token_sequences(tokenizer,['hi','world'],block_size=2)
        self.assertTrue(all(len(left)==len(right)<=2 for left,right in result))
        self.assertEqual(result[0][0],tuple(tokenizer.encode('hi')))
        self.assertEqual(result[0][1][-1],tokenizer.eos_id)
        with self.assertRaises(ValueError): token_sequences(tokenizer,['🙂'])

    def test_config_pairs_and_artifacts_are_bounded(self):
        model,tokenizer,pairs=initialized()
        for kwargs in ({'max_steps':True},{'block_size':0},{'max_seconds':float('inf')}):
            with self.assertRaises(ValueError): TrainingConfig(**kwargs)
        with self.assertRaises(ValueError): fit(model,[([999],[1])],TrainingConfig(max_steps=1))
        with self.assertRaises(ValueError): fit(model,pairs,TrainingConfig(max_steps=1),state=TrainingState(step=1))
        with TemporaryDirectory() as directory:
            good=Path(directory)/'good.tft';bad=Path(directory)/'bad.tft'
            save_training_checkpoint(good,model,tokenizer,TrainingConfig(),TrainingState())
            with zipfile.ZipFile(good) as archive: contents={name:archive.read(name) for name in archive.namelist()}
            manifest=json.loads(contents['manifest.json']);manifest['architecture']['n_embd']=1000000000
            contents['manifest.json']=json.dumps(manifest).encode()
            with zipfile.ZipFile(bad,'w') as archive:
                for name,raw in contents.items(): archive.writestr(name,raw)
            with self.assertRaises(ValueError): load_training_checkpoint(bad)

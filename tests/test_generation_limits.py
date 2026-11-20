import unittest
from unittest.mock import Mock

from thoughtformula import ModernMicroGPT, NGramDraftModel


class GenerationLimitTests(unittest.TestCase):
    def model(self):
        return ModernMicroGPT(8,n_embd=8,n_heads=2,n_kv_heads=1,n_layers=1,n_experts=1,moe_top_k=1,expert_hidden=8,lora_rank=2)

    def methods(self,model,prompt,budget):
        drafter=NGramDraftModel();drafter.fit([[1,2,3,4]])
        options=dict(max_new_tokens=budget,temperature=0,use_int8=False)
        return [lambda:model.generate(prompt,**options), lambda:list(model.generate_stream(prompt,**options)),
                lambda:model.generate_speculative(prompt,drafter,**options),
                lambda:list(model.generate_speculative_stream(prompt,drafter,**options)),
                lambda:model.generate_medusa(prompt,**options),lambda:list(model.generate_medusa_stream(prompt,**options))]

    def test_zero_budget_does_no_forward_cache_or_medusa_work(self):
        model=self.model()
        model.forward=Mock(side_effect=AssertionError('Unexpected forward'))
        model.init_kv_cache=Mock(side_effect=AssertionError('Unexpected cache'))
        model.init_medusa=Mock(side_effect=AssertionError('Unexpected Medusa initialization'))
        values=[method() for method in self.methods(model,[1,2],0)]
        self.assertEqual(values,[[1,2],[],([1,2],0,0,0),[],([1,2],0,0,0),[]])
        self.assertFalse(model.forward.called)

    def test_invalid_budgets_prompts_and_context_fail_before_forward(self):
        for prompt,budget in [([1],-1),([1],True),([1],1.5),([True],1),([-1],1),([8],1),([1]*256,1),([1],256)]:
            for method in self.methods(self.model(),prompt,budget):
                with self.assertRaises(ValueError): method()

    def test_empty_stream_shapes_and_small_context_capacity(self):
        self.assertEqual([method() for method in self.methods(self.model(),[],0)], [[],[],([],0,0,0),[],([],0,0,0),[]])
        model=self.model();model.max_seq_len=4
        for index,method in enumerate(self.methods(model,[1,2],2)):
            value=method()
            output=value[0] if isinstance(value,tuple) else value
            self.assertEqual(len(output),2 if index%2 else 4)

    def test_streaming_and_nonstream_work_with_paged_cache(self):
        model=self.model();original=model.init_kv_cache
        model.init_kv_cache=lambda:original(use_paged=True,page_size=2)
        for index,method in enumerate(self.methods(model,[1,2],3)):
            value=method()
            output=value[0] if isinstance(value,tuple) else value
            self.assertEqual(len(output),3 if index%2 else 5)

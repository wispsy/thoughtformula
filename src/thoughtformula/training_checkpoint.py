"""Atomic lossless training artifacts, independent of quantized inference exports."""
from array import array
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile
import zipfile

from .model import ModernMicroGPT
from .tokenizer import BPETokenizer
from .pretraining import TrainingConfig,TrainingState,integer,parameter_count,parameters

FORMAT='thoughtformula-training/1'
MAX_BYTES=64*1024*1024
FIELDS=('vocab_size','n_embd','n_heads','n_kv_heads','n_layers','n_experts','moe_top_k','expert_hidden','lora_rank','lora_alpha','moe_aux_loss_coeff','use_flash_attention','flash_block_size')


def _pairs(items):
    result={}
    for key,value in items:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def _decode(raw):
    return json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda _v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def validate_model_config(value):
    if type(value) is not dict or set(value)!=set(FIELDS): raise ValueError('Invalid model configuration')
    bounds={'vocab_size':(4,8192),'n_embd':(4,256),'n_heads':(1,32),'n_kv_heads':(1,32),
            'n_layers':(1,8),'n_experts':(1,16),'moe_top_k':(1,16),'expert_hidden':(1,2048),'lora_rank':(1,64),'flash_block_size':(1,256)}
    for name,(minimum,maximum) in bounds.items(): integer(value[name],name,minimum,maximum)
    d=value['n_embd'];heads=value['n_heads'];kv=value['n_kv_heads']
    if d%heads or (d//heads)%2 or heads%kv or value['moe_top_k']>value['n_experts']:
        raise ValueError('Incompatible Transformer dimensions')
    for name in ('lora_alpha','moe_aux_loss_coeff'):
        if type(value[name]) not in (int,float) or not math.isfinite(value[name]) or not 0<=value[name]<=128:
            raise ValueError('Invalid finite model setting')
    if type(value['use_flash_attention']) is not bool: raise ValueError('Invalid attention flag')
    vocab=value['vocab_size'];layers=value['n_layers'];rank=value['lora_rank']
    base=2*vocab*d+vocab+layers*(2*d*d+2*d*(d//heads*kv)+d*value['n_experts']+3*d*value['expert_hidden']*value['n_experts'])
    estimate=base+max(6,rank)*(d+vocab)+layers*8*d*rank
    if estimate>2000000: raise ValueError('Training artifact exceeds the selected parameter ceiling')
    return dict(value)


def save_training_checkpoint(path,model,tokenizer,config,state,*,metadata=None,overwrite=False):
    """Publish a complete artifact; overwrite is explicit for an owned run slot."""
    path=Path(path)
    if type(config) is not TrainingConfig or type(state) is not TrainingState or type(overwrite) is not bool:
        raise TypeError('Select typed training configuration/state and an explicit overwrite flag')
    architecture=validate_model_config({key:getattr(model,key) for key in FIELDS})
    if model.medusa_drafter is not None: raise ValueError('Auxiliary Medusa heads are outside this training artifact profile')
    if tokenizer.vocab and len(tokenizer.vocab)!=model.vocab_size: raise ValueError('Tokenizer vocabulary differs from model')
    counter=model.lora_steps if config.lora_only else model.total_steps
    if counter!=state.step or state.step>config.max_steps: raise ValueError('Model and training progress differ')
    tensors=parameters(model)
    if any(any(value!=0 for value in tensor.grad) for tensor in tensors.values()): raise ValueError('Save training state only at committed optimizer boundaries')
    if parameter_count(model)>2000000: raise ValueError('Too many model parameters')
    blocks={};manifest={}
    for index,(name,tensor) in enumerate(tensors.items()):
        entries={}
        for field in ('data','m','v'):
            value=getattr(tensor,field)
            if value is None: entries[field]=None;continue
            if len(value)!=tensor.rows*tensor.cols or any(not math.isfinite(number) for number in value):
                raise ValueError('Invalid finite parameter/optimizer state')
            raw=struct.pack('<'+str(len(value))+'d',*value)
            member=f'tensors/{index}-{field}.bin';blocks[member]=raw
            entries[field]={'member':member,'sha256':hashlib.sha256(raw).hexdigest()}
        manifest[name]={'rows':tensor.rows,'cols':tensor.cols,'requires_grad':tensor.requires_grad,**entries}
    payload={'format':FORMAT,'architecture':architecture,'context_capacity':model.max_seq_len,
             'config':asdict(config),'state':asdict(state),'total_steps':model.total_steps,'lora_steps':model.lora_steps,
             'tokenizer':tokenizer.to_dict(),'tensors':manifest,'metadata':metadata or {}}
    header=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    if len(header)>1024*1024 or sum(map(len,blocks.values()))+len(header)>MAX_BYTES:
        raise ValueError('Training artifact exceeds byte limits')
    temporary=None
    try:
        with NamedTemporaryFile(prefix='.training-',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json',header)
                for name,raw in blocks.items(): archive.writestr(name,raw)
            stream.flush();os.fsync(stream.fileno())
        if overwrite: os.replace(temporary,path)
        else: os.link(temporary,path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def load_training_checkpoint(path):
    """Read bounded passive state; no pickle/import/download or code execution."""
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist()
        if len(members)>4096 or len({m.filename for m in members})!=len(members) or sum(m.file_size for m in members)>MAX_BYTES:
            raise ValueError('Invalid bounded training container')
        if archive.getinfo('manifest.json').file_size>1024*1024: raise ValueError('Manifest too large')
        payload=_decode(archive.read('manifest.json'))
        expected={'format','architecture','context_capacity','config','state','total_steps','lora_steps','tokenizer','tensors','metadata'}
        if type(payload) is not dict or set(payload)!=expected or payload['format']!=FORMAT:
            raise ValueError('Unsupported training artifact')
        architecture=validate_model_config(payload['architecture'])
        config=TrainingConfig(**payload['config']);state=TrainingState(**payload['state'])
        integer(payload['context_capacity'],'Context capacity',1,256)
        integer(payload['total_steps'],'Full steps',0,1000000);integer(payload['lora_steps'],'LoRA steps',0,1000000)
        tokenizer=BPETokenizer.from_dict(payload['tokenizer'])
        if len(tokenizer.vocab)!=architecture['vocab_size']: raise ValueError('Tokenizer/model mismatch')
        model=ModernMicroGPT(**architecture)
        tensors=parameters(model)
        if type(payload['tensors']) is not dict or set(payload['tensors'])!=set(tensors): raise ValueError('Tensor identity mismatch')
        expected_members={'manifest.json'}
        for name,tensor in tensors.items():
            entry=payload['tensors'][name]
            if set(entry)!={'rows','cols','requires_grad','data','m','v'} or entry['rows']!=tensor.rows or entry['cols']!=tensor.cols or type(entry['requires_grad']) is not bool:
                raise ValueError('Tensor shape/type mismatch')
            size=tensor.rows*tensor.cols
            for field in ('data','m','v'):
                block=entry[field]
                if block is None:
                    if field=='data': raise ValueError('Missing model parameters')
                    setattr(tensor,field,None);continue
                if type(block) is not dict or set(block)!={'member','sha256'}: raise ValueError('Invalid tensor block')
                member=block['member']
                if type(member) is not str or not member.startswith('tensors/') or member in expected_members: raise ValueError('Invalid tensor member')
                expected_members.add(member)
                if archive.getinfo(member).file_size!=size*8: raise ValueError('Tensor byte size mismatch')
                raw=archive.read(member)
                if hashlib.sha256(raw).hexdigest()!=block['sha256']: raise ValueError('Tensor digest mismatch')
                values=array('d',struct.unpack('<'+str(size)+'d',raw))
                if any(not math.isfinite(value) for value in values): raise ValueError('Nonfinite tensor')
                setattr(tensor,field,values)
            if (tensor.m is None)!=(tensor.v is None): raise ValueError('Incomplete Adam state')
            tensor.requires_grad=entry['requires_grad'];tensor.quantize_int8()
        if expected_members!={m.filename for m in members}: raise ValueError('Unexpected artifact entries')
        model.max_seq_len=payload['context_capacity'];model.total_steps=payload['total_steps'];model.lora_steps=payload['lora_steps']
        counter=model.lora_steps if config.lora_only else model.total_steps
        if counter!=state.step or state.step>config.max_steps: raise ValueError('Incompatible optimizer/progress state')
        return model,tokenizer,config,state,payload['metadata']

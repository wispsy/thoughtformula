"""Bounded next-token pretraining over the existing autograd and Adam engine."""
from dataclasses import dataclass, replace
from itertools import islice
import math
import random
import time

from .tensor import FlatTensor


def integer(value,name,minimum,maximum):
    if type(value) is not int or not minimum<=value<=maximum:
        raise ValueError(f'{name} is outside its integer bound')


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 100
    block_size: int = 32
    learning_rate: float = 0.003
    max_seconds: float = 120
    seed: int = 1
    shuffle: bool = True
    lora_only: bool = False

    def __post_init__(self):
        integer(self.max_steps,'Steps',0,1000000)
        integer(self.block_size,'Block size',1,256)
        integer(self.seed,'Seed',0,2**32-1)
        for value,name,maximum in ((self.learning_rate,'Learning rate',1),(self.max_seconds,'Time limit',86400)):
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=maximum:
                raise ValueError(f'{name} requires a positive finite value')
        if type(self.shuffle) is not bool or type(self.lora_only) is not bool:
            raise TypeError('Training mode flags must be Boolean')


@dataclass(frozen=True)
class TrainingState:
    step: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0

    def __post_init__(self):
        integer(self.step,'Completed steps',0,1000000)
        integer(self.tokens,'Trained tokens',0,256000000)
        if type(self.elapsed_seconds) not in (int,float) or not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds<0:
            raise ValueError('Elapsed time must be finite and nonnegative')


@dataclass(frozen=True)
class StepMetric:
    step: int
    tokens: int
    loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    state: TrainingState
    status: str
    last_loss: float | None


def parameters(model):
    result={'weights/'+name:value for name,value in model.weights.items()}
    for name,(left,right) in model.lora.items():
        result['lora/'+name+'/A']=left
        result['lora/'+name+'/B']=right
    return result


def parameter_count(model):
    return sum(value.rows*value.cols for value in parameters(model).values())


def reset_optimizer(model):
    """A new specialization keeps weights but starts a new optimizer trajectory."""
    model.total_steps=model.lora_steps=0
    for tensor in parameters(model).values():
        tensor.m=tensor.v=None
        for index in range(len(tensor.grad)): tensor.grad[index]=0


def checked_pairs(pairs,model,block_size):
    values=tuple(islice(iter(pairs),8193))
    if not values or len(values)>8192:
        raise ValueError('Select 1–8192 token sequences')
    checked=[]
    for pair in values:
        if type(pair) not in (tuple,list) or len(pair)!=2: raise ValueError('Sequences require input/target pairs')
        left,right=pair
        if type(left) not in (tuple,list) or type(right) not in (tuple,list) or not 1<=len(left)==len(right)<=min(block_size,model.max_seq_len):
            raise ValueError('Input and target lengths must match the context budget')
        if any(type(token) is not int or not 0<=token<model.vocab_size for token in (*left,*right)):
            raise ValueError('Training token is outside vocabulary')
        checked.append((tuple(left),tuple(right)))
    return tuple(checked)


def fit(model,pairs,config,*,state=None,observer=None,stop=None,checkpoint=None):
    """Train to a total step/time budget; resume uses the same data/config/order.

    Stop callbacks run at step boundaries. Observers/checkpoints run after each
    committed optimizer step; their exceptions propagate with partial progress.
    The caller owns source/config identity and checkpoint publication policy.
    """
    if type(config) is not TrainingConfig: raise TypeError('Use TrainingConfig')
    state=state if state is not None else TrainingState()
    if type(state) is not TrainingState or state.step>config.max_steps:
        raise ValueError('Incompatible training progress')
    counter=model.lora_steps if config.lora_only else model.total_steps
    if counter!=state.step: raise ValueError('Model optimizer counter and training progress differ')
    values=checked_pairs(pairs,model,config.block_size)
    model.set_training_mode(lora_only=config.lora_only)
    started=time.monotonic();prior_seconds=state.elapsed_seconds
    order=None;epoch=-1;last_loss=None;status='completed'
    while state.step<config.max_steps:
        elapsed=prior_seconds+time.monotonic()-started
        if elapsed>=config.max_seconds:
            status='limited';break
        if stop is not None and stop():
            status='paused';break
        current_epoch,position=divmod(state.step,len(values))
        if current_epoch!=epoch:
            order=list(range(len(values)))
            if config.shuffle: random.Random(config.seed+current_epoch).shuffle(order)
            epoch=current_epoch
        inputs,targets=values[order[position]]
        logits=model.forward(inputs,use_int8=False)
        loss=FlatTensor.cross_entropy(logits,targets)
        last_loss=loss.data[0]
        if not math.isfinite(last_loss): raise ValueError('Nonfinite training loss')
        loss.backward()
        model.step_adam(config.learning_rate,lora_only=config.lora_only)
        state=TrainingState(state.step+1,state.tokens+len(targets),prior_seconds+time.monotonic()-started)
        if checkpoint is not None: checkpoint(state)
        if observer is not None: observer(StepMetric(state.step,state.tokens,last_loss,state.elapsed_seconds))
    state=replace(state,elapsed_seconds=prior_seconds+time.monotonic()-started)
    return TrainingResult(state,status,last_loss)


def evaluate(model,pairs,*,block_size=256,max_sequences=32):
    integer(max_sequences,'Evaluation sequences',1,8192)
    integer(block_size,'Evaluation block size',1,256)
    values=checked_pairs(pairs,model,block_size)[:max_sequences]
    flags=[(tensor,tensor.requires_grad) for tensor in parameters(model).values()]
    total=0.0;tokens=0
    try:
        for tensor,_flag in flags: tensor.requires_grad=False
        for inputs,targets in values:
            loss=FlatTensor.cross_entropy(model.forward(inputs,use_int8=False),targets).data[0]
            if not math.isfinite(loss): raise ValueError('Nonfinite evaluation loss')
            total+=loss*len(targets);tokens+=len(targets)
    finally:
        for tensor,flag in flags: tensor.requires_grad=flag
    mean=total/tokens
    return {'loss':mean,'perplexity':math.exp(mean),'tokens':tokens,'sequences':len(values)}


def token_sequences(tokenizer,documents,*,block_size=32,max_sequences=8192):
    """Pack strict next-token windows within documents, never across boundaries."""
    integer(block_size,'Block size',1,256)
    integer(max_sequences,'Sequence count',1,8192)
    result=[]
    for index,text in enumerate(islice(iter(documents),1025)):
        if index>=1024: raise ValueError('Too many selected documents')
        if type(text) is not str or len(text)>65536: raise ValueError('Select bounded text documents')
        ids=tokenizer.encode(text)
        if tokenizer.decode(ids)!=text: raise ValueError('Text is not exactly representable by the selected tokenizer')
        ids.append(tokenizer.eos_id)
        for offset in range(0,len(ids)-1,block_size):
            chunk=ids[offset:offset+block_size+1]
            if len(chunk)>1: result.append((tuple(chunk[:-1]),tuple(chunk[1:])))
            if len(result)>=max_sequences: return tuple(result)
    if not result: raise ValueError('No nonempty token sequences')
    return tuple(result)

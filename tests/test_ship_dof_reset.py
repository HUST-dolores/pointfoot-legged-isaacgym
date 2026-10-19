"""Run the real base reset method with physical/estimated velocity buffers."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


def method():
    source=Path(__file__).resolve().parents[1]/'legged_gym/envs/base/base_task.py'
    cls=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='BaseTask')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_reset_dofs')
    namespace={'torch':torch,'gymtorch':SimpleNamespace(unwrap_tensor=lambda x:x),
               'torch_rand_float':lambda lo,hi,shape,device:torch.zeros(shape,device=device)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['_reset_dofs']


@pytest.mark.parametrize('estimated',[False,True])
def test_reset_clears_physical_velocity_even_when_estimated_cache_is_detached(estimated):
    state=torch.zeros(3,4,2);state[:,:,1]=3.
    calls=[]
    gym=SimpleNamespace(set_dof_state_tensor_indexed=lambda sim,data,indices,n:calls.append((data.clone(),indices.clone(),n)))
    e=SimpleNamespace(num_envs=3,num_dof=4,device='cpu',sim=object(),gym=gym,
        cfg=SimpleNamespace(init_state=SimpleNamespace(dof_pos_randomization_range=[-.02,.02])),
        default_dof_pos=torch.full((3,4),.1),dof_state=state,dof_pos=state[:,:,0],
        dof_vel=torch.full((3,4),2.) if estimated else state[:,:,1],actor_indices=torch.tensor([0,2,4]))
    method()(e,torch.tensor([0,2]))
    assert len(calls)==1
    assert torch.count_nonzero(calls[0][0][[0,2],:,1])==0
    assert torch.count_nonzero(e.dof_vel[[0,2]])==0
    torch.testing.assert_close(state[1,:,1],torch.full((4,),3.))
    torch.testing.assert_close(state[[0,2],:,0],torch.full((2,4),.1))

"""Pure actual probe metrics: distinct survivors, repeats and first-hit timing."""
import ast
from pathlib import Path
import pytest
import torch

def helpers():
    source=Path(__file__).resolve().parents[1]/'legged_gym/scripts/ship_motion_probe.py'
    tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('survival_checkpoint','record_first_failure')]
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual survival metrics>','exec'),ns)
    return ns

@pytest.mark.parametrize('counts,survivors,resets',[
    ([0,0,0],3,0),([0,1,2,0],2,3),([1,2,3],0,6)])
def test_survival_counts_unique_environments_and_all_failure_resets(counts,survivors,resets):
    values=torch.tensor(counts);before=values.clone()
    r=helpers()['survival_checkpoint'](values,6000,.02)
    assert r=={'steps':6000,'simulated_seconds':120.,'survivors':survivors,
               'survival_fraction':survivors/len(counts),'failure_resets':resets}
    torch.testing.assert_close(values,before,rtol=0,atol=0)

def test_first_failure_never_overwrites_previous_hit_or_unfailed_env():
    first=torch.tensor([-1,4,-1,-1]);mask=torch.tensor([True,True,False,False]);old_mask=mask.clone()
    update=helpers()['record_first_failure'];update(first,mask,3000)
    assert first.tolist()==[3000,4,-1,-1]
    update(first,torch.tensor([True,False,True,False]),6000)
    assert first.tolist()==[3000,4,6000,-1]
    torch.testing.assert_close(mask,old_mask,rtol=0,atol=0)

def test_same_trajectory_survival_can_only_decrease_as_failures_accumulate():
    metrics=helpers()['survival_checkpoint'];counts=torch.zeros(4,dtype=torch.long)
    a=metrics(counts,3000,.02)
    counts[0]+=1;b=metrics(counts,6000,.02)
    counts[0]+=1;counts[1]+=1;c=metrics(counts,9000,.02)
    assert [r['survivors'] for r in [a,b,c]]==[4,3,2]
    assert [r['failure_resets'] for r in [a,b,c]]==[0,1,3]

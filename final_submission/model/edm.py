"""Inference adapter for the Festgen EDM 100K teacher (not paper architecture)."""
from pathlib import Path
import torch
from .unet import ClassConditionalUNet

ROOT=Path(__file__).resolve().parents[1]

def load_model(device='cpu', checkpoint=None):
    path=Path(checkpoint) if checkpoint else ROOT/'checkpoints/edm_100k_ema.pt'
    st=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    c=st['student_config']
    keys=('image_size','in_channels','num_classes','model_channels','channel_mult','num_res_blocks','attention_resolutions','dropout')
    model=ClassConditionalUNet(**{k:c[k] for k in keys},activation_checkpoint_resolutions=(64,32,16,8))
    model.load_state_dict(st['state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==89577673
    return model.to(device).eval()

def denoise(model,x,sigma,labels):
    sigma=sigma.reshape(-1,1,1,1).float();sd=.5;denom=sigma.square()+sd**2
    return sd**2/denom*x.float()+sigma*sd/denom.sqrt()*model(denom.rsqrt()*x.float(),sigma.flatten().log()*.25*250,labels).float()

@torch.inference_mode()
def sample(model,labels,seed=42,steps=32,cfg=2.5,sigma_min=.002,sigma_max=80.,rho=7.):
    if steps<2: raise ValueError('Heun sampler requires steps >= 2')
    device=labels.device
    if labels.dtype!=torch.long or labels.ndim!=1 or torch.any((labels<0)|(labels>=151)): raise ValueError('labels must be int64 class IDs 0..150')
    i=torch.arange(steps,device=device,dtype=torch.float64)
    levels=(sigma_max**(1/rho)+i/(steps-1)*(sigma_min**(1/rho)-sigma_max**(1/rho)))**rho
    levels=torch.cat([levels,torch.zeros(1,device=device,dtype=torch.float64)])
    x=torch.randn((len(labels),3,64,64),device=device,generator=torch.Generator(device=device).manual_seed(seed))*levels[0].float()
    calls=0
    def guided(z,sig):
        nonlocal calls
        s=torch.full((len(labels),),float(sig),device=device)
        u=denoise(model,z,s,torch.full_like(labels,151));c=denoise(model,z,s,labels);calls+=2
        return u+cfg*(c-u)
    for j in range(steps):
        a,b=levels[j].float(),levels[j+1].float();d=(x-guided(x,a))/a;new=x+(b-a)*d
        if j<steps-1:
            d2=(new-guided(new,b))/b;new=x+(b-a)*(d+d2)/2
        x=new
    assert calls==2*(2*steps-1)
    return x.clamp(-1,1),calls

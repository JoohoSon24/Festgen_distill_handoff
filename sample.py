import argparse,torch
from model.edm import load_model,sample
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--labels',nargs='+',type=int,default=[0]);p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=32);p.add_argument('--cfg',type=float,default=2.5);p.add_argument('--output',default='samples.pt');a=p.parse_args()
m=load_model(a.device);y=torch.tensor(a.labels,dtype=torch.long,device=a.device)
with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
 images,nfe=sample(m,y,a.seed,a.steps,a.cfg)
torch.save({'images':images.cpu(),'labels':y.cpu(),'nfe':nfe,'seed':a.seed},a.output);print('saved',tuple(images.shape),'NFE',nfe)

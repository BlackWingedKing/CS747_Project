import torch
import torch.nn as nn
from collections import OrderedDict
from pyramix import * 
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import torchvision
import pandas as pd

#path = "./tmp/model12.pt"
#solves = pd.read_csv('pyraminxValues.csv')
#pol = [list(map(int, i.split('-'))) for i in solves['policy']]
#dl_mdl = dict(zip(solves['index'],zip(solves['value'],pol)))

class Qval():
    def __init__(self):
        self.Q = {}
        self.Qkey = {}
        
    def key(self,val):
        return ''.join([''.join([str(j) for j in i]) for i in val])
    
    def __getitem__(self,k):
        val = self.key(k)
        if val in self.Q.keys():
            return self.Q[val]
        else:
            self.Qkey[val]= k
            self.Q[val] = (torch.tensor(0),torch.zeros(1,8))
            return self.Q[val]
        
    def __setitem__(self,key,val):
        key = self.key(key)
        self.Q[key]=val
    def __repr__(self):
        return str(self.Q)

def key(puzzle):
    return ''.join([''.join([str(j) for j in i]) for i in puzzle])
    
def reiter(model,optimizer,path):
      checkpoint = torch.load(path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      model.train()
      return (model,optimizer)

def convert(puzzle,faceColor= ['R', 'G', 'Y', 'B']):
    p = [[faceColor.index(j) for j in i] for i in puzzle]
    op = torch.zeros((9,4,4))
    for i,l in enumerate(p):
        for j,k in enumerate(l):
            op[j,i,k] = 1
    return op
    
class NN1(nn.Module):
    def __init__(self):
        super(NN1,self).__init__()
        self.bodySizes = [9*4*4, 512, 256]
        self.valueSizes = [256, 1]
        self.policySizes = [256, 8]
        self.elu = nn.ELU(alpha = 0.1)
        self.sigm = nn.Sigmoid()
        
        self.body = nn.ModuleList([nn.Linear(prev,nxt) for prev,nxt in zip(self.bodySizes[:-1],self.bodySizes[1:])])
        self.value = nn.ModuleList([nn.Linear(prev,nxt) for prev,nxt in zip(self.valueSizes[:-1],self.valueSizes[1:])])
        self.policy = nn.ModuleList([nn.Linear(prev,nxt) for prev,nxt in zip(self.policySizes[:-1],self.policySizes[1:])])
        
        for m in self.body:
            torch.nn.init.xavier_uniform_(m.weight,gain = 1.0)
            torch.nn.init.constant_(m.bias,0)
            
        for m in self.value:
            torch.nn.init.xavier_uniform_(m.weight,gain = 1.0)
            torch.nn.init.constant_(m.bias,0)
            
        for m in self.policy:
            torch.nn.init.xavier_uniform_(m.weight,gain = 1.0)
            torch.nn.init.constant_(m.bias,0)
            
        self.sm = nn.Softmax(dim=0)
        
    def forward(self,x):
        for m in self.body:
                x = self.elu(m(x))
        V = x #torch.mm(torch.eye(x.shape[0]),x)#x.clone().detach().requires_grad_(True)
        P = x 
        for m in self.value:
                V = self.elu(m(V))
        for m in self.policy:
                P = self.sigm(m(P))
        P = self.sm(P)
        return (V[0],P.view(1,8))
        
class NN2(nn.Module):
    def __init__(self):
        super(NN2,self).__init__()
        self.bodySizes = [9*4*4, 512, 512, 512]
        self.valueSizes = [512, 1]
        self.policySizes = [512, 8]
        self.elu = nn.ELU(alpha = 0.1)
        self.sigm = nn.Sigmoid()
        
        self.body = nn.ModuleList([nn.Linear(prev,nxt) for prev,nxt in zip(self.bodySizes[:-1],self.bodySizes[1:])])
        self.value = nn.ModuleList([nn.Linear(prev,nxt) for prev,nxt in zip(self.valueSizes[:-1],self.valueSizes[1:])])
        self.policy = nn.ModuleList([nn.Linear(prev,nxt) for prev,nxt in zip(self.policySizes[:-1],self.policySizes[1:])])
        
        for m in self.body:
            torch.nn.init.xavier_uniform_(m.weight,gain = 1.0)
            torch.nn.init.constant_(m.bias,0)
            
        for m in self.value:
            torch.nn.init.xavier_uniform_(m.weight,gain = 1.0)
            torch.nn.init.constant_(m.bias,0)
            
        for m in self.policy:
            torch.nn.init.xavier_uniform_(m.weight,gain = 1.0)
            torch.nn.init.constant_(m.bias,0)
            
        self.sm = nn.Softmax(dim=0)
        
    def forward(self,x):
        for m in self.body:
                x = self.elu(m(x))
        V = x #torch.mm(torch.eye(x.shape[0]),x)#x.clone().detach().requires_grad_(True)
        P = x 
        for m in self.value:
                V = self.elu(m(V))
        for m in self.policy:
                P = self.sigm(m(P))
        P = self.sm(P)
        return (V[0],P.view(1,8))
        
if __name__ =='__main__':
  face_colors = ['R', 'G', 'Y', 'B']
  actions = ['u','b','l','r',"u'","b'","l'","r'"]
  terminalPos = ''.join([''.join([f'{i}' for j in range(9)]) for i in face_colors])
  
  puzzle = None
  moves = []
  puzzle = pyraminx_puzzle(moves,puzzle)
  n = NN().cuda()
  epsilon = 0.1
  alpha = 0.5
  gamma = 1
  no_Samples = 50
  depth = 11
  # Q[puzzle] = [alpha for i in range(8)]
  Loss = nn.MSELoss()
  BCE = nn.BCELoss()
  # BCE = torchvision.transforms.Lambda(lambda x,y,z: torch.mm(BCE(x,y),z))
  
  Q = Qval()
  Q[pyraminx_puzzle([])]=(torch.tensor(1.0),torch.zeros(1,8))
  optimizer = torch.optim.Adam(n.parameters(), lr=5e-6)
  rate = 1.0
  run_loss = 0
  reiter(n,optimizer,path)
  l_array = []
  for itr in range(200):
      '''for puzzle,length,movs in allScrambles:
          #print(puzzle,length)
          reward = torch.tensor([1 if Q.key(pyraminx_puzzle([a],puzzle)) == terminalPos else -1 for a in actions]).type(torch.FloatTensor).cuda()
  #         if torch.any(reward == 1): print('here')
          xv = torch.tensor([n(torch.flatten(torch.tensor(convert(pyraminx_puzzle([a],puzzle)))).cuda())[0] for a in actions]).cuda()
          yv = torch.max(reward+xv).cuda()
          yp = torch.argmax(reward+xv).cuda()
  #         print(yv,yp)
          P = torch.zeros(1,8).cuda()
          P[0][yp]=1
          Q[puzzle]=(yv,P)
          #print(yv,torch.argmax(P),movs)'''
          
      allMoves = []
      allScrambles=[]
      for i in range(no_Samples):
          mv =[]
          for j in range(depth):
              mv.append(np.random.choice(actions))
          allMoves.append(mv)
      
      for mv in allMoves:
          for i,a in enumerate(mv):
              allScrambles.append((pyraminx_puzzle(mv[:i+1]),i+1,mv[:i+1]))
      
      allErrors = []
      for puzzle,length,_ in allScrambles:
          weight = rate/length
          val,pos = n(torch.flatten(torch.Tensor(convert(puzzle))).cuda())
          #error = (Loss(val,Q[puzzle][0])+BCE(pos,Q[puzzle][1]))*weight 
          error = (Loss(val,torch.Tensor([dl_mdl[key(puzzle)][0]]).cuda())+BCE(pos,torch.Tensor(dl_mdl[key(puzzle)][1]).cuda()))*weight 
          allErrors.append(error)
          error.backward()
          optimizer.step()
      netError = torch.stack(allErrors)
      run_loss = (itr*run_loss+torch.mean(netError))/(itr+1)
      l_array.append((run_loss.data.cpu(),torch.mean(netError).data.cpu()))
      print(itr,torch.mean(netError).data.cpu(),run_loss.data.cpu())
  #torch.save({'model_state_dict': n.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, './tmp/model.pt')
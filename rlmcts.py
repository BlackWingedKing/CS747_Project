import torch
import torch.nn as nn
from collections import OrderedDict
from pyramix import * 
from copy import deepcopy as dc
import predictor
import random
import math
#from policyEval import Index,invertIndex
from tqdm import tqdm
import pandas as pd
import numpy as np
N={}
W={}
P={}
L={}
nu=2
c=0.1

Actions = ['u','b','l','r',"u'","b'","l'","r'"]
solves = pd.read_csv('pyraminxValues.csv')
pol = [list(map(int, i.split('-'))) for i in solves['policy']]
dl_mdl = dict(zip(solves['index'],zip(solves['value'],pol)))
model = predictor.NN2()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-8)

def scramble(numScrambles):
    puzz = None
    for _ in range(numScrambles):
        puzz = pyraminx_puzzle([random.choice(Actions)], puzz)
    return puzz

def isSolved(puzzle):
    solvedPuzzle = pyraminx_puzzle([],None)
    if solvedPuzzle == puzzle:
        return True
    else:
        return False
        
def simulate_pyraminx(noSamples, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    pbar = tqdm(total=maxSolveDistance*noSamples,desc='dist/episode:', leave=True)
    solveMCTS = mctsState()
    for dist in range(1,maxSolveDistance+1):
        noSolved = 0
        for episode in range(noSamples):
            scrambledPuzzle = scramble(dist)
            #result, noMoves, solvedPath = solveMCTS(scrambledPuzzle, 6 * dist + 1)
            #result, noMoves, solvedPath = solveGreedy(scrambledPuzzle, 6 * dist + 1)
            result, noMoves, solvedPath = solve_nmcts(scrambledPuzzle,6*dist+1)
            # print(dist,episode)
            if result:
                noSolved += 1
            pbar.update(1)
            pbar.set_description(f"dist/episode: {dist}/{episode}:{result}")
        percentageSolved = float(noSolved)/noSamples
        data[dist] = percentageSolved
    pbar.close()
    return data

def solve_nmcts(scramble,maxmoves,dpt = 4,ideal=1):
    n=0
    solvedPath=[]
    while(n<=maxmoves):
        if isSolved(scramble):
            return True,n,solvedPath
        solvedPath.append(key(scramble))
        # print(scramble)
        a=MCTS_with_depth(scramble,dpt,ideal)
        scramble=Move(scramble,a)
        n+=1
    return False,n, solvedPath
    
def solveGreedy(scramble,maxMoves,ideal=1):
    n = 0
    solvedPath = []
    while n <= maxMoves:
        if scramble == pyraminx_puzzle([]):
            return True,n,solvedPath
        a = greedy(scramble,ideal)
        n+=1
        solvedPath.append(a)
        scramble = pyraminx_puzzle([a],scramble)
    return False, n, solvedPath

def puct(puzzle, ideal):
    # puzzle is a str rep of state
    model.cuda()
    uct=[]
    nsub=0.0
    for i,a in enumerate(Actions):
        nx = key(Move(invertIndex(puzzle), a))
        if(nx in N):
            nval = N[nx]
        else:
            N[nx] = 0.0
            nval=N[nx]
        nsub+=nval        
    for i,a in enumerate(Actions):
        nx = key(Move(invertIndex(puzzle), a))
        # print(type(nx))
        if ideal:
          (_,P) = model(torch.flatten(torch.tensor(convert(invertIndex(nx)).cuda()))) # here also model isnt ...s
        else:
          _,P = dl_mdl[nx]
          P = torch.tensor([P])
          print('Easter Egg!!')
        if(not(nx in W)):
            W[nx] = 0.0
        if(not(nx in L)):
            L[nx] = 0.0
        ucti = c*P[0][i]*np.sqrt(nsub/(N[nx]+1.0)) + W[nx] - L[nx]
        uct.append(ucti)
    return uct.index(max(uct))

def MCTS_with_depth(scramble,ideal,depth = 4):
    Tree={}
    solved = False
    for a in Actions:
        puzz = Move(scramble,a)
        if(isSolved(puzz)): return a
        Tree[key(puzz)]= []    #key() is to convert value to string
    for i,k in enumerate(Tree.keys()):
        a_star= puct(k,ideal)
        # if(Tree[k])
        if(isSolved(Move(invertIndex(k),Actions[a_star]))):
            solved = True
            return Actions[i]
        Tree[k].append(key(Move(invertIndex(k),Actions[a_star])))
        for d in range(depth-1):
            a_star=puct(Tree[k][-1],ideal)
            # print(Tree[k][-1])
            if(isSolved(Move(invertIndex(Tree[k][-1]),Actions[a_star]))):
                continue
                # break
            Tree[k].append(key(Move(invertIndex(Tree[k][-1]),Actions[a_star])))
        # Tree[k][-1]
    for k in Tree.keys():
        for j,a in enumerate(Actions):
            child = Tree[k][-1]
            tmv = Move(invertIndex(child),a)
            N[key(tmv)] = 0
            W[key(tmv)] = 0
            L[key(tmv)] = 0
    Q = []
    for k in Tree.keys():
        stau = Tree[k][-1]
        if ideal:
            (vstau,_) = model(torch.flatten(torch.tensor(convert(invertIndex(stau)).cuda()))) # model isnt defined yet
        else:
            vstau,_ = dl_mdl[stau]
        for j in Tree[k]:
            # W update
            if(j in W):
                W[j] = max(W[j], vstau)
            else:
                W[j] = vstau
            # L update
            if(j in L):
                L[j]-=nu
            else:
                L[j] = 0.0 # check  for initial value once (L,N)
            # N update
            if(j in N):
                N[j]+=1
            else:
                N[j]=0.0 
        if(k in W):
            W[k] = max(W[k], vstau)
        else:
            W[k] = vstau
        
        Q.append(W[k]-L[k])
    index = Q.index(max(Q))
    return Actions[index]

def greedy(scramble,ideal):
    maxV=torch.zeros(len(Actions))
    for i,a in enumerate(Actions):
        newPuzzle = pyraminx_puzzle([a],scramble)
        #idx = solves.loc[solves['index'] == Index(newPuzzle)].index.tolist()[0]
        if ideal:
            value,_ = model(torch.flatten(torch.tensor(convert(newPuzzle))))
        else:
            value,_ = dl_mdl[key(newPuzzle)]
        maxV[i] = value
    return Actions[torch.argmax(maxV).item()]

def reiter(model,optimizer,path):
      checkpoint = torch.load(path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      model.eval()
      return (model,optimizer)

model,optimizer = reiter(model,optimizer,'./tmp/model4.pt')

def test_pyraminx(dist):
    global model
    global optimizer
    global epsilon
    global const_Virtualloss
    c1 = 50
    rs = []
    scrambledPuzzle = scramble(dist)
    print(key(scrambledPuzzle))
    solveMCTS = mctsState()
    model = predictor.NN1()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-8)
    model,optimizer = reiter(model,optimizer,'./tmp/model12Adi.pt')
    result, noMoves, solvedPath = solveMCTS(scrambledPuzzle, c1)
    print('For ModelNN1+ADI on MCTS: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solveGreedy(scrambledPuzzle, c1)
    print('For ModelNN1+ADI on Greedy: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solve_nmcts(scrambledPuzzle, c1)
    print('For ModelNN1+ADI on pMCTS: '+str(noMoves))
    rs.append(noMoves)
    
    model = predictor.NN2()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-8)
    model,optimizer = reiter(model,optimizer,'./tmp/model4.pt')
    result, noMoves, solvedPath = solveMCTS(scrambledPuzzle, c1)
    print('For ModelADI4 on MCTS: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solveGreedy(scrambledPuzzle, c1)
    print('For ModelADI4 on Greedy: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solve_nmcts(scrambledPuzzle, c1)
    print('For ModelADI4 on pMCTS: '+str(noMoves))
    rs.append(noMoves)
    
    model = predictor.NN1()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-8)
    model,optimizer = reiter(model,optimizer,'./tmp/model12.pt')
    result, noMoves, solvedPath = solveMCTS(scrambledPuzzle, c1)
    print('For ModelNN1 on MCTS: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solveGreedy(scrambledPuzzle, c1)
    print('For ModelNN1 on Greedy: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solve_nmcts(scrambledPuzzle, c1)
    print('For ModelNN1 on pMCTS: '+str(noMoves))
    rs.append(noMoves)
    
    model = predictor.NN2()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-8)
    model,optimizer = reiter(model,optimizer,'./tmp/modelTrueValue.pt')
    result, noMoves, solvedPath = solveMCTS(scrambledPuzzle, c1)
    print('For ModelNN2 on MCTS: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solveGreedy(scrambledPuzzle, c1)
    print('For ModelNN2 on Greedy: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solve_nmcts(scrambledPuzzle, c1)
    print('For ModelNN2 on pMCTS: '+str(noMoves))
    rs.append(noMoves)
    
    epsilon = 1e10
    const_Virtualloss = 0.001
    result, noMoves, solvedPath = solveMCTS(scrambledPuzzle, c1, 0)
    print('For Tabular on MCTS: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solveGreedy(scrambledPuzzle, c1, 0)
    print('For Tabular on Greedy: '+str(noMoves))
    rs.append(noMoves)
    result, noMoves, solvedPath = solve_nmcts(scrambledPuzzle, c1, 4 ,0)
    print('For Tabular on pMCTS: '+str(noMoves))
    rs.append(noMoves)
    
    prdctr=['ModelNN1+ADI','ModelADI4','ModelNN1','ModelNN2','Tabular']
    slvr=['MCTS','Greedy','pMCTS']
    with open('results_compareModels.txt','a') as page:
        page.write(key(scrambledPuzzle)+':\n')
        cnt = 0
        for prd in prdctr:
          for slv in slvr:
            page.write('P: '+ prd +'; S: '+slv+'; ')
            page.write(str(rs[cnt])+'\n')
            cnt+=1
            
def convert(puzzle,faceColor= ['R', 'G', 'Y', 'B']):
    p = [[faceColor.index(j) for j in i] for i in puzzle]
    op = torch.zeros((9,4,4))
    for i,l in enumerate(p):
        for j,k in enumerate(l):
            op[j,i,k] = 1
    return op

def key(puzzle):
    return ''.join([''.join([str(j) for j in i]) for i in puzzle])
    
def invertIndex(idx):
    temp = []
    for i in range(4):
        temp.append(list(idx[i*9:i*9+9]))
    return temp

################################################################################

def Move(puzzle, action):
  return pyraminx_puzzle([action], puzzle)    #action needs to be converted to ulrb form
  
class mctsState():
    def __init__(self):
        #self.index = Index(puzzle)
        #self.input = convert(puzzle)
        # self.visited = False
        
        self.solvedPath = []
        self.solvedAction = []
        self.treeStates = set()
        self.seenStates = set()
        self.Ns = {}
        self.Ws = {}
        self.Ps = {}
        self.Ls = {}
        
    @property
    def puzzle(self):
      return self._puzzle
    @puzzle.setter
    def puzzle(self,p):
      self._puzzle = p
      self._puzzleState = key(p)
    @property
    def puzzleState(self):
      return self._puzzleState
    @puzzleState.setter
    def puzzleState(self,p):
      self._puzzle = invertIndex(p)
      self._puzzleState = p
      
    
    def isSolved(self,p):
      solvedPuzzle = pyraminx_puzzle([],None)
      if solvedPuzzle == p:
          return True
      else:
          return False

                  
    def initializeStateVals(self, probs, ideal,puzzleState = None):
      if puzzleState == None: puzz = self.puzzleState
      else: puzz = puzzleState
      self.Ns[puzz] = {}
      self.Ws[puzz] = {}
      self.Ps[puzz] = {}
      self.Ls[puzz] = {}
      for i,a in enumerate(Actions):
          self.Ns[puzz][a] = 0
          self.Ws[puzz][a] = 0
          self.Ls[puzz][a] = 0
          if ideal:
              self.Ps[puzz][a] = probs[0][i]
          else:
              self.Ps[puzz][a] = probs[i]
            
    def __call__(self,puzzle,maxMoves,ideal=1):
      noMoves = 0
      self.puzzle = puzzle
      if ideal:
          _,probs = model(torch.flatten(torch.tensor(convert(self.puzzle))))
      else:
          _,probs = dl_mdl[self.puzzleState]
      self.initializeStateVals(probs,ideal)
      self.seenStates.add(self.puzzleState)
      self.solvedPath.append(self.puzzle)
      while (noMoves <= maxMoves):
        if self.isSolved(self.puzzle):
          return True, noMoves, self.solvedPath
        if self.puzzleState not in self.treeStates:
          for a in Actions:
            child = Move(self.puzzle,a)
            if key(child) not in self.seenStates:
              if ideal:
                  _,probs = model(torch.flatten(torch.tensor(convert(child))))
              else:
                  _,probs = dl_mdl[key(child)]
              self.initializeStateVals(probs,ideal,puzzleState = key(child))
              self.seenStates.add(key(child))
          if ideal:
              value,_ = model(torch.flatten(torch.tensor(convert(self.puzzle))))
          else:
              value,_ = dl_mdl[self.puzzleState]
          for i,s in enumerate(self.solvedPath):
            state = key(s)
            if i < len(self.solvedAction):
              self.Ws[state][self.solvedAction[i]] = max(self.Ws[state][self.solvedAction[i]], value)
              self.Ns[state][self.solvedAction[i]] += 1
              self.Ls[state][self.solvedAction[i]] -= const_Virtualloss
          self.treeStates.add(self.puzzleState)
        else:
          Qs = np.zeros(len(Actions))
          totalStateCount = 0
          for a in Actions:
            totalStateCount += self.Ns[self.puzzleState][a]
          for i,a in enumerate(Actions):
            q = self.Ws[self.puzzleState][a] - self.Ls[self.puzzleState][a]
            u = epsilon * self.Ps[self.puzzleState][a] * math.sqrt(totalStateCount)/(1+self.Ns[self.puzzleState][a])
            Qs[i] = u + q
          ai_star = np.argmax(Qs)
          a_star = Actions[ai_star]
          self.Ls[self.puzzleState][a_star] += const_Virtualloss
          self.solvedAction.append(a_star)
          self.puzzle = Move(self.puzzle,a_star)
          self.solvedPath.append(self.puzzle)
          noMoves += 1
      return False, noMoves, self.solvedPath        

#################################################################################3


if __name__ == '__main__':
    epsilon = 10
    const_Virtualloss = 2
    
## For computing percentage solved scrambles
#    output = simulate_pyraminx(100,8)
###    textInput = input()
###    print(output)
#    with open('results_pmcts.txt','a') as page:
#      page.write('P: Model12Adi; S: pmcts'+'\n')
#      page.write(str(output)+'\n')

## For choosing the hyperparameters in MCTS algorithm
#    epsilon_list = [10**i * 1e-2 for i in range(10)]  # UCT exploration factor
#    const_Virtualloss_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10] # Virtural loss constant
#    output = []
#    for i in epsilon_list:
#      for j in const_Virtualloss_list:
#        epsilon = i  # UCT exploration factor
#        const_Virtualloss = j # Virtural loss constant
#        o = simulate_pyraminx(100,3)
#        print(o)
#        output.append(o)
#        textInput = str(i) + ' ' + str(j) + ' ' + 'P:Model12 ' + 'S:MCTS'
#        with open('results_2independentv.txt','a') as page:
#          page.write(textInput+'\n')
#          page.write(str(o)+'\n')
    
## For testing on all model at once on a given scramble   
    test_pyraminx(100)
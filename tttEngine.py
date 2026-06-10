import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class ttt:
  def __init__(self):
    self.board = torch.zeros((1,9), dtype=torch.float32)
    # these are all the possible winning paths in the game...
    self.winpaths= [torch.tensor([0,1,2]),torch.tensor([3,4,5]),torch.tensor([6,7,8]),
                   torch.tensor([0,3,6]),torch.tensor([1,4,7]),torch.tensor([2,5,8]),
                   torch.tensor([0,4,8]),torch.tensor([2,4,6])]

    # game play...
    self.picked = []
    self.scboard= [['_','_','_'],
                   ['_','_','_'],
                   ['_','_','_']] # this the dumb terminal scoreboard you can see the model's moves on.
    self.round  = 0
    self.won    = None
    self.start  = 'x'

  def regame(self):
    self.picked = []
    self.scboard= [['_','_','_'],
                   ['_','_','_'],
                   ['_','_','_']]
    self.round  = 0
    self.won    = None

  # playgame function handles training the model too because the model learns while interacting...
  def playgame(self,x,o,plays=1, prnt=False,prnt_probs=False,train=False,who='',loop=1,lr=-0.0001):
    for i in range(plays):
      self.regame()
      x.regame()
      o.regame()
      i += 1
      if self.start == 'x':
        self.turn = 'x'
      else:
        self.turn = 'o'
        
      def play(x,o):
        def checkwin(x,o):
          for k in self.winpaths:
            x.win = []
            o.win = []
            for p in x.paths:
              if p in k:
                x.win.append(p)
                if len(x.win) > 2:
                  for i in x.win:
                    x.lprobs[x.paths.index(i)] *= torch.tensor([2.0])
                  return 'x'
            for q in o.paths:
              if q in k:
                o.win.append(q)
                if len(o.win) > 2:
                  for i in o.win:
                    o.lprobs[o.paths.index(i)] *= torch.tensor([2.0])
                  return 'o'
                  
        for r in range(9):
          self.round += 1
          if r == 0:
            if self.turn == 'x':
              self.start = 'o'
            else: 
              self.start = 'x'
              
          if self.turn == 'x':
            xix = x.forward(self.board.clone())
            self.turn = 'o'
            self.board[0][xix] = 1
            if prnt_probs:
              ps = x.probs.tolist()
              print(ps.index(max(ps)))
            
            if xix.item() <= 2:
              self.scboard[0][xix.item()] = 'x'
            elif xix.item() <= 5:
              self.scboard[1][(xix.item()-3)] = 'x'
            else:
              self.scboard[2][(xix.item()-6)] = 'x'
          else:
            oix = o.forward(self.board.clone())
            self.turn = 'x'
            self.board[0][oix] = -1
            
            if oix.item() <= 2:
              self.scboard[0][oix.item()] = 'o'
            elif oix.item() <= 5:
              self.scboard[1][(oix.item()-3)] = 'o'
            else:
              self.scboard[2][(oix.item()-6)] = 'o'

          if r >= 4:
            score = checkwin(x,o)
            if score:
              self.won,self.end = score,score
              if score == 'x':
                x.loss=(torch.stack(x.lprobs)*torch.tensor([1.45])).sum()
                o.loss=(torch.stack(o.lprobs)*torch.tensor([-1.45])).sum()
              if score == 'o':
                x.loss=(torch.stack(x.lprobs)*torch.tensor([-1.45])).sum()
                o.loss=(torch.stack(o.lprobs)*torch.tensor([1.45])).sum()
              break
    
        if self.won:
          if self.won == 'x':
            if prnt:
              print("X WON!")
              print("----------")
              print(self.scboard[0])
              print(self.scboard[1])
              print(self.scboard[2])
          
          else:
            if prnt:
              print("O WON!")
              print("----------")
              print(self.scboard[0])
              print(self.scboard[1])
              print(self.scboard[2])
          
        else:       
          x.loss=(torch.stack(x.lprobs)*torch.tensor([0.45])).sum()
          o.loss=(torch.stack(o.lprobs)*torch.tensor([0.45])).sum()
            
          if prnt:
            print("TIE")
            print("----------")
            print(self.scboard[0])
            print(self.scboard[1])
            print(self.scboard[2])
            
      play(x,o)
      stepi.append(i)
      xtries.append(x.tries)
      otries.append(o.tries)
      if train:
        for l in range(loop):
          if not who:
            # backward pass
            for p in x.parameters:
              p.grad=None
            x.loss.backward()
            for p in o.parameters:
              p.grad=None
            o.loss.backward()
    
            # optimizing...
            for p in x.parameters:
              p.data += (lr * p.grad)
            for p in o.parameters:
              p.data += (lr * p.grad)
          if who == 'x':
            for p in x.parameters:
              p.grad=None
            x.loss.backward()
            # optimizing...
            for p in x.parameters:
              p.data += (lr * p.grad)
          if who == 'o':
            for p in o.parameters:
              p.grad=None
            o.loss.backward()
            # optimizing...
            for p in o.parameters:
              p.data += (lr * p.grad)
          
      if i % (plays/5) == 0:
        print(f"{i}:plays...")
  class player:
    def __init__(self, parent):
      self.g   = torch.Generator().manual_seed(223341)
      self.P   = parent
      # model learning parameters
      self.w1     = torch.randn((9,18),     generator=self.g) * 0.01
      self.b1     = torch.randn((1,18),     generator=self.g) * 0.01
      self.w2     = torch.randn((18,18),    generator=self.g) * ((5/3) / (((18+18)/2)**0.5))
      self.w3     = torch.randn((18,15),    generator=self.g) * ((5/3) / (((18+15)/2)**0.5))
      self.w4     = torch.randn((15,12),    generator=self.g) * ((5/3) / (((15+12)/2)**0.5))
      self.w5     = torch.randn((12,9),     generator=self.g) * 0.000001
      # layer Norm parameters...
      self.gamma1 = torch.ones((1,18),   dtype=torch.float32)
      self.beta1  = torch.zeros((1,18),  dtype=torch.float32)
      self.gamma2 = torch.ones((1,15),   dtype=torch.float32)
      self.beta2  = torch.zeros((1,15),  dtype=torch.float32)
      self.gamma3 = torch.ones((1,12),   dtype=torch.float32)
      self.beta3  = torch.zeros((1,12),  dtype=torch.float32)
      self.gamma4 = torch.ones((1,9),    dtype=torch.float32)
      self.beta4  = torch.zeros((1,9),   dtype=torch.float32)
      
      self.parameters = [self.w1, self.b1, self.w2, self.w3, self.w4, self.w5,
                        self.gamma1, self.beta1, self.gamma2, self.beta2,self.gamma3,
                        self.beta3, self.gamma4, self.beta4]
      for p in self.parameters:
        p.requires_grad=True
      self.allparams  = sum([i.nelement() for i in self.parameters])
      self.name = ''
      # game play...
      self.wins   = 0
      self.ties   = 0
      self.losses = 0
      self.tries  = 0
      self.orgtrs = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
      self.paths  = []
      self.lprobs = []

    def regame(self):
      self.tries  = 0
      self.paths  = []
      self.lprobs = []
      self.orgtrs = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
      
    # model's decision...
    def forward(self,x): # forward pass is really messy for full control and simulation...
      hidden  = x @ self.w1 + self.b1
      hidden2 = hidden @ self.w2
      h2mean  = hidden2.mean()
      h2std   = hidden2.std() + 1e-05
      hidden2 = self.gamma1 * ((hidden2-h2mean) / h2std) + self.beta1
      h2tanh  = hidden2.tanh()
      hidden3 = h2tanh @ self.w3
      h3mean  = hidden3.mean()
      h3std   = hidden3.std() + 1e-05
      hidden3 = self.gamma2 * ((hidden3-h3mean)/h3std) + self.beta2
      h3tanh  = hidden3.tanh()
      hidden4 = h3tanh @ self.w4
      h4mean  = hidden4.mean()
      h4std   = hidden4.std() + 1e-05
      hidden4 = self.gamma3 * ((hidden4-h4mean)/h4std) + self.beta3
      h4tanh  = hidden4.tanh()
      hidden5 = h4tanh @ self.w5
      h5mean  = hidden5.mean()
      h5std   = hidden5.std() + 1e-05
      hidden5 = self.gamma4 * ((hidden5-h5mean)/h5std) + self.beta4
      h5tanh  = hidden5.tanh()
      self.h2tanh = h2tanh
      # output layer:
      probs   = F.softmax(h5tanh[0], -1)
      self.probs = probs
      def sample():
        ix    = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.g)
        if ix not in self.P.picked:
          self.P.picked.append(ix)
          return ix
        return
      ix      = sample()
      notTensor = type(ix) == type(None)
      while notTensor:
        self.tries += 1
        self.orgtrs[self.P.round] += 1
        ix = sample()
        notTensor = type(ix) == type(None)

      self.paths.append(ix)
      self.lprobs.append(torch.log(probs[ix]))
      return ix

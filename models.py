from turtle import forward
import torch
import torch.nn.functional as F
from math import floor


class PGNetwork(torch.nn.Module):
    '''A policy gradient neural network to produce a probability of going up. Can  be  instantiated  with  an 
    arbitrary number of linear hidden layers, with a minimum number of 1. Can also use leaky or  conventional
    relu  activation  in the hidden layers, as well as linear or sigmoid activation  in the final layer. This
    network  requires inputs to be 1D, meaning a 2D frame must be flattened before input. 

    Input:
    hparams- an argparse parameter object containing runtime parameters
    inputSize- the 1D size of inputs to the network
    outputSize- number of outputs expected from the network, defaults to 1
    '''
    def __init__(self, hparams, inputSize, outputSize, DEVICE):
        super(PGNetwork,self).__init__()
        #class attributes
        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.hiddenSize = hparams.hidden
        self.outputSize = outputSize
        self.softmax = hparams.softmax
        self.DEVICE = DEVICE
        self.dropout_layer = GaussianDropout if 'GAUSS' in hparams.dropout_type.upper() else BernoulliDropout
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-8, dtype=float)
        self.activations = {False:F.relu, True:F.leaky_relu}    #allows user to specify hidden activations
        self.layers = torch.nn.ModuleList()                     #this will store the layers of the network

        self.layers.append( torch.nn.Linear(inputSize, self.hiddenSize) )
        self.layers.append( self.dropout_layer(hparams.dropout,hparams.seed,DEVICE) )

        if hparams.num_hiddens <= 1:
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )
        else:
            for _ in range(2,hparams.num_hiddens+1):
                self.layers.append( torch.nn.Linear(self.hiddenSize, self.hiddenSize) )
                self.layers.append( self.dropout_layer(hparams.dropout,hparams.seed,DEVICE) )
            self.layers.append( torch.nn.Linear(self.hiddenSize, outputSize) )

        self.num_layers = len(self.layers)
        print(self.num_layers,flush=True)  #should be twice the number of specified hidden layers due to dropout

    def forward(self,x):
        '''Takes a 1D input vector and outputs a probability.'''
        #network for probability
        for lyr in range(0,self.num_layers-1,2):
            x = self.layers[lyr+1](self.activations[self.leaky](self.layers[lyr](x)))
        
        p = self.layers[self.num_layers-1](x)
        if self.sigmoid:
            p = torch.sigmoid(p / self.temperature)
        return p

    def evaluate(self,x,m=10):
        '''After training using m=1 for Monte Carlo, use a larger m-value to get more accurate result. 
        Default m-value is 10.'''
        inferences = torch.zeros(m).to(self.DEVICE)
        for i in range(m):
            inferences[i] = self.forward(x).detach()
        return torch.mean(inferences)
        
class CNN_PG(torch.nn.Module):
    '''Uses a deep CNN to implement a policy gradient network. Compares current state'''
    def __init__(self,hparams,w,h,outputSize,DEVICE):
        super(CNN_PG,self).__init__()
        self.leaky = hparams.leaky
        self.sigmoid = hparams.sigmoid
        self.outputSize = outputSize
        self.DEVICE = DEVICE
        self.temperature = torch.tensor(hparams.temperature if hparams.temperature > 0 else 1e-8, dtype=float)
        self.dropout_layer = GaussianDropout if 'GAUSS' in hparams.dropout_type.upper() else BernoulliDropout
        self.activations = {False:torch.nn.ReLU(), True:torch.nn.LeakyReLU()}    #allows user to specify hidden activations

        #layer definitions
        self.layers = torch.nn.ModuleList()

        if hparams.cells < 1:
            hparams.cells = 2

        firstLayer = True
        for _ in range(hparams.cells):
            if not firstLayer:
                self.layers.append( torch.nn.Sequential(torch.nn.Conv2d(in_channels=64,out_channels=64, kernel_size=5,padding=2),
                                                    self.activations[self.leaky])
                                                    )
            else:
                self.layers.append( torch.nn.Sequential(torch.nn.Conv2d(in_channels=2,out_channels=64, kernel_size=5,padding=2),
                                                    self.activations[self.leaky])
                                                    )
                firstLayer = False
            w,h = self.__outSize((w,h),kSize=5,padLength=2)  #shape after batch norm layer

        self.linear1 =  torch.nn.Sequential(torch.nn.Linear(64*w*h, 100), 
                                            self.activations[self.leaky] 
                                            )  #linear layer takes a 1D tensor length: out_channels * w * h, requires flattened tensor
        self.linear2 = torch.nn.Linear(100,1)

    def forward(self,x):
        #utilize dropout during training
        skip = False
        for layer in self.layers:
            if skip:
                x = layer(x) + x
            else:
                x = layer(x)    
                skip = True
        x = self.linear1(torch.flatten(x))
        x = self.linear2(x)
        if self.sigmoid:
            x = torch.sigmoid(x/self.temperature)
        return x
            

    def evaluate(self,x,m=10):
        '''After training using m=1 for Monte Carlo, use a larger m-value to get more accurate result. 
        Default m-value is 10.'''
        inferences = torch.zeros(m).to(self.DEVICE)
        for i in range(m):
            inferences[i] = self.forward(x).detach()
        return torch.mean(inferences)

        
    def __outSize(self, inputSize, kSize, padLength=0, stride=1, dilation=1):
        return floor( ((inputSize[0] + 2*padLength - dilation * (kSize - 1) - 1) / stride) + 1 ), floor( ((inputSize[1] + 2*padLength - dilation * (kSize - 1) - 1) / stride) + 1 )

    def __poolSize(self,inputSize,kSize,padLength=0,stride=None):
        if stride==None:
            return floor( ((inputSize[0] + 2*padLength - kSize[0])/kSize[0])+1 ), floor( ((inputSize[1] + 2*padLength - kSize[1])/kSize[1])+1 )
        return floor( ((inputSize[0] + 2*padLength - kSize[0])/stride)+1 ), floor( ((inputSize[1] + 2*padLength - kSize[1])/stride)+1 )

class GaussianDropout(torch.nn.Module):
    '''Applies noise to every element of the input tensor from sampling a mean=1, stddev=p/(1-p) Normal
    distribution. 
    References:
    https://discuss.pytorch.org/t/gaussiandropout-implementation/151756/4
    https://stackoverflow.com/questions/65501878/gaussiandropout-vs-dropout-vs-gaussiannoise-in-keras
    
    Input:
    p_drop- the probability of an element in the input tensor to be zeroed out
    seed- an integer that is used to seed the prng'''
    def __init__(self,p_drop=.5,seed=1,DEVICE=None):
        super(GaussianDropout,self).__init__()
        self.DEVICE = DEVICE
        self.alpha = p_drop / (1 - p_drop)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x):
        '''Applies noise to every element of the input tensor from sampling a mean=1, stddev=p/(1-p) Normal
        distribution.
        
        Input:
        x- a tensor of activations/values to apply dropout to
        
        Output:
        x- the tensor result of multiplicative Gaussian noise being applied to the input tensor'''
        noise = torch.normal(mean=1,std=self.alpha,generator=self.generator,size=x.size()).to(self.DEVICE)
        x = torch.mul(x,noise)
        return x

class BernoulliDropout(torch.nn.Module):
    '''Creates a dropout layer using a Bernoulli distribution with a probability of sampling 1
    defined as (1 - p_drop). 
    
    Input:
    p_drop- the probability of an element in the input tensor to be zeroed out'''
    def __init__(self,p_drop=.5,seed=1,DEVICE=None):
        super(BernoulliDropout,self).__init__()
        self.DEVICE = DEVICE
        self.p_drop = p_drop
        self.bern = torch.distributions.bernoulli.Bernoulli(probs=(1-p_drop))
    
    def forward(self,x):
        '''Applies noise to every element of the input tensor from sampling a Bernoulli distribution.
        
        Input:
        x- a tensor of activations/values to apply dropout to
        
        Output:
        x- the tensor result of Bernoulli noise being applied to the input tensor'''
        noise = self.bern.sample(sample_shape=x.size()).to(self.DEVICE)
        x = torch.mul(x,noise)
        return x
        
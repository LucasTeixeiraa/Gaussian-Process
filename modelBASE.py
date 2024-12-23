"""
This is a baseline model class to map real parameters to a TTModel implemented in CUDA.

@author: yanbw
"""


###Base e modelo 
import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import chaospy as cp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from SALib.sample import saltelli
from SALib.analyze import sobol
import timeit
import re
import collections
import os
import six

class TTCellModel:
    tf=100
    ti=0
    dt=0.1
    dtS=1
    parametersN=["ki","ko","gna","gca","atp"]
    
    K_o_default=5.40e+00
    g_CaL_default=1.750e-04
    g_Na_default=1.48380e+01
    K_i_default=138.3
    atp_default=5.6
    g_K1_defaults=5.4050e+00
    g_Kr_defaults=0.096
    g_Ks_defaults=0.245
    g_to_defaults=2.940e-01
    g_bca_defaults=5.920e-04

    @staticmethod
    def parseR(name="out.txt"):  
        
      
       
      
      
        X=[]
        file = open(name, 'r')
        for row in file:
           aux=[]
           for x  in row.split(' '):
               try:
                   aux.append(float(x))
               except:
                   aux.append(-100)
           ads=TTCellModel.ads(aux[:-10],[0.5,0.9],aux[-10] )
         
           try:
               k={"Wf": aux[:-2] ,"dVmax":aux[-1],"ADP90":ads[1],"ADP50":ads[0],"Vreps":aux[-10],"tdV":aux[-2]}
           except:
             k={"Wf": aux[:-2] }
             #plt.plot(aux[:-2])
             #plt.show()
            # print("ADCALCERROR ",ads)
           X.append(k)
   
        return X
    
      
    @staticmethod
    def cofs(ps):

        params=[
           (1-0.25*ps[2])*TTCellModel.g_Na_default,
           
           (1-0.25*ps[2])*TTCellModel.g_CaL_default,
            
           TTCellModel.K_i_default - 13.3*ps[2],
           
           TTCellModel.K_o_default + 4.6*ps[1],
           
           TTCellModel.atp_default - 3 * ps[0], ##Atp
           
           TTCellModel.g_K1_defaults, 
           TTCellModel.g_Kr_defaults, 
           TTCellModel.g_Ks_defaults,
            
           TTCellModel.g_to_defaults,
           TTCellModel.g_bca_defaults,
           
            
            ]
     
        return np.array(params)
    
    @staticmethod 
    def prepareinput(P,cofs):
        parametersS=[cofs(p) for p in P]
        with open('m.txt','wb') as f:
             np.savetxt(f,parametersS, fmt='%.8f')
             
    @staticmethod
    def run(P="",use_gpu=False, regen=True,name="out.txt",cofsF=cofs.__func__):  
       with open(name, "w") as file:
        file.write("")  # Creates an empty file
       
       count=0
       countL=0
       output=name
       try:
           with open(name, 'r') as f:
                 for line in f:
                      count += 1
                      countL=0
                      for i in line:
                          countL+=1
       except:
           count=0
           
       try:  
            countP=0
            with open(P, 'r') as f:
                for line in f:
                     countP += 1
       except:
            try:
                countP=np.shape(P)[0]
            except:
                print("  Input distribution or file not given, reusing existing results")
                countP=count
            
       
       if(count==countP and regen==False):
            print("  Using existing results at ",name) ##!!! NO GUARANTEE SIZE PARAMETERS ARE THE SAME!!!!
       else:
   
            if False==isinstance(P, six.string_types): ##P is file or dist
                print("Solving from scracth for P(",countP,")")
                print("  Generating Input file")
                TTCellModel.prepareinput(P,cofsF)
            else:
                print("  Solving from input file ",P)
                print("  Using given input file")
                inpt=P
            TTCellModel.callCppmodel(countP,use_gpu,output,"m.txt")
            
            print ("  model output ready at",name," parsing it...")
            
            
       
       # Delete the file "out.txt" if it exists
       parsedR=TTCellModel.parseR(name)
       print("  parsing done")
       if os.path.exists(name):
            os.remove(name)
            print("  Output File  deleted.")
       else:
            print("  Cant clean Output File")
       return parsedR
        

    @staticmethod
    def setParametersOfInterest(parametersN):
        TTCellModel.parametersN=parametersN
      
    

    @staticmethod
    def getSimSize(): #Returns size of result vector for given simulation size parameters, usefull for knowing beforehand the number of datapoints to compare
        n=0#(tf-ti)/dt
        return n
    
    @staticmethod
    def setSizeParameters(ti,tf,dt,dtS):
        TTCellModel.ti=ti
        TTCellModel.tf=tf
        TTCellModel.dt=dt
        TTCellModel.dtS=dtS
        
    @staticmethod   #returns the time points at wich there is evalution
    def getEvalPoints():
        
        t=TTCellModel.ti
        ts=0
        ep=[]
        while(t<TTCellModel.tf):
            
            if (ts >= TTCellModel.dtS) :
                ep.append(t)
               	ts = 0
            t=t+TTCellModel.dtS
            ts=ts+TTCellModel.dtS
            
        return ep    
    
    @staticmethod      
    def ads(sol,repoCofs,repos): ##calculo da velocidade de repolarização
        k=0
        i=0;
        out={}
        x=sol
        flag=0
  
        x=np.array(sol)
        index=0
        idxmax=0

        for value in x:
           index+=1  
           if(value==x.max()):
                        flag=1                
                        out[len(repoCofs)]=index  + TTCellModel.ti
                        idxmax=index
           if(flag==1):
                        k+=1
           if(flag==1 and repoCofs[i]*repos >= value):
                        out[i]= (k)
                        i+=1
           if(i>=len(repoCofs)):
               
                        break

         
        return out

    @staticmethod
    def callCppmodel(N,use_gpu=False,outpt="out.txt",inpt="m.txt"):  
     #   print("Calling solver")
        name="c.exe"
        args=name +" --tf="+str(TTCellModel.tf)+" --ti="+str(TTCellModel.ti)+" --dt="+str(TTCellModel.dt)+" --dt_save="+str(TTCellModel.dtS) +" --n="+str(N)+" --i="+inpt+" --o="+outpt  
       
        if(use_gpu):
            args=args+" --use_gpu=1"
     
        
        print("  kernel call:",args)
        output = subprocess.Popen(args,stdout=subprocess.PIPE,shell=True)
        string = output.stdout.read().decode("utf-8")
        # Delete the file "out.txt" if it exists
        if os.path.exists(inpt):
            os.remove(inpt)
            print("  Input File deleted.")
        else:
            print("File 'out.txt' does not exist.")


    @staticmethod
    def getDist(low=0,high=1):
            
        hypox=cp.Uniform(low,high)
        hyper=cp.Uniform(low,high)    
        acid=cp.Uniform(low,high) 
        dist = cp.J(hypox,hyper,acid)
        return dist

    @staticmethod
    def getNPar():
        return 3


    
    
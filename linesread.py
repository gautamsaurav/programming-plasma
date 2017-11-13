from itertools import islice
import numpy as np
ns=14
mobility=np.zeros((1,ngrid0),float)#diffusion coefficient at each grid points
diffusion=np.zeros((1,ngrid0),float)#mobility at grid points, 1 row=1
sourcee=np.zeros((ns,ngrid0),float)#reaction rate at each grid points
#31+bol_indx*3+(bol_indx-1)*1000
mobind=np.array([(31+3*3+2*ngrid0),(31+3*3+2*ngrid0)+ngrid0+1])
difind=np.array([(1*2+(1-1)*ngrid0),(1*3+(1-1)*ngrid0+ngrid0+1)])
with open('bolsigplus032016-linux/output.txt') as lines:
    mobility[0,:]= np.transpose(np.genfromtxt(islice(lines,int(mobind[0]),int(mobind[1])))[:,1])
    diffusion[0,:]=np.transpose(np.genfromtxt(islice(lines, difind[0],difind[1]))[:,1])
    sourcee[0,:]=np.transpose(np.genfromtxt(islice(lines,13*3+2+13*ngrid0,14*3+13*ngrid0+1001))[:,1])
    for indd in np.arange(ns-1):
        sourcee[indd+1,:]=np.transpose(np.genfromtxt(islice(lines,2,1004))[:,0])
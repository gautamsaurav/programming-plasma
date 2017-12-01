#==== Python code on 1D Helium plasma, (7-11-2017) ====
#==================Saurav Gautam========================= 
import numpy as np
import sys
import math
from itertools import islice
import os
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as la

def laplaceDensityPotential(nxp, b, k1=-1, k2=0, k3=1):
    a = np.ones(nxp)*(0.5);a[0]=0.
    b[0]=1.;b[-1]=1.
    c = np.ones(nxp)*(0.5);c[-1]=0.
    return sparse.dia_matrix(([c,b,a],[k3,k2,k1]),shape=(nxp,nxp)).transpose().tocsc() 
    


ee=1.6*10**(-19) #electronic charge
e0=8.54187817*10**(-12) #epsilon
townsendunit=1.0/((2.5*10**(25))*(10)**(-21))

#*** Importing the value of MOBILITY, DIffusion and reaction rate from a text file
#----------------------------------------------------------------------------------------------------
#inpfile='input.txt'
oupfile='output.txt'
emobility=np.zeros((1,996),float)
ediffusion=np.zeros((1,996),float)
esourcee=np.zeros((1,996),float)
imobility=np.zeros((1,996),float)
idiffusion=np.zeros((1,996),float)
file=open(oupfile)
line=file.readline()
for data in np.arange(996):
   line=file.readline()
   lineSplit=line.split()
   emobility[0,data]=float(lineSplit[1])
print('done with emobility')
line=file.readline(); line=file.readline();
for data in np.arange(996):
   line=file.readline()
   lineSplit=line.split()
   ediffusion[0,data]=float(lineSplit[1])
print('done with ediffusion')
line=file.readline(); line=file.readline();
for data in np.arange(996):
   line=file.readline()
   lineSplit=line.split()
   esourcee[0,data]=float(lineSplit[1])
print('done with esourcee')
line=file.readline(); line=file.readline();
for data in np.arange(996):
   line=file.readline()
   lineSplit=line.split()
   imobility[0,data]=float(lineSplit[1])
print('done with imobility')
line=file.readline(); line=file.readline();
for data in np.arange(996):
   line=file.readline()
   lineSplit=line.split()
   idiffusion[0,data]=float(lineSplit[1])
print('done with idiffusion')
   
'''
with open('output.txt') as lines:
    
    emobility[0,:]=np.transpose(np.genfromtxt(islice(lines,1,997))[:,1])
    ediffusion[0,:]=np.transpose(np.genfromtxt(islice(lines, 2,998))[:,1])
    esourcee[0,:]=np.transpose(np.genfromtxt(islice(lines,2,998))[:,1])
    imobility[0,:]=np.transpose(np.genfromtxt(islice(lines,2,998))[:,1])
    idiffusion[0,:]=np.transpose(np.genfromtxt(islice(lines,2,998))[:,1])
'''
#*** Description of plasma reactor
#---------------------------------------------------------------------------------------------------
width=11.0     #space between two dielectric in mm
ngrid0=300     #Number of grid points (between two dielectric)
wd1=3.0        #width of first dielectric in mm
wd2=3.0        #width of second dielectric in mm
volt=200000.0    #Interelectrode voltage (peak not RMS)
gasdens=2.504e25          #number density of gas at NTP (unit: m^-3)
dx=width*10**(-2)/(ngrid0+1)#Grid size in meter
nwd1=int(wd1*10**(-2)/dx)       #number of grid points in first dielectric
nwd2=int(wd2*10**(-2)/dx)       #Number of grid points in second dielectric
wd1=nwd1*dx                 #Making wd1 as exact multiple of dx
wd2=nwd2*dx                 #making wd2 as exact multiple of dx
inelec=width*10**(-2)+wd1+wd2#total interelectrode separation
ngrid=int(ngrid0+2+nwd1+nwd2)    #total number of grid points(2 dielectrics +gas medium + all edge points)
dt=1.e-7 #small time interval

#*** Initialization
#-----------------------------------------------------------------------------------------------------
ns=2          #Number of species (Taking helium gas, one is electron and another is helium ion)
ndensity=np.zeros((ns,ngrid0+2),float) #Density of each species in all grid points between dielectric
ncharge=np.array([-1,1])  #Charge of the each species
netcharge=np.zeros((1,ngrid),float) #net charge at all grid points used to solve poission equation
potentl=np.zeros((1,ngrid),float) #potential at each grid points
uu=np.zeros((1,ngrid),float) #potential at each grid points
efield=np.zeros((1,ngrid0+2),float) #electric field at each grid points
mobegrid=np.zeros((1,ngrid0+2),float)#mobility at grid points, 1 row=1 tyepe of gaseous species
difegrid=np.zeros((1,ngrid0+2),float)#diffusion coefficient at each grid points
sourceegrid=np.zeros((1,ngrid0+2),float)#reaction rate at each grid points
mobigrid=np.zeros((1,ngrid0+2),float)
difigrid=np.zeros((1,ngrid0+2),float)
sig_e_left=0   #Electron surface charge density at left dielectric
sig_e_right=0  #Electron surface charge density at right dielectric
sig_i_left=0   #Ion surface charge density at left dielectric      
sig_i_right=0  #Ion surface charge density at right dielectric

ndensity = np.random.randint(1.e3, size=(ns, ngrid0+2))

#=======================Time Loop======================================================================
flogfile= open("density.txt","w")
for time in range(1,20000):
    #poission equation
    #--------------------------------------------------------------------------------------------------	
    toll= 1.e-6 #tolerence
    netcharge[:,:]=0.0 #clear the garbage value from pervious loop
    for i in np.arange(ns):
        netcharge[0,nwd1:nwd1+2+ngrid0]+=ee*ncharge[i]*ndensity[i,:]  #calculating the net charge at each grid points
  
    #boundary condition for potential
    if time==0: potentl[:,:]=0 # delete the garbage value of potential, if exists
    potentl[0,0]=volt*math.cos(((math.pi)/180)*(time*10**(-8))*10**(2)*360) # 1 cycle at 1 microsecond
    potentl[0,:]=potentl[0,0]*(ngrid-np.arange(ngrid)-1.0)/(ngrid-1.0)
    RHS=netcharge[0].copy();RHS[0]=1.;RHS[-1]=1.    
    A=laplaceDensityPotential(ngrid,RHS)
    iterations=0
    err=np.ones(ngrid);maxErr=1.
    while maxErr>toll:
        uu[0]=A.dot(potentl[0])
        err=(uu[0]-potentl[0])
        iterations+=1
        maxErr=np.max(np.abs(err))
        #print(maxErr)
        #plt.plot(err)
        #plt.show()
    plt.plot(netcharge[0])
    plt.show()
    print('time:',time, 'iterations',iterations)
   
    #**calculate electric field as negative gradient of potential (Expressed in Townsend Unit)
    efield[:,:]=townsendunit*(potentl[0,nwd1+1:nwd1+3+ngrid0]-potentl[0,nwd1-1:nwd1+1+ngrid0])/(-2.0*dx)
    if any(efield[0,:]>990):#All the reaction coefficients are calculated for efield<990. Value more than that will imply that the there is something wrong in the simulation
       f= open("logfile.txt","w+")
       f.write("Error!! The value of Electric field exceeded limit. Something might be wrong!!")
       sys.exit()
	
    #calculating the coefficients (Interpolation..)
    indlocate=efield[:,:].astype(int)
    mobegrid[:,:]=emobility[0,indlocate]+(emobility[0,indlocate+1]-emobility[0,indlocate])*(efield-indlocate)
    difegrid[:,:]=ediffusion[0,indlocate]+((ediffusion[0,indlocate+1]-ediffusion[0,indlocate])*(efield-indlocate))
    sourceegrid[:,:]=gasdens*(esourcee[0,indlocate]+(esourcee[0,indlocate+1]-esourcee[0,indlocate])*(efield-indlocate))
    mobigrid[:,:]=imobility[0,indlocate]+(imobility[0,indlocate+1]-imobility[0,indlocate])*(efield-indlocate)
    difigrid[:,:]=idiffusion[0,indlocate]+((idiffusion[0,indlocate+1]-idiffusion[0,indlocate])*(efield-indlocate))

    #continuity equation
    #---------------------------------------------------------------------------------------------------
    #electron
    ndensity[0,1:ngrid0+1]=(sourceegrid[0,1:ngrid0+1]+ndensity[0,1:ngrid0+1]*(efield[0,2:ngrid0+2]*mobegrid[0,2:ngrid0+2]-efield[0,0:ngrid0]*mobegrid[0,0:ngrid0])/(2*dx)-efield[0,1:ngrid0+1]*mobegrid[0,1:ngrid0+1]*(ndensity[0,2:ngrid0+2]-ndensity[0,0:ngrid0])/(2*dx)+difegrid[0,1:ngrid0+1]*(ndensity[0,2:ngrid0+2]-2*ndensity[0,1:ngrid0+1]+ndensity[0,0:ngrid0])/(dx*dx)+(ndensity[0,2:ngrid0+2]-ndensity[0,0:ngrid0])*(difegrid[0,2:ngrid0+2]-difegrid[0,0:ngrid0])/(4*dx*dx))*dt+ndensity[0,1:ngrid0+1]
    #ion
    ndensity[1,1:ngrid0+1]=(sourceegrid[0,1:ngrid0+1]-ndensity[1,1:ngrid0+1]*(efield[0,2:ngrid0+2]*mobigrid[0,2:ngrid0+2]-efield[0,0:ngrid0]*mobigrid[0,0:ngrid0])/(2*dx)-efield[0,1:ngrid0+1]*mobigrid[0,1:ngrid0+1]*(ndensity[1,2:ngrid0+2]-ndensity[1,0:ngrid0])/(2*dx)+difigrid[0,1:ngrid0+1]*(ndensity[1,2:ngrid0+2]-2*ndensity[1,1:ngrid0+1]+ndensity[1,0:ngrid0])/(dx*dx)+(ndensity[1,2:ngrid0+2]-ndensity[1,0:ngrid0])*(difigrid[0,2:ngrid0+2]-difigrid[0,0:ngrid0])/(4*dx*dx))*dt+ndensity[1,1:ngrid0+1]

    #charge accumulation at surface of dielectric
    #---------------------------------------------------------------------------------------------------
    sig_e_left=sig_e_left+dt*(ndensity[0,0]*mobegrid[0,0]*efield[0,0]-10*sig_e_left-10**(-6)*sig_e_left*sig_e_right)
    sig_e_right=sig_e_right+dt*(ndensity[0,ngrid0+1]*mobegrid[0,ngrid0+1]*efield[0,ngrid0+1]-10*sig_e_right-10**(-6)*sig_e_left*sig_e_right)
    sig_i_left=sig_i_left+dt*((1+0.01)*ndensity[1,0]*mobigrid[0,0]*efield[0,0]-10**(-6)*sig_i_left*sig_i_right)
    sig_i_right=sig_i_right+dt*((1+0.01)*ndensity[1,ngrid0+1]*mobigrid[0,ngrid0+1]*efield[0,ngrid0+1]-10**(-6)*sig_i_left*sig_i_right)
    ndensity[0,0]=(sig_e_left)*dx
    ndensity[0,ngrid0+1]=(sig_e_right)*dx     
    ndensity[1,0]=(sig_i_left)*dx
    ndensity[1,ngrid0+1]=(sig_i_right)*dx

    #export data of number density at regular intervals
    #print(time)
    if 1:
        plt.plot(ndensity[0]);plt.plot(ndensity[1]);
        plt.show()
#np.savetxt(flogfile, ndensity, delimiter=" ", newline = "\n", fmt="%s")
flogfile.close()
		
#end for

#==== Python code on 1D Helium plasma, (7-11-2017) ====
#==================Saurav Gautam=========================
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import scipy.sparse as sparse

def poissonmatrix(nxp, k1=-1, k2=0, k3=1):
    a = np.ones(nxp)*(2.0)
    b = np.ones(nxp-1)*(-1)
    return np.diag(b, k1) + np.diag(a, k2) + np.diag(b, k3)
    
def tridiagSparse(nxp, k1=-1, k2=0, k3=1):
    a = np.ones(nxp)*(-1)
    b = np.ones(nxp)*(2)
    return sparse.dia_matrix(([a,b,a],[k1,k2,k3]),shape=(nxp,nxp)).tocsc()

def readBoltzmannParameters(npoints,oupfile='output.txt'):
    #print('*** Importing the value of MOBILITY, DIffusion and reaction rate from a text file')
    emobility=np.zeros(npoints,float)
    ediffusion=np.zeros(npoints,float)
    esourcee=np.zeros(npoints,float)
    imobility=np.zeros(npoints,float)
    idiffusion=np.zeros(npoints,float)
   
    file=open(oupfile)
    line=file.readline()
    for data in np.arange(npoints):
       line=file.readline()
       lineSplit=line.split()
       emobility[data]=float(lineSplit[1])
    #print('done with emobility')
    line=file.readline(); line=file.readline();
    for data in np.arange(npoints):
       line=file.readline()
       lineSplit=line.split()
       ediffusion[data]=float(lineSplit[1])
    #print('done with ediffusion')
    line=file.readline(); line=file.readline();
    for data in np.arange(npoints):
       line=file.readline()
       lineSplit=line.split()
       esourcee[data]=float(lineSplit[1])
    #print('done with esourcee')
    line=file.readline(); line=file.readline();
    for data in np.arange(npoints):
       line=file.readline()
       lineSplit=line.split()
       imobility[data]=float(lineSplit[1])
    #print('done with imobility')
    line=file.readline(); line=file.readline();
    for data in np.arange(npoints):
       line=file.readline()
       lineSplit=line.split()
       idiffusion[data]=float(lineSplit[1])
       #print(data,idiffusion[data])
       #print('done with idiffusion')
   
    return(emobility,ediffusion,esourcee,imobility,idiffusion)
      
      
def fourPlots(ones,titleone,two,titletwo,three,titlethree,four,titlefour):      
        f, axarr = plt.subplots(2, 2)
        for field in ones:
            axarr[0,0].plot(field);
        axarr[0,0].set_title(titleone)       
        axarr[0,1].plot(netcharge)
        axarr[0,1].set_title(titletwo)
        axarr[1,1].plot(potentl)
        axarr[1,1].set_title(titlethree)       
        axarr[1,0].plot(efield)      
        axarr[1,0].set_title(titlefour)
        f.subplots_adjust(hspace=0.3)
        plt.show()

parameterSize=996       
(emobility,ediffusion,esourcee,imobility,idiffusion) = readBoltzmannParameters(parameterSize,'output.txt')


#*** Parameters for the of plasma reactor
#-------------------------------------------------------------------------------------------------------------
width=5.0     #space between two dielectric in mm
ngrid0=2000     #Number of grid points (between two dielectric)
wd1=0.5        #width of first dielectric in mm
wd2=0.5        #width of second dielectric in mm
dx=width*10**(-3)/(ngrid0+1.0)#Grid size in meter
nwd1=int(wd1*10**(-3)/dx)       #number of grid points in first dielectric
nwd2=int(wd2*10**(-3)/dx)       #Number of grid points in second dielectric
wd1=nwd1*dx                 #Making wd1 as exact multiple of dx
wd2=nwd2*dx                 #making wd2 as exact multiple of dx
inelec=width*10**(-3)+wd1+wd2#total interelectrode separation
ngrid=int(ngrid0+2+nwd1+nwd2)    #total number of grid points(2 dielectrics +gas medium + all edge points)
#--------------------------------------------------------------------------------------------------------------
volt=6000.0    #Interelectrode voltage (peak not RMS)
gasdens=2.504e25          #number density of gas at NTP (unit: m^-3)
dt=1.0e-9 #small tyme interval
dt1=1.0*dt
frequencySource = 20000 #30KHz
ee=1.6*10**(-19) #electronic charge
e0=8.54187817*10**(-12) #epsilon
townsendunit=1.0/((2.5*10**(25))*(10)**(-21))

#*** Initialization
#-----------------------------------------------------------------------------------------------------
ns=2          #Number of species (Taking helium gas, one is electron and another is helium ion)
ndensity=np.zeros((ns,ngrid0+2),float) #Density of each species in all grid points between dielectric
ncharge=np.array([-1,1])  #Charge of the each species
netcharge=np.zeros(ngrid,float) #net charge at all grid points used to solve poission equation
potentl=np.zeros(ngrid,float) #potential at each grid points
uu=np.zeros(ngrid,float) #potential at each grid points
efield=np.zeros(ngrid0+2,float) #electric field at each grid points
mobegrid=np.zeros(ngrid0+2,float)#mobility at grid points, 1 row=1 tyepe of gaseous species
difegrid=np.zeros(ngrid0+2,float)#diffusion coefficient at each grid points
sourceegrid=np.zeros(ngrid0+2,float)#reaction rate at each grid points
mobigrid=np.zeros(ngrid0+2,float)
difigrid=np.zeros(ngrid0+2,float)
sig_e_left=0.0   #Electron surface charge density at left dielectric
sig_e_right=0.0  #Electron surface charge density at right dielectric
sig_i_left=0.0   #Ion surface charge density at left dielectric     
sig_i_right=0.0  #Ion surface charge density at right dielectric

Maat1=poissonmatrix(ngrid-2)
Maat2=tridiagSparse(ngrid-2)
invertedmat=la.inv(Maat2)
numberOfSteps = 1000000000

#ndensity = np.ones((2,ngrid0+2),float)*1000.0
ndensity=1000*np.random.rand(2,ngrid0+2)

#=======================tyme Loop======================================================================
#storeResults=np.zeros((numberOfSteps,5,ngrid),float)
oo=1
tymestep2=0.0
tyme=0.0

for tymeStep in range(1,numberOfSteps):
    tyme=tyme+dt
#    if leftPot>2450: 
#        dt=10e-14
#    else: 
#        dt=10e-10
   #poission equation
    #==================================================================================================================
    netcharge[:]=0.0 #clear the garbage value from pervious loop
    for i in np.arange(ns):
        netcharge[nwd1:nwd1+2+ngrid0]+=ee*ncharge[i]*ndensity[i,:]  #calculating the net charge at each grid points
    #boundary condition for potential
    leftPot=1.0*volt*np.sin(2*np.pi*tyme*frequencySource) # frequency of the source is given by a parameter
    rightpot=0.0*volt*np.sin(2*np.pi*tyme*frequencySource)
    #potentl[:]=leftPot*(ngrid-np.arange(ngrid)-1.0)/(ngrid-1.0)
    chrgg=(netcharge[1:-1]/e0)*dx*dx
    chrgg[0]=chrgg[0]+leftPot
    chrgg[-1]=chrgg[-1]+rightpot
    potentl[0]=leftPot
    potentl[-1]=rightpot
    #solvedsparse=la.spsolve(Maat2,chrgg)
    #%solvpot=np.linalg.solve(Maat1, chrgg)
    solvpot=invertedmat.dot(chrgg)
    
    potentl[1:-1]=solvpot 
    #**calculate electric field as negative gradient of potential (Expressed in Townsend Unit)
    efield[:]=-townsendunit*(potentl[nwd1+1:nwd1+3+ngrid0]-potentl[nwd1-1:nwd1+1+ngrid0])/(2.0*dx)
    efield[0]=-townsendunit*(potentl[nwd1+1]-potentl[nwd1])/dx
    efield[-1]=-townsendunit*(potentl[nwd1+1+ngrid0]-potentl[nwd1+ngrid0])/dx
    if any(abs(efield[:])>1000):#All the reaction coefficients are calculated for efield<npoints. Value more than that will imply that the there is something wrong in the simulation
       f= open("logfile.txt","w+")
       f.write("Error!! The value of Electric field exceeded limit. Something might be wrong!!")
       sys.exit()
   
    #calculating the coefficients (Interpolation..)-------------------------------------------------------------------------------------
    indlocate=abs(efield[:]).astype(int)
    mobegrid=-1.0*(emobility[indlocate]+(emobility[indlocate+1]-emobility[indlocate])*(abs(efield)-indlocate))
    difegrid=1.0*(ediffusion[indlocate]+((ediffusion[indlocate+1]-ediffusion[indlocate])*(abs(efield)-indlocate)))
    sourceegrid=1.0*(esourcee[indlocate]+(esourcee[indlocate+1]-esourcee[indlocate])*(abs(efield)-indlocate))
    mobigrid=1.0*(imobility[indlocate]+(imobility[indlocate+1]-imobility[indlocate])*(abs(efield)-indlocate))
    difigrid=1.0*idiffusion[indlocate]+((idiffusion[indlocate+1]-idiffusion[indlocate])*(abs(efield)-indlocate))

    #---------------------Flux Correction----------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------------------
    velocitye=np.zeros(ngrid0+6,float)
    velocityi=np.zeros(ngrid0+6,float)
    rhoe=np.zeros(ngrid0+6,float)
    rhoi=np.zeros(ngrid0+6,float)
    velocitye[2:ngrid0+4]=mobegrid*efield
    velocityi[2:ngrid0+4]=mobigrid*efield
    rhoe[3:ngrid0+3]=ndensity[0,1:-1].copy()
    rhoi[3:ngrid0+3]=ndensity[1,1:-1].copy()
    
    Avelocitye=np.zeros(ngrid0+3,float)
    Avelocityi=np.zeros(ngrid0+3,float)
    Avelocitye=0.5*(velocitye[1:ngrid0+4]+velocitye[2:ngrid0+5])
    Avelocityi=0.5*(velocityi[1:ngrid0+4]+velocityi[2:ngrid0+5])
            #step1-------------------------------#time calculation
    flowe=(0.5*(velocitye[1:ngrid0+4]*rhoe[1:ngrid0+4]+velocitye[2:ngrid0+5]*rhoe[2:ngrid0+5])-0.5*abs(Avelocitye[:])*(rhoe[2:ngrid0+5]-rhoe[1:ngrid0+4]))*dt
    flowi=(0.5*(velocityi[1:ngrid0+4]*rhoi[1:ngrid0+4]+velocityi[2:ngrid0+5]*rhoi[2:ngrid0+5])-0.5*abs(Avelocityi[:])*(rhoi[2:ngrid0+5]-rhoi[1:ngrid0+4]))*dt
            #Step2--------------------------------
    fhighe=((7.0/12)*(velocitye[1:ngrid0+4]*rhoe[1:ngrid0+4]+velocitye[2:ngrid0+5]*rhoe[2:ngrid0+5])-(1.0/12)*(velocityi[0:ngrid0+3]*rhoi[0:ngrid0+3]+velocityi[3:ngrid0+6]*rhoi[3:ngrid0+6]))*dt
    fhighi=((7.0/12)*(velocityi[1:ngrid0+4]*rhoi[1:ngrid0+4]+velocityi[2:ngrid0+5]*rhoi[2:ngrid0+5])-(1.0/12)*(velocityi[0:ngrid0+3]*rhoi[0:ngrid0+3]+velocityi[3:ngrid0+6]*rhoi[3:ngrid0+6]))*dt
            #step3--------------------------------
    adife=fhighe-flowe
    adifi=fhighi-flowi
            #step4--------------------------------
    qtde=np.zeros(ngrid0+6,float)
    qtdi=np.zeros(ngrid0+6,float)
    qtde[2:ngrid0+4]=ndensity[0,:]-(flowe[1:ngrid0+3]-flowe[0:ngrid0+2])/dx
    qtdi[2:ngrid0+4]=ndensity[1,:]-(flowi[1:ngrid0+3]-flowi[0:ngrid0+2])/dx
    
            #step5--------------------------------
    signne=np.sign(adife)
    signni=np.sign(adifi)
    ace=signne*np.maximum(np.zeros(ngrid0+3,float),np.minimum(abs(adife),np.minimum(signne*(qtde[3:]-qtde[2:-1])*dx,signne*(qtde[1:-2]-qtde[:-3])*dx)))
    aci=signni*np.maximum(np.zeros(ngrid0+3,float),np.minimum(abs(adifi),np.minimum(signni*(qtdi[3:]-qtdi[2:-1])*dx,signni*(qtdi[1:-2]-qtdi[:-3])*dx)))

            #step6--------------------------------
    ndene=qtde[2:ngrid0+4]-(ace[1:ngrid0+3]-ace[0:ngrid0+2])/dx
    ndeni=qtdi[2:ngrid0+4]-(aci[1:ngrid0+3]-aci[0:ngrid0+2])/dx
        
        #diffusion and source term
    ndentemp=np.zeros((ns,ngrid0+4),float)
    difetemp=np.zeros(ngrid0+4,float)
    difitemp=np.zeros(ngrid0+4,float)
    ndentemp[0,1:-1]=ndensity[0,:].copy()
    ndentemp[1,1:-1]=ndensity[1,:].copy()
    difetemp[1:-1]=difegrid[:].copy()
    difitemp[1:-1]=difigrid[:].copy()
    ndene[1:-1]=ndene[1:-1]+(  sourceegrid[1:-1]*abs(ndene[1:-1]) +((difetemp[3:-1]-difetemp[1:-3])/(2.0*dx))*((ndentemp[0,3:-1]-ndentemp[0,1:-3])/(2.0*dx))+difetemp[2:-2]*(ndentemp[0,3:-1]-2*ndentemp[0,2:-2]+ndentemp[0,1:-3])/(dx*dx)  )*dt
    ndeni[1:-1]=ndeni[1:-1]+(  sourceegrid[1:-1]*abs(ndene[1:-1]) +((difitemp[3:-1]-difitemp[1:-3])/(2.0*dx))*((ndentemp[1,3:-1]-ndentemp[1,1:-3])/(2.0*dx))+difitemp[2:-2]*(ndentemp[0,3:-1]-2*ndentemp[0,2:-2]+ndentemp[0,1:-3])/(dx*dx)  )*dt
    ndensity[0,1:-1]=ndene[1:-1].copy()
    ndensity[1,1:-1]=ndeni[1:-1].copy()
    
    ndensity[ndensity<1000]=1000
    #time calculation
    #charge accumulation at surface of dielectric
    #-----------------------------------------------------------------------------------------------------
    efluxleft=-0.5*(ndene[0]+ndene[1])*(mobegrid[1])*efield[1]
    if efluxleft<0:
        efluxleft=0
    sig_e_left= sig_e_left+dt*(efluxleft-10*sig_e_left-10**(-10)*sig_e_left*sig_e_right)
    
    efluxright=0.5*(ndene[-1]+ndene[-2])*(mobegrid[-2])*efield[-2]
    if efluxright<0:
        efluxright=0
    sig_e_right=sig_e_right+dt*(efluxright-10*sig_e_right-10**(-10)*sig_e_left*sig_e_right)
    
    ifluxleft=-(1+0.01)*0.5*(ndeni[0]+ndeni[1])*(mobigrid[1])*efield[1]
    if ifluxleft<0:
        ifluxleft=0
    sig_i_left=sig_i_left+dt*(ifluxleft-10**(-10)*sig_i_left*sig_i_right)
    ifluxright=(1+0.01)*0.5*(ndeni[-1]+ndeni[-2])*(mobigrid[-2])*efield[-2]
    if ifluxright<0:
        ifluxright=0
    sig_i_right=sig_i_right+dt*(ifluxright-10**(-10)*sig_i_left*sig_i_right)
    #print (sig_e_left,sig_e_right,sig_i_left,sig_i_right)plasma_5000k_Fast_1
    
    ndensity[0,0]=0.5*(ndensity[0,1]+ndensity[0,2])+(sig_e_left)/dx
    ndensity[0,-1]=0.5*(ndensity[0,-2]+ndensity[0,-3])+(sig_e_right)/dx    
    ndensity[1,0]=0.5*(ndensity[1,1]+ndensity[1,2])+(sig_i_left)/dx
    ndensity[1,-1]=0.5*(ndensity[1,-2]+ndensity[1,-3])+(sig_i_right)/dx
    #dt=dt127e3
    #tymestep2+=1
    
    
    mobsta=0.5*(mobegrid[1:]+mobegrid[:-1])
    difsta=0.5*(difegrid[1:]+difegrid[:-1])
    eefsta=0.5*(efield[1:]+efield[:-1])/townsendunit
    gstability=max(abs(mobsta*(efield[1:]-efield[:-1])/(dx*townsendunit)+eefsta*(mobegrid[1:]-mobegrid[:-1])/dx+mobsta*eefsta/(2*dx)      +4*difsta/(dx*dx) +(difegrid[1:]-difegrid[:-1])/(dx*dx)))
    dt=1.0/(gstability)    
    
    if (tymeStep % 1000 == 0): #and leftPot>1080 ):
        #print (efluxleft,ifluxleft,efluxright,ifluxright)
        #print()
        f = open('testing.txt', 'ab')
        np.savetxt(f, ndensity[0,:])
        np.savetxt(f, ndensity[1,:])
        np.savetxt(f, netcharge)
        np.savetxt(f, efield)
        np.savetxt(f, potentl);
        f.close()
#        fourPlots((ndensity[0,:],ndensity[1,:]),'ndensity',netcharge,'netcharge',potentl[180:510],'electric potential',efield,'')
        #print('eleft', efluxleft, 'eritht', efluxright, 'ileft', ifluxleft, 'iright', ifluxright)
        #print('eleft', efield[2], 'eritht', efield[-2], 'ileft', efield[2], 'iright', efield[-2])
        #print(ndensity[0,100],ndensity[1,100])
        #print(dt)
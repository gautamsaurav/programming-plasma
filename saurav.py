#==== Python code on 1D Nitrogen plasma, (7-11-2017) ====
#==================Saurav Gautam========================= 
import numpy as np
from itertools import islice
import os
ee=1.6*10**(-19)
e0=854187817*10**(-12)

#*** description of plasma reactor
#---------------------------------------------------------------------------------------------------
width=11.0     #space between two dielectric in mm
ngrid0=1000     #Number of grid points (between two dielectric)
wd1=3.0        #width of first dielectric in mm
wd2=3.0        #width of second dielectric in mm
volt=1000.0    #Interelectrode voltage (peak not RMS)
freq=50        #frequency of the AC voltage (Hz)
cyc=3          #Total number of AC cycles for which simulation is done. Sumulation_Time=(1/freq)*cyc
gasdens=2.5*10**25          #number density of gas at NTP (unit: m^-3)
dx=width*10**(-2)/(ngrid0+1)#Grid size in meter
nwd1=wd1*10**(-2)//dx       #number of grid points in first dielectric
nwd2=wd2*10**(-2)//dx       #Number of grid points in second dielectric
wd1=nwd1*dx                 #Making wd1 as exact multiple of dx
wd2=nwd2*dx                 #making wd2 as exact multiple of dx
inelec=width*10**(-2)+wd1+wd2#total interelectrode separation
ngrid=ngrid0+2+nwd1+nwd2    #total number of grid points(2 dielectrics +gas medium + all edge points)

#*** Initialization
#-----------------------------------------------------------------------------------------------------
ns=14          #Number of species
ndensity=np.zeros((ns,ngrid),float) #Density of each species in all grid points between dielectric
ncharge=np.array[1,1,1,0,0,1,1,1,-1,1-,1,-2,1,1]  #Charge of the each species
netcharge=np.zeros((1,ngrid),float) #net charge at all grid points used to solve poission equation
potentl=np.zeros((1,ngrid),float) #potential at each grid points
efield=np.zeros((1,ngrid0),float) #electric field at each grid points
sig_e_left=0   #Electron surface charge density at left dielectric
sig_e_right=0  #Electron surface charge density at right dielectric
sig_i_left=0   #Ion surface charge density at left dielectric      
sig_i_right=0  #Ion surface charge density at right dielectric

#*** 'Bolsig+' Input/output file description
#----------------------------------------------------------------------------------------------------
inpfile='input.txt'
oupfile='output.txt'
diffusion=np.zeros((1,ngrid0),float)#mobility at grid points, 1 row=1 tyepe of gaseous species
mobility=np.zeros((1,ngrid0),float)#diffusion coefficient at each grid points
source=np.zeros((ns,ngrid0),float)#reaction rate at each grid points


#some calculations done before starting the time loop to reduce recurrent calculations inside the loop
#-----------------------------------------------------------------------------------------------------
posdielec1=nwd1 #array index of left dielectric surface
posdielec2=ngrid-nwd2-1 #array index of right dielectric surface  
posrtelectrode=ngrid-1 #array index of right electrode
sizeouu=ngrid-2 #grid points excluding electrode surfaces
ngM1=ngrid-1 #ngrid minus 1
ngM2=ngrid-2 # ''
ngM3=ngrid-2 # ''
dxdxB2=dx*dx/2 # ''
nwd1P2=nwd1+2 #used while calculating EField
nwd1P2Pk=nwd1+2+ngrid0 #used while calculating EField
nwd1Png0=nwd1+ngrid0 #used while calculating EField


#=======================Time Loop======================================================================
for time in np.arange(#let me decide later):
	#poission equation
	#--------------------------------------------------------------------------------------------------
	alpha=0.5 #relaxation parameter for SOR algorithm
	toll= 10**(-8) #tolerence
	netcharge[:]=0 #clear the garbage value from pervious loop
	for i=np.arange(ns):
		netcharge[0,:]+=ee*ncharge(i)*ndensity(i,:)  #calculating the net charge at each grid points
	#end for
	#boundary conditions at dielectric surfaces
	netcharge[0,posdielec1]=0   	#might be something in terms of sigma (but sigma is related to area not volume)
	netcharge[0,posdielec2]=0	#might be something in terms of sigma (but sigma is related to area not volume)	
	#boundary condition for potential
	potentl[:,:]=0 # delete the garbage value of potential, if exists
	potentl[0,0]=sin(#omega t or something)# Potential at left electrode increases sinusoidally
	potentl[0,posrtelectrode]=0 #potential at right electrode
	# SOR(successive over relaxation method) Poission equation
	uu=np.zeros((1,sizeouu),float)
	flagg=0
	while (flag==0):
		uu=(potentl[0:ngM3]+potentl[2:ngM1])/2-(netcharge[1:ngM2]/e0)*dxdxB2
		if ((sum(uu[uu<=tol]))<ngrid/40): #to skip convergence at some grid points and make iteration a bit faster
			flagg=1
		potentl[1:ngM2]=uu+alpha*(uu-potentl[1:ngM2]
	#end while
	#**calculate electric field as negative gradient of potential
	efield[:,:]=(potentl[nwd1P2:nwd1P2Pk]-potentl[nwd1:nwd1Png0])/(-2*dx)


	#Obtain reaction and transport cofficients from solver software named  BOLSIG+
	#--------------------------------------------------------------------------------------------------	
	#create input file
	froof=open("bolsigplus032016-linux/roof.txt")
	ffloor=open("bolsigplus032016-linux/floor.txt")
	with open("bolsigplus032016-linux/input.txt", "w") as f1:
		for line in froof:
			f1.write(line)
		froof.close()
		np.savetxt(f1, efield)
		for line in ffloor:
			f1.write(line)
		ffloor.close()
	#execute bolsig+ software
	os.system("./bolsigplus032016-linux/bolsigminus bolsigplus032016-linux/input.txt")
	#import necessary parameters from bolsig output file
	mobind=np.array([(31+3*3+2*ngrid0),(31+3*3+2*ngrid0)+ngrid0+1])
	difind=np.array([(1*2+(1-1)*ngrid0),(1*3+(1-1)*ngrid0+ngrid0+1)])
	with open('bolsigplus032016-linux/output.txt') as lines:
		mobility[0,:]= np.transpose(np.genfromtxt(islice(lines,int(mobind[0]),int(mobind[1])))[:,1])
		diffusion[0,:]=np.transpose(np.genfromtxt(islice(lines, difind[0],difind[1]))[:,1])
		sourcee[0,:]=np.transpose(np.genfromtxt(islice(lines,13*3+2+13*ngrid0,14*3+13*ngrid0+1001))[:,1])
		for indd in np.arange(ns-1):
			sourcee[indd+1,:]=np.transpose(np.genfromtxt(islice(lines,2,1004))[:,0])

	#continuity equation
	#---------------------------------------------------------------------------------------------------

	#charge accumulation at surface of dielectric
	#---------------------------------------------------------------------------------------------------
#end forr

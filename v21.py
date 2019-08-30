# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:55:42 2019

@author: Administrator
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import gridspec


alllines = []
path = "TT"
filelst = os.listdir(path)
for file in filelst:
   f = open(os.path.join(path + "/"+file),"r")
   lines = f.readlines()
   data = [i.replace("\n","") for i in lines]
   data = [i.split(",") for i in data]
   data_x = [i[0] for i in data]
   data_y = [i[1] for i in data]
   data_x = np.array(data_x,dtype = "float64")*1000
   data_y = np.array(data_y,dtype = "float64")*1000
   data = [file]+ [data_x] + [data_y]
   alllines.append(data)
   f.close()

np.save('allines',alllines)

# =============================================================================
# 
# =============================================================================

def linear_fit(x,m,b):
    return m*x+b

# =============================================================================
# 
# =============================================================================
#alllines = np.load('allines.npy')
c_index = []
for i in range(len(alllines)):
    if alllines[i][0] == 'dark signal.txt':
        d = i
    elif alllines[i][0] == 'background.txt':
        b = i
    elif alllines[i][0][0] == 'c':
        if alllines[i][0][1] == '0':
            c_0 = i
        c_index.append(i)

fig,(ax,ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios":[1, 5]})
ax2.plot(alllines[b][1],alllines[b][2],lw = 0.5, label = "Background signal")
ax.plot(alllines[d][1],alllines[d][2],".g",markersize = 0.5,fillstyle = 'full',label ="Dark signal")
ax.set_xlim(min(alllines[d][1]),max(alllines[d][1]))
ax2.set_ylim(min(alllines[b][2])-5,-150+10)  
ax.set_ylim(0.2, .65)  
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  
ax2.xaxis.tick_bottom()
ax2.set_xlabel("Time $t$/ms")
dd = .01  
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-dd, +dd), (-dd*5, +dd*5),lw = 0.5, **kwargs)        
ax.plot((1 - dd, 1 + dd), (-dd*5, +dd*5),lw = 0.5, **kwargs)  
kwargs.update(transform=ax2.transAxes)  
ax2.plot((-dd, +dd), (1 - dd, 1 + dd),lw = 0.5, **kwargs)  
ax2.plot((1 - dd, 1 + dd), (1 - dd, 1 + dd), lw = 0.5, **kwargs)  
plt.subplots_adjust(hspace = 0.06)
hand, leg = ax.get_legend_handles_labels() 
hand2, leg2 = ax2.get_legend_handles_labels()
plt.xlabel("Time $t$/ms")
plt.ylabel("Voltage $U$/mV")
plt.legend(hand+hand2, leg+leg2,loc = "best")
plt.savefig('Background',dpi = 300)
#plt.show()

# =============================================================================
# 
# =============================================================================
#allines = np.load('allines')
dark_signal = np.average(alllines[d][2])
dark_signal_err = np.std(alllines[d][2])/np.sqrt(len(alllines[d][2]))
background = np.average(alllines[b][2][0:200])
background_err = np.std(alllines[b][2][0:200])/np.sqrt(200)

data_m = []
for i in c_index:
    data_m.append(alllines[i])

data_m.sort()

data = []
plt.figure()
for i in data_m:
    lab = i[0][1:-8]
    y = i[2] - dark_signal 
    err = np.std(i[2][0:200])/np.sqrt(200)
    y_err = np.sqrt(err**2 + dark_signal_err**2)
    plt.plot(i[1],y,"-",lw = 0.5, label = lab + " mbar")
    data.append([lab,i[1],y,y_err])
plt.xlabel("Time $t$/ms")
plt.ylabel("Voltage $U$/mV")
plt.xlim(min(i[1]),max(i[1]))
plt.legend(ncol = 3,loc = 0)
plt.savefig('Measurements',dpi = 300)
#plt.show()

# =============================================================================
# 
# =============================================================================
d_cell = .1
P = 50.5 
c_I2 = P/8.314/(27.4+273.15)
c_I2_err = P/8.314/(27.4+273.15)**2*0.5
I_0 = np.average(alllines[c_0][2][0:100])-dark_signal
I_0_err = np.std(alllines[c_0][2][0:100])/np.sqrt(100)
print(I_0)
epsilon_I2 = -np.log(I_0/(background-dark_signal))/d_cell/c_I2
epsilon_I2_err = np.sqrt((np.log(I_0/(background-dark_signal))*c_I2_err/c_I2**2/d_cell/np.log(10))**2
                         +(np.log(-I_0)/(background-dark_signal)/c_I2/d_cell/np.log(10))**2
                         +(np.log(-background+dark_signal)/I_0/c_I2/d_cell/np.log(10))**2)
print(epsilon_I2)

def c_I2(x,x0):
    return 2/epsilon_I2/d_cell*np.log(x/x0)


#plt1,ax1 = plt.subplots()
#plt2,ax2 = plt.subplots()
#plt3,ax3 = plt.subplots()
fit = []
for i in range(len(data)):
    fig,ax = plt.subplots()
    indices = [n for n, x in enumerate(data[i][1]) if x == 0.004999949]
    start = indices[0]
    color = ['k','b','c']   
    U_0 = np.average(data[i][2][:indices[0]-10])
    print(data[i][0])
    print(U_0)
    
#    for n in range(start+1,len(data[i][2])):
#        if data[i][2][n] > U_0-1:
#            end = n
#            break 
    y = 1/c_I2(data[i][2][start:start+500],U_0)
    x = np.array(data[i][1][start:start+500])
#    if i%3 == 0:
#        fig = ax1
#    if i%3 == 1:
#        fig = ax2
#    if i%3 == 2:
#        fig = ax3
    plt.plot(x,y,".",ms = 1,fillstyle = 'full',mec = color[i//3],mfc =color[i//3])
    popt,pcov = curve_fit(linear_fit,x[100:175],y[100:175])
    perr = np.sqrt(np.diag(pcov))
    plt.plot(np.arange(0,11), linear_fit(np.array(np.arange(0,11)),*popt) ,ls = '--', lw = 1,color = color[i//3])
    plt.plot(-50,-50,'.--',color = color[i//3], label = str(data[i][0]) + " mbar")
    fit.append([data[i][0],popt,perr])
    plt.xlabel("Time $t$/ms")
    plt.ylabel("Concentration 1/[I]/ m$^3$ mol$^{-1}$")
    plt.legend()
    plt.xlim(0,max(x)*1.1)
    plt.ylim(0,max(y)*1.1)
    ax.ticklabel_format(axis='y', style='sci',scilimits=(-3,3))
    plt.savefig(str(i),dpi = 300)
    plt.show()
    

#ax1.set_xlabel("Time $t$/ms")
#ax1.set_ylabel("Concentration 1/[I]/ m$^3$ mol$^{-1}$")
#ax1.ticklabel_format(axis='y', style='sci',scilimits=(-3,3))
#ax1.set_xlim(0,10)
##ax1.set_ylim(0,200)
#ax1.legend()
#ax2.set_xlabel("Time $t$/ms")
#ax2.set_ylabel("Concentration 1/[I]/ m$^3$ mol$^{-1}$")
#ax2.ticklabel_format(axis='y', style='sci',scilimits=(-3,3))
#ax2.set_xlim(0,10)
##ax2.set_ylim(0,200)
#ax2.legend()
#ax3.set_xlabel("Time $t$/ms")
#ax3.set_ylabel("Concentration 1/[I]/ m$^3$ mol$^{-1}$")
#ax3.ticklabel_format(axis='y', style='sci',scilimits=(-3,3))
#ax3.set_xlim(0,10)
#ax3.set_ylim(0,200)
#ax3.legend()
#plt1.savefig('1',dpi = 300)
#plt2.savefig('2',dpi = 300)
#plt3.savefig('3',dpi = 300)
#plt.show()

# =============================================================================
# 
# =============================================================================


f = open('fit.txt','w')
f.write('epsilon_I2\n')
f.write(str(epsilon_I2)+'\n')
f.write('dark\n')
f.write(str(dark_signal)+'\n')
f.write(str(dark_signal_err)+'\n')
f.write('background\n')
f.write(str(background)+'\n')
f.write(str(background_err)+'\n')

for i in fit:
    f.write(str(i)+'\n')

T = [32,33,33,34.5,34.5,35,35,35,33]
plt.figure()
x = [float(i[0]) for i in fit ]
x = np.array(x)
x = x*100/8.314/(np.array(T)+273.15)
x_err = np.sqrt((1*100/8.314/(np.array(T)+273.15))**2+(x*100*0.5/8.314/(np.array(T)+273.15)**2)**2)
y = [float(i[1][0])*1000 for i in fit]
print(y)
y_err = [float(i[2][0])*1000 for i in fit]

plt.errorbar(x,y,yerr = y_err, fmt='x',ecolor = 'r',label = '$k_{tot}$')
popt,pcov = curve_fit(linear_fit,x[:-2],y[:-2])
perr = np.sqrt(np.diag(pcov))
plt.plot(np.arange(0,21), linear_fit(np.array(np.arange(0,21)),*popt) ,ls = '-', lw = 1,label = 'Linear fit')
plt.xlim(0,20)
#plt.ylim(-100,5000)
plt.xlabel("Concentration [Ar]/mol m$^{-3}$")
plt.ylabel("Total rate constant $k_{tot}$/mol m$^{-3}$ s$^{-1}$")
plt.ticklabel_format(axis='y', style='sci' ,scilimits=(-3,3))
plt.legend()
plt.savefig('10',dpi = 300)
#plt.show()

f.write(str(x)+'\n')
f.write(str(x_err)+'\n')
f.write(str(popt)+'\n')
f.write(str(perr)+'\n')
f.close()

k_I2 = fit[0][1][0]/P*8.314*(27.4+273.15)*1e3
ratio = k_I2/popt[0]
k_I2_err = ((fit[0][2][0]/P*8.314*(27.4+273.15))**2+
                      (fit[0][1][0]*c_I2_err/fit[0][1][0]/P**2*8.314**2*(27.4+273.15)**2)**2)
ratio_err = np.sqrt((((k_I2_err/popt[0]*1e3))/popt[0]**2)+(k_I2*perr[0]/popt[0]**2)**2)

















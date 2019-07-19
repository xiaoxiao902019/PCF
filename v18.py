# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:25:47 2019

@author: xiaox
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def fileread(x):
    filename = 'E:\\PCF\V18\\'+ x + '.asc'
    f = open(filename, 'r')   
    lines = f.readlines()
    f.close()
    data_1 = [i for i in lines]
    data_1 = [i.replace('\n','') for i in data_1]
    data_1 = [i.split(',') for i in data_1]
    xaxis = []
    yaxis = []
    for n in range(len(data_1)):
        try:
            x = int(data_1[n][0])
            y = int(data_1[n][1])
            xaxis.append(x)
            yaxis.append(y)  
        except ValueError:
            pass
    return xaxis,yaxis

def pickpeak(x,intensity,j):
    peak = []
    peak_int = []
    for n in range(5,len(x)-5):
        if yaxis[n] > yaxis[n-1] and yaxis[n] > yaxis[n+1] and yaxis[n] > intensity and yaxis[n] - yaxis[n-5] > j and yaxis[n] - yaxis[n+5] > j:
            peak.append(x[n])
            peak_int.append(yaxis[n])
    return peak,peak_int,yaxis[0]

def linear_fit(x,m,b):
    return m*x+b

def curv_fit(x,a,b,c):
    return a + b*(x+1/2) + c*(x+1/2)**2

#calibration
plt.figure(1)
peaks = []
n = 0
for x in ['calib1-20_HeNe','calib1-20_Hg','calib1-20_lamp']:
    xaxis = fileread(x)[0]
    yaxis = fileread(x)[1]
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    peak = pickpeak(xaxis,300,5)
    plt.plot(xaxis, np.array(yaxis)/max(peak[1])*100+700+20*n-peak[2], lw=0.5, 
             label = str(x))
    n = n+ 1
#    peaks.extend(peak[0])

    
y_cali = [632.8,546.07, 576.96, 579.07, 541.756, 546.0735, 576.9598, 579.0663, 
          580.3, 587.4, 593.2,599.5,611.3,631.1, 650.4]
x_cali = [740, 160, 359, 373, 132, 160, 359, 373, 382, 429, 467, 511, 591, 729, 868]
popt, pcov = curve_fit(curv_fit,x_cali,y_cali)
aa = popt[0]
bb = popt[1]
cc = popt[2]
perr = np.sqrt(np.diag(pcov))
print(perr)
plt.errorbar(x_cali, y_cali, yerr = perr[0], fmt='.', markersize = 1, ecolor='r',
            elinewidth =1, label = 'Signal')
plt.plot(range(1000), curv_fit(np.array(range(1000)),aa,bb,cc), '-k', lw = 0.5, label = 'Fitting')
plt.xlim(0,1000)
plt.xlabel('Pixel')
plt.ylabel('Wavelength/nm')
plt.legend()
plt.savefig('cali.pdf', format = 'PDF')
plt.show()
#
#plot background
plt.figure(2)
for x in ['20-bg','15-bg','5-bg','0-bg','-5-bg','-10-bg','-15-bg',
          '-20-bg']:
    xaxis = fileread(x)[0]
    yaxis = fileread(x)[1]
    xaxis = aa + bb*(np.array(xaxis)+0.5)+cc*(np.array(xaxis))**2
    plt.plot(xaxis, yaxis, lw=0.5, label = str(x))
    leg = plt.legend(loc='best', ncol=4, mode="expand",  fancybox=True)
#    leg.get_frame().set_alpha(1)
plt.xlim(525,675)
plt.ylim(150,1000)
plt.xlabel('Wavelength/nm')
plt.ylabel('Counts')
plt.savefig('BG.pdf', format = 'PDF')
plt.show()

#plot spectrum
yaxis = 0
for x in range(3,4):
    spe_peak = []
    plt.figure(4+x)
    x = 'iod-' + str(200000+x)
    xaxis = fileread(x)[0]
    yaxis_1 = fileread(x)[1]
    xaxis = aa + bb*(np.array(xaxis)+0.5)+cc*(np.array(xaxis))**2
    yaxis = np.array(yaxis_1) + yaxis
    plt.plot(xaxis, yaxis, lw=0.5, label = str(x))
#    plt.ylim(150,600)
    plt.xlim(522,675)
    plt.xlabel('Wavelength/nm')
    plt.ylabel('Counts')
    spe_peak.extend(pickpeak(xaxis,150,5)[0])
    f = open('E:\\PCF\V18\\peaks.txt', 'a') 
    f.write(str(x) + '%d\r\n:' + str(spe_peak) + '\n')
    f.close()
    plt.savefig(str(x))
#    plt.show()


#Gv = [533.9121151288845, 539.3976424661158, 545.1796847945487, 551.2582421141834,
#      557.3367994338181, 563.5636142490536, 570.0869440554908, 576.6102738619279, 
#      583.4301186595668, 590.2499634572057, 604.7791980260886, 619.9014625773748, 
#      627.9073673398204, 636.061529597867, 644.5122068471151, 653.1111415919642, 
#      661.858333832414, 671.0502985596664]
Gv = [532.1450959821694, 538.048400974759, 544.2282200388976, 550.6778294647503, 
      557.0791373682455, 563.5864917146756, 570.3493557228728, 577.0565910890209, 
      584.0092218183814, 590.9010518212049,  605.3809004631152,   620.1586785476524, 
      627.8611813447991, 635.6201935090066, 643.5696332718401, 651.5627096581195, 
       659.5944229861733, 667.9267480233785]
#Gv = [533.9,  539.4,  547.1, 551.3,  
#      557.3, 563.6,  570.0,  576.6,  
#      583.4,  590.3,   604.8, 619.9,  
#      627.9,   636.1,  644.5, 653.1,   
#      662.0,  671.1]

#Gv = spe_peak
#Gv.remove(Gv[20])
#Gv.remove(Gv[13])
#Gv.remove(Gv[12])
Gv.remove(Gv[0])
Gv = 10**7/1.00028/np.array(Gv)
G_diff = []
v = list(range(20))
v.remove(v[12])
v.remove(v[10])
v.remove(v[0])
v = np.array(v)+0.5
for n in range(18-2):
    G_diff.append((Gv[n]-Gv[n+1])/(v[n+1]-v[n]))


plt.figure(3)
popt, pcov = curve_fit(curv_fit,v,Gv)
a = popt[0]
b = popt[1]
c = popt[2]
perr = np.sqrt(np.diag(pcov))
yerr = v**2*perr[1]+v*perr[2]+perr[0]
plt.errorbar(v, Gv, yerr = perr[0], fmt='.', markersize = 5, ecolor='r', 
             elinewidth =1, label = 'Signal')
plt.plot(range(30), curv_fit(np.array(range(30)),a,b,c), '-k', lw = 0.5, label = 'Fitting' )
plt.xlim(0,30)
plt.xlabel('Energy Level/ v + '+r'$\frac{1}{2}$')
plt.ylabel('Energy/$cm^{-1}$')
plt.savefig('poly.pdf', format = 'PDF')
plt.legend()
print(popt,perr)
plt.show()

plt.figure(4)
v = list(range(19))
v.remove(v[12])
v.remove(v[10])
v.remove(v[0])
v = np.array(v)+0.5
popt, pcov = curve_fit(linear_fit,v,np.array(G_diff))
a = popt[0]
b = popt[1]
perr = np.sqrt(np.diag(pcov))
yerr = perr[0]*v + perr[1]
plt.errorbar(v, G_diff, yerr = yerr, fmt='.', markersize = 5, ecolor='r', 
             elinewidth=1, label = 'Signal')
plt.plot(range(30), linear_fit(np.array(range(30)),a,b), '-k', lw = 0.5, label = 'Fitting')
#plt.xlim(0,30)
#plt.ylim(60,220)
plt.ylabel('Energy Difference/$cm^{-1}$')
plt.xlabel('Energy Level/ v + '+r'$\frac{1}{2}$')
plt.legend()
print(popt,perr)
plt.savefig('bir.pdf', format = 'PDF')
plt.show()


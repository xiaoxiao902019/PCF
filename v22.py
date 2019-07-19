# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np  

R = 8.314

# =============================================================================
# read file
# =============================================================================
filename = '20190705_14.csv'
#input('Filename: ') #typ in the file name in the cmd or terminal
f = open(filename, 'r')
lines = f.readlines()
f.close()

data = [i.split(';') for i in lines]
data = [i for i in data[2:]]

# =============================================================================
# define functions
# =============================================================================
def linear_fit(x,m,b):
    return m*x+b


# =============================================================================
# data sortion
# =============================================================================
ang_10 = [[],[],[],[]]
ang_35 = [[],[],[]]
ang_rt = [[],[],[]]
ang_35_0 = [[],[]]
ang_35_f = [[],[]]
ang_rt_0 = [[],[]]
ang_rt_f = [[],[]]
for i in data[1:]:
    index = i[7].split('-')
    if len(index) == 1:
        offset = float(i[6])
    elif index[2] == '20cm':
        frac = float(index[1][:-1])/10**(len(index[1])-2)
        c = frac*20.535/60
        c_err = frac*np.sqrt(((0.01/60)**2+(20.535*1/60**2)**2)/frac)
        ang_20 = [float(i[6]),float(index[3][:3])/10,c,c_err]
    elif index[2] == '10cm':
        frac = float(index[1][:-1])/10**(len(index[1])-2)
        c = frac*20.535/60
        c_err = frac*np.sqrt(((0.01/60)**2+(20.535*1/60**2)**2)/frac)
        ang_10[0].append(float(i[6]))
        ang_10[1].append(float(index[3][:3])/10)
        ang_10[2].append(c)
        ang_10[3].append(c_err)
    elif index[3] == '35"':
        if index[2] == 'k':
            ang_35[0].append(float(i[6]))
            ang_35[1].append(float(i[5]))
            ang_35[2].append(float(i[10])/2)
        elif index[2] == 't0':
            ang_35_0[0].append(float(i[6]))
            ang_35_0[1].append(float(i[5]))
        elif index[2] == 'tf':
            ang_35_f[0].append(float(i[6]))
            ang_35_f[1].append(float(i[5]))        
    elif index[3] == 'rt"' or 'RT"':
        if index[2] == 'k':
            ang_rt[0].append(float(i[6]))
            ang_rt[1].append(float(i[5]))
            ang_rt[2].append(float(i[10]))
        elif index[2] == 't0':
            ang_rt_0[0].append(float(i[6]))
            ang_rt_0[1].append(float(i[5]))
        elif index[2] == 'tf':
            ang_rt_f[0].append(float(i[6]))
            ang_rt_f[1].append(float(i[5]))   
            
       
# =============================================================================
# 
# =============================================================================
plt.figure('Determination of the specific rotation')
y = list(np.array(ang_10[0])-offset)
y.append((np.array(ang_20[0])-offset)/2)
x = list(ang_10[2])
x.append(ang_20[2])
popt_r, pcov_r = curve_fit(linear_fit,x,y)
perr_r = np.sqrt(np.diag(pcov_r))
plt.plot(np.arange(0,max(x),0.01),linear_fit(np.array(np.arange(0,max(x),0.01)),*popt_r),c = 'k',ls = '--',label = 'Linear fit')
plt.errorbar(ang_10[2],np.array(ang_10[0])-offset,xerr = ang_10[3],fmt ='x',ecolor = 'r',label = 'With 10 cm sample cell')
plt.errorbar(ang_20[2],(np.array(ang_20[0])-offset)/2,xerr = ang_20[3],fmt ='o',ecolor = 'r',label = 'With 20 cm sample cell')
plt.xlabel('Concentration/ g mL$^{-1}}$')
plt.ylabel('Rotation angle/cell length °dm$^{-1}}$')
plt.xlim(0,0.35)
plt.ylim(0,20)
plt.legend()
plt.savefig('Determination of the specific rotation',fmt = 'png',dpi = 300)
plt.show()


# =============================================================================
# 
# =============================================================================
plt.figure('Determination of the rate constant')
ang_35_t0 = np.average(ang_35_0[0])
ang_35_t0_err = np.std(ang_35_0[0])/2
ang_35_tf = np.average(ang_35_f[0])
ang_35_tf_err = np.std(ang_35_f[0])/2
ang_rt_t0 = np.average(ang_rt_0[0])
ang_rt_t0_err = np.std(ang_rt_0[0])/2
ang_rt_tf = np.average(ang_rt_f[0])
ang_rt_tf_err = np.std(ang_rt_f[0])/2

y_35 = np.log((ang_35[0]-ang_35_tf)/(ang_35_t0-ang_35_tf))
y_rt = np.log((ang_rt[0]-ang_rt_tf)/(ang_rt_t0-ang_rt_tf))
popt_35, pcov_35 = curve_fit(linear_fit,ang_35[2],y_35)
perr_35 = np.sqrt(np.diag(pcov_35))
popt_rt, pcov_rt = curve_fit(linear_fit,ang_rt[2],y_rt)
perr_rt = np.sqrt(np.diag(pcov_rt))
plt.plot(np.arange(0,60),linear_fit(np.arange(0,60),*popt_35),ls = '--',label = 'Linear fit for reaction at 35 °C')
plt.plot(np.arange(0,60),linear_fit(np.arange(0,60),*popt_rt),ls = '--',label = 'Linear fit for reaction at 25 °C')
plt.plot(ang_35[2],y_35,'x', markersize = 3,label = 'Reaction at 35 °C')
plt.plot(ang_rt[2],y_rt,'o', markersize = 3,label = 'Reaction at 25 °C')
plt.legend()
plt.xlim(0,60)
plt.xlabel('Time/ minute')
plt.ylabel('ln($c_S/c_{S0}$)')
plt.savefig('Determination of the rate constant',fmt = 'png',dpi = 300)
plt.show()

# =============================================================================
# 
# =============================================================================
E_A = R*np.log(popt_35[0]/popt_rt[0])*(35+273.15)*(25+273.15)/(35-25)
E_A_err= R*np.sqrt((np.log(-popt_35[0])*perr_rt[0]/popt_rt[0]*(35+273.15)*(25+273.15)/(35-25))**2
                   +(np.log(-popt_rt[0])*perr_35[0]/popt_35[0]*(35+273.15)*(25+273.15)/(35-25))**2
                   +(np.log(popt_35[0]/popt_rt[0])*(35+273.15)**2/(35-25)**2*np.std(ang_35[1]))**2/59
                   +(np.log(popt_35[0]/popt_rt[0])*(25+273.15)**2/(35-25)**2*np.std(ang_rt[1]))**2/59)

# =============================================================================
# 
# =============================================================================
fig,(ax1,ax2) = plt.subplots(nrows = 2,ncols = 1)
ax1.plot(np.arange(1,31,0.5),ang_35[1])
ax2.plot(np.arange(1,61),ang_rt[1])
ax1.set_xlim(1,30)
ax2.set_xlabel('Time/ minute')
fig.text(0.02, 0.5, 'Temperature/ °C', va='center', rotation='vertical')
plt.xlim(1,60)
plt.savefig('Temperature changes',fmt = 'png',dpi = 300)
plt.show()
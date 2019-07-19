# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:11:35 2019

@author: xiaox
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

h = 6.626e-34
c = 3e8
rho1 = 1.288
rho2 = 0.779
M1 = 147
M2 = 84.16
k_B = 1.38e-23
N_A = 6.022e23
epsilon_0 = 8.85e-12


def linear_fit(x,m,b):
    return m*x+b

filename = 'V10_190515_temperaturedependence_DCB1.csv'
f = open(filename, 'r')   
lines = f.readlines()
f.close()
yaxis = []
for n in range(len(lines)):
    y = float(lines[n])
    yaxis.append(y)

    
def temperature(x):
    t = 273.15 + 21 + x//600
    return t

fig,ax1 = plt.subplots()
x = np.array(np.arange(n+1))
ax1.set_xlabel('Time/s')
plot1 = ax1.plot(x,np.array(yaxis)*1e9,'.b',markersize = 0.5, label = 'The period time record')
ax1.set_ylabel('Period Time $t$/ns')
#ax2 = ax1.twinx()
#ax2.set_ylabel('Temperamture/K')
#plot2 = ax2.plot(x[:-10],temperature(x[:-10]),'-r',label = 'The programmed temperature rising')
#fig.tight_layout()
plt.xlim(0,3320)
plt.axvline(x = x[450],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[700],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[1000],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[1300],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[1700],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[2000],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[2300],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[2600],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[2900],c ='r',lw = 0.5,ls ='--')
plt.axvline(x = x[3200],c ='r',lw = 0.5,ls ='--')
t21 = np.average(yaxis[450:700])*1e9
del_t21 = np.std(yaxis[450:700])*1e9
t22 = np.average(yaxis[1000:1300])*1e9
del_t22 = np.std(yaxis[1000:1300])*1e9
t23 = np.average(yaxis[1700:2000])*1e9
del_t23 = np.std(yaxis[1700:2000])*1e9
t24 = np.average(yaxis[2300:2600])*1e9
del_t24 = np.std(yaxis[2300:2600])*1e9
t25 = np.average(yaxis[2900:3200])*1e9
del_t25 = np.std(yaxis[2900:3200])*1e9
plt.text((x[450]+x[700])/2-100,t21+.05,'21 °C')
plt.text((x[1000]+x[1300])/2-100,t22+.05,'22 °C')
plt.text((x[1700]+x[2000])/2-100,t23+.05,'23 °C')
plt.text((x[2300]+x[2600])/2-100,t24+.05,'24 °C')
plt.text((x[2900]+x[3200])/2-100,t25+.05,'25 °C')
plot = plot1 
labs = [l.get_label() for l in plot]
ax1.legend(plot,labs,loc = 'best')
plt.savefig('Temperature vs period',dpi = 600)
#plt.show()

t21 = np.average(yaxis[450:700])*1e9
del_t21 = np.std(yaxis[450:700])*1e9
t22 = np.average(yaxis[1000:1300])*1e9
del_t22 = np.std(yaxis[1000:1300])*1e9
t23 = np.average(yaxis[1700:2000])*1e9
del_t23 = np.std(yaxis[1700:2000])*1e9
t24 = np.average(yaxis[2300:2600])*1e9
del_t24 = np.std(yaxis[2300:2600])*1e9
t25 = np.average(yaxis[2900:3200])*1e9
del_t25 = np.std(yaxis[2900:3200])*1e9


n_cali =[2.0243,1.00054,2.2735]
T_cali =[579.1566667,511.1666667,595.5833333]
T_cali_err = [0.4,0.006,0.6]

def linear_err(x,err,m,b,m_err,b_err):    
    return np.sqrt((x*m_err)**2+(err*m)**2+b_err**2)

plt.figure('Calibration')
plt.errorbar(T_cali,n_cali,xerr = T_cali_err,fmt='x', markersize = 3,  ecolor = 'r',
                elinewidth =1, label = 'Assigned Values with Error')
popt_cali,pcov_cali = curve_fit(linear_fit,T_cali,n_cali)
perr_cali = np.sqrt(np.diag(pcov_cali))
plt.plot(range(500,600),linear_fit(np.array(range(500,600)),*popt_cali),lw =0.5,label = 'Linear Fitting')
plt.legend()
plt.xlabel('Period Time $t$/ns')
plt.ylabel('Relative permittivity $ϵ_r$')
plt.savefig('Calibration',dpi = 600)
plt.show()
print('cali_fit:')
print(popt_cali)
print(perr_cali)

T = [582.653333333333,587.94,594.143333333333,599.02,]
T_err = [1.00,0.80,2.0,0.8]

t= [t21,t22,t23,t24,t25]
t_err = [del_t21,del_t22,del_t23,del_t24,del_t25]
print('Temperatrue dependent data:')
print(t)
print(t_err)

epsilon = linear_fit(np.array(T),*popt_cali)
epsilon = np.insert(epsilon, 0,2.0243)
epsilon_err = linear_err(np.array(T),np.array(T_err),*popt_cali,*perr_cali)
epsilon_err = np.insert(epsilon_err, 0,0)
print('epsilon:')
print(epsilon)
print(epsilon_err)

Mass = [[84.8726,86.1383,117.1198],
[76.6666,79.2501,110.2455],
[62.5500,67.6788,98.6451],
[63.1736,69.6020,100.5417]]

rho = [rho2]
rho_err = [0]
M = [M2]
M_err = [0]
frac = [0]
frac_err = [0]
for i in Mass:
    m1 = i[1]-i[0]
    m2 = i[2]-i[1]
    rho_i = (i[2]-i[0])/(m2/rho2+m1/rho1)
    rho_i_err =rho1*rho2*(rho1-rho2)*0.001*np.sqrt(m1**2+m2**2)/(rho1*m2+rho2*m1)**2
    rho.append(rho_i)
    rho_err.append(rho_i_err)
    M_i = (i[2]-i[0])/(m2/M2+m1/M1)
    M_i_err =M1*m2*(M1-M2)*0.001*np.sqrt(m1**2+m2**2)/(M1*m2+M2*m1)**2
    M.append(M_i)
    M_err.append(M_i_err)
    frac_i = m1/M1/(m2/M2+m1/M1)
    frac_i_err = m1*0.0014/M1/M2/(m2/M2+m1/M1)**2
    frac.append(frac_i)
    frac_err.append(frac_i_err)

epsilon = np.array(epsilon)
epsilon_err = np.array(epsilon_err)
rho = np.array(rho)
rho_err = np.array(rho_err)
M = np.array(M)
M_err = np.array(M_err)
print('rho and Mass:')
print(rho)
print(rho_err)
print(M)
print(M_err)


P_m = (epsilon-1)/(epsilon+2)*M/rho
P_m_err = np.sqrt((3*epsilon_err*M/rho/(epsilon+2)**2)**2 + 
          (M_err*(epsilon-1)/(epsilon+2)/rho)**2 + (rho_err*M*(epsilon-1)/rho**2/(epsilon+2))**2)
print('P_m:')
print(P_m)
print(P_m_err)


plt.figure('Molar polarization')
plt.errorbar(frac,P_m,yerr = P_m_err,xerr = frac_err,fmt ='x',markersize = 3,  ecolor = 'r',
             elinewidth =1, label =' Data point with Error')
popt_P_m,pcov_P_m = curve_fit(linear_fit,frac,P_m)
perr_P_m = np.sqrt(np.diag(pcov_P_m))
plt.plot(np.arange(0,0.125,0.01),linear_fit(np.array(np.arange(0,0.125,0.01)),*popt_P_m),lw =0.5,
         label = 'Linear Fitting ')
plt.legend()
plt.xlabel('Molar fraction $x_1$')
plt.ylabel('Molar polarization $P_m$/ml mol$^{-1}$')
plt.savefig('Molar polarization',dpi = 600)
#plt.show()

print('P_m linear fitting')
print(popt_P_m)
print(perr_P_m)

n = [1.4246,1.4272,1.4301,1.4353,1.4387]
n_err = [0.00026,0.000128,0.00006,0.00023,0.00039]
n = np.array(n)
n_err = np.array(n_err)
R_m = (n**2-1)/(n**2+2)*M/rho
R_m_err = np.sqrt((6*n_err*n**2*M/rho/(n**2+2)**2)**2 + 
          (M_err*(n**2-1)/(n**2+2)/rho)**2 + (rho_err*M*(n**2-1)/rho**2/(n**2+2))**2)

print('R_m:')
print(R_m)
print(R_m_err)

plt.figure('Molar refraction')
plt.errorbar(frac,R_m,yerr = R_m_err,xerr = frac_err,fmt ='x',markersize = 3,  ecolor = 'r',
             elinewidth =1, label =' Data point with Error')
popt_R_m,pcov_R_m = curve_fit(linear_fit,frac,R_m)
perr_R_m = np.sqrt(np.diag(pcov_R_m))
plt.plot(np.arange(0,0.125,0.01),linear_fit(np.array(np.arange(0,0.125,0.01)),*popt_R_m),lw =0.5,
         label = 'Linear Fitting ')
plt.legend()
plt.xlabel('Molar fraction $x_1$')
plt.ylabel('Molar refraction $R_m$/ml mol$^{-1}$')
plt.savefig('Molar refraction',dpi = 600)
#plt.show()

print('R_mlinear fitting')
print(popt_R_m)
print(perr_R_m)

epsilon_t = linear_fit(np.array(t),*popt_cali)
epsilon_t_err = linear_err(np.array(t),np.array(t_err),*popt_cali,*perr_cali)
Tem = [294.15,295.15,296.15,297.15,298.15]
Tem_err = 0.1
P_m_t = (epsilon_t-1)/(epsilon_t+2)*M[1]/rho[1]
P_m_err_t = np.sqrt((3*epsilon_t_err*M[1]/rho[1]/(epsilon_t+2)**2)**2 + 
                  (M_err[1]*(epsilon_t-1)/(epsilon_t+2)/rho[1])**2 + 
                  (rho_err[1]*M[1]*(epsilon_t-1)/rho[1]**2/(epsilon_t+2))**2)
P_m_T = (P_m_t-P_m[0]*(1-frac[1]))/frac[1]
P_m_T_err = np.sqrt((P_m_t-P_m[0])**2/frac[1]**4*frac_err[1]**2
                    + (P_m_err_t**2 + P_m_err[0]**2)/frac[1]**2 
                    +P_m_err[1]**2)
Tem = np.array(Tem)
plt.figure('Molar polarization dependant on Temperature1')
plt.errorbar(1/Tem,P_m_t,yerr = P_m_err_t,xerr = Tem_err/Tem**2 ,fmt ='x',markersize = 3,  ecolor = 'r',
             elinewidth =1, label =' Data point with Error')
popt_P_m_t,pcov_P_m_t = curve_fit(linear_fit,1/Tem,P_m_t)
perr_P_m_t = np.sqrt(np.diag(pcov_P_m_t))
plt.plot(np.arange(1/300,1/290,1e-5),linear_fit(np.array(np.arange(1/300,1/290,1e-5)),*popt_P_m_t),lw =0.5,
         label = 'Linear Fitting ')
plt.legend()
plt.xlabel('Temperature $1/T$ /K$^{-1}$')
plt.ylabel('Molar polarization $P_m$/ml mol$^{-1}$')
plt.savefig('Molar polarization dependant on Temperature1',dpi = 600)
#plt.show()

print('Temperature dependent data:')
print(epsilon_t)
print(epsilon_t_err)
print(Tem)
print(Tem_err)
print(P_m_t)
print(P_m_err_t)
print(popt_P_m_t)
print(perr_P_m_t)

plt.figure('Molar polarization dependant on Temperature2')
plt.errorbar(1/Tem,P_m_T,yerr = P_m_T_err,xerr = Tem_err/Tem**2 ,fmt ='x',markersize = 3,  ecolor = 'r',
             elinewidth =1, label =' Data point with Error')
popt_P_m_T,pcov_P_m_T = curve_fit(linear_fit,1/Tem,P_m_T)
perr_P_m_T = np.sqrt(np.diag(pcov_P_m_T))
plt.plot(np.arange(1/300,1/290,1e-5),linear_fit(np.array(np.arange(1/300,1/290,1e-5)),*popt_P_m_T),lw =0.5,
         label = 'Linear Fitting ')
plt.legend()
plt.xlabel('1/Temperature $1/T$ /K$^{-1}$')
plt.ylabel('Molar polarization $P_m$/ml mol$^{-1}$')
plt.savefig('Molar polarization dependant on Temperature2',dpi = 600)
#plt.show()

print('Temperature dependant fitting')
print(popt_P_m_T)
print(perr_P_m_T)

P = sum(popt_P_m)
P_err = np.sqrt(sum(perr_P_m**2))
R = sum(popt_R_m)
R_err = np.sqrt(sum(perr_R_m**2))

def dipl(P,R,P_err,R_err):
    a = 0.0128*np.sqrt((P - 1.1*R)*Tem[0])
    b = 0.0128*np.sqrt(Tem[0]/4/(P - 1.1*R)*P_err**2 
                       + Tem[0]/4/(P - 1.1*R)*R_err**2
                       + (P - 1.1*R)/4/Tem[0]*Tem_err**2)    
    return a,b

print('P,R,Diploe moment: ')
print(P,P_err,R,R_err)
print(dipl(P,R,P_err,R_err))


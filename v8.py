#!/bin/ python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

V_flask = 1280 #cm³
p = 800 #Torr
delta_p = 2 #Torr
r_B = 1.6 #cm
h = 15 #cm butyl phthalate
delta_h = 2 #cm
R = 8.314

filename = 'v8.csv'
#input('Filename: ') #typ in the file name in the cmd or terminal
f = open(filename, 'r')
lines = f.readlines()
f.close()
data = [i.replace('\t',',') for i in lines]
data = [i.replace('\n','') for i in data]
data = [i.replace(' ','') for i in data]
data = [i.split(' ') for i in data]
data = [i[0].split(',') for i in data]
data = [i for i in data if i != ['']]
data_info = data[0]
data = data[1:]
data = [list(map(float, i)) for i in data]

v_flow_N2 = 0.62*h - 6.4e-3*h**2 + 4.9e-5*h**3
delta_v_flow_N2 = 0.62 - 2*6.4e-3*delta_h + 3*4.9e-5*delta_h**2
T = float(data_info[3]) + 273.15

 
def v_N2(x):
    a = 1/np.pi/r_B**2
    b = p*v_flow_N2/(x*0.750062)
    c = np.sqrt(delta_v_flow_N2**2*p**2/x**2+v_flow_N2**2*p**2*(0.1*0.750062)**2/x**4+1*v_flow_N2**2/(x*0.750062)**2)
    return a*b,a*c

def linear_fit(x,m,b):
    return m*x+b

plt.figure(figsize= [16,9])
n = 1
k_tot = []
k_tot_err = []
I_0 = []
I_0_err = []
t = []
t_err = []
for i in data:
    i = np.array(i)
    color1 = np.random.random_sample(3,)
    color2 = np.random.random_sample(3,)
    x_forward = list(range(3,33,3))
    x_forward = np.array(x_forward)
    x_forward = x_forward/v_N2(i[1])[0]
    x_forward_err = x_forward/v_N2(i[1])[0]**2*v_N2(i[1])[1]  
    y_1 = 1/np.sqrt(i[2:12]/50)
    y_1_err = 1/2/(i[2:12]/50)**1.5*1/50
    y_2 = 1/np.sqrt(i[11:-1]/50)
    y_2_err = 1/2/(i[11:-1]/50)**1.5*1/50
    plt1 = plt.errorbar(x_forward,y_1,xerr = x_forward_err,yerr = y_1_err,fmt = 'x',markersize = 5,mec = 'k', markerfacecolor ='none', ecolor = 'r',elinewidth =1)
    popt_f, pcov_f = curve_fit(linear_fit,x_forward,y_1)
    perr_f = np.sqrt(np.diag(pcov_f))
    plt2 = plt.errorbar(x_forward[::-1],y_2,xerr = x_forward_err[::-1],yerr = y_2_err,fmt = 'o',markersize = 5,markerfacecolor ='none', mec = 'k',  ecolor = 'r')
    popt_b, pcov_b = curve_fit(linear_fit,x_forward[::-1],y_2)
    perr_b = np.sqrt(np.diag(pcov_b))
    plt3 = plt.plot(np.arange(0,0.15,0.02), linear_fit(np.array(np.arange(0,0.15,0.02)),*popt_f) ,ls = '--',c = color2, lw = 1)
    plt.plot(np.arange(0,0.15,0.02), linear_fit(np.array(np.arange(0,0.15,0.02)),*popt_b) ,ls = '-',c =color2, lw = 1,label='Measurement '+str(n))
    k_tot.append([popt_f[0],popt_b[0]])
    k_tot_err.append([perr_f[0],perr_b[0]])
    I_0.append([popt_f[1],popt_b[1]])
    I_0_err.append([perr_f[1],perr_b[1]])
    t.append(x_forward)
    t_err.append(x_forward_err)
    n = n + 1
plt.xlim(0,0.14)
plt.ylim(0.4,1.1)
plt.xlabel('Time $t$/s')
plt.ylabel('1/$\sqrt {U}$/ Arbitrary units')
plt.legend(loc = 0,ncol=5, mode="expand",  fancybox=True)
plt.savefig('V8_1',fmt ='png',dpi = 600)
plt.show()


def v_NO(x):
    a = V_flask*2*0.750062/800/x
    b = V_flask/800*np.sqrt((0.1*0.750062/x)**2+(2*0.750062*0.250/x**2)**2)
    return a,b

k_tot_C = []
k_tot_C_err = []
C = []
C_err = []
C_M = []
C_M_err = 0.1*100e-6/R/T
for n in range(len(k_tot)):
    v_N_1 = v_NO(data[n][0])
    v_N_2 = v_NO(data[n][21])
    k_tot_C.append(k_tot[n][0]/I_0[n][0])
    k_tot_C.append(k_tot[n][1]/I_0[n][1])
    k_tot_C_err.append(np.sqrt((k_tot_err[n][0]/I_0[n][0])**2+(k_tot[n][0]*I_0_err[n][0]/I_0[n][0]**2)**2))
    k_tot_C_err.append(np.sqrt((k_tot_err[n][1]/I_0[n][1])**2+(k_tot[n][1]*I_0_err[n][1]/I_0[n][1]**2)**2))
    C_1 = v_N_1[0]/(v_flow_N2+v_N_1[0])*data[n][1]*100e-6/R/T
    C_2 = v_N_2[0]/(v_flow_N2+v_N_2[0])*data[n][1]*100e-6/R/T
    C_m = data[n][1]*100e-6/R/T
    C.append(C_1)
    C.append(C_2)
    C_M.append(C_m-C_1)
    C_M.append(C_m-C_2)
    C_1_err = 100e-6/R/T*np.sqrt((v_N_1[1]*data[n][1]/v_flow_N2)**2+(v_N_1[0]*delta_v_flow_N2*data[n][1]/v_flow_N2**2)**2+(v_N_2[0]*0.1/v_flow_N2)**2)
    C_2_err = 100e-6/R/T*np.sqrt((v_N_2[1]*data[n][1]/v_flow_N2)**2+(v_N_2[0]*delta_v_flow_N2*data[n][1]/v_flow_N2**2)**2+(v_N_2[0]*0.1/v_flow_N2)**2)
    C_err.append(C_1_err)
    C_err.append(C_2_err)

plt.figure()
x = np.array(C)*np.array(C_m)
y = k_tot_C
x_err = np.sqrt((np.array(C_err)*np.array(C_M))**2+np.array(C)**2*(C_M_err**2+np.array(C_err)**2))
plt.errorbar(x,y,yerr= k_tot_C_err,xerr = x_err,ecolor = 'r',fmt= '.', label = 'Total rate constants dependent\non different concentration')
popt_k, pcov_k = curve_fit(linear_fit,x,np.array(y)/1e18)
perr_k = np.sqrt(np.diag(pcov_k))
plt.plot(np.arange(2e-16,1e-15,1e-16),  linear_fit(np.arange(2e-16,1e-15,1e-16),*popt_k)*1e18 ,ls = '--', c='k', lw = 1,label='Linear fit')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Concentration/ mol cm$^{-3}$')
plt.ylabel('Total rate constant/ s$^{-1}$')
plt.legend()
plt.savefig('V8_2',fmt = 'png',dpi = 300)
plt.show()


f = open('databaken','w+')
f.write('volume flow N2\n')
f.write(str(v_flow_N2))
f.write(str(delta_v_flow_N2))
f.write('slope\n')
for i in k_tot:
    f.write(str(i)+'\n')
for i in k_tot_err:
    f.write(str(i)+'\n')
f.write('intrcept\n')
for i in I_0:
    f.write(str(i)+'\n' )
for i in I_0_err:
    f.write(str(i)+'\n' )
f.write('time\n')
for i in range(len(t)):
    f.write(str(t[i])+'\n' )
    f.write(str(t_err[i])+'\n' )
f.write('concentration\n')
for i in C:
    f.write(str(i)+'\n' )
for i in C_err:
    f.write(str(i)+'\n' )
f.write('concentration_M\n')
for i in C_M:
    f.write(str(i)+'\n' )
f.write('total rate constant\n')
for i in k_tot_C:
    f.write(str(i)+'\n' )
for i in k_tot_C_err:
    f.write(str(i)+'\n')
f.write('fit parameters\n')
f.write(str(popt_k))
f.write(str(perr_k))
f.close()

 


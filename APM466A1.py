#!/usr/bin/env python
# coding: utf-8

# In[408]:


# import necessary packages
import scipy.optimize as optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import math


# write the function to calculate ytm for each day point
""" Get yield-to-maturity of a bond """
def bond_ytm(cur_price, face_val, T, coup_rate, freq=2, guess=0.05):
    freq = float(freq)
    periods = T*freq
    # calculat the coupon for each semi-annual
    coupon = coup_rate*face_val/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + face_val/(1+y/freq)**(freq*T) - cur_price
        
    return optimize.newton(ytm_func, guess) *100


    
    
# load the data of chosen bonds
bonds = pd.read_excel (r'/Users/minhuixu/Desktop/APM466/A1/CHOSEN BOND.xlsx') 

# format the issue date and the maturity date
bonds['issue date'] = pd.to_datetime(bonds['issue date'])
bonds['maturity date'] = pd.to_datetime(bonds['maturity date'])


# format the chosen date from 2020-01-02 to 2020-01-15
cur_time_list = list(bonds.columns)[5:15]
format_cur_time = []
for i in range(0,10):
    format_cur_time.append(pd.to_datetime(cur_time_list[i])) 
    


# calculate the T for each chosen date
T_lst_total = []
for curr in format_cur_time:
    T_lst = []
    for item in bonds["maturity date"]:
        delta = item - curr
        left = delta.days
        T_lst.append(int(left // 180 +1))
    T_lst_total.append(T_lst)
    
    


# prepare other parameters needed in the ytm calculation
coupon = list(bonds["Coupon"])
face_val = 100
num_bonds = len(bonds["maturity date"])

# apply the bond_ytm function
ytm_total =[]
for i in range(len(cur_time_list)):
    ytm_lst = []
    cur_price_lst = bonds[cur_time_list[i]]
    cur_T_list = T_lst_total[i]
    for j in range(num_bonds):
        cur_price = cur_price_lst[j]
        cur_T = cur_T_list[j]
        cur_coupon = coupon[j]
        cur_ytm = bond_ytm(cur_price, face_val, cur_T,cur_coupon)
#         print(cur_ytm)
        ytm_lst.append(cur_ytm)
    ytm_total.append(ytm_lst)

# organize the ytm
ytm_dict = {}
for idx in range(len(cur_time_list)):
    ytm_dict[cur_time_list[idx]]= ytm_total[idx]
ytm_frame = pd.DataFrame(data = ytm_dict)  

print(ytm_frame)


    

if __name__ == "__main__":
    ytm = bond_ytm(95.0428, 100, 1.5, 0.0575, 2)
    print(ytm)


# In[352]:


# plot the ytm
FIVE_YEARS = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

    

    
# Make a data frame    
df=pd.DataFrame({'x': FIVE_YEARS, 
                 'Jan 2': ytm_total[0], 
                 'Jan 3': ytm_total[1], 
                 'Jan 6': ytm_total[2], 
                 'Jan 7': ytm_total[3], 
                 'Jan 8': ytm_total[4], 
                 'Jan 9': ytm_total[5], 
                 'Jan 10': ytm_total[6], 
                 'Jan 13': ytm_total[7], 
                 'Jan 14': ytm_total[8], 
                 'Jan 15': ytm_total[9] })    

plt.figure(figsize=(15,5), dpi= 80)
# style
plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num +=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)
 
# Add titles
plt.title("Yield to Maturity", loc='center', fontsize=20, fontweight=0, color='orange')
plt.xlabel("Maturity")
plt.ylabel("ytm")


# In[430]:


from datetime import datetime
from dateutil.relativedelta import relativedelta
# calcualte the first spot rate 
def cal_spot_0(price, face_Val, T):
    spot = - math.log(price / face_Val)/T
    return spot * 100


spot_rate_0 = []

for i in range(10):
    T = (bonds['maturity date'][0]- cur_time_list[i]).days / 365
    price = bonds[cur_time_list[i]][0]
    face_Val = 100
    spot_rate = cal_spot_0(price, face_Val, T)
    spot_rate_0.append(spot_rate)


# calculate the dirty price
def dirty_price(curr_date, last_coupon_date, coupon, face_val, curr_price):
    n = (curr_date - last_coupon_date).days
    accured_int = n / 365 * coupon * face_val
    dirty_pri = accured_int + curr_price
    return dirty_pri
    
    

# bootstrap for following spot rates
def bootstrap(dirty_price, coupon, face_val, spot_1, t_1, t_2):
    payment_1 = coupon * face_val
    payment_2 = (1 + coupon) * face_val
    in_log = (dirty_price - payment_1 * math.exp(-spot_1 * t_1)) / payment_2
    spot_2 = - math.log(in_log) / t_2    
    return spot_2

# bootstrapping
total_spot_rate = []
mature_list = bonds['maturity date']
coupon_list = bonds['Coupon']
for i in range(10):
    last_coupon_date = mature_list[0] - relativedelta(months = 6)
    cur_time = cur_time_list[i]
    cur_spot = spot_rate_0[i]
    spots = [spot_rate_0[i]]
    face_val = 100
    for j in range(9):
        if j == 0:
            t_1 = (mature_list[0] - cur_time).days /365
            t_2 = (mature_list[1] - cur_time).days /365
            spot_1 = cur_spot
            coupon = coupon_list[1]
            mature_date = mature_list[1]
            curr_price = bonds[cur_time][0]
            dirty_pri = dirty_price(mature_date, last_coupon_date, coupon, face_val, curr_price)
            spot_2 = bootstrap(dirty_pri, coupon, face_val, spot_1, t_1, t_2)
            spots.append(spot_2 * 100)
            cur_spot = spot_2
            last_coupon_date = mature_list[1]
        else:
            t_1 = (mature_list[j] - mature_list[j - 1]).days /365
            t_2 = (mature_list[j + 1] - mature_list[j - 1]).days /365
            coupon = coupon_list[j + 1] 
            mature_date = mature_list[j + 1]
            curr_price = bonds[cur_time][j + 1]
            spot_1 = cur_spot
            dirty_pri = dirty_price(mature_date, last_coupon_date, coupon, face_val, curr_price)
            spot_2 = bootstrap(dirty_pri, coupon, face_val, spot_1, t_1, t_2)
            spots.append(spot_2 * 100)
            cur_spot = spot_2
            last_coupon_date = mature_list[j + 1]        
    total_spot_rate.append(spots)
        

        
spot_dict = {}
for idx in range(len(cur_time_list)):
    spot_dict[cur_time_list[idx]]= total_spot_rate[idx]
spot_frame = pd.DataFrame(data = spot_dict)  
print(spot_frame)



# In[355]:


# Plot the spot rate
FIVE_YEARS = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
df=pd.DataFrame({'x': FIVE_YEARS, 
                 'Jan 2': total_spot_rate[0], 
                 'Jan 3': total_spot_rate[1], 
                 'Jan 6': total_spot_rate[2], 
                 'Jan 7': total_spot_rate[3], 
                 'Jan 8': total_spot_rate[4], 
                 'Jan 9': total_spot_rate[5], 
                 'Jan 10': total_spot_rate[6], 
                 'Jan 13': total_spot_rate[7], 
                 'Jan 14': total_spot_rate[8], 
                 'Jan 15': total_spot_rate[9] })    
# size
plt.figure(figsize=(15,5), dpi= 80)
# style
plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num +=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)

    # Add legend
plt.legend(loc=2, ncol=2)
 
# Add titles
plt.title("Spot Rate", loc='center', fontsize=20, fontweight=0, color='orange')
plt.xlabel("Maturity")
plt.ylabel("spot rate")


# In[411]:


# calculate the one year forward rate
def one_year_forward_rates(spot_rates):
    t_a = 2
    t_b = 1
    forward_rates = []
    for i in range(4):
        r_a = spot_rates[(i + 1) * 2]
        r_b = spot_rates[i * 2]
        forward_rate = (1+r_a / 100)**t_a/(1+r_b / 100)**t_b - 1
        forward_rates.append(abs(forward_rate) * 100)
        t_a += 1
        t_b += 1
    return forward_rates 
        
#          

total_forward_rate = []
for i in range(10):
    forward_rate = one_year_forward_rates(total_spot_rate[i])
    total_forward_rate.append(forward_rate)
print(total_forward_rate)

for_dict = {}
for idx in range(len(cur_time_list)):
    for_dict[cur_time_list[idx]]= total_forward_rate[idx]
for_frame = pd.DataFrame(data = for_dict)  


# In[369]:


# plot the forward rate
FIVE_YEARS = [i for i in range(2,6)]
df=pd.DataFrame({'x': FIVE_YEARS, 
                 'Jan 2': total_forward_rate[0], 
                 'Jan 3': total_forward_rate[1], 
                 'Jan 6': total_forward_rate[2], 
                 'Jan 7': total_forward_rate[3],
                 'Jan 8': total_forward_rate[4],
                 'Jan 9': total_forward_rate[5], 
                 'Jan 10': total_forward_rate[6], 
                 'Jan 13': total_forward_rate[7], 
                 'Jan 14': total_forward_rate[8],
                 'Jan 15': total_forward_rate[9]
                })    
# size
plt.figure(figsize=(15,5), dpi= 80)
# style
plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num +=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)

    # Add legend
plt.legend(loc=2, ncol=2)
 
# Add titles
plt.title("Forward Rate", loc='center', fontsize=20, fontweight=0, color='orange')
plt.xlabel("Maturity")
plt.ylabel("forward rate")



# In[429]:


from numpy import *

cov_mat2 = np.zeros([9, 5])
for i in range(0,5):
    for j in range(1, 10):
        X_ij = math.log((ytm_frame.iloc[i*2, j]) / (ytm_frame.iloc[i*2, j-1]))
        cov_mat2[j-1, i] = X_ij
ytm_cov = np.cov(cov_mat2.T)
eig_val_ytm, eig_vec_ytm = np.linalg.eig(ytm_cov)
print(ytm_cov)
print(eig_val_ytm, eig_vec_ytm)
print(eig_val_ytm[0]/sum(eig_val_ytm))


cov_mat2 = np.zeros([9, 4])
for i in range(0,4):
    for j in range(1, 10):
        X_ij = math.log((for_frame.iloc[i, j]) / (for_frame.iloc[i, j-1]))
        cov_mat2[j-1, i] = X_ij
forward_cov = np.cov(cov_mat2.T)
eig_val_for, eig_vec_for = np.linalg.eig(forward_cov)
print(forward_cov)
print(eig_val_for, eig_vec_for)
print(eig_val_for[0]/sum(eig_val_for))


import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import pickle
import random
import time
import csv
from operator import add

start_time = time.time()

bess_max_capacity = 12.15 # powerwall specs
bess_min_capacity = 1.35
max_discharge_rate = 5 # kw
max_discharge = 2.5 # kWh
hours_in_day = 24

hour_diff = 0.5

peak = 0.44539
shoulder = 0.24431
off_peak = 0.30734

elec_rate = 0.3773 #$/kWh
fit = 0.16 #$/kWh

yearly_amber_fee = 180
other_bill_costs = 0.2168
market_env_costs = 0.031

load_scaling = 1
pv_scaling = 5.0156

pickle_in = open("time_single.pickle", "rb")
timestamp = pickle.load(pickle_in)

list_length = len(timestamp)

df = pd.read_csv('sa_ACTUAL.csv')

rrp= df.RRP.tolist()

raise6sec_price = df.RAISE6SECRRP.tolist()
raise60sec_price  = df.RAISE60SECRRP.tolist()
raise5min_price  = df.RAISE5MINRRP.tolist()
raisereg_price  = df.RAISEREGRRP.tolist()

lower6sec_price  = df.LOWER6SECRRP.tolist()
lower60sec_price  = df.LOWER60SECRRP.tolist()
lower5min_price  = df.LOWER5MINRRP.tolist()
lowerreg_price  = df.LOWERREGRRP.tolist()


global instant_bess, bess , import_grid, import_cost, export_grid , fit_rev, FCAS_rev 

instant_bess = [0]*list_length #battery SOC list at the given time period
bess = [0]*list_length #battery SOC list which accumulates overtime
import_grid = [0]*list_length
import_cost = [0]*list_length
export_grid = [0]*list_length
fit_rev = [0]*list_length
FCAS_rev = [0]*list_length

raise6sec = [0]*list_length
raise60sec = [0]*list_length
raise5min = [0]*list_length
raisereg = [0]*list_length
lower6sec = [0]*list_length
lower60sec = [0]*list_length
lower5min = [0]*list_length
lowerreg = [0]*list_length

FCAS_max_price_list = [0]*list_length

bess_raise_discharge = 5 * hour_diff # maximum discharging within 30 min period in kWh
bess_lower_charge = 5 * hour_diff # maximum charging within 30 min period in kWh
bess_FCAS_min = bess_min_capacity + (1 * bess_raise_discharge)
bess_FCAS_max = bess_max_capacity - (1 * bess_lower_charge)

pickle_in = open("solar_homes.pickle", "rb")
df = pickle.load(pickle_in)

column_len = len(df.columns)

row_len = len(df.index) # 268557

original_bill_flat = 0
original_bill_tou = 0

from_row=0
to_row=17520

array_rows, array_cols = (300, 22) 
arr = [ [ 0 for i in range(array_cols) ] for j in range(array_rows) ]

CL = []
GC = []
GG = []

raise_commitment = 5
lower_commitment = 5

def bess_pv_load_plot():
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Battery SOC (kWh)')
    ax1.plot(timestamp[from_row:to_row], bess[from_row:to_row],color=color, label = "Battery SOC")
    ax1.tick_params(axis='y')
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Power (kW)')  # we already handled the x-label with ax1
    ax2.plot(timestamp[from_row:to_row], load[from_row:to_row], label = "Load")
    ax2.plot(timestamp[from_row:to_row], pv[from_row:to_row], label = "PV generation")
    ax2.tick_params(axis='y')

    plt.legend(loc="upper right")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_import_export():

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Spot Price ($/kWh)')
    ax1.plot(timestamp[from_row:to_row], rrp[from_row:to_row],color='k', label = "Spot price")
    ax1.tick_params(axis='y')
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Revenue and cost ($/trading interval)')  # we already handled the x-label with ax1
    ax2.plot(timestamp[from_row:to_row], import_cost[from_row:to_row], color = 'r', label = "Import cost")
    ax2.plot(timestamp[from_row:to_row], fit_rev[from_row:to_row], color = 'g', label = "FiT revenue")
    ax2.tick_params(axis='y')

    plt.legend(loc="upper right")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_bess_only():
    plt.plot(timestamp[from_row:to_row], bess[from_row:to_row],color='g')
    
    plt.xlabel("Date")
    plt.ylabel("Battery SOC (kWh)")
    plt.show()

def raise_period(duration): # duration in minutes
    bess[i] = bess[i] - raise_commitment * (duration/60)
    if bess[i] < bess_min_capacity:
        bess[i] = bess_min_capacity

def lower_period(duration):
    bess[i] = bess[i] + lower_commitment * (duration/60)
    if bess[i] > bess_max_capacity:
        bess[i] = bess_max_capacity

def contingency():
    if (timestamp[i].day == 3 and timestamp[i].month == 9 and
        timestamp[i].hour == 1 and timestamp[i].minute == 0):
        # print ("1")
        lower_period(8)
    elif (timestamp[i].day == 9 and timestamp[i].month == 10 and
        timestamp[i].hour == 8 and timestamp[i].minute == 0):
        # print("2")
        raise_period(3)               
    elif (timestamp[i].day == 16 and timestamp[i].month == 11 and
        timestamp[i].hour == 18 and timestamp[i].minute == 0):
        # print("3")
        lower_period(15)
    elif (timestamp[i].day == 10 and timestamp[i].month == 12 and
        timestamp[i].hour == 13 and timestamp[i].minute == 30):
        # print("4")
        lower_period(2)
    elif (timestamp[i].day == 10 and timestamp[i].month == 12 and
        timestamp[i].hour == 14 and timestamp[i].minute == 0):
        # print("5")
        raise_period(8)
    elif (timestamp[i].day == 2 and timestamp[i].month == 1 and
        timestamp[i].hour == 15 and timestamp[i].minute == 30):
        # print("6")
        raise_period(0.2)
    elif (timestamp[i].day == 20 and timestamp[i].month == 1 and
        timestamp[i].hour == 13 and timestamp[i].minute == 0):
        # print("7")
        raise_period(13)
    elif (timestamp[i].day == 23 and timestamp[i].month == 1 and
        timestamp[i].hour == 11 and timestamp[i].minute == 30):
        # print("8")
        raise_period(10)
    elif (timestamp[i].day == 28 and timestamp[i].month == 1 and
        timestamp[i].hour == 17 and timestamp[i].minute == 0):
        # print("9a")
        lower_period(30)
    elif (timestamp[i].day == 28 and timestamp[i].month == 1 and
        timestamp[i].hour == 17 and timestamp[i].minute == 30):
        # print("9b")
        lower_period(30)
    elif (timestamp[i].day == 30 and timestamp[i].month == 1 and
        timestamp[i].hour == 17 and timestamp[i].minute == 30):
        # print("10")
        raise_period(10)
    elif (timestamp[i].day == 14 and timestamp[i].month == 2 and
        timestamp[i].hour == 10 and timestamp[i].minute == 0):
        # print("11")
        raise_period(10)
    elif (timestamp[i].day == 5 and timestamp[i].month == 4 and
        timestamp[i].hour == 15 and timestamp[i].minute == 30):
        # print("12")
        raise_period(10)
    elif (timestamp[i].day == 10 and timestamp[i].month == 4 and
        timestamp[i].hour == 17 and timestamp[i].minute == 0):
        # print("13")
        raise_period(7)
    elif (timestamp[i].day == 16 and timestamp[i].month == 4 and
        timestamp[i].hour == 18 and timestamp[i].minute == 0):
        # print("14a")
        raise_period(30)
    elif (timestamp[i].day == 16 and timestamp[i].month == 4 and
        timestamp[i].hour == 18 and timestamp[i].minute == 30):
        # print("14b")
        raise_period(30)
    elif (timestamp[i].day == 6 and timestamp[i].month == 5 and
        timestamp[i].hour == 16 and timestamp[i].minute == 30):
        # print("15")
        raise_period(10)
    elif (timestamp[i].day == 12 and timestamp[i].month == 5 and
        timestamp[i].hour == 14 and timestamp[i].minute == 30):
        # print("16")
        raise_period(7)
    elif (timestamp[i].day == 19 and timestamp[i].month == 5 and
        timestamp[i].hour == 14 and timestamp[i].minute == 30):
        # print("17")
        raise_period(10)
    elif (timestamp[i].day == 1 and timestamp[i].month == 6 and
        timestamp[i].hour == 12 and timestamp[i].minute == 30):
        # print("18")
        raise_period(10)

def calc_pv_ratio():
    global pv_consumption_ratio
    pv_consumption_ratio = ((sum(load)*0.5) - sum(import_grid))/(sum(export_grid))

def empty_lists():
    global instant_bess, bess , import_grid, import_cost, export_grid , fit_rev, FCAS_rev 
    instant_bess = [0]*list_length #battery SOC list at the given time period
    bess = [0]*list_length #battery SOC list which accumulates overtime
    import_grid = [0]*list_length
    import_cost = [0]*list_length
    export_grid = [0]*list_length
    fit_rev = [0]*list_length
    FCAS_rev = [0]*list_length


row=0
check=0
customer_min = 1
customer_max = 300
while (row < row_len): #row_len
    if (df.iloc[row,0] >= customer_min and df.iloc[row,0] <= customer_max) :
        customer_no = df.iloc[row,0]
        temp_list = list(df.iloc[row,4:column_len])
        if df.iloc[row,2] == "CL":
            CL.extend(temp_list)
        if df.iloc[row,2] == "GC":
            GC.extend(temp_list)
        if df.iloc[row,2] == "GG":
            GG.extend(temp_list)
        
        if ( customer_no != 300 and (df.iloc[row+1,0] == customer_no+1)): # once the list hits the full year mark, we move onto the next customer
            if (len(GG) == 17520 and len(GC) == 17520 and customer_no !=68 and customer_no!=95 and customer_no !=161 and customer_no != 187
                                and customer_no != 248 and customer_no != 272 and customer_no!=284 and customer_no!=289 and customer_no!=293 and customer_no!=294):
                # check = check+1
                # print(len(GC), len(GG), customer_no)
                if not CL: # if CL list is empty, then the load list is just GC, general consumption
                    load = GC
                else: # otherwise, add CL (controlled load) and GC together
                    load = list(map(add,CL,GC))
                load = [i*load_scaling for i in load]
                pv = GG
                pv = [i*pv_scaling for i in pv]
                
                original_bill_flat = 0
                original_bill_tou = 0

                i=0
                for times in timestamp:  # BASE FLAT
                    if i>0:
                        excess_in_kW = pv[i] - load[i]
                        excess_in_kWh = excess_in_kW * (hour_diff)
                        original_bill_flat = original_bill_flat + load[i]*hour_diff*elec_rate
                        if excess_in_kWh > 0: #if PV is greater than load
                            if bess[i-1] == bess_max_capacity:
                                export_grid[i] =  excess_in_kWh
                                fit_rev[i] = export_grid[i] * fit #paid at the feed in tariff price
                                bess[i] = bess_max_capacity
                            else:
                                bess[i] = bess[i-1] + excess_in_kWh
                                if bess[i] > bess_max_capacity:
                                    export_grid[i] = bess[i] - bess_max_capacity
                                    fit_rev[i] = export_grid[i] * fit 
                                    bess[i] = bess_max_capacity
                        else: #if load is greater than PV
                            if bess[i-1] > bess_min_capacity: #if the battery has charge
                                bess[i] = bess[i-1] - abs(excess_in_kWh)
                                if bess[i] < bess_min_capacity:
                                    import_grid[i] =  bess_min_capacity-bess[i] #importing electricity from grid
                                    import_cost[i] = import_grid[i] * elec_rate # paying for importing electricity
                                    bess[i] = bess_min_capacity
                            else: 
                                import_grid[i] = -excess_in_kWh
                                import_cost[i] = import_grid[i] * elec_rate
                                bess[i] = bess_min_capacity
                    i = i + 1
                
                calc_pv_ratio()
                # print (pv_consumption_ratio)

                # customer_no - 1 because the 1st customer is in the 0th row
                arr[customer_no-1][0] = original_bill_flat 
                arr[customer_no-1][1] = sum(import_cost)
                arr[customer_no-1][2] = sum(fit_rev)
                arr[customer_no-1][16] = pv_consumption_ratio

                # print(original_bill_flat,sum(import_cost), sum(fit_rev))

                # bess_pv_load_plot()
                # plot_bess_only()

                empty_lists()

                i=0
                for times in timestamp: # BASE TOU
                    if i>0:
                        excess_in_kW = pv[i] - load[i]
                        excess_in_kWh = excess_in_kW * (hour_diff)

                        # BASE TOU CODE
                        if (timestamp[i].hour >= 15 and timestamp[i].hour < 24) or timestamp[i].hour == 0 or (timestamp[i].hour >= 6 and timestamp[i].hour < 10):
                            tou_elec_rate = peak
                        if (timestamp[i].hour >= 10 and timestamp[i].hour < 15 ): # if shoulder period
                            tou_elec_rate = shoulder
                        if (timestamp[i].hour >= 1 and timestamp[i].hour < 6): # if off peak 
                            tou_elec_rate = off_peak

                        original_bill_tou = original_bill_tou + load[i]*hour_diff*tou_elec_rate # iterating to find the original bill without PV and battery

                        if (tou_elec_rate == peak): # if peak period
                            if excess_in_kWh > 0: #if PV is greater than load
                                if bess[i-1] == bess_max_capacity:
                                    export_grid[i] =  excess_in_kWh
                                    fit_rev[i] = export_grid[i] * fit #paid at the feed in tariff price
                                    bess[i]=bess[i-1]
                                else:
                                    bess[i] = bess[i-1] + excess_in_kWh
                                    if bess[i] > bess_max_capacity:
                                        export_grid[i] = bess[i] - bess_max_capacity
                                        fit_rev[i] = export_grid[i] * fit 
                                        bess[i] = bess_max_capacity
                            else: #if load is greater than PV
                                
                                if bess[i-1] > bess_min_capacity: #if the battery has charge
                                    bess[i] = bess[i-1] - abs(excess_in_kWh)
                                    if bess[i] < bess_min_capacity:
                                        import_grid[i] =  bess_min_capacity-bess[i] #importing electricity from grid
                                        import_cost[i] = import_grid[i] * tou_elec_rate # paying for importing electricity
                                        bess[i] = bess_min_capacity
                                else: 
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * tou_elec_rate
                                    bess[i] = bess_min_capacity

                        else: #if not peak time
                            if excess_in_kWh > 0: #if PV is greater than load
                                if bess[i-1] == bess_max_capacity:
                                    export_grid[i] =  excess_in_kWh
                                    fit_rev[i] = export_grid[i] * fit
                                    bess[i]=bess[i-1]
                                else:
                                    bess[i] = bess[i-1] + excess_in_kWh
                                    if bess[i] > bess_max_capacity:
                                        export_grid[i] = bess[i] - bess_max_capacity
                                        fit_rev[i] = export_grid[i] * fit
                                        bess[i] = bess_max_capacity
                            else: #if load is greater than PV
                                if (tou_elec_rate == shoulder): # if shoulder period
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * tou_elec_rate
                                    bess[i] = bess[i-1]
                                if (tou_elec_rate == off_peak): # if off peak 
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * tou_elec_rate
                                    bess[i] = bess[i-1]    
                                                
                    i=i+1

                calc_pv_ratio()
                # print (pv_consumption_ratio)

                arr[customer_no-1][3] = original_bill_tou
                arr[customer_no-1][4] = sum(import_cost)
                arr[customer_no-1][5] = sum(fit_rev)
                arr[customer_no-1][17] = pv_consumption_ratio

                # print(original_bill_tou,sum(import_cost), sum(fit_rev))

                # bess_pv_load_plot()
                # plot_bess_only()

                empty_lists()

                i=0
                for times in timestamp: #FCAS FLAT
                    if i>0:
                        excess_in_kW = pv[i] - load[i]
                        excess_in_kWh = excess_in_kW * (hour_diff)

                        FCAS_rev[i] = ( (raise6sec_price[i] + raise60sec_price[i] + raise5min_price[i]) * bess_raise_discharge +
                                    (lower6sec_price[i] + lower60sec_price[i] + lower5min_price[i]) * bess_lower_charge )

                        if excess_in_kWh > 0: #if PV is greater than load
                            if bess[i-1] == bess_FCAS_max:
                                export_grid[i] =  excess_in_kWh
                                fit_rev[i] = export_grid[i] * fit #paid at the feed in tariff price
                                bess[i] = bess_FCAS_max
                            else:
                                bess[i] = bess[i-1] + excess_in_kWh
                                if bess[i] > bess_FCAS_max:
                                    export_grid[i] = bess[i] - bess_FCAS_max
                                    fit_rev[i] = export_grid[i] * fit 
                                    bess[i] = bess_FCAS_max
                        else: #if load is greater than PV
                            if bess[i-1] > bess_FCAS_min: #if the battery has charge
                                bess[i] = bess[i-1] - abs(excess_in_kWh)
                                if bess[i] < bess_FCAS_min:
                                    import_grid[i] =  bess_FCAS_min-bess[i] #importing electricity from grid
                                    import_cost[i] = import_grid[i] * elec_rate # paying for importing electricity
                                    bess[i] = bess_FCAS_min
                            else: 
                                import_grid[i] = -excess_in_kWh
                                import_cost[i] = import_grid[i] * elec_rate
                                bess[i] = bess_FCAS_min

                        contingency()
                    i = i + 1
                
                calc_pv_ratio()
                # print (pv_consumption_ratio)

                arr[customer_no-1][6] = sum(import_cost)
                arr[customer_no-1][7] = sum(fit_rev)
                arr[customer_no-1][8] = sum(FCAS_rev)
                arr[customer_no-1][18] = pv_consumption_ratio

                # bess_pv_load_plot()
                
                # print(sum(import_cost), sum(fit_rev), sum(FCAS_rev))

                # plt.plot(timestamp[from_row:to_row], FCAS_rev[from_row:to_row])
                # plt.show()
                # print (sum(FCAS_rev[10320:11712])) # calculating revenue in Feb only

                empty_lists()

                i= 0
                for times in timestamp: #FCAS TOU
                    if i>0:
                        excess_in_kW = pv[i] - load[i]
                        excess_in_kWh = excess_in_kW * (hour_diff)

                        FCAS_rev[i] = ( (raise6sec_price[i] + raise60sec_price[i] + raise5min_price[i]) * bess_raise_discharge +
                                    (lower6sec_price[i] + lower60sec_price[i] + lower5min_price[i]) * bess_lower_charge )

                        if (timestamp[i].hour >= 15 and timestamp[i].hour < 24) or timestamp[i].hour == 0 or (timestamp[i].hour >= 6 and timestamp[i].hour < 10):
                            tou_elec_rate = peak
                        if (timestamp[i].hour >= 10 and timestamp[i].hour < 15 ): # if shoulder period
                            tou_elec_rate = shoulder
                        if (timestamp[i].hour >= 1 and timestamp[i].hour < 6): # if off peak 
                            tou_elec_rate = off_peak

                        if (tou_elec_rate == peak): # if peak period
                            if excess_in_kWh > 0: #if PV is greater than load
                                if bess[i-1] == bess_FCAS_max:
                                    export_grid[i] =  excess_in_kWh
                                    fit_rev[i] = export_grid[i] * fit #paid at the feed in tariff price
                                    bess[i]=bess_FCAS_max
                                else:
                                    bess[i] = bess[i-1] + excess_in_kWh
                                    if bess[i] > bess_FCAS_max:
                                        export_grid[i] = bess[i] - bess_FCAS_max
                                        fit_rev[i] = export_grid[i] * fit 
                                        bess[i] = bess_FCAS_max
                            else: #if load is greater than PV
                                if bess[i-1] > bess_FCAS_min: #if the battery has charge
                                    bess[i] = bess[i-1] - abs(excess_in_kWh)
                                    if bess[i] < bess_FCAS_min:
                                        import_grid[i] =  bess_FCAS_min-bess[i] #importing electricity from grid
                                        import_cost[i] = import_grid[i] * tou_elec_rate # paying for importing electricity
                                        bess[i] = bess_FCAS_min
                                else: 
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * tou_elec_rate
                                    bess[i] = bess_FCAS_min

                        else: #if not peak time
                            if excess_in_kWh > 0: #if PV is greater than load
                                if bess[i-1] == bess_FCAS_max:
                                    export_grid[i] =  excess_in_kWh
                                    fit_rev[i] = export_grid[i] * fit
                                    bess[i]=bess[i-1]
                                else:
                                    bess[i] = bess[i-1] + excess_in_kWh
                                    if bess[i] > bess_FCAS_max:
                                        export_grid[i] = bess[i] - bess_FCAS_max
                                        fit_rev[i] = export_grid[i] * fit
                                        bess[i] = bess_FCAS_max
                            else: #if load is greater than PV
                                if (tou_elec_rate == shoulder): # if shoulder period
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * tou_elec_rate
                                    bess[i] = bess[i-1]
                                if (tou_elec_rate == off_peak): # if off peak 
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * tou_elec_rate
                                    bess[i] = bess[i-1]

                        contingency()
                    i = i + 1
                
                
                calc_pv_ratio()
                # print (pv_consumption_ratio)

                arr[customer_no-1][9] = sum(import_cost)
                arr[customer_no-1][10] = sum(fit_rev)
                arr[customer_no-1][11] = sum(FCAS_rev)
                arr[customer_no-1][19] = pv_consumption_ratio

                # print(sum(import_cost), sum(fit_rev), sum(FCAS_rev))

                # bess_pv_load_plot()

                empty_lists()

                i= 0
                for times in timestamp: #ENERGY FLAT
                    if i>0:
                        excess_in_kW = pv[i] - load[i]
                        excess_in_kWh = excess_in_kW * (hour_diff)

                        if excess_in_kWh > 0: #if PV is greater than load
                            if bess[i-1] == bess_max_capacity:
                                export_grid[i] =  excess_in_kWh 
                                fit_rev[i] = export_grid[i] * (rrp[i]+market_env_costs) #paid at the feed in tariff price
                                bess[i] = bess_max_capacity
                            else:
                                bess[i] = bess[i-1] + excess_in_kWh
                                if bess[i] > bess_max_capacity:
                                    export_grid[i] = bess[i] - bess_max_capacity
                                    fit_rev[i] = export_grid[i] * (rrp[i]+market_env_costs)
                                    bess[i] = bess_max_capacity
                        else: #if load is greater than PV
                            if bess[i-1] > bess_min_capacity: #if the battery has charge
                                bess[i] = bess[i-1] - abs(excess_in_kWh)
                                if bess[i] < bess_min_capacity:
                                    import_grid[i] =  bess_min_capacity-bess[i] #importing electricity from grid
                                    import_cost[i] = import_grid[i] * (rrp[i]+other_bill_costs)# paying for importing electricity
                                    bess[i] = bess_min_capacity
                            else: 
                                import_grid[i] = -excess_in_kWh
                                import_cost[i] = import_grid[i] * (rrp[i]+other_bill_costs)
                                bess[i] = bess_min_capacity
                    i=i+1

                calc_pv_ratio()
                # print (pv_consumption_ratio)

                arr[customer_no-1][12] = sum(import_cost) + yearly_amber_fee
                arr[customer_no-1][13] = sum(fit_rev)
                arr[customer_no-1][20] = pv_consumption_ratio

                # print(sum(import_cost), sum(fit_rev))

                # bess_pv_load_plot()
                # plot_bess_only()

                # print(max(rrp))

                # plot_import_export()

                # plt.plot(timestamp, import_cost)
                # plt.xlabel("Date")
                # plt.ylabel("Electricity cost ($/Trading Interval)")
                # plt.show()

                empty_lists()

                i=0
                for times in timestamp: # ENERGY TOU
                    if i>0:
                        excess_in_kW = pv[i] - load[i]
                        excess_in_kWh = excess_in_kW * (hour_diff)

                        # BASE TOU CODE
                        if (timestamp[i].hour >= 15 and timestamp[i].hour < 24) or timestamp[i].hour == 0 or (timestamp[i].hour >= 6 and timestamp[i].hour < 10):
                            tou_elec_rate = peak
                        if (timestamp[i].hour >= 10 and timestamp[i].hour < 15 ): # if shoulder period
                            tou_elec_rate = shoulder
                        if (timestamp[i].hour >= 1 and timestamp[i].hour < 6): # if off peak 
                            tou_elec_rate = off_peak

                        if (tou_elec_rate == peak): # if peak period
                            if excess_in_kWh > 0: #if PV is greater than load
                                if bess[i-1] == bess_max_capacity:
                                    export_grid[i] =  excess_in_kWh
                                    fit_rev[i] = export_grid[i] * (rrp[i]+market_env_costs) #paid at the feed in tariff price
                                    bess[i]=bess[i-1]
                                else:
                                    bess[i] = bess[i-1] + excess_in_kWh
                                    if bess[i] > bess_max_capacity:
                                        export_grid[i] = bess[i] - bess_max_capacity
                                        fit_rev[i] = export_grid[i] * (rrp[i]+market_env_costs)
                                        bess[i] = bess_max_capacity
                            else: #if load is greater than PV
                                
                                if bess[i-1] > bess_min_capacity: #if the battery has charge
                                    bess[i] = bess[i-1] - abs(excess_in_kWh)
                                    if bess[i] < bess_min_capacity:
                                        import_grid[i] =  bess_min_capacity-bess[i] #importing electricity from grid
                                        import_cost[i] = import_grid[i] * (rrp[i]+other_bill_costs) # paying for importing electricity
                                        bess[i] = bess_min_capacity
                                else: 
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * (rrp[i]+other_bill_costs)
                                    bess[i] = bess_min_capacity

                        else: #if not peak time
                            if excess_in_kWh > 0: #if PV is greater than load
                                if bess[i-1] == bess_max_capacity:
                                    export_grid[i] =  excess_in_kWh
                                    fit_rev[i] = export_grid[i] * (rrp[i]+market_env_costs)
                                    bess[i]=bess[i-1]
                                else:
                                    bess[i] = bess[i-1] + excess_in_kWh
                                    if bess[i] > bess_max_capacity:
                                        export_grid[i] = bess[i] - bess_max_capacity
                                        fit_rev[i] = export_grid[i] * (rrp[i]+market_env_costs)
                                        bess[i] = bess_max_capacity
                            else: #if load is greater than PV
                                if (tou_elec_rate == shoulder): # if shoulder period
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * (rrp[i]+other_bill_costs)
                                    bess[i] = bess[i-1]
                                if (tou_elec_rate == off_peak): # if off peak 
                                    import_grid[i] = -excess_in_kWh
                                    import_cost[i] = import_grid[i] * (rrp[i]+other_bill_costs)
                                    bess[i] = bess[i-1]          
                    i=i+1

                calc_pv_ratio()
                # print (pv_consumption_ratio)

                arr[customer_no-1][14] = sum(import_cost) + yearly_amber_fee
                arr[customer_no-1][15] = sum(fit_rev)
                arr[customer_no-1][21] = pv_consumption_ratio

                # print(sum(import_cost), sum(fit_rev))

                # bess_pv_load_plot()
                # plot_bess_only()

                # plot_import_export()

                empty_lists()

                CL = []
                GC = []
                GG = []
            else :
                CL = []
                GC = []
                GG = []

    # where all the operations and number crunching finishes, lists get cleaned and ready for the next customer
    row=row+1


with open("results.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(arr)


print("Process finished --- %s seconds ---" % (time.time() - start_time))

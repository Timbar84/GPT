# -*- coding: utf-8 -*-
#%% Importations
import simpy
import random
from datetime import datetime, timedelta
from functools import partial, wraps
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
import statistics
import scipy.stats

#%% Parameters

TODAY = datetime(2023,5,1,0,0,0)
DAYS_OF_SIMULATION = 7*4

def param_settings():
    global PARAM
    global OEE_FILLING
    global OEE_DECASING
    PARAM = dict()
    
    # Filling
    if OEE_FILLING == 0.3:
        filling_time = dict(#FILLING
                     A_FILLING_TIME_AVG = 3.54,
                     A_FILLING_TIME_STD = 0.87,
                     A_FILLING_TIME_MIN = 2,
                     A_FILLING_TIME_MAX = 5.92,
                    
                     B_FILLING_TIME_AVG = 3.86,
                     B_FILLING_TIME_STD = 1.05,
                     B_FILLING_TIME_MIN = 2.08,
                     B_FILLING_TIME_MAX = 5.92,
                    
                     C_FILLING_TIME_AVG = 4.36,
                     C_FILLING_TIME_STD = 1.19,
                     C_FILLING_TIME_MIN = 2.67,
                     C_FILLING_TIME_MAX = 8.92)
    elif OEE_FILLING == 0.45:
        filling_time = dict(#Filling
                     A_FILLING_TIME_AVG = 2.75,
                     A_FILLING_TIME_STD = 0.85,
                     A_FILLING_TIME_MIN = 1.50,
                     A_FILLING_TIME_MAX = 4,
                    
                     B_FILLING_TIME_AVG = 2.75,
                     B_FILLING_TIME_STD = 0.85,
                     B_FILLING_TIME_MIN = 1.50,
                     B_FILLING_TIME_MAX = 4,
                    
                     C_FILLING_TIME_AVG = 3.44,
                     C_FILLING_TIME_STD = 0.85,
                     C_FILLING_TIME_MIN = 2,
                     C_FILLING_TIME_MAX = 5.5)
    
    # STERILIZATION
    sterilization_time = dict(STERILIZATION_TIME = 2)
    
    # DECASING
    if OEE_DECASING == 0:
        decasing_time = dict(A_DECASING_TIME_AVG = 2.69,
                             A_DECASING_TIME_STD = 0.55,
                             A_DECASING_TIME_MIN = 1.93,
                             A_DECASING_TIME_MAX = 4.17,
                            
                             B_DECASING_TIME_AVG = 2.99,
                             B_DECASING_TIME_STD = 0.76,
                             B_DECASING_TIME_MIN = 1.68,
                             B_DECASING_TIME_MAX = 4.77,
                            
                             C_DECASING_TIME_AVG = 3.67,
                             C_DECASING_TIME_STD = 0.88,
                             C_DECASING_TIME_MIN = 2.40,
                             C_DECASING_TIME_MAX = 5.92)
    else:
        decasing_time = dict(A_DECASING_TIME_AVG = 2.69*(1-OEE_DECASING/100),
                             A_DECASING_TIME_STD = 0.65,
                             A_DECASING_TIME_MIN = max(1.35,1.68*(1-OEE_DECASING/100)),
                             A_DECASING_TIME_MAX = 4.77*(1-OEE_DECASING/100),
                            
                             B_DECASING_TIME_AVG = 2.69*(1-OEE_DECASING/100),
                             B_DECASING_TIME_STD = 0.65,
                             B_DECASING_TIME_MIN = max(1.35,1.68*(1-OEE_DECASING/100)),
                             B_DECASING_TIME_MAX = 4.77*(1-OEE_DECASING/100),
                            
                             C_DECASING_TIME_AVG = 3.67*(1-OEE_DECASING/100),
                             C_DECASING_TIME_STD = 0.8,
                             C_DECASING_TIME_MIN = max(1.80,2.40*(1-OEE_DECASING/100)),
                             C_DECASING_TIME_MAX = 5.92*(1-OEE_DECASING/100))

    PARAM.update(filling_time)
    PARAM.update(sterilization_time)
    PARAM.update(decasing_time)

# Factors
NUM_BOXES = 450+250
OEE_FILLING = 0.45 # 0.3 or 0.45
OEE_DECASING = 10 # 0 or 10 FOR NOW IT IS THE DECREASE IN % OF THE DECASING TIME
param_settings()
MAX_QUEUE = 4
DATA_COLLECTION = True # To collect the data for the gantt, level of boxes, level of the queue, processing time
PLOT = False # To plot the results
#%% Variables
if DATA_COLLECTION:
    data_gantt = []
data_idle_time_filling = []
data_batch_produced = 0
data_boxes = []
data_queue_fill_ster = []
data_queue_ster_decas = []

#%% Classes

class Entity:
    def __init__(self, env, batch_number, sub_batch, filling_station, sterilization_station, decasing_station, boxes, queue_fill_ster, queue_ster_decas):
        # Simulation attributes
        self.env = env
        self.action = env.process(self.run(filling_station, sterilization_station, decasing_station, boxes, queue_fill_ster, queue_ster_decas))
        # Batch informations
        self.batch_number = batch_number
        self.sub_batch = sub_batch
        self.name = 'Batch ' + str(self.batch_number) + ' sub-' + str(self.sub_batch)
        # Processing times
        self.filling_time = filling_time(self.sub_batch)
        self.decasing_time = decasing_time(self.sub_batch)
        self.sterilization_time = PARAM['STERILIZATION_TIME']
        self.decasing_overlap = False # Boolean for the gantt data
        # Waiting time data
        self.waiting_time_due_to_boxes = 0
        self.waiting_time_due_to_queue = 0
    
    def wait_the_night(self):
        self.decasing_overlap = True
        self.decasing_first_end = format_time(self.env.now)
        yield self.env.timeout(8) # Wait the night
        self.decasing_second_start = format_time(self.env.now)
        
    def run(self, filling_station, sterilization_station, decasing_station, boxes, queue_fill_ster, queue_ster_decas):
        if self.batch_number == 1 and self.sub_batch == 1:
            yield self.env.timeout(6) # Begin the simulation at Monday 6 am
        
        amount_boxes = 106 if self.sub_batch == 3 else 84
        
        # ************** Checking if the production of a new batch can start **************
        
        current_day, current_time = self.env.now // 24 % 7, self.env.now % 24
        if self.sub_batch == 1 and current_day >= 4: # First sub_batch on a friday or saturday
            estimated_ending_time = current_time + PARAM["A_FILLING_TIME_AVG"]+PARAM["B_FILLING_TIME_AVG"]+PARAM["C_FILLING_TIME_AVG"]
            waiting_time = 0
            if current_day == 4: # On friday
                if estimated_ending_time > 24 and estimated_ending_time - 24 > 6: # Cannot produce a batch
                    waiting_time = 24 - current_time + 24+24+6
            else: # On saturday
                if estimated_ending_time > 6: # We cannot produce a batch
                    waiting_time = 6 - current_time + 24+24
            if waiting_time: # if it will end after 6am of Saturday
                yield self.env.timeout(waiting_time)
        
        # ************** Entity waits for boxes **************
        
        start = self.env.now
        yield boxes.get(amount_boxes//2) # Make busy half of the needed boxes
        self.waiting_time_due_to_boxes += self.env.now - start
        
        # ************** Entity enters filling process **************
        
        prio = 1 if self.sub_batch == 3 else 0 # The priority helps to manage the setup
        with filling_station.request(prio) as request:
            yield request # Request the filling station
            self.filling_start = self.env.now # Save starting filling time
            yield self.env.timeout(self.filling_time*0.5) # Fill the sub-batch
            yield boxes.get(amount_boxes//2) # Make busy the rest of the needed boxes
            self.waiting_time_due_to_boxes += self.env.now - (self.filling_start + self.filling_time/2)
            yield self.env.timeout(self.filling_time*0.5) # Fill the second half of the sub-batch
            self.filling_end = self.env.now # Save ending filling time
            filling_station.idle_time += self.waiting_time_due_to_boxes
            yield queue_fill_ster.get(1)
        
        # ************** Entity enters sterilization process **************
        
        with sterilization_station.request() as request:
            yield request # Request the sterilization station
            yield queue_fill_ster.put(1)
            self.waiting_time_due_to_boxes += self.env.now - self.filling_end
            # Generate a new sub-batch at the start of the filling phase
            next_batch = self.batch_number + 1 if self.sub_batch == 3 else self.batch_number
            next_sub_batch = 1 if self.sub_batch == 3 else self.sub_batch + 1
            Entity(self.env, next_batch, next_sub_batch, filling_station, sterilization_station, decasing_station, boxes, queue_fill_ster, queue_ster_decas)
            
            self.sterilization_start = format_time(self.env.now) # Save starting time
            yield self.env.timeout(self.sterilization_time) # Sterilize the sub-batch
            self.sterilization_end = self.env.now # Save ending time
            if queue_ster_decas.level > MAX_QUEUE: # Physical space of maximum of trolley that can wait in the sterilization queue
                yield self.env.timeout(1/60/2) # wait 30 seconds
            self.waiting_time_due_to_queue += self.env.now - self.sterilization_end
            filling_station.idle_time += self.waiting_time_due_to_queue
            # Put the sub_batch in the queue
            yield queue_ster_decas.get(1)
        
        # ************** Entity is queuing before decasing **************
        ## Queue managed by the shared resource queue_ster_decas
        # ************** Entity enters decasing process **************
        
        with decasing_station.request(1) as request: # Priority meant to consider the weekly cleaning
            yield request # Request the decasing station
            yield queue_ster_decas.put(1)
            # Checking the working hour of the decasing station
            current_time, start_time, end_time = self.env.now % 24, 6, 22
            if current_time < start_time or current_time > end_time:
                waiting_time = start_time + 24 - current_time if current_time >= end_time else start_time - current_time
                yield self.env.timeout(waiting_time) # Wait until the decasing is working
            current_time = self.env.now % 24
            self.decasing_start = format_time(self.env.now)
            if current_time + self.decasing_time > 22 and current_time + self.decasing_time/2 < 22: # Case 1: overlap, half decased before 22
                yield self.env.timeout(self.decasing_time/2) # Decase half
                yield boxes.put(amount_boxes//2) # Release half of the boxes
                yield self.env.timeout(22 - (current_time + self.decasing_time/2)) # Decase until 22
                yield self.env.process(self.wait_the_night()) # Wait the night
                yield self.env.timeout(self.decasing_time - (22 - current_time)) # Decase rest
                yield boxes.put(amount_boxes//2) # Release half of the boxes
            elif current_time + self.decasing_time > 22 and current_time + self.decasing_time/2 > 22: # Case 2: overlap, half decased after 6
                yield self.env.timeout(22 - current_time) # Decase until 22
                yield self.env.process(self.wait_the_night()) # Wait the night
                yield self.env.timeout(self.decasing_time/2 - (22 - current_time)) # Decase until half
                yield boxes.put(amount_boxes//2) # Release half of the boxes
                yield self.env.timeout(self.decasing_time/2) # Decase rest
                yield boxes.put(amount_boxes//2) # Release half of the boxes
            else: # Case 3: no overlap, decasing during the day
                yield self.env.timeout(self.decasing_time/2) # Decase half
                yield boxes.put(amount_boxes//2) # Release half of the boxes
                yield self.env.timeout(self.decasing_time/2) # Decase rest
                yield boxes.put(amount_boxes//2) # Release half of the boxes
            self.decasing_end = format_time(self.env.now) # Save time
        
        # --------------- Data compiling ----------------------
        
        if DATA_COLLECTION:
            # Add data for the gantt
            data_gantt.append(dict(Task=self.name, Start=format_time(self.filling_start), Finish=format_time(self.filling_end), Resource="Filling"))
            data_gantt.append(dict(Task=self.name, Start=self.sterilization_start, Finish=format_time(self.sterilization_end), Resource="Sterilization"))
            if self.decasing_overlap == True:
                data_gantt.append(dict(Task=self.name, Start=self.decasing_start, Finish=self.decasing_first_end, Resource="Decasing"))
                data_gantt.append(dict(Task=self.name, Start=self.decasing_second_start, Finish=self.decasing_end, Resource="Decasing"))
            else:
                data_gantt.append(dict(Task=self.name, Start=self.decasing_start, Finish=self.decasing_end, Resource="Decasing"))
        
        data_idle_time_filling.append((self.name,self.waiting_time_due_to_boxes+self.waiting_time_due_to_queue))
        global data_batch_produced
        data_batch_produced += 1/3
        
class Filling_station(simpy.Resource):
    def __init__(self, env, capacity):
        super().__init__(env, capacity=capacity)
        self.idle_time = 0
        self.clean_type = "Complete" # or "Reduced
        
    def request(self, priority, *args, **kwargs):
        # Check if cleaning is needed
        if priority == 1:
            self._env.process(self.clean())
        return super().request(*args, **kwargs)
            
    def release(self, *args, **kwargs):
        return super().release(*args, **kwargs)
    
    def clean(self):
        with self.request(priority=-1) as req:
            yield req
            self.clean_type = "Reduced" if self.clean_type == "Complete" else "Complete"
            if self.clean_type == "Complete":
                self.cleaning_duration = self.compute_cleaning_duration()
                yield self._env.timeout(self.cleaning_duration)
            elif self.clean_type == "Reduced":
                yield self._env.timeout(0.5)

    def compute_cleaning_duration(self):
        # min_cleaning_filling, max_cleaning_filling = 2.5, 6
        # avg_cleaning_filling, std_cleaning_filling = 3.8, 1.5
        # return min(max_cleaning_filling,
        #             max(min_cleaning_filling,
        #                 random.gauss(avg_cleaning_filling, std_cleaning_filling)))
        return 3.4

class Decasing_station(simpy.Resource):
    def __init__(self, env, capacity):
        super().__init__(env, capacity=capacity)
        self.cleaning_duration = 4.53
        self.need_to_clean = False
        self.toogle = True
        self.idle_time = 0
        
    def request(self, priority, *args, **kwargs):
        # Check if cleaning is needed on tuesdays
        if priority != -1:
            if self._env.now // 24 % 7 + 1 == 1 and self.toogle == True: # Be on Monday
                self.need_to_clean = True # Toogle the need to be cleaned
                self.toogle = False
            elif self._env.now // 24 % 7 + 1 == 2: # Be on Tuesday
                if self.need_to_clean == True: # Need to be cleaned
                    self.need_to_clean = False
                    self.toogle = True
                    # self._env.process(self.clean()) # Clean
        
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        return super().release(*args, **kwargs)

    def clean(self):
        with self.request(priority=-1) as req:
            yield req
            yield self._env.timeout(self.cleaning_duration)

#%% Support Functions

def filling_time(sub_batch):
    if sub_batch == 1:
        return min(PARAM['A_FILLING_TIME_MAX'], max(PARAM['A_FILLING_TIME_MIN'], random.gauss(PARAM['A_FILLING_TIME_AVG'], PARAM['A_FILLING_TIME_STD'])))
    elif sub_batch == 2:
        return min(PARAM['B_FILLING_TIME_MAX'], max(PARAM['B_FILLING_TIME_MIN'], random.gauss(PARAM['B_FILLING_TIME_AVG'], PARAM['B_FILLING_TIME_STD'])))
    elif sub_batch == 3:
        return min(PARAM['C_DECASING_TIME_MAX'], max(PARAM['C_FILLING_TIME_MIN'], random.gauss(PARAM['C_FILLING_TIME_AVG'], PARAM['C_FILLING_TIME_STD'])))

def decasing_time(sub_batch):
    if sub_batch == 1:
        return min(PARAM['A_DECASING_TIME_MAX'], max(PARAM['A_DECASING_TIME_MIN'], random.gauss(PARAM['A_DECASING_TIME_AVG'], PARAM['A_DECASING_TIME_STD'])))
    elif sub_batch == 2:
        return min(PARAM['B_DECASING_TIME_MAX'], max(PARAM['B_DECASING_TIME_MIN'], random.gauss(PARAM['B_DECASING_TIME_AVG'], PARAM['B_DECASING_TIME_STD'])))
    elif sub_batch == 3:
        return min(PARAM['C_DECASING_TIME_MAX'], max(PARAM['C_DECASING_TIME_MIN'], random.gauss(PARAM['C_DECASING_TIME_AVG'], PARAM['C_DECASING_TIME_STD'])))

def format_time(now):
    return datetime.fromtimestamp(TODAY.timestamp() + now * 3600)

def patch_resource(resource, pre=None, post=None):
     def get_wrapper(func):
         # Generate a wrapper for put/get/request/release
         @wraps(func)
         def wrapper(*args, **kwargs):
             if pre:
                 pre(resource)
             ret = func(*args, **kwargs)
             if post:
                 post(resource)
             return ret
         return wrapper
     # Replace the original operations with our wrapper
     for name in ['put', 'get', 'request', 'release']:
         if hasattr(resource, name):
             setattr(resource, name, get_wrapper(getattr(resource, name)))

def monitor_level_resource(data, resource):
    data.append((format_time(resource._env.now), resource.level))
    
def avg_list_timedelta(liste):
    res = liste[0]
    for x in liste[1:]:
        res += x
    return res / (len(liste) + 1)

def flatten_list_2D(liste2D):
    res = []
    for x in liste2D:
        for y in x:
            res.append(y)
    return res
    
def gen_state():
    return 'OEE filling {}, OEE decasing {}, boxes {}'.format(OEE_FILLING, OEE_DECASING, NUM_BOXES)

def plot_gantt():
    title = gen_state()
    df1 = pd.DataFrame(data_gantt)
    # fig = px.timeline(df1, x_start="Start", x_end="Finish", y="Resource", color="Task")
    # fig = px.timeline(df1, x_start="Start", x_end="Finish", y="Resource", color="Resource",title=title)
    fig1 = px.timeline(df1, x_start="Start", x_end="Finish", y="Task", color="Resource",title=title)
    plot(fig1)

def plot_level_container(data, title_suffix=""):
    title = gen_state()
    fig2 = px.line(data, x='date', y='level',title=title+title_suffix)
    plot(fig2)
    
def plot_hist_time(df, col):
    plt.hist(df[col], bins=30)
    plt.title(label=gen_state())
    plt.show()
    
def save_df(df):
    date_hour_str = datetime.now().strftime('%Y%m%d_%H%M%S') # Get the current date and hour
    filename = f'./output/output_{date_hour_str}.xlsx' # Create the filename with the date and hour
    df.to_excel(filename, index=False) # Save the DataFrame to an excel file with the date and hour in the filename

#%% Core Functions
def simulation():
    
    global data_boxes
    global data_queue_fill_ster
    global data_queue_ster_decas
    global data_idle_time_filling
    # Create an environment and start the simulation
    env = simpy.Environment()
    
    # Create shared resources
    filling_station = Filling_station(env, capacity=1)
    sterilization_station = simpy.Resource(env, capacity=1)
    decasing_station = Decasing_station(env, capacity=1)
    boxes = simpy.Container(env, init=NUM_BOXES, capacity=NUM_BOXES)
    queue_fill_ster = simpy.Container(env, init=1, capacity=1)
    queue_ster_decas = simpy.Container(env, init=MAX_QUEUE, capacity=MAX_QUEUE)
    # Patching resources    
    monitor_boxes = partial(monitor_level_resource, data_boxes)
    monitor_queue_ster = partial(monitor_level_resource, data_queue_fill_ster)
    monitor_queue_decas = partial(monitor_level_resource, data_queue_ster_decas)
    patch_resource(boxes, pre= monitor_boxes, post=monitor_boxes)
    patch_resource(queue_fill_ster, pre=monitor_queue_ster, post=monitor_queue_ster)
    patch_resource(queue_ster_decas, pre=monitor_queue_decas, post=monitor_queue_decas)
    
    # Generate first entity
    Entity(env, 1, 1, filling_station, sterilization_station, decasing_station, boxes, queue_fill_ster, queue_ster_decas)
    
    # Run the simulation for 24 hours
    env.run(until=24*DAYS_OF_SIMULATION)

    data_boxes = pd.DataFrame(data_boxes, columns=['date', 'level'])
    data_queue_fill_ster = pd.DataFrame(data_queue_fill_ster, columns=['date', 'level'])
    data_queue_ster_decas = pd.DataFrame(data_queue_ster_decas, columns=['date', 'level'])
    data_idle_time_filling = pd.DataFrame(data_idle_time_filling, columns=['Sub-batch', 'idle time'])


def compile_results():
    res = {} # Init
    nb_wait = len(data_idle_time_filling) - [x[1] for x in data_idle_time_filling].count(0)
    tot_idle_time = np.sum(np.array([x[1] for x in data_idle_time_filling]))
    res['nb_wait'] = nb_wait
    res['tot_waiting_time_before_filling'] = tot_idle_time
    if nb_wait != 0:
        res['avg_waiting_time_before_filling'] = tot_idle_time / nb_wait
    else:
        res['avg_waiting_time_before_filling'] = 0
    if DATA_COLLECTION:
        res['waiting_time_before_filling'] = data_idle_time_filling
    res['avg_queue'] = data_queue #avg_list_timedelta(data_queue).total_seconds() / 24 # Transform into decimal hours
    res['batch_produced'] = data_batch_produced
    return res

def MCS(N, verbose = False):
    # Initialisation of the output variables
    avg_nb_wait = 0
    avg_tot_waiting_time_before_filling = 0
    avg_waiting_time_before_filling = 0
    avg_queue = 0
    avg_batch_produced = 0
    
    for i in range(N):
        # Reinitialisation of the global variables
        global data_waiting_time_before_filling
        global data_queue
        global data_batch_produced
        data_waiting_time_before_filling = []
        data_queue = []
        data_batch_produced = 0
        
        simulation()
        
        results = compile_results()
        
        avg_nb_wait += results['nb_wait']/N
        avg_tot_waiting_time_before_filling += results['tot_waiting_time_before_filling']/N
        avg_waiting_time_before_filling += results['avg_waiting_time_before_filling']/N
        avg_queue += results['avg_queue']/N
        avg_batch_produced += results['batch_produced']/N

    return [OEE_FILLING, OEE_DECASING, NUM_BOXES,avg_batch_produced,avg_nb_wait,avg_tot_waiting_time_before_filling,avg_waiting_time_before_filling,avg_queue]

def produce_results(number_replicates = 1, list_OEE = [0.3,0.45], list_prod = [0,10], list_boxes = [450,550,650,750,850,950]):
    global OEE_FILLING
    global OEE_DECASING
    global NUM_BOXES
    data = []
    for rep in range(number_replicates):
        for OEE_FILLING in list_OEE:
            for OEE_DECASING in list_prod:
                for NUM_BOXES in list_boxes:
                    param_settings()
                    L = MCS(143, False)
                    L.insert(0,rep)
                    data.append(L)
                    print('MCS', 'rep {}'.format(rep), gen_state())
    res = pd.DataFrame(data, columns=['Replication','OEE filling', 'OEE decasing', 'Number of boxes', 
                                      'Number of batch produced', 'Number of waits', 'Average total waiting', 'Average waiting', 'Average queue']) #, 'Data waiting time'])
    return res

#%% Start the timer
starting_time = datetime.now()

#%% One shot simulation

## Simulation
simulation()
# results = compile_results()

## Visualisation
# pio.renderers.default='browser' # Render the graphs in the browser
pio.renderers.default='svg' # Render the graphs in the spyder

#%% Plot section
if PLOT:
    plot_gantt()
    plot_level_container(data_boxes, title_suffix=" level of the boxes")
    plot_level_container(data_queue_fill_ster, title_suffix=" level of the queue fill/ster")
    plot_level_container(data_queue_ster_decas, title_suffix=" level of the queue ster/decas")
    print("Total filling idle time:", data_idle_time_filling['idle time'].sum())
tot_idle_time = data_idle_time_filling['idle time'].sum()
#%% Number of run computation
def compute_number_of_run():
    N = 10
    alpha = 0.05
    plt.figure()
    x, y = [], []
    list_output = []
    while True and N < 200:
        for i in range(N):
            # Reinit global variables
            global data_batch_produced 
            global data_boxes 
            global data_queue_fill_ster 
            global data_queue_ster_decas 
            global data_idle_time_filling 
            data_batch_produced = 0
            data_boxes = []
            data_queue_fill_ster = []
            data_queue_ster_decas = []
            data_idle_time_filling = []
            
            simulation()
            # Output is the nb_wait
            tot_idle_time = data_idle_time_filling['idle time'].sum()
            list_output.append(tot_idle_time)
        s = statistics.variance(list_output)
        quantile = scipy.stats.t.ppf(1-alpha/2, N-1)
        c = quantile*(s/N)**0.5
    
        x.append(N)
        y.append(statistics.mean(list_output))
        if c <= alpha*statistics.mean(list_output):
            break
        else:
            N = N + 1
        print("N: %d, c: %.4f || COMP: %.4f" % (N, c, alpha*statistics.mean(list_output)))
    plt.plot(x, y)
    plt.ylabel('Average output value')
    plt.xlabel("Number of runs")
    plt.title('{}, N: {} for int conf={}%'.format(gen_state(), N, (1-alpha)*100))
    return N, list_output

compute_number_of_run()
#%% Monte-Carlo simulation
# df = produce_results(2,[0.3,0.45],[0,5,10,15],[1,2,3,4,0])
# df = produce_results(1,[0.3],[0],[0], [450])
#%% save results
# save_df(df.iloc[:,:])


#%% Calculate the elapsed time
print("Execution time: {} minutes".format(int((datetime.now() - starting_time).total_seconds()/60*100)/100))
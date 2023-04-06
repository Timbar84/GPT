#%% Importation

import random
import numpy
import simpy
import pandas
import matplotlib.pyplot as plt
import statistics
import scipy.stats

#%% Simulation Parameters

BOOKING = False
# BOOKING = True
# DISPATCH = "FIFO" 
DISPATCH = "Detention fees"

#%% Parameters

# Trucks arrival parameters
UNIF_PARAM = 12.03 # Trucks per hours
POIS_PARAM_HOUR_1 = 17.017 # 06 am – 07 am Poisson ∼ (17.017)
POIS_PARAM_HOUR_2 = 18.194 # 07 am – 08 am Poisson ∼ (18.194)
POIS_PARAM_HOUR_3 = 17.518 # 08 am – 09 am Poisson ∼ (17.518)
POIS_PARAM_HOUR_4 = 16.588 # 09 am – 10 am Poisson ∼ (16.588)
POIS_PARAM_HOUR_5 = 17.448 # 10 am – 11 am Poisson ∼ (17.448)
POIS_PARAM_HOUR_6 = 17.034 # 11 am – 12 am Poisson ∼ (17.034)
POIS_PARAM_HOUR_7 = 17.077 # 12 pm – 13 pm Poisson ∼ (17.077)
POIS_PARAM_HOUR_8 = 16.881 # 13 pm – 14 pm Poisson ∼ (16.881)
POIS_PARAM_HOUR_9 = 15.804 # 14 pm – 15 pm Poisson ∼ (15.804)

NUM_DOCK = 13 # Number of docks
MEAN_PROC_TIME = 47.33 # Mean processing time in minutes
DETENTION_FEE = 40/60 # €/min
CONTRACTUAL_FREE_TIME = 120 # Min
SIM_TIME = 60*9 # 9 Hours of work

# Scenari parameters
TRIANGULAR_TYPE = 'Forte'

#%% Classes

class Truck():
    
    def __init__(self, env, name, docks, verbose=True):
        self.verbose = verbose
        self.env = env
        self.name = name
        self.docks = docks
        self.arrival_time = self.env.now
        self.unloading_time = random.triangular(30, 100, MEAN_PROC_TIME)
        # The unloading_time is computed at the beginning for the dynamic dispatch
        self.in_system = True
        
        # Initializing variable attributes
        self.departure_time = 0
        self.detention_fee = 0
        self.priority = 0
        self.history = []
        
    def disp(self, msg):
        """Prints a msg that is a string if verbose = True and saves it in the history list."""
        self.history.append(msg)
        if self.verbose: print(msg)
        
    def process(self):
        self.disp('%s arrives and asks a dock at %.2f.' % (self.name, self.env.now))
        
        # Conditionnal computation of the initial priority of the truck 
        if DISPATCH == "FIFO":
            self.priority = 0
        elif DISPATCH == "Detention fees":
            current_waiting_time = self.env.now - self.arrival_time 
            remaining_free_time = CONTRACTUAL_FREE_TIME - current_waiting_time
            self.priority = remaining_free_time - self.unloading_time
            
        # Requesting a dock from the Docks
        with self.docks.request(priority = self.priority) as request:
            yield request # Trucks get assigned a dock
            
            self.disp('%s starts unloading at the dock at %.2f.' % (self.name, self.env.now))
            yield self.env.timeout(self.unloading_time) # Truck gets unloaded
            
        
        # Truck exits the system
        self.in_system = False
        self.departure_time = self.env.now
        
        # Computation of the detention fees
        self.detention_fee = max(0, (self.departure_time - self.arrival_time - CONTRACTUAL_FREE_TIME)*DETENTION_FEE)
        self.disp('%s leaves the system at %.2f, produced %.2f fees and spent %.2f min in the system' % (self.name, self.env.now, self.detention_fee, self.departure_time-self.arrival_time))
        
        # Remove the non-picklable attributes
        delattr(self,"env")
        delattr(self,"docks")
        

class Dock(simpy.PriorityResource):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dic_PQueue = dict() # Dictionnary of the request events pending in the queue of the resource
        # Structure of the distionary: 
        # [key]: (value)
        # REQUEST_ADRESSE: (INITIAL_REQUEST_TIME, CURRENT_PRIORITY)
        self.idle_time = 0 # Initializing the idle of all docks
        self.time_previous_interaction = 0 # Initializing the time of the previous interaction
        # The idle time is computed as
        # number of free docks * time since the last interaction
        
    def request(self, priority):
        # Update the idle time
        self.idle_time += (self.capacity - len(self.users)) * (self._env.now - self.time_previous_interaction)

        # Update the time of the previous interaction with the time of the current interaction
        self.time_previous_interaction = self._env.now

        # Request the resource
        req = super().request(priority)
        
        if DISPATCH == "Detention fees":
            # Remove the items from the dictionary if they use the resource, therefore they are not in the queue
            for req_adr in [hex(id(x)) for x in self.users]: # Loop on the users of the resource
                if req_adr in self.dic_PQueue.keys(): # If they are in the distionary of the queue
                    self.dic_PQueue.pop(req_adr) # Remove the item from the dictionary 
            
            # Add the request with data if the request got into the queue
            if len(self.put_queue) != 0: # If the queue is not empty
                for pending_req in self.queue: # Loop on the request in the queue
                    if not hex(id(pending_req)) in self.dic_PQueue.keys(): # Check if it is not in the dict
                        self.dic_PQueue[hex(id(pending_req))] = self._env.now, priority # Create a new item
            
            # Update the priorities dynamically
            self.update_priorities()
        
        return req

    def release(self, req):
        # Update the idle time
        self.idle_time += (self.capacity - len(self.users)) * (self._env.now - self.time_previous_interaction )

        # Update the time of the previous interaction with the time of the current interaction
        self.time_previous_interaction = self._env.now 

        if DISPATCH == "Detention fees":
            # Remove the item from the dictionary if it got in the users
            for req_adr in [hex(id(x)) for x in self.users]:
                if req_adr in self.dic_PQueue.keys():
                    self.dic_PQueue.pop(req_adr)
                    
            # Remove the item from the dictionary if it is the one getting released
            if hex(id(req)) in self.dic_PQueue.keys():
                self.dic_PQueue.pop(hex(id(req)))
                
            # Update the priorities dynamically
            self.update_priorities()
        
        # Release the resource from the request
        return super().release(req)

    def update_priorities(self):
        for pending_req in self.put_queue: # Loop on the request in the queue
            # if hex(id(pending_req)) in self.dic_PQueue.keys(): 
            """ Score(T1) = Score(T0) + T0 - T1 /// with T1 > T0 """
            pending_req.priority = self.dic_PQueue[hex(id(pending_req))][1] + self.dic_PQueue[hex(id(pending_req))][0] - self._env.now
            # pending_req.time = self._env.now
        
        # Sort the queue by their priority attribute
        self.queue.sort(key= lambda x: (x.priority))

#%% Functions

def setup(env, docks, list_truck, verbose=True):
    """Generator of the trucks."""

    # Initializing the index of the trucks
    i = 0 
    
    # Create 1 initial trucks
    list_truck.append(Truck(env, 'Truck %d' % i, docks, verbose)) # Create a truck
    env.process(list_truck[0].process()) # Process the truck

    # Create more trucks while the simulation is running
    while True:
        if BOOKING == True:
            if env.now < SIM_TIME:
                yield env.timeout(60/random.uniform(UNIF_PARAM-2,UNIF_PARAM+2))
            else:
                return
        elif BOOKING == False:
            if env.now < 60: # Hour 1
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_1))               
            elif 60 <= env.now < 120: # Hour 2
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_2))               
            elif 120 <= env.now < 180: # Hour 3
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_3))
            elif 180 <= env.now < 240: # Hour 4
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_4))
            elif 240 <= env.now < 300: # Hour 5
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_5))
            elif 300 <= env.now < 360: # Hour 6
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_6))
            elif 360 <= env.now < 420: # Hour 7
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_7))
            elif 420 <= env.now < 480: # Hour 8
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_8))
            elif 480 <= env.now < 540: # Hour 9
                yield env.timeout(60/numpy.random.poisson(POIS_PARAM_HOUR_9))
            else:
                return 
        elif BOOKING == 'Triangular':
            if env.now < SIM_TIME:
                if TRIANGULAR_TYPE == 'Base':
                    yield env.timeout(60/random.triangular(10,14,12))
                elif TRIANGULAR_TYPE == 'Forte':
                    yield env.timeout(60/random.triangular(8,16,12))
            else:
                return
            
        i += 1# Increment the index of the trucks
        list_truck.append(Truck(env, 'Truck %d' % i, docks, verbose)) # Create a truck
        env.process(list_truck[-1].process()) # Process the truck

def simulation(verbose=False):
    
    # Initialize the list of truck
    list_truck = []
    
    # Create an environment and start the setup process
    env = simpy.Environment()
    
    # Create the docks
    docks = Dock(env, NUM_DOCK)
    
    # Setup the generator of trucks
    env.process(setup(env, docks, list_truck, verbose))
    
    # Run the simulation until SIM_TIME
    env.run(until=SIM_TIME + 60*2)
    
    return docks, list_truck

def compute_detention_fees(truck_list):
    Detention_fees = 0
    for truck in truck_list:
        Detention_fees += truck.detention_fee
    return Detention_fees

#%% Main

dock, list_truck = simulation()
fee = compute_detention_fees(list_truck)

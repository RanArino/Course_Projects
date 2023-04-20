import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from datetime import *
from scipy.stats import *
from distfit import distfit
# import python file
import math_help

# assuming we observe 200m*200m area; center is (0, 0)
MONITOR_AREA = 200

# Traffic Light Time (seconds)
#  this number is taken into account for the yellow time (3s) and change interval (2s) at each time
TL_TIME = 45

# Roundabout
#  this number indicates that if a car's speed is more than 20km/h during rounding, stop accelerating
#  any integer should be assigned here. (used in line 927) 
LIMIT_ROUNDING_SPEED = 20

# car's speed
CAR_SPEED = 55.  # Base of car max speed; 55km/h; assigned 50~60km/h
CAR_ACC = 6.  # Base of car acceleration speed; 6km/h per second; assigned 4~6km/h
CAR_DCL = {'normal': -10, 'stronger': -15, 'maximum': -20}  # Bases of car deceleration rate; different among people

# Car's Spaces (assuming the spaces between car's central points)
#  it may be wider than real-situation, but we need to take into account for car size (around 2m)
CAR_SPACE = {"idling": 4.0, "others": 7.0}

# driver's reaction start distance (this is base)
# "straight": drivers carefully monitor the front car who is within this distance
#               however, the lower speed of a front car, the earlier the driver reaction
# "blinker": drivers start showing blinker sign within this distance
#            As for the roundabout, "Right" blinker is shown 15m before leaving the rounding area
# "turn"  : driver cannot turn if other cars are within this range;
#             however, it depends on the others' current speed and active mode
# "traffic_light": drivers start braking from this distance when a traffic light is yellow or red
REACT_DIST = {"straight": 25, 'blinker': 15, 'turn': 30, "traffic_light": 40}

# Gas pollution (g/second), which are referred from the project document
GAS_POLLUTION = {"driving": 2.5, "accelerating": 5, "idling": 1, "braking": 1}

# DICT FOR CONVERTING
# tracking car's behavior by color
ACTIVE_TO_COLOR = {'driving': 'green', 'braking': 'red', 'accelerating': 'yellow', 'idling': 'dimgrey'}
# direction to xy_vector; this is the actual direction on the x-y canvas
DIR_TO_XY_VECTOR = {'NB': (0, 1), 'SB': (0, -1), 'EB': (1, 0), 'WB': (-1, 0)}
# direction & lane to initial position
INITIAL_XY_POS = {
    "NB": {'Left': (3., -100.), 'Right': (9., -100.)},
    "SB": {'Left': (-3., 100.), 'Right': (-9., 100.)},
    "EB": {'Left': (-100., -3.), 'Right': (-100., -9.)},
    "WB": {'Left': (100., 3.), 'Right': (100., 9.)}
}
# the stop line before the pedestrian road
FIRST_STOP_DICT = {  # by simulation type, lane, direction
    'TL': {
        'Left': {'NB': (3, -20), 'SB': (-3, 20), 'EB': (-20, 3), 'WB': (20, -3)},
        'Right': {'NB': (9, -20), 'SB': (-9, 20), 'EB': (-20, 3), 'WB': (20, -3)}
    },
    'RA': {
        'Left': {'NB': (3, -31), 'SB': (-3, 31), 'EB': (-31, 3), 'WB': (31, -3)},
        'Right': {'NB': (9, -31), 'SB': (-9, 31), 'EB': (-31, 3), 'WB': (31, -3)}
    }
}
# when a car will turn, which direction and lane driver should watch out (only in "TL") by turn and direction
TURN_CHECK_DICT = {
    # those who go straight don't have to care for their sides
    'Straight': {
        direction: None for direction in ['NB', 'SB', 'EB', 'WB']},
    'Left': {
        # e.g., a driver (turn 'Left' and heads 'NB') cares for
        #  those who are heading "SB", going "Straight", and locating at "Left" & "Right" lane)
        'NB': {"dir": 'SB', "turn": "Straight", "lane": ["Left", "Right"]},
        'SB': {"dir": 'NB', "turn": "Straight", "lane": ["Left", "Right"]},
        'EB': {"dir": 'WB', "turn": "Straight", "lane": ["Left", "Right"]},
        'WB': {"dir": 'EB', "turn": "Straight", "lane": ["Left", "Right"]}, },
    'Right': {
        'NB': {"dir": 'EB', "turn": "Straight", "lane": ["Right"]},
        'SB': {"dir": 'WB', "turn": "Straight", "lane": ["Right"]},
        'EB': {"dir": 'SB', "turn": "Straight", "lane": ["Right"]},
        'WB': {"dir": 'NB', "turn": "Straight", "lane": ["Right"]}, }
}

# GLOBAL STATISTICS
STAT_WAIT = {}  # total wait time on each car (Definition: the time when car speed is less than 5km/h)
STAT_PASS_TIME = {}  # time to pass through area (from enter to leave road)
STAT_GAS1 = {}  # total gas pollution on each car
STAT_GAS2 = {"driving": 0, "accelerating": 0, "idling": 0, "braking": 0}  # total gas pollution on each car active mode


class Car:
    """
    The organizer of
    - all behaviors of each driver (reaction time/distance for braking and traffic light)
    - each driver's plan (direction and turn)
    - each driver's characteristics (speed, acceleration/deceleration rate, spaces, )
    - basic statistics of each driver (velocity, waiting/passing time, gas pollutions)

    Connection to other classes; Road(), TurnControl(), SpeedControl()
    """
    def __init__(self, id, road, turn, direction, sim):
        self.id = id  # id number for each car, based on the entry time on the road
        self.turn = turn
        self.dir = direction
        self.active = False  # either 'driving', 'accelerating', 'braking', 'idling'
        self.turning = False  # whether a car is turning or not; assigned value after cars enter in road
        self.blinker = None  # car's blinker ("Left" or "Right" sign before drivers start turning)

        # initialization
        self.x = None
        self.y = None

        # VARIABLES FOR BRAKING
        self.front = None  # assigned dictionary by SpeedControl; assuming a driver keeps monitoring its car
        self.side_clear = False  # assigned by SpeedControl; whether it's safe enough to start turning
        self.front_car_brake = False  # True if a front car starts braking (not started reacting yet)
        self.gap_time_brake = 0  # track time until reaching a driver's reaction time
        self.realize_front_car_brake = False  # True if a driver starts braking
        self.dist_must_stop = 8  # a driver must stop within 8m after crossing the stop line (for waiting for turning)

        # VARIABLES FOR TRAFFIC LIGHT
        self.traffic_light_stop = False  # True if traffic light is changed to red/yellow (driver hasn't recognized yet)
        self.gap_time_traffic_light = 0  # track time until reaching a driver's reaction time
        self.realize_traffic_light = False  # True if a driver starts braking after realizing the red or yellow.

        # INITIAL DIRECTION/POSITION
        self.xy_vector = DIR_TO_XY_VECTOR[self.dir]  # assign car's xy-vector
        self.lane = "".join([self.turn if self.turn != 'Straight' else random.choice(['Left', 'Right'])])  # lane
        self.initial_pos = INITIAL_XY_POS[self.dir][self.lane]  # starting xy coordinates (initial position)
        self.focus_traffic_light = "NS" if self.dir in ["NB", "SB"] else "EW"  # driver's focusing traffic light

        # DRIVER CHARACTERISTICS
        self.max_speed = CAR_SPEED + random.randint(-5, 5)  # define driver's max speed on each driver (km/h)
        self.current_speed = self.max_speed - random.randint(10, 15)  # initial car speed
        self.acc_rate = CAR_ACC + random.uniform(-1, 1)  # define driver's acceleration speed (km/h)
        self.dcl_rate = {k: v + random.uniform(-1, 1) for k, v in CAR_DCL.items()}  # driver's deceleration rate
        self.car_space = {  # define driver's space from a front car
            "idling": CAR_SPACE["idling"] + random.uniform(-0.5, 0.5),
            "others": CAR_SPACE["others"] + random.uniform(-1, 1)}
        self.react_dist = {k: v + random.randint(-2, 2) for k, v in REACT_DIST.items()}  # driver's reaction distance
        # total reaction time (second), which includes recognition & reaction time
        # e.g., a driver started braking 0.2 (minimum) seconds after a front car started braking
        self.react_time = {k: random.uniform(0.2, 0.8) for k in ['front_brake', 'traffic_light']}

        # CONNECT TO OTHER CLASSES
        # situation of the road (class 'Road')
        self.road = road
        # the organizer of turning/rounding position & blinker
        self.turn_ctl = TurnControl(sim=sim, turn=self.turn, time_step=self.road.time_step)
        # the organizer of car's speed and active mode
        self.speed_ctl = SpeedControl(sim=sim, car=self, time_step=self.road.time_step)

        # OBSERVATION (keeping updating or adding)
        self.entry_time = None
        self.track_speed = []  # car speed km/h
        self.wait_time = 0  # total wait time
        self.pass_time = 0  # total time to pass road (200M*200M)
        self.gas_pollution = 0  # total gas pollution

    def __repr__(self):
        return f"car{self.id}"

    def __str__(self):
        return f"car{self.id} ({self.x}, {self.y}, {self.turn})"

    # Whenever a car attempts to enter road, using this function.
    def enter_road(self, elapsed_time):
        # assign current position as the initial position (but not activated yet technically)
        self.x = round(self.initial_pos[0], 1)
        self.y = round(self.initial_pos[1], 1)
        # if a car can enter from its initial xy position,...
        if self.road.enter_road(self):
            self.active = 'driving'  # a car enter road by driving
            self.turning = self.turn_ctl.check(self)  # True or False
            self.entry_time = elapsed_time
            # print(f"Enter {str(self)} at {self.road.get_clock()}")

    # Whenever a car attempts leave on road, use this function
    def leave_road(self):
        if self.road.leave_road(self):
            self.observation_end()  # make sure to record statistics for each car
            self.active = False  # make sure a car's active False
            # print(f"Leave {str(self)} at {self.road.get_clock()}")

    def update(self, time_step):
        # updating the car's front and sides
        self.speed_ctl.check_front_sides(self)
        # update the current car speed
        self.speed_ctl.adjust_speed(self)
        # make sure that the minimum speed is 0
        self.current_speed = max(self.current_speed, 0)

        # if a car starts turning or rounding (return True)
        #  -> car's direction is not straight, it will be activated to True
        #  -> once it's activated, a behind car can overtake this car
        if self.turn_ctl.check(self):
            # decide the xy position for the next step
            movex, movey = self.turn_ctl.move(self)
            # if car turning completely finish;,
            if self.turn_ctl.turn_done:
                # finalize the updated variables
                self.turn_ctl.finalize(self)
        # if a car never turn or have not turned yet,
        else:
            # a car movement is based on current speed (make sure to convert km/h to m/s)
            velocity = np.array(self.xy_vector) * (self.current_speed * (1000 / 3600))
            movex = self.x + velocity[0] * time_step
            movey = self.y + velocity[1] * time_step

        # finalize the next movement
        self.road.move(self, movex, movey)

    # observations for each time step as long as cars are active
    def observation(self, time_step):
        # velocity
        self.track_speed.append(self.current_speed)
        # wait time (defined as car's speed is less than 5km/h)
        self.wait_time += time_step if self.current_speed <= 5 else 0.
        # passing time on the road (in our case, 200m*200m area)
        self.pass_time += time_step
        # total gas pollution
        self.gas_pollution += GAS_POLLUTION[self.active] * time_step
        # total gas pollution on each active mode
        STAT_GAS2[self.active] += GAS_POLLUTION[self.active] * time_step

    # observations at the time when a car leaves
    def observation_end(self):
        STAT_WAIT[self] = self.wait_time
        STAT_PASS_TIME[self] = self.pass_time
        STAT_GAS1[self] = self.gas_pollution
        self.road.each_car_stat[self] = [self.entry_time, self.wait_time, self.pass_time, self.gas_pollution]


class Road:
    """
    The organizer of
    - all car's entry timing based on the randomly generated car entry time and leave timing
    - the current positions for all active cars on the road
    - running the simulation step to update and observe car's behaviors on each step
    - tracking the traffic light signal changes
    - tracking the simulation time and the current time
    """
    def __init__(self, sim, time_step, car_generation, traffic_light_ctl=None):
        self.car_id_track = 0  # tracking the car id
        self.sim = sim  # simulation type
        self.time_step = time_step  # time step of each simulation
        self.car_generation = car_generation  # CarGeneration class
        self.car_entry_dic = {}  # car's generation time assigned by "organize_entry_time()" function in CarGeneration
        self.traffic_light_ctl = traffic_light_ctl  # TrafficLightControl class
        self.traffic_light_color = {}  # assigned by TrafficLightControl class; key: "NS" or "EW", value: light color
        self.clock = datetime.strptime('00:00 AM', '%H:%M %p')
        self.map = dict()  # tracking the positions of all cars; key: Car class, value: position (x,y)
        self.waited_enter_lst = []  # list of car class which is waiting to enter on the road.
        # list of cars who are waiting for turning around a further area of the stop line (assigned by SpeedControl)
        #  each key shows the car's initial direction
        #  for example, NB: [car1] -> car1 is waiting for the South intersection (because of heading NB initially)
        self.waiting_turn_dic = {key: {"Left": [], "Right": []} for key in ['NB', 'SB', 'EB', 'WB']}

        # statistics of each time
        self.num_car = []
        self.total_wait_car = []
        self.avg_car_speed = []
        self.total_gas = []

        # statistics of each car
        # e.g., car0: [entry_time, wait_time, pass_time, gas_amount]
        self.each_car_stat = {}

    def get_clock(self):
        return f"{self.clock:%I:%M:%S %p}"

    # Cars can enter the road if there are no other cars within the minimum front distance
    def enter_road(self, car):
        # check the car's front
        car.speed_ctl.check_front_sides(car)
        # if there is another car in the front within the maintained car space
        if car.front['dist'] <= car.car_space["others"]:
            # add list (whenever it's available, enter a car)
            self.waited_enter_lst.append(car)
            return False
        # if no cars within front distance, enter car on road
        else:
            self.map[car] = (round(car.x, 1), round(car.y, 1))
            return True

    # Cars will leave road when they reached above the 200m*200m area
    def leave_road(self, car):
        # if a car's current position is the outside of the certain length
        if (abs(car.x) > MONITOR_AREA / 2) or (abs(car.y) > MONITOR_AREA / 2):
            # delete a car from the map
            del self.map[car]
            return True
        else:
            return False

    # organizing a car's current position
    def move(self, car, movex, movey):
        # delete a car's position from the map before updating
        del self.map[car]
        # assign new position
        car.x = round(movex, 1)
        car.y = round(movey, 1)
        self.map[car] = (car.x, car.y)

    # just getting the active cars
    def get_active_cars(self):
        return list(self.map.keys())

    # running the simulation step
    # (1): Cars are attempted to enter the road, define car generation time is less than elapsed_time
    # (2): Waiting cars are attempted to enter the road whenever possible
    # (3): Running the "update()" function of the Car class each time
    # (4): Tracking the statistics by running "observation()" of the Car class each time
    # (5): Controlling each car's leave from the road
    def run_step(self, elapsed_time):
        # get all keys(defined car's entry times) if it's less than elapsed_time
        keys = [key for key in self.car_entry_dic.keys() if key < elapsed_time]
        for k in keys:
            # get value for each key; make sure that it's list type
            value = self.car_entry_dic[k]
            for v in value:
                # attempt to enter car
                car = Car(self.car_id_track, self, v['turn'], v['dir'], self.sim)
                car.enter_road(elapsed_time)
                self.car_id_track += 1
            # remove keys once traveling all items
            del self.car_entry_dic[k]

        # update the traffic light
        if self.sim == "TL":
            self.traffic_light_ctl.update()
            # decide signal colors for each traffic light
            self.traffic_light_color = self.traffic_light_ctl.observe()

        # attempt to enter waiting cars into the road
        for car in self.waited_enter_lst:
            self.waited_enter_lst.remove(car)
            car.enter_road(elapsed_time)

        # update & observe all cars on the road
        total_active_car = 0
        total_waiting_car = 0
        total_car_speed = 0
        total_gas_amount = 0
        for car in self.get_active_cars():
            car.update(self.time_step)
            car.observation(self.time_step)
            car.leave_road()
            if car.active != False:
                total_active_car += 1
                total_waiting_car += 1 if car.current_speed <= 5 else 0
                total_car_speed += car.current_speed
                total_gas_amount += GAS_POLLUTION[car.active] * self.time_step

        self.num_car += [total_active_car]
        self.total_wait_car += [total_waiting_car]
        self.avg_car_speed += [total_car_speed / total_active_car] if total_active_car != 0 else [0]
        self.total_gas += [total_gas_amount]


class CarGeneration:
    """
    The organizer of
    - Creating the assumption of the car's generation time in the real world from the provided data file.
        based on the number of entry car's on each hour, create car's entry time list segmented by turn and direction.
    - Finding the best-fitting distribution from some options and appropriate parameters used by distfit() module.
        based on acquired distribution, creating randomly generated car's entry time (IIDs) on each turn & direction
    - Comparing the assumed real car entry data and the randomly generated car entry time by comparing with histograms.
    - Finally, creating the car's generation time dictionary; key: entry time(int), value {"turn": "", "dir": ""}
        turn and dir shows driver's plan (turning or going straight and which direction a driver moves initially)
    """
    def __init__(self, data_file):
        # the provided Excel file is assumed that the real data
        self.real_data = pd.read_excel(data_file)
        # add new column "Seconds", which is converted from Hours to Seconds
        self.real_data['Seconds'] = self.real_data['Hour'] * 3600
        # unique value of turn column
        self.turn_lst = list(self.real_data['Turn'].unique())
        # unique value of direction column
        self.dir_lst = list(self.real_data['Direction'].unique())

        # this is the nested dictionary to show the car's generation time (seconds) in real
        #  segmented by car's turn and direction
        self.real_car_generation = {k: {} for k in self.turn_lst}
        # this is the randomly generated car entry time
        self.rand_car_generation = {k: {} for k in self.turn_lst}

        # distribution names and scipy.stats function to find a best-matching one on each distribution;
        self.select_dist = {'norm': norm, 'genextreme': genextreme, 'expon': expon, 'gamma': gamma, 'pareto': pareto,
                            'lognorm': lognorm, 'dweibull': dweibull, 'beta': beta, 't': t, 'uniform': uniform}

        # tracking the best-fitting distribution name and appropriate parameters
        self.fit_dist_params = {k: {} for k in self.turn_lst}

    # define the assumption of the car generation time in real
    # the provided data shows the number of entry car in each hour, so creating random numbers noted by seconds
    def initialize(self):
        for i, turn in enumerate(self.turn_lst):
            for j, direction in enumerate(self.dir_lst):
                # subset segmented by turn and direction
                sub_data = self.real_data[
                    (self.real_data.Turn == turn) & (self.real_data.Direction == direction)].reset_index()
                # create car's generation time list
                real_car_time_lst = []
                for count, start in zip(sub_data['Count'], sub_data["Seconds"]):
                    # create random number based on hour and count
                    # note that each min and max number (expressed by seconds) is followed by a range of each hour
                    real_car_time_lst += [float(random.uniform(start, start + 3600)) for _ in range(count)]

                # store the data
                self.real_car_generation[turn].update({direction: real_car_time_lst})

    # fit distribution for each data (segmented by turn and direction
    #  the output figure is the comparison between the real car entry data and the randomly generated entry data
    def fit_distribution(self):
        print("Fitting distributions right now...")
        # prepare for the output figure
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
        bin_range = [i for i in range(0, 86401, 3600)]
        for i, turn in enumerate(self.real_car_generation.keys()):
            for j, direction in enumerate(self.real_car_generation[turn].keys()):
                each_data = np.array(self.real_car_generation[turn][direction])
                # start fitting distribution from our define options, which is the same as distr = "popular"
                #  also turning off the fitting information in this case
                dist = distfit(distr=list(self.select_dist.keys()), verbose=0)
                dist.fit_transform(each_data)
                # tracking the model name and some parameters
                self.fit_dist_params[turn].update({direction: dist.model})
                # generate random number based on generated distribution, parameters
                dist_name = dist.model['name']
                dist_params = dist.model['params']
                # generate the random car enter time (make sure that the data range)
                rand_car_time_lst = self.select_dist[dist_name].rvs(*dist_params, size=len(each_data))
                # add random car generation time
                self.rand_car_generation[turn].update({direction: rand_car_time_lst})
                # real data
                real_car_time_lst = self.real_car_generation[turn][direction]

                # for visualization
                # real data
                axes[i, j].hist(real_car_time_lst, color='r', alpha=0.3, bins=bin_range)
                # randomly generated data based on fitted distribution and params
                axes[i, j].hist(rand_car_time_lst, color='b', alpha=0.5, bins=bin_range)
                axes[i, j].set_xticklabels([])
                # set title
                axes[i, j].set_title(f"{turn} & {direction}\nfit-dist: {dist_name}")
                # basic statistics to show the comparison between random and real
                diff_mean = abs(np.mean(rand_car_time_lst) - np.mean(real_car_time_lst))
                diff_std = abs(np.std(rand_car_time_lst) - np.std(real_car_time_lst))
                axes[i, j].text(0.99, 0.99, f"difference\nmean: {diff_mean:.1f}\nstd: {diff_std:.1f}",
                                fontsize=7.5, ha="right", va="top", transform=axes[i, j].transAxes)

        print("Here are the histograms to compare the randomly generated car and the real car entry data.")
        print("Once you closed the figure, the simulation starts.")
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle('Comparison of Car Enter Time (Random: Blue, Real: Red)')
        plt.show()

    # get the dictionary "car_generation_time"
    #  key: enter time (int): {"turn": str, "dir": str}
    def organize_entry_time(self):
        car_generation_time = {}  # the output
        for turn in self.rand_car_generation.keys():
            for direction in self.rand_car_generation[turn].keys():
                for enter_time in self.rand_car_generation[turn][direction]:
                    # if enter_time is exceeded the range (less than 0 or greater than 86400=24hour),
                    #  reselected randomly, make sure that the time should be close to either start or end time
                    if enter_time < 0:
                        # assigned somewhere within the first 1 hour
                        enter_time = random.randint(0, 3600)
                    elif enter_time > 86400:
                        # assigned somewhere within the last 1 hour
                        enter_time = random.randint(82800, 86400)

                    # if it has the same key
                    if car_generation_time.get(int(enter_time)):
                        # adding items to the list
                        car_generation_time[int(enter_time)] += [{'turn': turn, "dir": direction}]
                    else:
                        # assigned a new item as a list
                        car_generation_time[int(enter_time)] = [{'turn': turn, "dir": direction}]

        return car_generation_time


class TrafficLightControl:
    """
    The organizer of
    - Controlling the signal color on each traffic light (noted by "NS"; north&south and "EW": east&west)
    - Passing the data to show the current signal color on each traffic light
    """

    def __init__(self, time_step):
        self.time_step = time_step
        self.green = {"NS": False, "EW": False}  # show which "NS" or "EW" light is currently 'green' (-> True),
        self.yellow = ""  # show "NS" or "EW" if either one shows yellow signal
        self.next = None  # the light will be Green for the next step
        self.interval = 0  # tracking the time of the traffic light
        self.activeYellow = False
        self.activeAllRed = False

    def initialize(self):
        # randomly assign the green light
        self.green[random.choice(["NS", "EW"])] = True
        self.yellow = ""
        # the next should be the key whose value is minimum (=False < True)
        self.next = min(self.green, key=self.green.get)
        # make sure the time of changing "Yellow" and signal change interval
        self.interval = random.randint(0, TL_TIME-5)
        self.activeYellow = False
        self.activeAllRed = False

    def update(self):
        # if passed TL_TIME-5 seconds & not yet activated the yellow light
        if self.interval >= TL_TIME-5 and not self.activeYellow:
            # change condition
            self.activeYellow = True
            # assign the current green location as yellow
            self.yellow = max(self.green, key=self.green.get)
            self.green[self.yellow] = False
            # show notification
            # print(f"Traffic light in {self.yellow[0]}B & {self.yellow[1]}B turned to Yellow")
        # if passed TL_TIME-2 seconds & still assigned the yellow light
        elif self.interval >= TL_TIME-2 and not self.activeAllRed:
            # show notification
            # print(f"Traffic light in {self.yellow[0]}B & {self.yellow[1]}B turned to Red")
            # change condition
            self.activeAllRed = True
            # reset yellow
            self.yellow = ""
        # if passed TL_TIME seconds
        elif self.interval >= TL_TIME:
            # show notification
            # print(f"Traffic light in {self.next[0]}B & {self.next[1]}B turned to Green")
            # change the next location to green light
            self.green[self.next] = True
            # green light for the next should be the currently Red = False
            self.next = min(self.green, key=self.green.get)
            # reset variables
            self.interval = 0
            self.activeYellow = False
            self.activeAllRed = False

        # keep increasing the time
        self.interval += self.time_step

    def observe(self):
        # if either one location is green
        if True in list(self.green.values()):
            green = max(self.green, key=self.green.get)
            red = min(self.green, key=self.green.get)
            return {green: "green", red: 'red'}

        else:
            # if self.yellow has a variable
            if not self.yellow == "":
                red_idx = 1 - list(self.green.keys()).index(self.yellow)
                return {self.yellow: "yellow", list(self.green.keys())[red_idx]: "red"}
            # otherwise, both should be red
            else:
                return {"NS": 'red', "EW": 'red'}


class SpeedControl:
    """
    The organizer of
    - Car's active modes (braking, accelerating, driving, and idling) based on the current speed(velocity)
    -
    """
    def __init__(self, sim, car, time_step):
        self.sim = sim
        self.time_step = time_step
        self.first_stop = FIRST_STOP_DICT[sim][car.lane][car.dir]  # define the stop line
        # 'turn_check' shows other cars that a driver should look at careful before the start of turning
        self.turn_check = TURN_CHECK_DICT[car.turn][car.dir] if self.sim == "TL" else None
        # car's distance from the first stop to the current position
        #  positive: the length(m) until reaching stop line, negative: the length(m) after reaching stop line
        self.dist_first_stop = np.inf
        # initialize the car space that a driver should maintain, which depends on "idling" or others
        self.maintain_car_space = 0

    # Searching the other cars given the specific conditions
    def search_other_cars(self, car, mode):
        # just searching the front side of other cars considering the distance
        if mode == 'straight':
            # car's position based on either x or y is 0 (the center of the intersection)
            #  each entry point is defined by -100, then the exit point is 100, regardless of direction)
            car_base_pos = np.dot((car.x, car.y), car.xy_vector)
            # key: Car class, values: distance
            other_pos_dist = {}
            # getting all active cars mapping in the road
            for other, pos in car.road.map.items():
                # make sure that other car's condition is matching (same direction and same lane)
                # also make sure that a driver can ignore if a front car has started turning
                #  although cars are apparently crashed in the animation, a car will avoid crash in real world.
                if (other.dir == car.dir) and (other.lane == car.lane) and (not other.turn_ctl.check(other)):
                    # distance between a car and other cars
                    car_other_dist = (np.dot(pos, other.xy_vector)) - car_base_pos
                    # the positive number shows that another car is in the front of car(negative shows behind)
                    # in this case, we want to focus on the front cars
                    if car_other_dist > 0:
                        # assign data to the dictionary
                        other_pos_dist[other] = car_other_dist

            return other_pos_dist

        # searching surroundings when a driver attempts to turn in Traffic Light (TL) simulation
        elif mode == 'TL_turn':
            for other, pos in car.road.map.items():
                # if other cars are matching the conditions (refer to self.turn_check)
                if (other.dir == self.turn_check['dir']) and other.turn == self.turn_check['turn']\
                        and other.lane in self.turn_check['lane']:
                    # other car's position based on either x or y is 0, referred to around code line 550
                    others_base_pos = np.dot(pos, other.xy_vector)

                    # a driver turns Left
                    if car.turn == 'Left':
                        # if others are within a driver's reaction distance for turning...
                        # in the case, a driver observes a further area from the intersection center
                        #  for example, looking at a range of 10m ~ 40m (the intersection center as 0)
                        if (others_base_pos * -1 <= car.react_dist['turn'] + 10) and (others_base_pos < - 10):
                            # if others are idling or braking, a driver will be able to start turning before they pass
                            #  in this case, if other cars are located in 20m further than the intersection center,
                            #  a driver decides to start turning
                            # make sure that uf it has not passed some time (3s) right after changing to "green" signal,
                            #  a driver should start turning, because those who go straight are priority.
                            if (other.active in ['idling', 'braking']) and (others_base_pos < -20) and \
                                    (car.road.traffic_light_ctl.interval >= 3):
                                return False
                            else:
                                return True
                    # a driver turns Right
                    else:
                        # if others are within a driver's reaction distance for turning
                        # note that drivers wait for passing others in a further distance from intersection center
                        # therefore, adjusting this distance by subtracting -5
                        if (others_base_pos * -1 - 5 <= car.react_dist['turn']) and (others_base_pos < 5):
                            # similar to the previous, if others are slow, a driver can turn faster
                            if (other.active == 'idling') and (others_base_pos < -15):
                                return False
                            else:
                                return True

            # if no other cars in within certain areas, return False
            return False

        # searching surroundings when a driver attempts to turn in Roundabout (RA) simulation
        elif mode == 'RA_turn':
            # define the necessary check lane of each car
            # for those who are in "Left" lane have to check both lanes to enter roundabout
            car_check_lane = ['Left', 'Right'] if car.lane == "Left" else ["Right"]
            for other, pos in car.road.map.items():
                # setting the conditions for searching other cars
                cond1 = not (other.dir == car.dir)  # 1: not the same direction as a car
                cond2 = other.lane in car_check_lane  # 2: lane of others is included in a car's necessary check lane
                cond3 = not other.turn_ctl.turn_done  # 3: not finished turning(rounding, in this case)
                cond4 = not (other.blinker == "Right")  # 4: not showing the "Right" blinker
                # if meeting all above conditions
                if cond1 and cond2 and cond3 and cond4:
                    # calculate distance from other cars
                    dist_others = math_help.get_distance((car.x, car.y), pos)
                    # if distance from other cars are within a driver's reaction distance when turning, return True
                    return dist_others <= car.react_dist['turn']

    # update the "car.front" and "car.side_clear" variable
    def check_front_sides(self, car):
        # car's position based on the intersection center, refer to around code lin 550
        car_base_pos = np.dot((car.x, car.y), car.xy_vector)
        # stop line's distance from 0 (focusing on car's changing position)
        stop_dist = np.dot(self.first_stop, car.xy_vector)
        # distance between the first stop line and the current car position
        #  if positive number: a distance(m) remained to reach the stop line.
        #  if negative number: a distance(m) passed from the stop line
        self.dist_first_stop = stop_dist - car_base_pos

        # (1) Define the first conditions
        # (1)-1: if car has never reached the stop line or car will keep going straight in 'TL',
        cond1_1 = ((self.dist_first_stop > 0) or (car.turn == "Straight")) and (self.sim == 'TL')
        # (1)-2: a car has not started turning yet in 'RA'
        cond1_2 = (not car.turn_ctl.check(car)) and (self.sim == "RA")
        if cond1_1 or cond1_2:
            # just checking the car's front
            others = self.search_other_cars(car, mode='straight')
            # get one other car whose distance from a driver is minimum considering driver's reaction distance
            # try/except statement considers that a dictionary "others" is empty
            try:
                other, dist = min(others.items(), key=lambda x: x[1])
                # the slower speed of a front car, the earlier the driver reaction time (base/minimum 10 km/h)
                weight = 0.5 + (10 / max(other.current_speed, 10))
                # if distance from a front car is within the calculated range based on a driver's reaction length
                if dist <= car.react_dist['straight'] * weight:
                    # assigned a front car; a driver carefully monitors this car
                    car.front = {"car": other, 'dist': dist}
                else:
                    car.front = {"car": None, "dist": np.inf}
            except ValueError:
                car.front = {"car": None, "dist": np.inf}

        # (2) Define the second condition (only in 'TL')
        if self.sim == 'TL':
            # if a driver will turn (not Straight) and a driver reached/crossed the stop line,
            #  start checking whether the side is clear (whether there are enough space to start turning)
            if (not car.turn == "Straight") and (self.dist_first_stop <= 0):
                # however, those who turn Left, need to keep checking after a passing a further 7.5m from stop line
                cond2 = (car.turn == "Left") and (self.dist_first_stop <= -7.5)
                # as long as "car.side_clear == False", keeping checking if it is True each time
                #  once getting True, no need to check, because a car will start turning and should accelerate quickly
                if car.side_clear and cond2:
                    car.side_clear = True
                else:
                    car.side_clear = not self.search_other_cars(car, mode='TL_turn')

            # once those who will go straight pass a further 10m from the stop line
            elif (car.turn == "Straight") and (self.dist_first_stop <= -10):
                # assign side clear True due to priority
                car.side_clear = True

        # (3) Define the third conditions (only in 'RA')
        # if a car reached the stop line
        if (self.sim == 'RA') and (self.dist_first_stop <= 0):
            # until reaching a further 7.5m line from the stop line, keep checking the others
            #  once get the "side_clear == True" & passing a further 7.5m from the stop line, return True
            if car.side_clear and (self.dist_first_stop <= -7.5):
                car.side_clear = True
            else:
                car.side_clear = not self.search_other_cars(car, mode='RA_turn')

    # checking whether those who attempt to turn is waiting for passing others
    def check_waiting_turn(self, car):
        # if a car is not listed in the list
        if car in car.road.waiting_turn_dic[car.dir][car.lane]:
            # once a car get "side_clear = True", delete a car from the list
            if car.side_clear:
                car.road.waiting_turn_dic[car.dir][car.lane].remove(car)
        else:
            # if a car has already crossed the stop line & car.side_clear is not True
            if (self.dist_first_stop < 0) and (not car.side_clear):
                # add Car to the list of waiting car in Road class
                car.road.waiting_turn_dic[car.dir][car.lane].append(car)

    # Adjusting a car's speed
    def adjust_speed(self, car):
        # get the previous car speed before adjusting
        previous_car_speed = car.current_speed

        # get a dictionary key to access the appropriate maintained car space based on the car's speed
        key_car_space = "idling" if previous_car_speed <= 10 else "others"
        # the car space that a driver must maintain
        self.maintain_car_space = car.car_space[key_car_space]

        # check whether a car is waiting for passing others in order to turn safely
        self.check_waiting_turn(car)

        # Braking:
        # For ensuring the front space (except for when a driver is not currently turning)
        if (not car.turn_ctl.check(car)) and self.brake_timing_front(car):
            # if a distance between a front is less than the maintained car space
            if car.front['dist'] <= self.maintain_car_space:
                # setting a stronger deceleration rate
                target_speed = max(previous_car_speed + car.dcl_rate['stronger'], 0)
            else:
                # setting a normal deceleration rate
                target_speed = max(previous_car_speed + car.dcl_rate['normal'], 0)

            # assume that a car should decrease speed to the defined target_speed within the distance
            dcl = math_help.deceleration(init_speed=previous_car_speed, target_speed=target_speed,
                                         distance=car.front['dist'] - self.maintain_car_space,
                                         max_dcl_kmh=car.dcl_rate['maximum'])
            # update the car's current speed
            car.current_speed += dcl * self.time_step

        # if there are no other cars in the front
        else:
            # For the reaction of the traffic light & the preparation for turning & the waiting for turning
            # (1): Setting deceleration rate so that a driver can stop before the stop line
            #  cond1: a driver realized the traffic signal (if it's red or yellow) based on car.react_dist['signal']
            cond1_1 = self.brake_timing_traffic_light(car)
            #  cond2: if there are 3 other cars who are waiting for turning,
            #         if a driver reached the reaction distance based on the stop line
            #         if a driver didn't cross the stop line and not idling
            #         a driver will start braking so that he/she can stop the line
            cond1_2 = (len(car.road.waiting_turn_dic[car.dir][car.lane]) >= 3) and \
                      (self.dist_first_stop <= car.react_dist['straight']) and \
                      (self.dist_first_stop > 0) and (not car.active == 'idling')
            # if meeting the above two conditions
            if cond1_1 or cond1_2:
                dcl = math_help.deceleration(init_speed=previous_car_speed, target_speed=0,
                                             distance=self.dist_first_stop, max_dcl_kmh=car.dcl_rate['maximum'])
                car.current_speed += dcl * self.time_step

            # (2): Other Decelerations (initial requirement; a car has not finished turning if they will do
            elif not car.turn_ctl.turn_done:
                # (2)-1: Preparing for turning, conditions;
                #  -> a car reached the certain distance to start preparing for turning based on react_dist['turn']
                #  -> a car has not reached the stop line
                #  -> a car's speed is greater than 20 km/h
                if (self.dist_first_stop <= car.react_dist['turn']) \
                        and (self.dist_first_stop > 0) and (car.current_speed > 20):
                    dcl = math_help.deceleration(
                        init_speed=previous_car_speed, target_speed=20,
                        distance=self.dist_first_stop, max_dcl_kmh=car.dcl_rate['maximum'])
                    # update the car's current speed
                    car.current_speed += dcl * self.time_step

                # (2)-2: Waiting for passing others, conditions;
                #  -> a driver has crossed the stop line
                #  -> a driver's side has not cleared yet
                #  -> a car is not idling
                if (self.dist_first_stop <= 0) and (not car.side_clear) and (not car.active == "idling"):
                    # change the must-stop distance
                    car.dist_must_stop -= previous_car_speed * 1000 / 3600 * self.time_step
                    # car should stop within the must-stop line when they are waiting for passing other cars
                    dcl = math_help.deceleration(init_speed=previous_car_speed, target_speed=0,
                                                 distance=car.dist_must_stop,
                                                 max_dcl_kmh=car.dcl_rate['maximum'])
                    # update the car's current speed
                    car.current_speed += dcl * self.time_step

        # Accelerating or Driving
        # no braking in this time step then start considering the acceleration or driving
        if car.current_speed == previous_car_speed:
            if self.acc_timing(car):
                # update current car speed
                car.current_speed += car.acc_rate

            # Otherwise, a car will keep the same speed as the previous step
            else:
                # update current car speed (no change)
                car.current_speed += 0

        # in the end, updating the car's active mode
        self.active_update(car, previous_car_speed)

    # Controlling the timing when a driver should start braking (traffic light signal)
    # this function is processing in only "TL" (Traffic Light) simulation
    def brake_timing_traffic_light(self, car):
        # if the simulation type is "RA", return False (no need to start braking due to the signal)
        # Also, those who will turn Right in "TL" can ignore the traffic light.
        if (self.sim == "RA") or (car.turn == 'Right' and self.sim == "TL"):
            return False
        # if a driver enters the range that he/she can recognize or start reacting the traffic light
        #  make sure that a car is now located within a 7.5m after the stop line.
        if (self.dist_first_stop <= car.react_dist['traffic_light']) and self.dist_first_stop > -7.5:
            # assign the current traffic light color
            traffic_light_color = car.road.traffic_light_color[car.focus_traffic_light]
            # once a driver checks the traffic light, then realize yellow or red
            if not car.traffic_light_stop and (traffic_light_color in ['red', 'yellow']):
                # change variable
                car.traffic_light_stop = True
                # tracking the gap time
                car.gap_time_traffic_light += self.time_step
                brake_prop = 0
            else:
                # if the traffic light is still 'red' or 'yellow' and
                # if a driver's recognition & reaction time exceeds at certain time
                if (traffic_light_color in ['red', 'yellow']) \
                        and (car.gap_time_traffic_light >= car.react_time['traffic_light']):
                    # a driver start braking
                    brake_prop = 1
                # if the current traffic light changes to green,
                elif car.road.traffic_light_color[car.focus_traffic_light] == 'green':
                    # reset variable
                    car.traffic_light_stop = False
                    car.gap_time_traffic_light = 0
                    # and no brake in terms of the traffic light
                    brake_prop = 0
                # otherwise, brake probability os 75%
                else:
                    brake_prop = 0.75

            return True if random.random() < brake_prop else False

        else:
            return False

    # Controlling the timing when a driver should start braking (there is a front car)
    # the probability of braking depends on distance between cars, car's speed, and a front's behavior
    def brake_timing_front(self, car):
        # if there is another car in front of each driver
        if isinstance(car.front['car'], Car):
            # the faster the speed, the higher the probability
            base_speed = 10  # set minimum speed
            weight = 0.75 + (1 - base_speed / max(car.current_speed, base_speed))
            # the smaller the distance that a car can reach, the higher the probability
            base_dist = self.maintain_car_space * 2  # set the minimum distance
            brake_prop = (base_dist / max(car.front['dist'] - base_dist, base_dist)) * weight
            # if a front car starts braking for the first time
            if car.front['car'].active == 'braking' and (not car.front_car_brake):
                # activate the brake light of the front car
                car.front_car_brake = True
                # tracking a driver's recognition & reaction time
                car.gap_time_brake += self.time_step
            # if a front car is still braking and passing certain times (car.react_time),
            if car.front['car'].active == 'braking' and (car.gap_time_brake >= car.react_time['front_brake']):
                # a driver start braking with 100%
                brake_prop = 1
            # if a front car's braking light is over, reset variable
            if car.front['car'].active != 'braking':
                car.front_car_brake = False
                car.gap_time_brake = 0

            return True if random.random() < brake_prop else False

        else:
            return False

    # Controlling the timing when a driver starts accelerating
    def acc_timing(self, car):
        # if the current speed has already reached the max speed, cannot accelerate anymore
        # and if the traffic light is red
        if car.current_speed >= car.max_speed:
            return False
        # if the traffic light is red or yellow except for those who will turn Right
        elif (self.sim == "TL") and (not car.turn == "Right") and \
                (car.road.traffic_light_color[car.focus_traffic_light] in ['red', 'yellow']):
            # if a car already moved more than a further 7.5m from the stop line, a car should exit swiftly
            if self.dist_first_stop <= -7.5:
                return True
            # otherwise, must stop
            else:
                return False

        # If LIMIT_ROUNDING_SPEED is defined (is not None or False) in RA, and a car hasn't finished rounding
        # and a car has reached the stop line and the current car speed is above the LIMIT_ROUNDING_SPEED
        elif (car.current_speed >= LIMIT_ROUNDING_SPEED) and (self.sim == "RA") and (not car.turn_ctl.turn_done) and (self.dist_first_stop <= 0):
            # stop accelerating
            return False

        # once car_side_clear is True (only applied for those who will turn), start accelerating quickly
        elif car.side_clear and not car.turn == "Straight":
            return True
        # otherwise, the acceleration timing is based on probabilities
        else:
            # if there is another car in the front
            if isinstance(car.front['car'], Car):
                # if a front car is braking, a driver won't start accelerating
                if car.front['car'].active == 'braking':
                    acc_prop = 0
                # if a front car is accelerating, a driver will start accelerating
                elif car.front['car'].active == 'accelerating':
                    acc_prop = 1
                else:
                    # the lower the current car speed, the higher the probability of accelerating (base 10)
                    base_speed = 10
                    weight = 1 + base_speed / max(car.current_speed, base_speed)
                    # the larger the distance that a car can reach, the higher the probability
                    base_dist = self.maintain_car_space  # set the minimum distance
                    acc_prop = (1 - base_dist / max(car.front['dist'] - base_dist, base_dist)) * weight
            # if there is no front car,
            else:
                # a driver start accelerating (100%)
                acc_prop = 1
            return True if random.random() < acc_prop else False

    # Controlling the car's active mode in the end
    def active_update(self, car, previous_car_speed):
        # if speed decreases or speed is less than or equal to 0,
        if (car.current_speed < previous_car_speed) or (car.current_speed <= 0):
            car.active = 'braking'
        # if speed increases and speed is greater than 10km/h,
        elif (car.current_speed > previous_car_speed) and car.current_speed > 10:
            car.active = 'accelerating'
        # if speed is the same and speed is greater than 10km/h,
        elif (car.current_speed == previous_car_speed) and car.current_speed > 10:
            car.active = 'driving'
        # otherwise,
        else:
            car.active = 'idling'


class TurnControl:
    """
    The Organizer of
    - Providing the next car position when it is turning or rounding.
        The next position can be calculated by solving two equations (common point of line and circle).
         line equation is based on a start point and the length(car's magnitude in one time step) of line
         circle equation is based on defined center and defined radius.
         So, the next car position should be on circle, and length from start point is equal to the car's magnitude.
        However, solving those two equation provides two answer due to finding common points between line and circle
         to select the appropriate answer, the system always choose the point that has minimum length from the target.
        Only for those who will turn Left in Roundabout, we set two targets to avoid choosing the shortest path.
         once a car reached the first target, automatically swifts to the final target.
         because of this way, getting an answer based on the minimum length is always working.
    - Controlling each driver's blinker, it should start around 15m before start turning.
        For those who will turn Left Roundabout will activate blinker around 15m before the final target point.
    - Assigning the new car.direction, car.turn, car.xy-vector after turning or rounding completely.
    """
    def __init__(self, sim, turn, time_step):
        self.time_step = time_step
        self.sim = sim
        # initial turn
        self.init_turn = turn
        # circle equation for turning
        self.circle_equ = None
        # when a cat finishes turning completely or never turned anymore -> True
        self.turn_done = False

        # variables after completing the turn
        self.chg_xy_vector = None
        self.chg_turn = None
        self.chg_dir = None

        # target xy vector, by turning direction (not lane)
        self.xy_vec = {
            "Straight": {'NB': (0, 1), 'SB': (0, -1), 'EB': (1, 0), 'WB': (-1, 0)},
            "Left": {'NB': (-1, 0), 'SB': (1, 0), 'EB': (0, 1), 'WB': (0, -1)},
            "Right": {'NB': (1, 0), 'SB': (-1, 0), 'EB': (0, -1), 'WB': (0, 1)}}

        # For Traffic Light simulation
        if self.sim == "TL":
            # if car will go straight, car will never turn
            if self.init_turn == 'Straight':
                self.turn_done = True
            # if car will turn "Left" or "Right", setting variables
            else:
                # Once car is within the range, turning is active, by each lane
                self.turn_threshold = {
                    "Left": 12,
                    "Right": 15}
                # cars will turn based on a circle center, by each lane and direction
                self.turn_center = {
                    "Left": {'NB': (-12, -12), 'SB': (12, 12), 'EB': (-12, 12), 'WB': (12, -12)},
                    "Right": {'NB': (15, -15), 'SB': (-15, 15), 'EB': (-15, -15), 'WB': (15, 15)}}
                # radius of a circle by each lane
                self.turn_radius = {"Left": 15, "Right": 6}
                # target of xy position, by lane and direction
                self.turn_xy_pos = {
                    "Left": {'NB': (-12, 3), 'SB': (12, -3), 'EB': (3, 12), 'WB': (-3, -12)},
                    "Right": {'NB': (15, -9), 'SB': (-15, 9), 'EB': (-9, -15), 'WB': (9, 15)}}

        # Car turning in Roundabout
        elif self.sim == "RA":
            # once car is within the range, round is activated, by each lane and direction
            self.turn_threshold = {
                "Left": 15,  # the changing position is <= 15(abs)
                "Right": 20}  # the changing position is <= 20(abs)

            # for car that will turn right (only turning)
            if self.init_turn == "Right":
                # car will turn right based on a circle center, by lane and direction
                self.turn_center = {
                    "Right": {'NB': (20, -20), 'SB': (-20, 20), 'EB': (-20, -20), 'WB': (20, 20)}}
                # radius of a circle, by lane
                self.turn_radius = {"Right": 11}  # only right lane for those who will turn right
                # target of xy position (only right lane)
                self.turn_xy_pos = {
                    "Right": {'NB': (20, -9), 'SB': (-20, 9), 'EB': (-9, -20), 'WB': (9, 20)}}

            # for car will go Straight and turn Left
            else:
                # cars will round based on a circle center (only "Straight" & "Left"); always (0,0)
                self.turn_center = {
                    "Left": {key: (0, 0) for key in ['NB', 'SB', 'EB', 'WB']},
                    "Right": {key: (0, 0) for key in ['NB', 'SB', 'EB', 'WB']}
                }
                # radius of a circle (when rounding), by each lane
                self.turn_radius = {"Left": 12, "Right": 18}

                # for car that will turn left (rounding & turning)
                if self.init_turn == 'Left':
                    # setting the first target for only those who will turn left in "RA"
                    self.reach_first_target = False
                    # 1st target of xy position
                    self.first_target = {
                        'NB': (3, 12), 'SB': (-3, -12), 'EB': (12, -3), 'WB': (-12, 3)
                    }
                    # target of xy position (turning), by only Left lane
                    self.turn_xy_pos = {
                        "Left": {'NB': (-12, 3), 'SB': (12, -3), 'EB': (3, 12), 'WB': (-3, -12)}
                    }
                # for those who will go straight (rounding & turning)
                elif self.init_turn == 'Straight':
                    # target of xy position (rounding), by each lane
                    self.turn_xy_pos = {
                        "Left": {'NB': (3, 12), 'SB': (-3, -12), 'EB': (12, -3), 'WB': (-12, 3)},
                        "Right": {'NB': (9, 16), 'SB': (-9, -16), 'EB': (16, -9), 'WB': (-16, 9)}
                    }

    # checking the blinker timing; if necessary, return "Left" or "Right"
    def blinker_check(self, car):
        #  the xy position where a car is currently located
        current_pos = (car.x, car.y)
        # if car blinker is not activated, (applied both in "TL" and "RA")
        if not car.blinker:
            stop_line = car.speed_ctl.first_stop  # car's stop line
            # distance between current position and stop line
            dist_car_stop = math_help.get_distance(current_pos, stop_line)
            # once its distance reached an individual blinker reaction distance
            if dist_car_stop <= car.react_dist['blinker']:
                # make sure that the blinker is only Left or Right
                return car.turn if not car.turn == "Straight" else None
            else:
                return None
        # applying a little more work for those who will turn "Straight" or "Left" in the Roundabout simulation
        elif (car.turn != "Right") and (self.sim == "RA"):
            # the xy position where a car's rounding is completely over
            target_pos = self.turn_xy_pos[car.lane][car.dir]
            # distance between its target and current position
            dist_car_target = math_help.get_distance(target_pos, current_pos)
            # if its distance reached an individual blinker reaction distance
            if dist_car_target <= car.react_dist['blinker']:
                # make sure that the blinker is always "Right" in Roundabout right before reaching the exit
                return "Right"
            else:
                # keeping doing the same blinker situation
                return car.blinker

        # otherwise, keep doing the same blinker condition
        else:
            return car.blinker

    # checking whether a car should start rounding or turning, also update the blinker.
    # if start turning, define the circle formula (this is fixed during all turning operations)
    def check(self, car):
        # if car has completed turning or don't have to turn(like "Straight" in "TL"), return False
        if self.turn_done:
            return False
        # if car has not turned yet, check whether a car should turn right now
        else:
            # update a car blinker
            car.blinker = self.blinker_check(car)

            # if changing position within certain threshold, start turning
            if abs(np.dot((car.x, car.y), car.xy_vector)) <= self.turn_threshold[car.lane]:
                # define circle equation for turning
                self.circle_equ = math_help.circle_by_center_radius(
                    center=self.turn_center[car.lane][car.dir],
                    radius=self.turn_radius[car.lane])
                return True
            else:
                return False

    def move(self, car):
        # car's magnitude (length of move at one step); make sure to convert m/s
        car_mag = car.current_speed * (1000 / 3600) * self.time_step
        # current xy position
        current_pos = (car.x, car.y)
        # target xy position
        target_pos = self.turn_xy_pos[car.lane][car.dir]
        # get arc length (m) between start and target
        arc_len = math_help.arc_length(current_pos, target_pos,
                                       self.turn_center[car.lane][car.dir],
                                       self.turn_radius[car.lane])
        # if arc length is greater than car magnitude,
        if arc_len > car_mag:
            line_equ = math_help.line_by_point_length(start_xy=current_pos, length=car_mag)
            # always get two answers
            sols = math_help.solve_two_equations(self.circle_equ, line_equ)

            # if this case below, the next xy position can be obtained by different criteria
            #  this process deals with the next car position is chosen by the minimum distance from start to target.
            #  when a car turns Left, the minimum distance should not always be chosen as for the final target.
            #  therefore, setting the first target
            if (self.sim == "RA") and (car.turn == "Left"):
                # setting the first target
                first_target_pos = self.first_target[car.dir]
                # get arc length between current and first target position
                arc_len2 = math_help.arc_length(current_pos, first_target_pos,
                                                self.turn_center[car.lane][car.dir],
                                                self.turn_radius[car.lane])
                # if car can move to first target position by one step
                if arc_len2 <= car_mag:
                    # we recognize that car can reach the first target
                    self.reach_first_target = True

                # if reaching the first target position
                if self.reach_first_target:
                    # setting the final target position
                    target_pos = self.turn_xy_pos[car.lane][car.dir]
                else:
                    # assign target_pos as the first target
                    target_pos = first_target_pos

            # get distance between target position and the possible next xy positions
            length_dict = {}  # key: distance, value: next xy position
            for sol in sols:
                length = math_help.get_distance(sol, target_pos)
                length_dict[length] = (sol[0], sol[1])
            # get the next xy position whose distance is minimum.
            next_xy_pos = length_dict[min(length_dict)]
            # return position of (x, y)
            return next_xy_pos

        # if car can reach a target position by one step, finish turning or rounding process
        else:
            self.turn_done = True
            # assign variable
            self.chg_xy_vector = self.xy_vec[car.turn][car.dir]
            self.chg_turn = "Straight"
            self.chg_dir = [key for key, value in DIR_TO_XY_VECTOR.items() if value == car.xy_vector][0]
            # next xy position (consider extra car magnitude)
            #  this extra length is defined by one-step car magnitude after deducing by arc length
            #  and add this extra length to the next changing position
            next_xy_pos = tuple(target_pos + np.array(self.chg_xy_vector) * int(car_mag - arc_len))
            # return position of (x, y)
            return next_xy_pos

    # this function will be invoked after a car finished turning completely
    def finalize(self, car):
        # update a car's variable
        car.xy_vector = self.chg_xy_vector
        car.turn = self.chg_turn
        car.dir = self.chg_dir
        car.blinker = None


# -----------------------------------------------------------------------------------------------------------------
# simulation
# set the passing time
elapsed_time = 0


def run_sim(data_file, sim, time_step, sim_times, animation=False, interval=200):
    """
    running the simulation.
    data_file: the traffic data file (assuming that this is the real car data)
    sim: simulation type; "TL" or "RA"
    time_step: the time_step
    sim_times: simulation times
    animation: True if tracking car's behavior (default; False)
    interval: the speed of the animation (default: 200)
    """
    if sim == "TL":
        # setting Road class with classes for car generation (common) and traffic light(only in "TL")
        road = Road(sim=sim, time_step=time_step,
                    car_generation=CarGeneration(data_file),
                    traffic_light_ctl=TrafficLightControl(time_step))  # set road
        road.traffic_light_ctl.initialize()  # initialize the traffic light
    else:
        road = Road(sim=sim, time_step=time_step, car_generation=CarGeneration(data_file))

    # generating the car entry time
    # initialize the data (clean the real data)
    road.car_generation.initialize()
    # search the best-fitted distribution and generate car entry time segmented by each car's turn and direction
    # generating distribution for each turn & direction, ensure or try to do IID (Independent & Identically Distributed)
    road.car_generation.fit_distribution()
    # create the dictionary
    # -> key: entry time (seconds), value: list of dictionary {"id": int, "turn": str, "dir"" str}
    road.car_entry_dic = road.car_generation.organize_entry_time()

    if animation:
        # Set up graphical display
        if sim == 'TL':
            img = plt.imread('Traffic_Intersection.png')
        else:
            img = plt.imread('Roundabout.png')

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-MONITOR_AREA / 2, MONITOR_AREA / 2)
        ax.set_ylim(-MONITOR_AREA / 2, MONITOR_AREA / 2)
        ax.imshow(img, extent=[-MONITOR_AREA / 2, MONITOR_AREA / 2, -MONITOR_AREA / 2, MONITOR_AREA / 2],
                  aspect='auto', zorder=-1, alpha=0.5)
        scatter = ax.scatter(x=[], y=[], s=10)

        def updatefigure(frames):
            global elapsed_time
            elapsed_time = round(elapsed_time + time_step, 2)
            # update the clock 1 seconds = 1000 milliseconds
            road.clock += timedelta(milliseconds=1000 * time_step)

            # per 30 min
            if elapsed_time % 900 == 0:
                print(f"Time at {road.get_clock()}")

            # run car simulation
            road.run_step(elapsed_time=elapsed_time)
            x, y, c = [], [], []  # xy positions & color
            for car in road.get_active_cars():
                x.append(car.x)
                y.append(car.y)
                c.append(ACTIVE_TO_COLOR[car.active])

            scatter.set_offsets(np.column_stack([x, y]))  # set all data points
            scatter.set_color(c)  # set color for all data points
            return scatter,

        anim = FuncAnimation(fig, updatefigure, frames=sim_times,
                             interval=interval, blit=True, repeat=False)
        plt.show()

    else:
        elapsed_time = 0
        print("start simulation")
        for _ in range(sim_times):
            elapsed_time = round(elapsed_time + time_step, 2)
            # 1 seconds = 1000 milliseconds
            road.clock += timedelta(milliseconds=1000 * time_step)
            # per 1 hour
            if elapsed_time % 3600 == 0:
                print(f"Time at {road.get_clock()}")

            road.run_step(elapsed_time=elapsed_time)

    # -----------------------------------------------------------------------------------------------------------
    # Statistics & Visualization
    print("Creating Visualizations...")
    # show crucial parameters
    print(f"Traffic Light Interval: {TL_TIME} for SIM = 'TL'.")
    print(f"Rounding Speed Restriction: {LIMIT_ROUNDING_SPEED} for SIM = 'RA'.")


    # (1): Show Statistics Focusing on Each Time Step
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), num=1)
    fig.suptitle('1 Hour Moving Average of Car Statistics')
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    xticks_interval_sec = [i for i in range(0, 86401, 10800)]
    xticks_interval_hour = [i for i in range(0, 25, 3)]
    car_data_each_step = {"Number of Cars": road.num_car, "Total Waiting Cars": road.total_wait_car,
                          "Avg Car Speed(km/h)": road.avg_car_speed, "Total Gas Pollution(g)": road.total_gas}
    idx = 0

    for i in range(2):
        for j in range(2):
            ax[i, j].plot(math_help.moving_average(list(car_data_each_step.values())[idx], 3600))
            ax[i, j].set_title(f"{list(car_data_each_step.keys())[idx]}")
            ax[i, j].set_xticks(xticks_interval_sec, xticks_interval_hour)
            idx += 1

    # (2): Show Statistics Focusing on Each Car
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), num=2)
    fig.suptitle('Statistics for Each Car')
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    xticks_interval_sec = [i for i in range(0, 86401, 10800)]
    xticks_interval_hour = [i for i in range(0, 25, 3)]
    each_car_data = {"Entry Time": [], "Waiting Time": [],
                     "Passing Time": [], "Total Gas Pollution(g)": []}

    # assigning data
    for v in road.each_car_stat.values():
        v_lst = list(v)
        each_car_data["Entry Time"] += [v_lst[0]]
        each_car_data["Waiting Time"] += [v_lst[1]]
        each_car_data["Passing Time"] += [v_lst[2]]
        each_car_data["Total Gas Pollution(g)"] += [v_lst[3]]

    # start visualization

    # first plot (entry time)
    ax[0, 0].hist(each_car_data["Entry Time"], bins=[i for i in range(0, 86401, 3600)])
    ax[0, 0].set_title("Histogram of Car Entry Time")

    # second plot(entry time vs Waiting Time)
    ax[0, 1].scatter(each_car_data["Entry Time"], each_car_data["Waiting Time"], s=1, alpha=0.2)
    ax[0, 1].set_title("Entry Time(x;hours) vs Wait Time(y; seconds)")
    ax[0, 1].set_xticks(xticks_interval_sec, xticks_interval_hour)
    mean_wait = sum(each_car_data["Waiting Time"]) / len(each_car_data["Waiting Time"])  # mean
    std_wait = np.std(each_car_data["Waiting Time"])  # standard deviation
    ax[0, 1].text(0.98, 0.98, f"mean: {mean_wait:.1f}\nstd: {std_wait:.1f}",
                  fontsize=10, ha="right", va="top", transform=ax[0, 1].transAxes)

    # third plot(entry time vs Passing Time)
    ax[1, 0].scatter(each_car_data["Entry Time"], each_car_data["Passing Time"], s=1, alpha=0.2)
    ax[1, 0].set_title("Entry Time(x;hours) vs Passing Time(y; seconds)")
    ax[1, 0].set_xticks(xticks_interval_sec, xticks_interval_hour)
    mean_pass = sum(each_car_data["Passing Time"]) / len(each_car_data["Passing Time"])  # mean
    std_pass = np.std(each_car_data["Passing Time"])  # standard deviation (assuming n is large)
    ax[1, 0].text(0.98, 0.98, f"mean: {mean_pass:.1f}\nstd: {std_pass:.1f}",
                  fontsize=10, ha="right", va="top", transform=ax[1, 0].transAxes)

    # forth plot(histogram of total gas pollution on each car)
    min_gas = min(each_car_data["Total Gas Pollution(g)"])
    max_gas = max(each_car_data["Total Gas Pollution(g)"])
    range_bins = [i for i in range(int(min_gas), int(max_gas), 5)]
    ax[1, 1].hist(each_car_data["Total Gas Pollution(g)"], bins=range_bins)
    ax[1, 1].set_title("Histogram of Total Gas Pollution(g)")
    mean_gas = sum(each_car_data["Total Gas Pollution(g)"]) / len(each_car_data["Total Gas Pollution(g)"])  # mean
    std_gas = np.std(each_car_data["Total Gas Pollution(g)"])  # standard deviation (assuming n is large)
    ax[1, 1].text(0.98, 0.98, f"mean: {mean_gas:.1f}\nstd: {std_gas:.1f}",
                  fontsize=10, ha="right", va="top", transform=ax[1, 1].transAxes)

    # (3) Historgram of Waiting Time
    print("Average Wait Time: ", mean_wait)
    print("Standard deviation of Wait Time: ", std_wait)
    print(f"95% of Wait Time included in ", np.percentile(each_car_data["Waiting Time"], [2.5, 97.5]))
    min_wait = min(each_car_data["Waiting Time"])
    max_wait = max(each_car_data["Waiting Time"])
    range_bins_wait = [i for i in range(int(min_wait), int(max_wait), 1)]
    plt.figure(3)
    plt.hist(each_car_data["Waiting Time"], bins=20)
    plt.title("Histogram of Wait Time")

    # (4) Histogram of Passing Time
    print("Average Pass Time: ", mean_pass)
    print("Standard deviation of Pass Time: ", std_pass)
    print(f"95% of Pass Time included in ", np.percentile(each_car_data["Passing Time"], [2.5, 97.5]))
    min_pass = min(each_car_data["Passing Time"])
    max_pass = max(each_car_data["Passing Time"])
    range_bins_pass = [i for i in range(int(min_pass), int(max_pass), 5)]
    plt.figure(4)
    plt.hist(each_car_data["Passing Time"], bins=20)
    plt.title("Histogram of Pass Time")

    print("---------------------------------------------------------------")
    print("Gas Statistics")
    print("Total Gas Pollution: ", sum(list(STAT_GAS1.values())))
    print("Average GAS Pollution: ", sum(STAT_GAS1.values()) / len(STAT_GAS1))
    print("Total Gas Pollution on Each Active: ", STAT_GAS2)

    # plot all the above visualizations
    plt.show()

    return road

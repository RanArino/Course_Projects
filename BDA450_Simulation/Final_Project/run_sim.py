from car_simulation import *
import math_help

DATA_FILE = "Traffic_Data.xlsx"
SIM = "TL"  # choose "TL" -> Traffic Light or "RA" -> Roundabout
TIME_STEP = 1  # seconds
TOTAL_SECONDS = 24*3600
SIM_TIMES = int(TOTAL_SECONDS/TIME_STEP) + 30  # extra time to exit the remained cars
ANIMATION = False
INTERVAL = 100

# START SIMULATION
road = run_sim(data_file=DATA_FILE, sim=SIM, time_step=TIME_STEP, sim_times=SIM_TIMES,
               animation=ANIMATION, interval=INTERVAL)


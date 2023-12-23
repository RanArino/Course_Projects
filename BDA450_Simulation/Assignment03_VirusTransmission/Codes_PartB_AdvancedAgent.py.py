# Base for simple simulation of virus transmission between pedestrians on a sidewalk.
# Written by Reid Kerr for use in an assignment in BDA450.
# Please note that this code is not written to be efficient/elegant/etc.!  It is written
# to be brief/simple/transparent, to a reasonable degree.
# There may very well be bugs present!  Moreover, there is very little validation of inputs
# to function calls, etc.  It is generally up to the developer who uses this to build their simulation
# to ensure that their valid state is maintained.

import math
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.animation import FuncAnimation
import random

rand = random.Random()
SIDEWALK_WIDTH = 10  # This is the y-dimension of the sidewalk
SIDEWALK_LENGTH = 200  # This is the x-dimension of the sidewalk
TRANSPROB = 0.1  # Probability of transmission of virus in 1 time step at distance 1
INFECTED_PROP = 0.1  # The proportion of people who enter the simulation already carrying the virus
INTERARRIVAL = 6  # Average number of time steps between arrivals (each side handled separately)
CAREFUL_PROP = 0.1  # The proportion of the population who are 'careful', trying to keep a distance from others

NUM_SIM_PEOPLE = 300  # Number of people that we would like to simulate

# Setup for graphical display
colourmap = colors.ListedColormap(["lightgrey", "green", "red", "yellow", "purple"])
normalizer = colors.Normalize(vmin=0.0, vmax=4.0)


# An agent representing a single person traversing the sidewalk.  Simple movement is demonstrated.  It is up
# to the user to implement behaviour according to the assignment specification, and to collect data as
# necessary.
# A person occupies an x,y position on the sidewalk.  While the person can access their own position, they
# cannot change their coordinates directly.  Instead, they must make movement requests to the sidewalk, which
# (if the move is valid) updates the person's x and y coordinates.
class Person:
    def __init__(self, id, sidewalk):
        self.id = id
        self.active = False
        self.sidewalk = sidewalk
        self.infected = True if rand.random() < INFECTED_PROP else False
        self.newlyinfected = False
        self.careful = True if rand.random() < CAREFUL_PROP else False

        if rand.choice([True, False]):
            self.startx = 0
            self.direction = 1
            # track people from left
            self.sidewalk.from_left.append(self)

        else:
            self.startx = SIDEWALK_LENGTH - 1
            self.direction = -1
            # track people from right
            self.sidewalk.from_right.append(self)

        self.starty = self.y = rand.randint(0, SIDEWALK_WIDTH - 1)

    def enter_sidewalk(self, x, y):
        if self.sidewalk.enter_sidewalk(self, x, y):
            self.active = True

    def leave_sidewalk(self, time_step):
        if self.sidewalk.leave_sidewalk(self):
            self.active = False
            print(f"Leave p_{self.id} at {time_step}")

    # This is the method that is called by the simulation once for each time step.  It is during this call
    # that the agent is active and can take action: examining surroundings, attempting to move, etc.
    # This method should only be called when the agent's 'active' flag is true, but you might want to check here
    # as well for safety.
    def step(self):
        # going straight would be prior to moving either left or right
        self.change_options = {'straight': (self.direction, 0)}
        # move either left or right, depending on direction
        left_right = [{'left': (0, -self.direction)}, {'right': (0, self.direction)}]
        # randomly shuffle the order of the list "left_right"
        random.shuffle(left_right)
        # simply add each of them
        for move in left_right:
            self.change_options.update(move)

        # behavious for those who are not careful
        if self.careful == False:
            # repeat change_options
            for change in self.change_options.values():
                desiredx = self.x + change[0]
                desiredy = self.y + change[1]

                # Ensure x and y don't go off edge of sidewalk
                desiredx = max(min(desiredx, SIDEWALK_LENGTH - 1), 0)
                desiredy = max(min(desiredy, SIDEWALK_WIDTH - 1), 0)
                # once person succeeded to move new position, ignore the other change options
                if self.sidewalk.attemptmove(self, desiredx, desiredy) == True:
                    break

        # behaviors for those who are careful
        else:
            # check the availability to move
            keys = list(self.change_options.keys()) # get direction name list, like ['straight',...]
            for i in range(len(keys)):
                desiredx = self.x + self.change_options[keys[i]][0]
                desiredy = self.y + self.change_options[keys[i]][1]
                # remove a desired position once facing the following 3 conditions
                if desiredx < 0 or desiredx >= SIDEWALK_LENGTH:
                    del self.change_options[keys[i]]
                    continue
                if desiredy < 0 or desiredy >= SIDEWALK_WIDTH:
                    del self.change_options[keys[i]]
                    continue
                if self.sidewalk.storage.isoccupied(desiredx, desiredy) == True:
                    del self.change_options[keys[i]]

            # if there are no options for the desired positions, person would stay
            if len(self.change_options) == 0:
                self.sidewalk.attemptmove(self, self.x, self.y)

            # otherwise, checking the surrounding areas
            else:
                # checking positions of other people within a square of radius 1, 2, 3 and additional front areas.
                result_check_surroundings = self.sidewalk.check_surroundings(self)
                # Decide the optimized desired direction based on dictionary of "result_check_surroundings"
                desired_direction = self.optimize_change_options(result_check_surroundings)
                # After finalizing the desired direction and positions, person will move to there
                desired_change = self.change_options[desired_direction]
                self.sidewalk.attemptmove(self, self.x + desired_change[0], self.y + desired_change[1])

    # Return the optimized desired direction, which is included "person.change_options"
    def optimize_change_options(self, result_check_surroundings):
        # get available desired direction e.g., ['straight', 'left', 'right']
        desired_directions = list(self.change_options.keys())
        # number of other people within a square of radius 1 and 2
        # e.g, radius_n=[0,0,0] indicates no one exists for all three desired directions within a square of radius n.
        radius_1 = np.array([len(d[1]) for d in result_check_surroundings])
        idx_min_r1 = list(np.where(radius_1 == min(radius_1))[0]) # get index of radius_1 with minimum value
        radius_2 = np.array([len(d[2]) for d in result_check_surroundings])
        idx_min_r2 = list(np.where(radius_2 == min(radius_2))[0]) # get index of radius_2 with minimum value

        # list of all other people positions within a square of radius 3; x-y coordinates (removed duplication)
        list_others_pos = list(set([pos for sublist in [d[3] for d in result_check_surroundings] for pos in sublist]))
        # calculating directions with fewer people (left, right, straight)
        optimized_directions = self.optimize_direction(list_others_pos)

        # (1): If there is only one desired direction where no one exists within a square of radius 1...
        #  return its direction because space for a square of radius 1 is priority.
        if len(idx_min_r1) == 1 and min(radius_1) == 0:
            # However, another person chases right after, change behavior
            # because another person is likely to go straight
            if self.sidewalk.storage.isoccupied(self.x-1, self.y):
                return optimized_directions[0]
            elif self.sidewalk.storage.isoccupied(self.x+1, self.y):
                return optimized_directions[0]
            else:
                return desired_directions[idx_min_r1[0]]

        # (2) If there are multiple desired direction where no one exists within a square of radius 1...
        elif len(idx_min_r1) > 1 and min(radius_1) == 0:
            # (2)-1:if only one desired direction where no one exists within a square of radius 2, return its position
            if len(idx_min_r2) == 1:
                return desired_directions[int("".join(map(str, idx_min_r2)))]
            # (2)-2: if multiple desired positions where no one exists within a square of radius 2...
            else:
                # a careful person prefers to be the direction with fewer people
                return optimized_directions[0]
        # Nene of above two cases, a careful person will move to the direction with fewer people
        else:
            return optimized_directions[0]

    # Calculate the direction with fewer people based on the location of a careful person
    def optimize_direction(self, list_others_pos):
        # list of all direction (no matter which direction is available at that time)
        dict_direction = {'straight': -0.5, 'left': 0, 'right': 0}
        # checking all positions of others
        for others_pos in list_others_pos:
            # calculate the distance between a careful person and other people (become weight)
            #  in order to prevent "ZeroDivisionError", adding a tiny float number
            distance = math.sqrt((others_pos[0] - self.x)**2 + (others_pos[1] - self.y)**2) + 0.00001
            # the closer other people exist, the more careful person feels anxiety (bigger weight).
            weight = (1/distance)
            # calculate the direction of degree
            direct_degree = math.degrees(math.atan2(others_pos[1] - self.y, others_pos[0] - self.x))
            # if other people exist in the front or back
            if (direct_degree == 0) or (direct_degree == 180):
                # if other people exist in the front or back, a person will be more reactive
                dict_direction['straight'] += weight
                dict_direction['left'] -= weight
                dict_direction['right'] -= weight

            # if other people exist in larger y position than a careful person
            elif direct_degree > 0:
                if self.direction == 1:
                    # for people whose direction is 1, other people locates at left side
                    #  lower direct_degree, higher reactive because those who direction = 1 see in their front
                    dict_direction['left'] += weight + 1/abs(direct_degree)
                    dict_direction['right'] -= weight * 0.2
                else:
                    # for people whose direction is -1, another person locates at right side
                    #  higher direct_degree, higher reactive because those who direction = -1 see in their front
                    dict_direction['right'] += weight + 1/abs(180-direct_degree)
                    dict_direction['left'] -= weight * 0.2

            elif direct_degree < 0:
                if self.direction == 1:
                    # for people whose direction is 1, another person locates at right side
                    dict_direction['right'] += weight + 1/abs(direct_degree)
                    dict_direction['left'] -= weight * 0.2

                else:
                    # for people whose direction is -1, another person locates at left side
                    # higher direct_degree higher reactive because a person sees in its front (direction = -1)
                    dict_direction['left'] += weight + 1/abs(180-direct_degree)
                    dict_direction['right'] -= weight * 0.2

        # sort the "dict_direction" by value(number)
        sorted_dict_direction = sorted(dict_direction, key=dict_direction.get)
        # return direction list sorted by number of other people
        return [direction for direction in sorted_dict_direction if direction in set(self.change_options.keys())]

    def __repr__(self):
        return "p_%s" % (self.id)

    def __str__(self):
        return "id: %s  x: %d  y: %d" % (self.id, self.x, self.y)


# The class representing the sidewalk itself.  Agents must enter the sidewalk to be active in the simulation.
# The sidewalk controls agents' positions/movement.
class Sidewalk:

    def __init__(self):
        # Tracking of positions of agents
        self.storage = SWGrid()

        # Tracking of 'Person class' from each side
        self.from_left = []
        self.from_right = []
        # Tracking of arrival time for each person
        self.person_arrival_dict = {}

        # Tracking of infection rate at certain times
        self.infection_rate_at_time = {}  # key: time, value: infection rate

        # Bitmap is for graphical display
        self.bitmap = [[0.0 for i in range(SIDEWALK_LENGTH)] for j in range(SIDEWALK_WIDTH)]

    # Assigning the arrival time for each agent.
    # Variable "arrival" is defined independently for each side & incremented randomly based on INTERARRIVAL
    def generate_arrival(self):
        # travel personlist from each side separately (ensure independent generation)
        for personlist in [self.from_left, self.from_right]:
            # set arrival time for the first agent
            arrival = int(rand.expovariate(1 / INTERARRIVAL))
            # travel every person and add arrival time
            for person in personlist:
                # taking care of duplicated keys
                if self.person_arrival_dict.get(arrival) == None:
                    self.person_arrival_dict[arrival] = [person]
                else:
                    self.person_arrival_dict[arrival] += [person]

                # increment the arrival time
                arrival += int(rand.expovariate(1 / INTERARRIVAL))

    # An agent must enter the sidewalk at one of the ends (i.e., with an x coordinate of either zero or
    # the maximum.  They may attempt to enter at any y coordinate.  The function returns true if successful, false
    # if the agent is not added to the sidewalk (e.g., if the desired square is already occupied.)
    # The method will set the agent's x and y position if the attempt is successful.
    def enter_sidewalk(self, person, x, y):
        # New entrant to the sidewalk, must attempt to start at one end
        # if x!=0 and x!=SIDEWALK_LENGTH-1:
        #     print("Must start at an end!")
        #     return False

        # Only allow move if space not currently occupied
        if self.storage.isoccupied(x, y):
            print("Move rejected: occupied")
            return False
        self.storage.add_item(x, y, person)
        person.x = x
        person.y = y
        return True

    # An agent must leave the sidewalk at one of the ends (i.e., with an x coordinate of either zero or
    # the maximum.  The function returns true if successful, false if not.
    # You should be sure to get any information you want from the agent before doing so, because you may not
    # have a handle for it afterwards.
    def leave_sidewalk(self, person):
        # Must attempt to leave at one end
        if person.x != 0 and person.x != SIDEWALK_LENGTH - 1:
            return False

        else:
            self.storage.remove_item(person)
            return True

    # Returns True if person successfully moved, False if not (e.g., the desired square is occupied).
    # An agent can only move one square in a cardinal direction from its current position; any other attempt will
    # be rejected.
    # The method will set the agent's x and y position if the attempt is successful.
    def attemptmove(self, person, x, y):

        # Reject move of more than 1 square
        if (abs(person.x - x) + abs(person.y - y)) > 1:
            print("Attempt to move more than one square!")
            return False

        # Only allow move if space not currently occupied
        if self.storage.isoccupied(x, y):
            # print("Move rejected: occupied")
            return False
        person.x = x
        person.y = y
        self.storage.move_item(x, y, person)
        return True

    # checking the surrounding areas
    def check_surroundings(self, person):
        # order of this list corresponds to order of person.change_options
        result_check_surroundings = []  # list of dictionaries "d"
        for desired_direct in person.change_options:
            desiredx = person.x + person.change_options[desired_direct][0]
            desiredy = person.y + person.change_options[desired_direct][1]
            radius = 3
            # d[1]: list of positions of other people within a square of radius 1
            # d[2]: list of positions of other people within a square of radius 2
            # d[3]: list of positions of other people within a square of radius 3 + additional areas (front)
            d = {1: [], 2: [], 3: []}
            # check a square of radius n, where n is 1~3
            for x in range(desiredx - radius, desiredx + radius + 1):
                for y in range(desiredy - radius, desiredy + radius + 1):
                    d[3] += [(x, y)] if self.isoccupied(x, y) and (x, y) != (person.x, person.y) else []
                    # count people within a square of radius 1
                    if (abs(x - desiredx) <= 1) and (abs(y - desiredy) <= 1):
                        d[1] += [(x, y)] if self.isoccupied(x, y) and (x, y) != (person.x, person.y) else []
                    if (abs(x - desiredx) <= 2) and (abs(y - desiredy) <= 2):
                        d[2] += [(x, y)] if self.isoccupied(x, y) and (x, y) != (person.x, person.y) else []

            # searching additional space (triangle-ish x-y coordinates for the person's front)
            for x in range(desiredx + 4 * person.direction, desiredx + 7 * person.direction):
                for y in range(desiredy - 2, desiredy + 3):
                    if x - abs(y) >= 4 or x - abs(y) <= -4:
                        d[3] += [(x, y)] if self.isoccupied(x, y) and (x, y) != (person.x, person.y) else []

            result_check_surroundings.append(d)

        return result_check_surroundings

    # When called, infects new agents (with some probability) who are in proximity to infected agents.  The risk
    # is equal to the simulation parameter at distance of 1, and decreases with greater distance.
    # You may add to this function, e.g., for gathering data, but do not modify the actual determination of infection.
    def spread_infection(self):
        for person in self.storage.get_list():
            currentx = person.x
            currenty = person.y
            if person.infected:
                # Find all agents within a square of 'radius' 2 of the infected agent
                for x in range(currentx - 2, currentx + 3):
                    for y in range(currenty - 2, currenty + 3):
                        target = self.storage.get_item(x, y)

                        # If target is not infected, infect with probability dependent on distance
                        if target is not None and not target.infected:
                            riskfactor = 1 / ((currentx - x) ** 2 + (currenty - y) ** 2)
                            tranmission_prob = TRANSPROB * riskfactor
                            if rand.random() < tranmission_prob:
                                target.infected = True
                                target.newlyinfected = True
                                print('New infection! %s' % target)

    # Monitoring the infection rate
    def monitor_infection(self, time_step):
        if time_step % 50 == 0 and time_step != 0:
            # get all active agents at a certain time
            personlist = self.storage.get_list()
            # number of agents at a certain time
            population = len(personlist)
            # infection rate at a certain time
            try:
                infection_rate = round(sum([person.infected for person in personlist]) / population, 3)
                self.infection_rate_at_time[time_step] = infection_rate
                print(f"Infection Rate at time {time_step}: {infection_rate}")
            except:
                pass

    # Updates the graphic for display
    def refresh_image(self):
        self.bitmap = [[0.0 for i in range(SIDEWALK_LENGTH)] for j in range(SIDEWALK_WIDTH)]
        for person in self.storage.get_list():
            x = person.x
            y = person.y
            colour = 1
            if person.newlyinfected:
                colour = 3
            elif person.infected:
                colour = 2
            elif person.careful:
                colour = 4
            self.bitmap[y][x] = colour

    # Function that is called at each time step, to execute the step.  Calls the step() function of every active
    # agent, spreads infection after the agents have moved, and updates the image for display.  You will need to add
    # code here to, for example, have new agents enter.
    def run_step(self, time_step):
        # enter new agents
        try:
            # retrieve person class whose arrival == time_step
            personlist = self.person_arrival_dict[time_step]
            [person.enter_sidewalk(person.startx, person.starty) for person in personlist]
            print(f'Arrived {", ".join(repr(p) for p in personlist)} at {time_step}')
        except KeyError:
            pass

            # update state for all active agents
        for person in self.storage.get_list():
            if person.active:
                person.step()
                person.leave_sidewalk(time_step)
        self.spread_infection()
        self.monitor_infection(time_step)
        self.refresh_image()

    # Returns true if x,y is occupied by an agent, false otherwise.  This is the only information that an agent
    # has about other agents; it can't (e.g.) see if other agents are infected!
    def isoccupied(self, x, y):
        return self.storage.isoccupied(x, y)


# Used to provide storage, lookup of occupants of sidewalk
class SWGrid:
    def __init__(self):
        self.dic = dict()  # key: position (x, y), value: Person class

    def isoccupied(self, x, y):
        # self.check_coordinates(x, y)
        return (x, y) in self.dic

    # Stores item at coordinates x, y.  Throws an exception if the coordinates are invalid.  Returns false if
    # unsuccessful (e.g., the square is occupied) or true if successful.
    def add_item(self, x, y, item):
        self.check_coordinates(x, y)
        if (x, y) in self.dic:
            return False
        self.dic[(x, y)] = item
        return True

    # Removes item from its current coordinates (which do not need to be provided) and stores it
    # at coordinates x, y.  Throws an exception if the coordinates are invalid or if the square is occupied.
    def move_item(self, x, y, item):
        self.check_coordinates(x, y)
        if self.isoccupied(x, y):
            raise Exception("Move to occupied square!")

        # Find and remove previous location.  Assumed state is valid (meaning only one entry per x,y key)
        oldloc = next(key for key, value in self.dic.items() if value == item)
        del self.dic[oldloc]
        self.add_item(x, y, item)

    # Removes item (coordinates do not need to be provided), item is agent
    # Throws an exception if the item doesn't exist.
    def remove_item(self, item):
        # Find and remove previous location.  Assumed state is valid (meaning only one entry per x,y key)
        oldloc = next(key for key, value in self.dic.items() if value == item)
        if oldloc is None:
            raise Exception('Attempt to remove non-existent item!')
        del self.dic[oldloc]

    def get_item(self, x, y):
        # self.check_coordinates(x, y)
        return self.dic.get((x, y), None)

    # Returns a list of all agents in the simulation.
    def get_list(self):
        return list(self.dic.values())

    def check_coordinates(self, x, y):
        if x < 0 or x >= SIDEWALK_LENGTH or y < 0 or y >= SIDEWALK_WIDTH:
            raise Exception("Illegal coordinates!")


#
# Run simulation - Part B
#
NUM_SIM_PEOPLE = 300

sw = Sidewalk()

# generate person (not active on the sidewalk yet)
personlist = [Person(i, sw) for i in range(NUM_SIM_PEOPLE)]

# initial infection rate
initial_infected = sum([person.infected for person in personlist])

# specify the arrival time for each person
sw.generate_arrival()

# Set up graphical display
display = plt.figure(figsize=(12, 5))
image = plt.imshow(sw.bitmap, cmap=colourmap, norm=normalizer, animated=True)

# Track time step
t = 0


# The graphical routine runs the simulation 'clock'; it calls this function at each time step.  This function
# calls the sidewalk's run_step function, as well as updating the display.  You should not implement your simulation
# here, but instead should do so in the run_step method.
def updatefigure(*args):
    global t
    t += 1

    if t % 100 == 0:
        print("Time: %d" % t)
    sw.run_step(t)
    sw.refresh_image()
    image.set_array(sw.bitmap)
    return image,


# Sets up the animation, and begins the process of running the simulation.  As configured below, it will
# run for 1000 steps.  After this point, it will simply stop, but the window will remain open.  You can close
# the window to proceed to the code below these lines (where you could add, for example, output of your statistics.
#
# You can change the speed of the simulation by changing the interval, and the duration by changing frames.
anim = FuncAnimation(display, updatefigure, frames=1000, interval=50, blit=True, repeat=False)
plt.show()
# After the figure is closed, showing the following
print("\n")
print("Done!")

# Show statistics
print("\n_____ Show Statistics _____\n")
# initial infection rate
print("Initial Infection Rate: ", initial_infected/NUM_SIM_PEOPLE)
# new infection rate
print("\nNew Infection Rate: ", sum([person.newlyinfected for person in personlist])/NUM_SIM_PEOPLE)
# infection rate at certain times
print("\nInfection Rate at Certain Times\n", sw.infection_rate_at_time)
# Total infected rate
print("\nTotal Infection Rate: ", sum([person.infected for person in personlist])/NUM_SIM_PEOPLE)


# plot infection rate at certain times
plt.plot(list(sw.infection_rate_at_time.keys()),
         [y * 100 for y in sw.infection_rate_at_time.values()])
plt.title('Percentage of Infected People at Certain Times')
plt.xlabel('Certain Time')
plt.ylabel('Infection Rate')
plt.show()
"""
Course: Simulation and Modelling
Course Code: BDA450NAA.06993.2231
Semester: Winter 2023

Name: Ran Arino
ID: 153073200
Subject: Assignment 01
Professor: Dr. Reid Kerr
Submission Date: Feb 2nd, 2023
"""

import matplotlib.pyplot as plt
import math
import astropy.coordinates

class rocketSim():
    def __init__(self):
        """
        Reset all variables and lists as 0 or blank.
        """

        # Rocket position
        self.current_xpos = 0
        self.xpos = []
        self.current_ypos = 0
        self.ypos = []
        self.current_zpos = 0
        self.zpos = []        
        self.current_rocket_pos = (0, 0, 0)
        self.rocket_pos = []        
        
        # magnitude,  bearing, trajetory
        self.current_magnitude = 0
        self.magnitudes = []
        self.current_bearing = 0
        self.bearings = []
        self.current_trajectory = 0
        self.trajectories = []
        
        # Rocket Velocity
        self.current_rocket_velocity = self.spherical_to_components(0, 0, 0)
        self.rocket_velocities = []    
       
        # engine
        self.thrustforce = 0
        self.thrusttime = 0
        self.thrust_acceleration_magnitude = 0
        
        # drag
        self.drag_force = 0
        self.drag_acceleration_magnitude = 0
        
        # gravity
        self.gravity_velocity = (0, 0, 0)
        
        # target & distance
        self.targerx = 0
        self.targety = 0
        self.distance = 0
        
        # time
        self.interval = 0
        self.t = 0.
        self.timesteps = []
    
    # Three functions below are the provided codes from a ptofessor.
    
    #Takes a 3D vector, and returns a tuple of the x, y, and z components
    def spherical_to_components(self, magnitude, bearing, trajectory):
        return astropy.coordinates.spherical_to_cartesian(magnitude, math.radians(trajectory), math.radians(bearing))
    
    #Takes the x, y, and z components of a 3D vector, and returns a tuple of magnitude, bearing, and trajectory
    def components_to_spherical(self, x, y, z):
        magnitude, trajectory, bearing = astropy.coordinates.cartesian_to_spherical(x, y, z)

        return magnitude, math.degrees(bearing.to_value()), math.degrees(trajectory.to_value())
    
    #Takes two 3D vectors (each specified by magnitude, bearing, and trajectory) and returns a
    #tuple representing the sum of the two vectors
    def add_spherical_vectors(self, magnitude1, bearing1, trajectory1, magnitude2, bearing2, trajectory2):
        x1, y1, z1 = self.spherical_to_components(magnitude1, bearing1, trajectory1)
        x2, y2, z2 = self.spherical_to_components(magnitude2, bearing2, trajectory2)

        return self.components_to_spherical(x1 + x2, y1 + y2, z1 + z2)

    
    def initialize(self,startingspeed,startingbearing,startingtrajectory,thrustforce,thrusttime,targetx,targety,timeinterval):
        """
        Reset all variables and lists that we will use as the initial value(s)
        """

        # Rocket position
        self.current_xpos = 0
        self.xpos = [self.current_xpos]
        self.current_ypos = 0
        self.ypos = [self.current_ypos]
        self.current_zpos = 2.2
        self.zpos = [self.current_zpos]        
        self.current_rocket_pos = (self.current_xpos, self.current_ypos, self.current_zpos)
        self.rocket_pos = [self.current_rocket_pos]        
        
        # magnitude,  bearing, trajetory
        self.current_magnitude = startingspeed
        self.magnitudes = [self.current_magnitude]
        self.current_bearing = startingbearing
        self.bearings = [self.current_bearing]
        self.current_trajectory = startingtrajectory
        self.trajectories = [self.current_trajectory]
        
        # Rocket Velocity
        self.current_rocket_velocity = self.spherical_to_components(self.current_magnitude, self.current_bearing, self.current_trajectory)
        self.rocket_velocities = [self.current_rocket_velocity]
        
        # engine
        self.thrustforce = thrustforce
        self.thrusttime = thrusttime        
        self.thrust_acceleration_magnitude = 0

        # drag
        self.drag_force = 0
        self.drag_acceleration_magnitude = 0
        
        # gravity
        self.gravity_velocity = (0, 0, 0)        
        
        # target
        self.targetx = targetx
        self.targety = targety
        
        # time
        self.interval = timeinterval
        self.t = 0.
        self.timesteps = [self.t]        
    
    def observe(self):
        # record positions of (x, y, z)
        self.xpos.append(self.current_xpos)
        self.ypos.append(self.current_ypos)
        self.zpos.append(self.current_zpos)
        self.rocket_pos.append(self.current_rocket_pos)
        
        # record velocity
        self.magnitudes.append(self.current_magnitude)
        self.bearings.append(self.current_bearing)
        self.trajectories.append(self.current_trajectory)
        
        # record rocket's position change based on the velocity vector
        self.rocket_velocities.append(self.current_rocket_velocity)
        
        # record the time
        self.timesteps.append(self.t)
        
    
    def update(self):       
        # get current (x, y, z)
        self.current_xpos = self.xpos[-1] + self.rocket_velocities[-1][0] * self.interval
        self.current_ypos = self.ypos[-1] + self.rocket_velocities[-1][1] * self.interval
        self.current_zpos = self.zpos[-1] + self.rocket_velocities[-1][2] * self.interval
        self.current_rocket_pos = (self.current_xpos, self.current_ypos, self.current_zpos)  
        
        # thrust
        if self.t < self.thrusttime:
            self.thrust_acceleration_magnitude = self.thrustforce / 5
            self.current_magnitude, self.current_bearing, self.current_trajectory = \
            self.add_spherical_vectors(*self.components_to_spherical(*self.rocket_velocities[-1]),
                                       self.thrust_acceleration_magnitude*self.interval, self.current_bearing, self.current_trajectory)

        
        # drag
        self.drag_force = 0.006 * self.current_magnitude**2
        self.drag_acceleration_magnitude = self.drag_force/5
        self.current_magnitude, self.current_bearing, self.current_trajectory = \
            self.add_spherical_vectors(self.current_magnitude, self.current_bearing, self.current_trajectory, \
                                       (-1)*self.drag_acceleration_magnitude*self.interval, self.current_bearing, self.current_trajectory)

        
        # gravity
        self.gravity_velocity = (9.8*self.interval, 0, -90)
        self.current_magnitude, self.current_bearing, self.current_trajectory = \
            self.add_spherical_vectors(self.current_magnitude, self.current_bearing, self.current_trajectory, *self.gravity_velocity)
        

        # update the rocket velocity
        self.current_rocket_velocity = self.spherical_to_components(self.current_magnitude, self.current_bearing, self.current_trajectory)
                
        # update the time
        self.t = self.t + self.interval
        
       
    def runsim(self, startingspeed, startingbearing, startingtrajectory, thrustforce, thrusttime, targetx, targety, timeinterval):
        self.initialize(startingspeed, startingbearing, startingtrajectory, thrustforce, thrusttime, targetx, targety, timeinterval)
        while self.zpos[-1] > 0.0:
            self.update()
            self.observe()
        
        self.distance = math.sqrt( (self.targetx-self.xpos[-1])**2 + (self.targety-self.ypos[-1])**2 )
        print("Distance from target: ", self.distance)
        

        plt.figure(1)
        plt.title("X Position vs. time")
        plt.xlabel("t (s)")        
        plt.ylabel("x (Distance) (m)")
        plt.plot(self.timesteps, self.xpos)
        
        plt.figure(2)
        plt.title("X Position vs. Z Position (Height)")
        plt.xlabel("x (Distance) (m)")
        plt.ylabel("z (Height) (m)")
        plt.plot(self.xpos, self.zpos)
        plt.xlim([0, self.xpos[-1]+100])
        
        plt.figure(3)
        plt.title("X Position vs. Y Position (Height)")
        plt.xlabel("x (Distance) (m)")
        plt.ylabel("y (Distance) (m)")
        plt.plot(self.xpos, self.ypos)
        plt.plot(self.targetx, self.targety, 'ro')
        
        plt.figure(4)
        ax = plt.axes(projection='3d')
        plt.title("3D Path of Rocket")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        plt.plot(self.xpos, self.ypos, self.zpos)
        plt.plot(targetx, targety, 0, 'ro')
        
        plt.figure(5)
        plt.title("Velocity vs. time")        
        plt.xlabel("time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.plot(self.timesteps, self.magnitudes)
        plt.xlim([0, self.t+5])
        
        plt.figure(6)
        plt.title("Trajectory vs. time")        
        plt.xlabel("time (s)")
        plt.ylabel("Trajectory (degrees)")
        plt.plot(self.timesteps, self.trajectories)
        plt.xlim([0, self.t+5])
        
        
if __name__ == '__main__':
    sim = rocketSim()
    # here is the exmaple
    # sim.runsim(5, 45, 85, 500, 10, 1000, 1200, 0.01)
    
    # here is the best parameter I found ; the distance less than 5m.
    sim.runsim(5, 57.95, 76.85, 500, 10, 1000, 1600, 0.01)
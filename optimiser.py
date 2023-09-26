# Import Gymnasium modules
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os
from math import pi

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# AIM
# Replicate race conditions for the BWSC in the Australian outback

SPEED_UP_TRIGGER = 2
SLOW_DOWN_TRIGGER = 0
STEP_MAGNITUDE = 5
MAX_SPEED = 100
MIN_SPEED = 0
SR7_CUMULATIVE_MASS = 760
INTERVAL_TIME = 30


class BSWCEnv(Env):
    def __init__(self):
        
        # Intialise assumptions as constants, can be changed later to replicate alternate conditions in the BWSC race
        self.stint_length = 100.0                           # Km
        self.full_battery = 136800000                       # Maximum battery capacity (J), sourced from constants.py
        self.remaining_battery = self.full_battery
        self.energy_safety_buffer = int(self.full_battery * 0.2) # 20% safety buffer to aim to not fall under before reaching a Control Stop
        self.velocity = 50.0                                # Km/h
        self.height = 0                                     # (m) Height from initial starting point, not sea elevation
        self.global_reward = 0 
        self.bwsc_speed_max = 100.0                         # Km/h
        self.deadline = 3600                                # Maximum stint time, this current config assumes 100km in an hour.
        self.kms = 0
        self.used_energy = 0
        self.time = 0
        
        
        self.action_space = Discrete(3)                     # 0: Slow down -5km/h, 1: Steady, 2: Speed up +5km/h
        self.observation_space = Dict({
            "remaining_kms": Box(low=0.0, high=self.stint_length),
            "battery_remaining": Box(low=self.energy_safety_buffer, high=self.full_battery),
            "height": Box(low=-50.0, high=50.0),    # Height in (m) from original position
            "velocity": Box(low=0.0, high=self.bwsc_speed_max)
        })
        
    def ecm(self, vf, vi, hf, hi):
        '''
        Limitiations
            - No consideration of friction
            - No regenerative braking consideration when going down a slope for example and we do not want to exceed local speed limits
            - No consideration of additional energy input for incident wind
            - No consideration of energy output pending on terrain type. Eg: tarmac would require less energy than a dirt road.
        '''
        return (0.5 * SR7_CUMULATIVE_MASS * (vf**2 - vi**2)) + (SR7_CUMULATIVE_MASS * 9.8 *(hf - hi))
    
    def update_obs(self):
        return {
            "remaining_kms": np.array([float((self.stint_length - self.kms))], dtype=np.float32),
            "battery_remaining": np.array([float(min(self.full_battery - self.used_energy, self.full_battery))], dtype=np.float32),
            "height": np.array([float(self.height - self.prev_height)], dtype=np.float32),
            "velocity": np.array([float(self.velocity - self.prev_velocity)], dtype=np.float32),
        }
        
    def step(self, action):
        
        done = False
        curr_reward = 0
        
        self.prev_velocity = self.velocity
        self.prev_height = self.height
        
        if action == SPEED_UP_TRIGGER:
            self.velocity += STEP_MAGNITUDE
        elif action == SLOW_DOWN_TRIGGER:
            self.velocity -= STEP_MAGNITUDE
        else: 
            pass
    
        # Reward adjustments in different circumstances pending observation state
        #   - If the new velcoity is over or under MAX or MIN, penalise by some factor.
        #   - If relative energy usage at the current instance exceeds the recommended energy safety buffer at that instance, penalise.
        #   - If the current time lags behind the recommended time to reach the Control Stop, penalise.
        #   - Else, if the car exists within these constraints, reward the progress.
        exceed_speed = False
        over_energy_buffer = False
        over_time_recommendation = False
        
        # If the new velcoity is over or under MAX or MIN, penalise by some factor.
        if self.velocity < MIN_SPEED: 
            self.velocity = MIN_SPEED
            curr_reward -= 1
            exceed_speed = True
        elif self.velocity > MAX_SPEED:
            self.velocity = MAX_SPEED
            curr_reward -= 1
            exceed_speed = True
        
        self.kms += (self.velocity * INTERVAL_TIME) / 3600
        
        # If relative energy usage at the current instance exceeds the recommended energy safety buffer at that instance, penalise.
        self.used_energy += self.ecm(self.velocity,
                               self.prev_velocity,
                               self.height,
                               self.prev_height)
        
        # Suggested Buffer: Shows recommended Js to use for the kms travelled
        # Actual Buffer: Shows the energy we have chewed up for the same kms
        self.remaining_battery -= self.used_energy
        actual_buffer = self.used_energy * self.kms
        suggested_buffer = ((self.full_battery - self.energy_safety_buffer) / self.stint_length) * self.kms
        energy_overuse_penalty = 20     # Heavy penalty since not making a Control Stop is highly frowned upon
        if actual_buffer > suggested_buffer:
            curr_reward -= energy_overuse_penalty
            over_energy_buffer = True
        
        # If the current time lags behind the recommended time to reach the Control Stop, penalise.
        self.time += INTERVAL_TIME          # Time in seconds conversion
        average_velocity = self.stint_length / self.deadline
        if average_velocity > self.velocity:
            over_time_recommendation = True
            curr_reward -= 5
        
        # If within all these bounds, reward the car
        if not exceed_speed and not over_energy_buffer and not over_time_recommendation:
            curr_reward += 5
            
        # Check done conditions
        if self.kms > self.stint_length or self.remaining_battery < self.energy_safety_buffer or self.time > self.deadline:
            done = True
            lateness = max(0, self.time - self.deadline)
            curr_reward += (0.99**lateness) / max(self.used_energy, 1.0)
            
        if done:
            print("Finished")
        
        self.global_reward += curr_reward
        
        updated_obs = self.update_obs()
        
        return updated_obs, curr_reward, done, False, {}
    
    def render(self):
        pass
    
    def update_reset_obs_type(self):
        pass
    
    def reset(self, seed=None):
        # Intialise assumptions as constants, can be changed later to replicate alternate conditions in the BWSC race
        self.stint_length = 100.0                           # Km
        self.full_battery = 136800000                       # Maximum battery capacity (J), sourced from constants.py
        self.remaining_battery = self.full_battery
        self.energy_safety_buffer = int(self.full_battery * 0.2) # 20% safety buffer to aim to not fall under before reaching a Control Stop
        self.velocity = 50.0                                # Km/h
        self.prev_velocity = 0
        self.height = 0                                     # (m) Height from initial starting point, not sea elevation
        self.prev_height = 0
        self.global_reward = 0 
        self.bwsc_speed_max = 100.0                         # Km/h
        self.deadline = 3600                                # Maximum stint time, this current config assumes 100km in an hour.
        self.kms = 0
        self.used_energy = 0
        self.time = 0

        return self.update_obs(), {}


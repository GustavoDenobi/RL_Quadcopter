import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10., 0., 0., 0.]) 

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)    
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0.
        weights = [0.05, 0.05, .9, 0.0, 0.0, 0.0]
        for i in range(3):
            error = abs(self.sim.pose[i] - self.target_pos[i])
            if error <= 1:
                reward += weights[i]*(-0.5*np.log(0.5*abs(error))) # penalties for being off the target
            else:
                reward += weights[i]*(-0.5*np.log(0.5*abs(error)))
        if self.sim.pose[2] <= 0.1:
            reward = -1 # if it crashes, reward clips to -1
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        return reward

#    def get_reward(self):
#        """Uses current pose of sim to return reward."""
#        reward = 0.
#        weights = [0.4, 0.4, 1.2, .0, .0, .0]
#        #weights = self.softmax(weights)
#        for i in range(3):
#            reward -= weights[i]*(np.tanh((abs(self.sim.pose[i] - self.target_pos[i])/5))) # penalties for being off the target
#        if self.sim.pose[2] <= 0.05:
#            reward = -1 # if it crashes, reward clips to -1
#        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
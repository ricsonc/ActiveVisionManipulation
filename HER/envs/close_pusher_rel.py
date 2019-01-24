from HER.envs import close_pusher
import numpy as np 

class BaxterEnv(close_pusher.BaxterEnv):
    
    def __init__(self, max_len=50):
        super(BaxterEnv, self).__init__(max_len=max_len)

    def _get_rel_ob(self, absolute_ob):
        gripper_pos = absolute_ob[:self.space_dim]
        object_pos = absolute_ob[self.space_dim:2*self.space_dim]
        target_pos = absolute_ob[-self.space_dim:]

        return np.concatenate((gripper_pos, object_pos - gripper_pos, target_pos - object_pos))

    def _get_abs_ob(self, rel_ob):
        gripper_pos = rel_ob[:self.space_dim]
        object_pos = rel_ob[self.space_dim:2*self.space_dim] + gripper_pos
        target_pos = rel_ob[-self.space_dim:] + object_pos

        return np.concatenate((gripper_pos, object_pos, target_pos))

    def reset_model(self,
        # initial config of the end-effector
            gripper_pos = np.array([0.6 , 0.3 , 0.12]),
            add_noise = False
        ):

        absolute_ob = super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos, add_noise = add_noise)
        return self._get_rel_ob(absolute_ob)

    def step(self, action):
        ob, total_reward, done, info = super(BaxterEnv, self).step(action)
        return self._get_rel_ob(ob), total_reward, done, info

    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = goal_state[self.space_dim:2*self.space_dim]    ## this is the absolute goal location = obj last loc
        # enter the last state in the list
        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []
        # make corrections in the first state 
        states[0][-self.space_dim:] = goal.copy() - states[0][self.space_dim:2*self.space_dim]
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()  - state[self.space_dim:2*self.space_dim]  # copy the new goal into state
            
            absolute_state = self._get_abs_ob(state)
            reward = self.calc_reward(absolute_state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards


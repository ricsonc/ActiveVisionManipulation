from HER.envs import reacher2d
import numpy as np

class BaxterEnv(reacher2d.BaxterEnv):
    
    def _get_rel_ob(self, absolute_ob):
        gripper_pos = absolute_ob[:self.space_dim]
        target_pos = absolute_ob[-self.space_dim:]

        return np.concatenate((gripper_pos, target_pos - gripper_pos))

    def _get_abs_ob(self, rel_ob):
        gripper_pos = rel_ob[:self.space_dim]
        target_pos = rel_ob[-self.space_dim:] + gripper_pos

        return np.concatenate((gripper_pos, target_pos))

    def reset_model(self,
        # initial config of the end-effector
            gripper_pos = np.array([0.6 , 0.3 , 0.15]),
            ctrl = None,
            add_noise = True
        ):

        absolute_ob = super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos, ctrl = ctrl, add_noise = add_noise)
        return self._get_rel_ob(absolute_ob)

    def step(self, action):
        ob, total_reward, done, info = super(BaxterEnv, self).step(action)
        return self._get_rel_ob(ob), total_reward, done, info

    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = goal_state[:self.space_dim]    ## this is the absolute goal location = gripper last loc
        # enter the last state in the list
        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []
        # make corrections in the first state 
        states[0][-self.space_dim:] = goal.copy() - states[0][:self.space_dim]
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()  - state[:self.space_dim]  # copy the new goal into state
            
            absolute_state = self._get_abs_ob(state)
            #print(state, absolute_state)
            reward = self.calc_reward(absolute_state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards


    # calc_reward is working with abs states as it is implemented in reacher2d


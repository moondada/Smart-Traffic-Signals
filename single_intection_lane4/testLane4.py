from __future__ import absolute_import
from __future__ import print_function
from select import select
import termios
import os
import sys
import optparse
import subprocess
import random
import time
#import cv2
import curses
#import readscreen3
import numpy as np
import pandas as pd
import datetime
from time import time
import matplotlib.pyplot as plt
from operator import add

import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
from parl.utils import logger

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

options = get_options()

# this script has been called from the command line. It will start sumo as a
# server, then connect and run

if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

# first, generate the route file for this simulation

# this is the normal way of using traci. sumo is started as a
# subprocess and then the python script connects and runs


print("TraCI Started")


# State = State_Lengths()
# print(State.get_tails())

# states = State.get_tails


# runner = Runner()
# print(Runner().run)


def getPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("0")
    phaseState = np.zeros((transition_time,num_lanes,num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState


def getState(transition_time):  # made the order changes
    newState = []
    avg_qlength = 0
    # transition_time_step_leftcount = 0
    # transition_time_step_rightcount = 0
    # transition_time_step_topcount = 0
    # transition_time_step_bottomcount = 0
    avg_leftcount = 0
    avg_rightcount = 0
    avg_bottomcount = 0
    avg_topcount = 0
    for _ in range(transition_time):
        traci.simulationStep()




        leftcount = 0
        rightcount = 0
        topcount = 0
        bottomcount = 0
        vehicleList = traci.vehicle.getIDList()

        print("Traffic : ")

        for id in vehicleList:
            x, y = traci.vehicle.getPosition(id)

            if x<110 and x>60 and y<130 and y>120:
                leftcount+=1
            else :
                if x<120 and x>110 and y<110 and y>600:
                    bottomcount+=1
                else :
                    if x<180 and x>130 and y<120 and y>110:
                        rightcount+=1
                    else :
                        if x<130 and x>120 and y<180 and y>130:
                            topcount+=1

        print("Left : ", leftcount)
        print("Right : ", rightcount)
        print("Top : ", topcount)
        print("Bottom : ", bottomcount)

        avg_topcount += topcount
        avg_bottomcount += bottomcount
        avg_leftcount += leftcount
        avg_rightcount += rightcount

        # transition_time_step_bottomcount+= bottomcount
        # transition_time_step_leftcount+= leftcount
        # transition_time_step_rightcount+= rightcount
        # transition_time_step_topcount+= topcount

        state = [bottomcount / 40,
                 rightcount / 40,
                 topcount / 40,
                 leftcount / 40
                 ]

        avg_qlength += ((bottomcount + rightcount + topcount + leftcount)/4)
        newState.insert(0, state)
    # print (state)

    # df = pd.DataFrame([[, 2]], columns=['a', 'b'])
    # params_dict =
    avg_qlength /= transition_time
    avg_leftcount /= transition_time
    avg_topcount /= transition_time
    avg_rightcount /= transition_time
    avg_bottomcount /= transition_time

    avg_lane_qlength = [avg_leftcount, avg_topcount, avg_rightcount, avg_bottomcount]
    newState = np.array(newState)
    phaseState = getPhaseState(transition_time)
    newState = np.dstack((newState, phaseState))
    newState = np.expand_dims(newState, axis=0)
    return newState, avg_qlength, avg_lane_qlength


print("here")
import traci


def makeMove(action, transition_time):
    if action == 1:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)


    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()

    return getState(transition_time)


def getReward(this_state, this_new_state):
    num_lanes = 4
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][0][i][0])
        qLengths2.append(this_new_state[0][0][i][0])

    qLengths11 = [x + 1 for x in qLengths1]
    qLengths21 = [x + 1 for x in qLengths2]

    q1 = np.prod(qLengths11)
    q2 = np.prod(qLengths21)

    # print("Old State with product : ", q1)
    #
    # print("New State with product : ", q2)
    #
    #
    # if q1 > q2:
    #     this_reward = 1
    # else:
    #     this_reward = -1
    this_reward = q1 - q2

    if this_reward > 0:
        this_reward = 1
    elif this_reward < 0:
        this_reward = -1
    elif q2 > 1:
        this_reward = -1
    else:
        this_reward = 0

    return this_reward

def getRewardAbsolute(this_state, this_new_state):
    num_lanes = 4
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][0][i][0])
        qLengths2.append(this_new_state[0][0][i][0])

    qLengths11 = [x + 1 for x in qLengths1]
    qLengths21 = [x + 1 for x in qLengths2]

    q1 = np.prod(qLengths11)
    q2 = np.prod(qLengths21)

    # print("Old State with product : ", q1)
    #
    # print("New State with product : ", q2)
    #
    #
    # if q1 > q2:
    #     this_reward = 1
    # else:
    #     this_reward = -1
    this_reward = q1 - q2

    return this_reward



def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


class Model(parl.Model):
    def __init__(self, act_dim):

        hid1_size = 128
        hid2_size = 64
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)

        return Q



from parl.algorithms import DQN

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 10  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim, 4, 5], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim, 4, 5], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim, 4, 5], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.4, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)



# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(agent, render=False):
    traci.start([sumoBinary, "-c", "data/test.sumocfg",#(lane4.sumocfg, test.sumocfg)
             "--tripinfo-output", "tripinfo.xml"])

    traci.trafficlight.setPhase("0", 0)

    AVG_Q_len_perepisode = []
    sum_q_lens = 0
    length_data_avg = []
    count_data = []
    delay_data_avg = []
    delay_data_min = []
    delay_data_max = []
    delay_data_time = []
    current_left_time = 0
    current_top_time = 0
    current_bottom_time = 0
    current_right_time = 0
    overall_lane_qlength = [0, 0, 0, 0]
    num_cycles = 0
    num_qlength_instances = 0

    traci.load(["--start", "-c", "data/test.sumocfg",#(lane4.sumocfg, test.sumocfg)
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)

    obs = getState(agent.obs_dim)
    while traci.simulation.getMinExpectedNumber() > 0:
        prev_phase = traci.trafficlight.getPhase("0")
        #print(obs)
        action = agent.predict(obs[0].flatten())  # 预测动作，只选最优动作

        next_obs, qlength, avg_lane_qlength = makeMove(action, transition_time)
        new_phase = traci.trafficlight.getPhase("0")

        #reward = getRewardAbsolute(obs, next_obs)
        #done = True #待定

        print("Previous phase = ", prev_phase)
        print("New phase = ", new_phase)
        vehicleList = traci.vehicle.getIDList()
        num_vehicles = len(vehicleList)
        print("Number of cycles = ", num_cycles)
        if num_vehicles:
            avg = 0
            max = 0
            mini = 100
            for id in vehicleList:
                time = traci.vehicle.getAccumulatedWaitingTime(id)
                if time > max:
                    max = time

                if time < mini:
                    mini = time

                avg += time
            avg /= num_vehicles
            delay_data_avg.append(avg)
            delay_data_max.append(max)
            delay_data_min.append(mini)
            length_data_avg.append(qlength)
            count_data.append(num_vehicles)
            delay_data_time.append(traci.simulation.getCurrentTime() / 1000)

            if traci.simulation.getCurrentTime() / 1000 < 2100:
                overall_lane_qlength = list(map(add, overall_lane_qlength, avg_lane_qlength))
                num_qlength_instances += 1
                if prev_phase == 3 and new_phase == 0:
                    num_cycles += 1
                if prev_phase == 0:
                    current_bottom_time += transition_time
                if prev_phase == 1:
                    current_right_time += transition_time
                if prev_phase == 2:
                    current_top_time += transition_time
                if prev_phase == 3:
                    current_left_time += transition_time
        obs = next_obs
    overall_lane_qlength[:] = [x / num_qlength_instances for x in overall_lane_qlength]
    current_right_time /= num_cycles
    current_top_time /= num_cycles
    current_left_time /= num_cycles
    current_bottom_time /= num_cycles
    avg_free_time = [current_left_time, current_top_time, current_right_time, current_bottom_time]

    plt.plot(delay_data_time, delay_data_avg, 'b-', label='avg')
    #plt.plot(delay_data_time, delay_data_min, 'g-', label='min')
    #plt.plot(delay_data_time, delay_data_max,'r-', label='max')
    plt.legend(loc='upper left')
    plt.ylabel('Waiting time per minute')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    plt.plot(delay_data_time, length_data_avg, 'b-', label='avg')
    plt.legend(loc='upper left')
    plt.ylabel('Average Queue Length')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    plt.plot(delay_data_time, count_data, 'b-', label='avg')
    plt.legend(loc='upper left')
    plt.ylabel('Average Number of Vehicles in Map')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    label = ['Obstacle Lane reward', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, avg_free_time, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Green Time per Cycle (in s)')
    plt.xticks(index, label)
    axes = plt.gca()
    axes.set_ylim([0,60])

    plt.figure()
    label = ['Obstacle Lane reward', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, overall_lane_qlength, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Q-length every 8 seconds')
    plt.xticks(index, label)
    axes = plt.gca()
    axes.set_ylim([0,20])
    plt.show()

    AVG_Q_len_perepisode.append(sum_q_lens / 702)
    sum_q_lens = 0

    #return np.mean(eval_reward)



LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 8000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 350  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.9 # reward 的衰减因子，一般取 0.9 到 0.999 不等

LEARNING_RATE = 0.00025 # 学习率


# 创建环境
action_dim = 2
transition_time = 8
obs_shape = (transition_time, 4, 5)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池


model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.99,  # 有一定概率随机选取动作，探索
    e_greed_decrement=2*1e-4)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
ckpt = 'amodel/lane4_model.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
agent.restore(ckpt)

# test part
evaluate(agent, render=False)  # render=True 查看显示效果

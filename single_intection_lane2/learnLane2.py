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

import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
from parl.utils import logger
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
    # transition_time_step_leftcount = 0
    # transition_time_step_rightcount = 0
    # transition_time_step_topcount = 0
    # transition_time_step_bottomcount = 0
    for _ in range(transition_time):
        traci.simulationStep()



        #应当是各车道车辆数
        leftcount = 0
        rightcount = 0
        topcount = 0
        bottomcount = 0
        vehicleList = traci.vehicle.getIDList()

        print("Traffic : ")

        for id in vehicleList:
            x, y = traci.vehicle.getPosition(id)

            if x < 110 and x > 60 and y < 130 and y > 120:
                leftcount += 1
            else:
                if x < 120 and x > 110 and y < 110 and y > 60:
                    bottomcount += 1
                else:
                    if x < 180 and x > 130 and y < 120 and y > 110:
                        rightcount += 1
                    else:
                        if x < 130 and x > 120 and y < 180 and y > 130:
                            topcount += 1

        print("Left : ", leftcount)
        print("Right : ", rightcount)
        print("Top : ", topcount)
        print("Bottom : ", bottomcount)

        # transition_time_step_bottomcount+= bottomcount
        # transition_time_step_leftcount+= leftcount
        # transition_time_step_rightcount+= rightcount
        # transition_time_step_topcount+= topcount

        state = [bottomcount / 40,
                 rightcount / 40,
                 topcount / 40,
                 leftcount / 40
                 ]


        newState.insert(0, state)
    # print (state)

    # df = pd.DataFrame([[, 2]], columns=['a', 'b'])
    # params_dict =
    newState = np.array(newState)
    phaseState = getPhaseState(transition_time)
    newState = np.dstack((newState, phaseState))
    newState = np.expand_dims(newState, axis=0)
    return newState


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
    this_reward_cubic = this_reward * this_reward * this_reward

    return this_reward_cubic



def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)

class Model(parl.Model):
    def __init__(self, act_dim):

        hid1_size = 32
        hid2_size = 32
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
        self.update_target_steps = 10  # 每隔10个training steps再把model的参数复制到target_model中

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
            0.3, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
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

# 训练一个episode
def run_episode(agent, rpm, episode = 1):
    total_reward = 0
    obs = getState(agent.obs_dim)
    step = 0
    sum_q_lens = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        #交通灯不能一直关着，否则惩罚
        same_action_count = 0
        for temp in reversed(rpm.buffer):
            if temp[1] == 0:
                same_action_count += 1
            else:
                break
        if same_action_count == 20:
            action = 1
            print("SAME ACTION PENALTY")

        next_obs = makeMove(action, agent.obs_dim)
        reward = getRewardAbsolute(obs, next_obs)
        done = True #待定
        rpm.append((obs, action, reward, next_obs, done))
        sum_q_lens += np.average(next_obs)

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        #if done:
            #break
    return total_reward, sum_q_lens


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = getState(agent.obs_dim)
        episode_reward = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            next_obs = makeMove(agent.obs_dim, action)
            reward = getRewardAbsolute(obs, next_obs)
            done = True #待定
            episode_reward += reward
            #if render:
                #env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)



LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 8000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 350  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.9 # reward 的衰减因子，一般取 0.9 到 0.999 不等

LEARNING_RATE = 0.0025 # 学习率


# 创建环境
action_dim = 2
transition_time = 8
obs_shape = (transition_time, 4, 5)

sum_q_lens = 0
AVG_Q_len_perepisode = []

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池


model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=2*1e-4)  # 随着训练逐步收敛，探索的程度慢慢降低

traci.start([sumoBinary, "-c", "data/lane2.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    if traci.simulation.getMinExpectedNumber() <= 0:
        traci.load(["--start", "-c", "data/lane2.sumocfg",
                    "--tripinfo-output", "tripinfo.xml"])
    run_episode(agent, rpm, 1)

max_episode = 100

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    traci.load(["--start", "-c", "data/lane2.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)
    # train part
    for i in range(1):#(0, 50):
        total_reward, sum_q_lens = run_episode(agent, rpm, episode)
        episode += 1
    AVG_Q_len_perepisode.append(sum_q_lens / 702)

print(AVG_Q_len_perepisode)
import matplotlib.pyplot as plt
x1=range(len(AVG_Q_len_perepisode))
plt.plot(x1,AVG_Q_len_perepisode,linewidth=3,color='r',marker='o',
markerfacecolor='blue',markersize=12)
plt.xlabel('episode')
plt.ylabel('AVG_Q_len')
plt.title('AVG_Q_len_per_episode')
plt.legend()
plt.show() 

# 训练结束，保存模型
save_path = './amodel/dqn_model.ckpt'
agent.save(save_path)


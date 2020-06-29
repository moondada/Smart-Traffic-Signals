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
#import readscreen3
import numpy as np
import pandas as pd
import datetime
from time import time
import matplotlib.pyplot as plt
from operator import add




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



def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


num_episode = 1
discount_factor = 0.9
#epsilon = 1
epsilon_start = 1
epsilon_end = 0.4
epsilon_decay_steps = 3000

Average_Q_lengths = []

params_dict = [] #for graph writing
sum_q_lens = 0
AVG_Q_len_perepisode = []

transition_time = 40
target_update_time = 20
replay_memory_init_size = 350
replay_memory_size = 8000
batch_size = 32
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

#generate_routefile_random(episode_time, num_vehicles)
#generate_routefile(290,10)
traci.start([sumoBinary, "-c", "data/test.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 2

total_t = 0
for episode in range(num_episode):

    traci.load(["--start", "-c", "data/test.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)


    counter = 0
    stride = 0

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

    while traci.simulation.getMinExpectedNumber() > 0:
        print("Episode # ", episode)
        # print("Waiting time on lane 1i_0 = ",getWaitingTime("1i_0"))

        print("Inside episode counter", counter)

        counter += 1
        total_t += 1
        # batch_experience = experience[:batch_history]
        prev_phase = traci.trafficlight.getPhase("0")
        new_state, qlength, avg_lane_qlength = makeMove(1, transition_time)
        new_phase = traci.trafficlight.getPhase("0")
        print("Previous phase = ", prev_phase)
        print("New phase = ", new_phase)
        vehicleList = traci.vehicle.getIDList()
        num_vehicles = len(vehicleList)
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

    overall_lane_qlength[:] = [x / num_qlength_instances for x in overall_lane_qlength]
    current_right_time /= num_cycles
    current_top_time /= num_cycles
    current_left_time /= num_cycles
    current_bottom_time /= num_cycles
    avg_free_time = [current_left_time, current_top_time, current_right_time, current_bottom_time]

    print(delay_data_time)
    print(delay_data_avg)
    print(length_data_avg)
    print(count_data)


    plt.plot(delay_data_time, delay_data_avg, 'b-', label='avg')
    #plt.plot(delay_data_time, delay_data_min, 'g-', label='min')
    #plt.plot(delay_data_time, delay_data_max, 'r-', label='max')
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
    label = ['Obstacle Lane Uniform', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, avg_free_time, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Green Time per Cycle (in s)')
    plt.xticks(index, label)
    axes = plt.gca()
    axes.set_ylim([0, 60])

    plt.figure()
    label = ['Obstacle Lane Uniform', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, overall_lane_qlength, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Q-length every 8 seconds')
    plt.xticks(index, label)
    axes = plt.gca()
    axes.set_ylim([0, 20])
    plt.show()









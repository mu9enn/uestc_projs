import skfuzzy as fuzz
import numpy as np

# Define membership functions for sludge
def define_sludge_membership():
    sludge = {}
    sludge['SD'] = fuzz.trimf(np.arange(0, 101, 1), [0, 0, 50])
    sludge['MD'] = fuzz.trimf(np.arange(0, 101, 1), [0, 50, 100])
    sludge['LD'] = fuzz.trimf(np.arange(0, 101, 1), [50, 100, 100])
    return sludge

# Define membership functions for grease
def define_grease_membership():
    grease = {}
    grease['NG'] = fuzz.trimf(np.arange(0, 101, 1), [0, 0, 50])
    grease['MG'] = fuzz.trimf(np.arange(0, 101, 1), [0, 50, 100])
    grease['LG'] = fuzz.trimf(np.arange(0, 101, 1), [50, 100, 100])
    return grease

# Define membership functions for washing time
def define_washing_time_membership():
    washing_time = {}
    washing_time['VS'] = fuzz.trimf(np.arange(0, 121, 1), [0, 0, 30])
    washing_time['S'] = fuzz.trimf(np.arange(0, 121, 1), [0, 30, 60])
    washing_time['M'] = fuzz.trimf(np.arange(0, 121, 1), [30, 60, 90])
    washing_time['L'] = fuzz.trimf(np.arange(0, 121, 1), [60, 90, 120])
    washing_time['VL'] = fuzz.trimf(np.arange(0, 121, 1), [90, 120, 120])
    return washing_time

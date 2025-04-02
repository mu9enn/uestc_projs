import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from membership import define_sludge_membership, define_grease_membership, define_washing_time_membership


# Define fuzzy rules
def define_fuzzy_rules(sludge, grease, washing_time):
    rule1 = ctrl.Rule(sludge['SD'] & grease['NG'], washing_time['VS'])
    rule2 = ctrl.Rule(sludge['SD'] & grease['MG'], washing_time['M'])
    rule3 = ctrl.Rule(sludge['SD'] & grease['LG'], washing_time['L'])
    rule4 = ctrl.Rule(sludge['MD'] & grease['NG'], washing_time['S'])
    rule5 = ctrl.Rule(sludge['MD'] & grease['MG'], washing_time['M'])
    rule6 = ctrl.Rule(sludge['MD'] & grease['LG'], washing_time['L'])
    rule7 = ctrl.Rule(sludge['LD'] & grease['MG'], washing_time['M'])
    rule8 = ctrl.Rule(sludge['LD'] & grease['LG'], washing_time['L'])
    rule9 = ctrl.Rule(sludge['LD'] & grease['LG'], washing_time['VL'])

    return [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]


def run_fuzzy_control_system(sludge_input, grease_input, _plot=True):
    # Define the variables
    sludge = ctrl.Antecedent(np.arange(0, 101, 1), 'sludge')
    grease = ctrl.Antecedent(np.arange(0, 101, 1), 'grease')
    washing_time = ctrl.Consequent(np.arange(0, 121, 1), 'washing_time')

    # Define the membership functions
    sludge_membership = define_sludge_membership()
    grease_membership = define_grease_membership()
    washing_time_membership = define_washing_time_membership()

    # Set the membership functions
    sludge['SD'] = sludge_membership['SD']
    sludge['MD'] = sludge_membership['MD']
    sludge['LD'] = sludge_membership['LD']

    grease['NG'] = grease_membership['NG']
    grease['MG'] = grease_membership['MG']
    grease['LG'] = grease_membership['LG']

    washing_time['VS'] = washing_time_membership['VS']
    washing_time['S'] = washing_time_membership['S']
    washing_time['M'] = washing_time_membership['M']
    washing_time['L'] = washing_time_membership['L']
    washing_time['VL'] = washing_time_membership['VL']

    # Define fuzzy rules
    rules = define_fuzzy_rules(sludge, grease, washing_time)

    # Create control system
    washing_ctrl = ctrl.ControlSystem(rules)
    washing = ctrl.ControlSystemSimulation(washing_ctrl)

    # Input values
    washing.input['sludge'] = sludge_input
    washing.input['grease'] = grease_input

    # Perform fuzzy reasoning
    washing.compute()

    # Get the output
    washing_time_output = washing.output['washing_time']
    print(f"Washing time: {washing_time_output} minutes")

    # Plot the results
    if _plot:
        washing_time.view(sim=washing)
    return washing_time_output

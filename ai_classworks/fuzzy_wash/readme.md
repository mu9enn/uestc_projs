The experiment you're working on requires structuring your fuzzy reasoning algorithm into different files for better readability and organization. Below is a breakdown of how the code can be structured and improved. Each part of the fuzzy reasoning system will be placed into separate files to ensure modularity and reusability.
```shell
  pip install -U scikit-fuzzy
```
see https://github.com/scikit-fuzzy/scikit-fuzzy

### File 1: `membership_functions.py`
This file defines the membership functions for sludge, grease, and washing time.

```python
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
```

### File 2: `fuzzy_rules.py`
This file defines the fuzzy rules for washing time based on the sludge and grease values.

```python
from skfuzzy import control as ctrl

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
```

### File 3: `fuzzy_control_system.py`
This file contains the fuzzy control system, which integrates the fuzzy rules and the membership functions to simulate the washing machine control.

```python
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from membership_functions import define_sludge_membership, define_grease_membership, define_washing_time_membership
from fuzzy_rules import define_fuzzy_rules

def run_fuzzy_control_system(sludge_input, grease_input):
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
    washing_time.view(sim=washing)
    return washing_time_output
```

### File 4: `main.py`
This is the main file where you will run the fuzzy control system with given inputs.

```python
from fuzzy_control_system import run_fuzzy_control_system

def main():
    # Example inputs
    sludge_input = 60  # Sludge level input
    grease_input = 70  # Grease level input
    
    # Run the fuzzy control system
    washing_time = run_fuzzy_control_system(sludge_input, grease_input)
    print(f"Predicted washing time: {washing_time} minutes")

if __name__ == "__main__":
    main()
```

### File 5: `visualization.py`
This file handles any dynamic visualizations or 3D plots of the fuzzy control system.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fuzzy_control_system import run_fuzzy_control_system

def plot_3d():
    x_stain = np.arange(0, 101, 1)
    x_oil = np.arange(0, 101, 1)
    X, Y = np.meshgrid(x_stain, x_oil)
    Z = np.zeros_like(X)

    # Calculate washing time for each (sludge, grease) pair
    for i in range(len(x_stain)):
        for j in range(len(x_oil)):
            Z[i, j] = run_fuzzy_control_system(x_stain[i], x_oil[j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Sludge')
    ax.set_ylabel('Grease')
    ax.set_zlabel('Washing Time')
    plt.show()

if __name__ == "__main__":
    plot_3d()
```

### How to Run:
1. Save each of the code snippets above into separate files as indicated.
2. Install the necessary libraries if you haven't already:
   ```bash
   pip install numpy skfuzzy matplotlib
   ```
3. Run the `main.py` to simulate the fuzzy control system:
   ```bash
   python main.py
   ```
4. To view the 3D plot of the fuzzy control system, run:
   ```bash
   python visualization.py
   ```

This modular approach will allow you to easily modify or extend any part of the system in the future without affecting other parts of the code.
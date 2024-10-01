import numpy as np
import matplotlib.pyplot as plt
from takeovers import compute_policy

def plot_c_vs_takeover_time(alpha0, k, t, beta0, max_steps, gamma, cost_range):
    take_overs = np.zeros_like(cost_range)
    for i,c in enumerate(cost_range):
        V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
        
        # Convert action to a list
        action_list = [action[i] for i in range(max_steps)]
        
        # Get the index of the first time that action_list is 'take_over'
        if 'take_over' in action_list:
            first_take_over = action_list.index('take_over')
        else:
            first_take_over = max_steps
            print(f"Cost: {c}, No takeover")
            
        take_overs[i] = first_take_over
    
    plt.figure()
    plt.plot(cost_range, take_overs)
    plt.xlabel('Cost per time step (c)')
    plt.ylabel('Take-Over Time')
    plt.title('Cost vs. Take-Over Time')
    plt.grid(True)
    plt.show()
    
def plot_t_vs_takeover_time(alpha0, k, t_range, beta0, max_steps, gamma, c):
    take_overs = np.zeros_like(t_range)
    for i,t in enumerate(t_range):
        V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
        
        # Convert action to a list
        action_list = [action[i] for i in range(max_steps)]
        
        # Get the index of the first time that action_list is 'take_over'
        if 'take_over' in action_list:
            first_take_over = action_list.index('take_over')
        else:
            first_take_over = max_steps
            print(f": {t}, No takeover")
            
        take_overs[i] = first_take_over
    
    plt.figure()
    plt.plot(t_range, take_overs)
    plt.xlabel('Utility of Take-Over (t)')
    plt.ylabel('Take-Over Time')
    plt.title('Utility of Take-Over vs. Take-Over Time')
    plt.grid(True)
    plt.show()

def plot_prior_heatmap(mean_range, k, t, var_range, max_steps, gamma, c):
    take_overs = np.zeros((len(mean_range), len(var_range)))
    for i, mean in enumerate(mean_range):
        for j, var in enumerate(var_range):
            alpha0 = mean * var
            beta0 = (1-mean) * var 
            
            V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
            
            # Convert action to a list
            action_list = [action[i] for i in range(max_steps)]
            
            # Get the index of the first time that action_list is 'take_over'
            if 'take_over' in action_list:
                first_take_over = action_list.index('take_over')
            else:
                first_take_over = max_steps
                
            take_overs[i,j] = first_take_over

    # Create heatmap
    plt.figure()
    X, Y = np.meshgrid(mean_range, var_range)
    plt.contourf(X, Y, take_overs, levels=100, cmap='viridis')
    plt.colorbar(label='Take-Over Time')
    plt.ylabel('Confidence (sample size)')
    plt.xlabel('Mean')
    plt.title('Prior Parameters vs. Take-Over Time')
    plt.show()

def main():
    # Parameters
    c = 1      # Cost per time step
    gamma = 1  # Discount factor (less than 1)
    k = 20        # Utility if the child succeeds
    t = 0         # Utility if the parent takes over
    alpha0 = 2    # Initial alpha parameter of the Beta prior
    beta0 = 2     # Initial beta parameter of the Beta prior
    max_steps = 100  # Maximum beta value to prevent infinite loops

    cost_range = np.linspace(.5, 2.0, 20)
    plot_c_vs_takeover_time(alpha0, k, t, beta0, max_steps, gamma, cost_range)

    t_range = np.linspace(-10, 10, 20)
    plot_t_vs_takeover_time(alpha0, k, t_range, beta0, max_steps, gamma, c)
    
    mean_range = np.linspace(0.01, 0.99, 50)
    var_range = np.linspace(.1, 5, 50)
    plot_prior_heatmap(mean_range, k, t, var_range, max_steps, gamma, c)
    
if __name__ == "__main__":
    main()

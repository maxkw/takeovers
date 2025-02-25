import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from takeovers import compute_policy
from load_data import load_and_process_data, plot_time_vs_takeovers
from itertools import product
import scipy as sp

sns.set_context('talk')
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
    sns.despine()
    plt.tight_layout()
    plt.savefig('cost_vs_takeover_time.pdf')
    
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
    sns.despine()
    plt.tight_layout()
    plt.savefig('t_vs_takeover_time.pdf')

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
    plt.title('Prior vs. Take-Over Time')
    plt.tight_layout()
    plt.savefig('prior_heatmap.pdf')

def plot_p_success_vs_takeover_rate(p_success_range, alpha0, beta0, k, t, gamma, c, max_steps):
    V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
    action_list = [action[i] for i in range(max_steps)]

    takeovers = []
    # get first time that policy is 'take_over'
    if 'take_over' in action_list:
        first_takeover = action_list.index('take_over')
    else:
        first_takeover = max_steps    
        
    for p_success in p_success_range:
        # use the cdf of a geometric distribution to get the probability of taking over at the first step
        p_takeover = (1-p_success)**(first_takeover-1)
        takeovers.append(p_takeover)
    
    plt.figure()
    plt.plot(p_success_range, takeovers)
    plt.xlabel('p_success')
    plt.ylabel('Takeover Rate')
    plt.title('p_success vs. Takeover Rate')
    sns.despine()
    plt.tight_layout()
    plt.savefig('p_success_vs_takeover_rate.pdf')

def plot_k_vs_takeover_rate(k_range, t=0, gamma=1, c=.001, max_steps=500):
    p_success_values = [0.01, 0.02, 0.05]
    takeovers = {p: [] for p in p_success_values}
    var = 1
    
    for k in k_range:
        for p_success in p_success_values:
            alpha0 = p_success * var
            beta0 = (1-p_success) * var
            
            V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
            action_list = [action[i] for i in range(max_steps)]

            # Get first time that policy is 'take_over'
            if 'take_over' in action_list:
                first_takeover = action_list.index('take_over')
            else:
                first_takeover = max_steps    
                
            # Use geometric distribution to get probability of taking over
            p_takeover = (1-p_success)**(first_takeover-1)
            takeovers[p_success].append(p_takeover)
    
    plt.figure(figsize=(5, 5))
    for p_success in p_success_values:
        plt.plot(k_range, takeovers[p_success], label=f'p = {p_success}')
    plt.xlabel(r'$r_{\text{success}}$')
    plt.title('')
    plt.ylim(0, 1.05)

    # plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.ylabel('')
    
    plt.savefig('k_vs_takeover_rate.pdf')

def plot_cost_vs_takeover_rate(c_range, k, t=0, gamma=1, max_steps=500):
    p_success_values = [0.01, 0.02, 0.05]
    takeovers = {p: [] for p in p_success_values}
    var = 1
    
    for c in c_range:
        for p_success in p_success_values:
            alpha0 = p_success * var
            beta0 = (1-p_success) * var
            
            V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
            action_list = [action[i] for i in range(max_steps)]

            # Get first time that policy is 'take_over'
            if 'take_over' in action_list:
                first_takeover = action_list.index('take_over')
            else:
                first_takeover = max_steps    
                
            # Use geometric distribution to get probability of taking over
            p_takeover = (1-p_success)**(first_takeover-1)
            takeovers[p_success].append(p_takeover)
    
    plt.figure(figsize=(5, 5))
    for p_success in p_success_values:
        plt.plot(c_range, takeovers[p_success], label=f'p = {p_success}')
    plt.xlabel(r'$r_c$')
    plt.title('')    
    plt.ylim(0, 1.05)

    # plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.ylabel('')
    
    plt.savefig('cost_vs_takeover_rate.pdf')

  
def plot_prior_vs_takeover_rate(mean_range, k, c, t= 0, var=1, gamma=1, max_steps=500):
    p_success_values = [0.01, 0.02, 0.05]
    take_over_probs = {p: np.zeros(len(mean_range)) for p in p_success_values}
    
    for i, mean in enumerate(mean_range):
            alpha0 = mean * var
            beta0 = (1-mean) * var 
            V, action = compute_policy(alpha0, k, t, beta0, max_steps, gamma, c)
            
            # Convert action to a list
            action_list = [action[i] for i in range(max_steps)]
            
            # Get the probability of 'take_over' action using the geometric cdf
            if 'take_over' in action_list:
                first_takeover = action_list.index('take_over')
            else:
                first_takeover = max_steps
            
            # Calculate takeover probability for each p_success value
            for p_success in p_success_values:
                take_over_probs[p_success][i] = (1 - p_success)**(first_takeover)

    # Create plot
    plt.figure(figsize=(5, 5))
    for p_success in p_success_values:
        plt.plot(mean_range, take_over_probs[p_success], label=f'p = {p_success}')
    plt.ylabel('P(Take Over)')
    plt.xlabel(r'$\mathbb{E}[b_0]$')
    plt.ylim(0, 1.05)
    plt.title('')
    plt.legend()
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('prior_vs_takeover_rate.pdf')


    
def plot_model_vs_data(df)  :
    df = df.groupby(['task', 'capable', 'condition']).mean('TO').reset_index()
    
    plt.figure()
    plt.plot(df['model_TO'], df['TO'], 'o')
    plt.xlabel('Model')
    plt.ylabel('Human')
    plt.title('Model vs. Human')
    sns.despine()
    plt.tight_layout()
    plt.savefig('model_vs_human.pdf')
        
def MSE(model_TO, data_TO):
    # Convert inputs to numpy arrays and ensure they're 1D
    model_TO = np.array(model_TO).flatten()
    data_TO = np.array(data_TO).flatten()
    return np.mean((model_TO - data_TO)**2)

def ll(model_TO, data_TO):
    # comput the log likelihood of the model_TO given the data_TO
    return np.sum(np.log(model_TO) * data_TO)   

def compute_random_TO(p_takeover, df, offset=0):
    model_data = []
    loss = []
    for row in df.itertuples():
        p_success = 1/row.TIME
        p_takeover_first = p_takeover / (p_takeover + p_success - p_takeover*p_success)
        model_data.append(p_takeover_first)
    
    model_data = np.array(model_data)+offset
    loss = float(MSE(model_data, df['TO']))
    return model_data, loss, np.corrcoef(model_data, df['TO'])[0,1]

def compute_model_TO(c, learning_multiplier, df, offset=0, lesion_belief=False, lesion_subtask_difficulty=False):
    max_steps = 200

    gamma = 1  # Discount factor (less than 1)
    take_over_utility = 0      # Utility if the parent takes over 
    k = 1
    var = 1
    k_learning = k * learning_multiplier
    max_capable = df['capable'].max()
    model_data = []
    for row in df.itertuples():

        cap = row.capable
        t = row.TIME
        condition = row.condition
    
    
        if lesion_subtask_difficulty:
            # set p_success to the average of all TIME values
            diff = (1/df['TIME']).mean()
        else:
            diff = (1/t)
            
        p_success = diff
        if lesion_belief:
            mean = p_success
        else:
            mean = p_success**(max_capable-cap+1)
            # mean = p_success * (cap / max_capable)


        alpha0 = mean * var
        beta0 = (1-mean) * var 

        k_value = k_learning if condition == 'learning' else k
        V, action = compute_policy(alpha0, k_value, take_over_utility, beta0, max_steps, gamma, c)
        
        # Convert action to a list
        action_list = [action[i] for i in range(max_steps)]
        
        # Get the probability of 'take_over' action using the geometric cdf
        if 'take_over' in action_list:
            first_takeover = action_list.index('take_over')
        else:
            first_takeover = max_steps
        TO = (1 - p_success)**(first_takeover)
        
        model_data.append(TO+offset)
        

    model_data = np.array(model_data)
    # loss = -np.corrcoef(model_data, df['TO'])[0,1]
    loss = MSE(model_data, df['TO'])
    # loss = -ll(model_data, df['TO'])
    
    if np.isnan(loss):
        loss = 0

    return model_data, loss, np.corrcoef(model_data, df['TO'])[0,1]

def main():
    df = load_and_process_data("comp_study2.csv")
    df.groupby(['task', 'capable', 'condition']).mean('TO').reset_index()
    
    scaling = 4
    plt.rcParams['font.size'] *= scaling
    plot_cost_vs_takeover_rate(np.linspace(0,0.04, 300), k=1, t=0, gamma=1, max_steps=500)    
    plot_k_vs_takeover_rate(np.linspace(0, 50, 100), c=.001)
    plot_prior_vs_takeover_rate(np.linspace(0, .5, 200), 1,.001)
    plt.style.use('default')  # Reset to default style first if needed

    fine_ranges = (slice(.0001, .001, .0001), 
              slice(10, 30, 5), 
              )   
    optimize = True 
    ##### Optimize full model ##### 
    if optimize:
        res = sp.optimize.brute(
            lambda x: compute_model_TO(x[0], x[1], df)[1], fine_ranges, full_output=True, finish=sp.optimize.fmin)[0]
    else:
        res = [7.86718750e-04, 2.62963867e+01]
    model_TO, loss, corr = compute_model_TO(res[0],  res[1], df)
    print('full model', loss, res, corr)
    df['model_TO'] = model_TO
    ##### End of Optimize full model ##### 
    
    ##### Optimize lesion belief ##### 
    lesion_belief = True
    if optimize:
        res = sp.optimize.brute(
            lambda x: compute_model_TO(x[0], x[1], df,
                                       lesion_belief=lesion_belief)[1], fine_ranges, full_output=True, finish=sp.optimize.fmin)[0]
    else:
        res = [1.04233557e-02, 1.12055359e+01]
    model_TO, loss, corr = compute_model_TO(res[0],  res[1], df,  lesion_belief=lesion_belief)
    print('lesion belief', loss, res, corr)
    df['lesionbelief_TO'] = model_TO
    ##### End of Optimize lesion belief ##### 
    
    ##### Optimize lesion sub-task difficulty ##### 
    lesion_subtask_difficulty = True
    if optimize:
        res = sp.optimize.brute(
            lambda x: compute_model_TO(x[0], x[1], df, lesion_subtask_difficulty=lesion_subtask_difficulty)[1], fine_ranges, full_output=True, finish=sp.optimize.fmin)[0]
    else:
        res = [1.26013184e-03, 7.50976562e+00]
        
    model_TO, loss, corr = compute_model_TO(
        res[0],  res[1], df, lesion_subtask_difficulty=lesion_subtask_difficulty)
    print('lesion sub-task difficulty', loss, res, corr)
    df['lesionsubtask_TO'] = model_TO
    ##### End of Optimize lesion sub-task difficulty ##### 
    
    #### Optimize p_takeover using sp.optimize.brute ####
    ranges = [slice(0, 1, .01),
            #   slice(.1, .5, .05)
              ]
    if optimize:
        p_takeover = sp.optimize.brute(
            lambda x: compute_random_TO(x[0], df)[1], ranges,     full_output=True, finish=sp.optimize.fmin)[0]
    else:
        p_takeover = [0.04025]
    model_data, loss, corr = compute_random_TO(p_takeover[0], df)
    df['random_TO'] = model_data
    print("random intervention", loss, p_takeover, corr)
    #### End of Optimize p_takeover using snp.optimize.brute ####
        
    plot_time_vs_takeovers(df, output_file='model_time_vs_takeovers.pdf')
    

    # # Run regressions for each model
    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf
    
    # # Full model regression
    # model_full = smf.logit('TO ~ model_TO', data=df).fit()
    # print("\nFull Model Regression:")
    # print(model_full.summary())
    
    # # Lesion belief regression
    # model_lesion_belief = smf.logit('TO ~ lesionbelief_TO', data=df).fit()
    # print("\nLesion Belief Model Regression:") 
    # print(model_lesion_belief.summary())
    
    # # Lesion subtask difficulty regression
    # model_lesion_subtask = smf.logit('TO ~ lesionsubtask_TO', data=df).fit()
    # print("\nLesion Subtask Difficulty Model Regression:")
    # print(model_lesion_subtask.summary())
    
    # # Random takeover regression
    # model_random = smf.logit('TO ~ random_TO', data=df).fit()
    # print("\nRandom Takeover Model Regression:")
    # print(model_random.summary())    
    # df['TO'] = df['model_TO']
    # plot_time_vs_takeovers(df, output_file='model_time_vs_takeovers.pdf')
    

    # gamma = 1  # Discount factor (less than 1)
    # take_over_utility = 0      # Utility if the parent takes over    
    # k_learning = k * learning_multiplier
    
    # model_data = []
    # for row in df.itertuples():
    #     cap = row['capable']
    #     t = row['TIME']
    #     condition = row['condition']
    
    #     p_success = 1/t
    #     mean = 1/t * (cap / max(capable))

    #     alpha0 = mean * var
    #     beta0 = (1-mean) * var 

    #     k_value = k_learning if condition == 'learning' else k
    #     V, action = compute_policy(alpha0, k_value, take_over_utility, beta0, max_steps, gamma, c)
        
    #     # Convert action to a list
    #     action_list = [action[i] for i in range(max_steps)]
        
    #     # Get the probability of 'take_over' action using the geometric cdf
    #     if 'take_over' in action_list:
    #         first_takeover = action_list.index('take_over')
    #     else:
    #         first_takeover = max_steps
            
    #     TO = (1 - p_success)**(first_takeover)
        
    #     model_data.append(TO)
        
    
    # # Parameters

    # alpha0 = 2    # Initial alpha parameter of the Beta prior
    # beta0 = 2     # Initial beta parameter of the Beta prior
    # max_steps = 100  # Maximum beta value to prevent infinite loops

    # mean_range = np.linspace(0.01, 0.99, 100)
    # p_success = 0.5
    # plot_prior_vs_takeover_rate(mean_range, p_success, k, t, gamma, c, max_steps)   

    # p_success_range = np.linspace(0, 1, 100)
    # plot_p_success_vs_takeover_rate(p_success_range, alpha0, beta0, k, t, gamma, c, max_steps)

    # cost_range = np.linspace(.5, 2.0, 20)
    # plot_c_vs_takeover_time(alpha0, k, t, beta0, max_steps, gamma, cost_range)

    # t_range = np.linspace(-10, 10, 20)
    # plot_t_vs_takeover_time(alpha0, k, t_range, beta0, max_steps, gamma, c)
    
    # mean_range = np.linspace(0.01, 0.99, 50)
    # var_range = np.linspace(.1, 5, 50)
    # plot_prior_heatmap(mean_range, k, t, var_range, max_steps, gamma, c)
    
if __name__ == "__main__":
    main()



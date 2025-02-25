import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

def load_and_process_data(filepath):
    """Load and process the CSV data."""
    comp_df = pd.read_csv(filepath)
    to_columns = [col for col in comp_df.columns if col.endswith('_TO')]
    # rename the to_columns to start with TO
    for col in to_columns:
        comp_df.rename(columns={col: 'TO_' + col}, inplace=True)
        
    time_columns = [col for col in comp_df.columns if col.endswith('_TIME')]
    for col in time_columns:
        comp_df.rename(columns={col: 'TIME_' + col}, inplace=True)
        
    id_vars = ['subject', 'condition', 'capable', 'exact_age', 'gender', 'parent', 'race', 'ethnicity', 'education', 'income']
    df1 = comp_df.melt(id_vars=id_vars, value_vars=[col for col in comp_df.columns if col.startswith('TO_')], var_name='task', value_name='TO')
    
    df2 = comp_df.melt(id_vars=id_vars, value_vars=[col for col in comp_df.columns if col.startswith('TIME_')], var_name='task2', value_name='TIME')
    
    df1 = df1.set_index(id_vars +  [df1.groupby(id_vars).cumcount()])
    df2 = df2.set_index(id_vars +  [df2.groupby(id_vars).cumcount()])

    comp_df = (pd.concat([df1, df2],axis=1)
            # .sort_index(level=2)
            # .reset_index(level=2, drop=True)
            .reset_index())    
    
    # delete column task2
    comp_df.drop(columns=['task2', 'level_10'], inplace=True)

    return comp_df

def plot_time_vs_takeovers(to_df, output_file='average_time_vs_takeovers.pdf', plot_model=False):
    """Plot average time vs takeovers for each task."""
    TO_vs_time = to_df.groupby(['task', 'capable', 'condition']).mean('TO').reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.set_context('talk')

    # if plot_model:
    #     # melt the data to long format
    #     TO_vs_time = TO_vs_time.melt(id_vars=['task', 'capable', 'condition', 'TIME'], value_vars=['TO', 'model_TO'], var_name='mh', value_name='TO_value')

    #     sns.lmplot(data=TO_vs_time, x='TIME', y='TO_value', hue='condition', col='capable', row='mh', height=4)
    # else:
    #     sns.lmplot(data=TO_vs_time, x='TIME', y='TO', hue='condition', col='capable', height=4)
    
    TO_vs_time = TO_vs_time.melt(id_vars=['task', 'capable', 'condition', 'TIME'], value_vars=['TO', 'model_TO', 'lesionbelief_TO', 'lesionsubtask_TO', 'random_TO'], var_name='mh', value_name='TO_value')
    
    plt.rcParams['text.usetex'] = True

    
    g = sns.FacetGrid(TO_vs_time, row='capable', col='mh', hue='condition', height=4, col_order=['TO', 'model_TO', 'lesionbelief_TO', 'lesionsubtask_TO', 'random_TO'])
    
    # Only plot scatter points for the first column (human data)
    for i, ax in enumerate(g.axes[:,0]):  # Get axes of first column
        data = TO_vs_time[TO_vs_time['mh'] == 'TO']  # Filter for human data
        data = data[data['capable'] == i+2]  # Filter for correct capability level
        sns.scatterplot(data=data, x='TIME', y='TO_value', style='task', hue='condition', ax=ax, legend=False)
    
    # g.map_dataframe(sns.scatterplot, x='TIME', y='TO_value', style='task', legend=False)
    
    
    g.map_dataframe(sns.regplot, x='TIME', y='TO_value', scatter=False, ci=68, line_kws={'alpha':0.7})
    

    g.set_titles('')
    g.facet_axis(0, 0).set_title(r'{\huge \bf Human Data}' + '\n\n')
    g.facet_axis(0, 1).set_title(r'{\huge \bf Full Model}' + '\n\n')
    g.facet_axis(0, 2).set_title(r'{\huge \bf Lesion: Skill Beliefs}' + '\n\n' + f'Skill Belief= 2/5')
    g.facet_axis(0, 3).set_title(r'{\huge \bf Lesion: Subtask Difficulty}'+ '\n\n')
    g.facet_axis(0, 4).set_title(r'{\huge \bf Random Intervention}' + '\n\n')

    for i in range(1,4):
        j = 2
        # for j in range(5):
        bel = 2+i
        g.facet_axis(i, j).set_title(f'Skill Belief = {bel}/5')            
                
    # Add grey rectangles spanning all columns for each row
    beliefs = ['Skill Belief', 'Skill Belief', 'Skill Belief', 'Skill Belief']
    for i, belief in enumerate(beliefs):
        rect = plt.Rectangle((.048, .935 - i*.229), .94, 0.02, 
                           transform=g.fig.transFigure, 
                           facecolor='lightgrey', edgecolor='black', alpha=0.3)
        g.fig.add_artist(rect)

    

    g.set_xlabels('Subtask Difficulty (secs)')
    g.set_ylabels('P(Take Over)')
    g.tight_layout()
    g.add_legend(title='Condition', bbox_to_anchor=(.14, .88), loc='upper left')

    # if plot_model:
    #     # g.map_dataframe(sns.scatterplot, x='TIME', y='model_TO', style='task', palette='viridis')
    #     g.map_dataframe(sns.regplot, x='TIME', y='model_TO', scatter=False, color='black', line_kws={'alpha':0.7})
        
    # result = smf.ols(formula="TO ~ capable * TIME", data=TO_vs_time).fit()        
    # import pdb; pdb.set_trace()

    # plt.xlabel('Average Time to Complete (Secs)')
    # plt.ylabel('Average % Takeovers')
    # plt.title('Average Time to Complete vs. Average % Takeovers for Each Task')
    # plt.legend(title='condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(-.05, 1.05)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return TO_vs_time

def plot_takeover_by_skill(comp_df, output_file='takeover_by_skill.pdf'):
    """Plot takeover by belief about child skill."""
    plt.figure(figsize=(10, 6))
    sns.set_context('talk')
    sns.regplot(data=comp_df, x='capable', y='takeover_score_per', 
                scatter=True, color='black', line_kws={'alpha':0.2}, 
                x_jitter=0.05, scatter_kws={'alpha':0.5, 's':20})
    plt.xlabel('Belief about Child Skill')
    plt.ylabel('% Takeovers')
    plt.title('Takeover by Belief about Child Skill')
    plt.ylim(0, 1)
    plt.xlim(1, 5.2)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    import pdb; pdb.set_trace()

def main():
    # Load and process data
    comp_df = load_and_process_data("comp_study2.csv")
    
    # Generate plots and print statistics
    TO_vs_time = plot_time_vs_takeovers(comp_df)
    print("Average TO for each task:")
    print(TO_vs_time)
    
    print("\nTakeover scores by capability:")
    print(comp_df[['capable', 'takeover_score_per']])
    
    plot_takeover_by_skill(comp_df)

if __name__ == "__main__":
    main()




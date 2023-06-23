import ast
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def extract_distribution_view_data(trajectory_df):
    credit_dist_start_1 = trajectory_df["credit_score_distribution-group_1"].head(1)
    credit_dist_end_1 = trajectory_df["credit_score_distribution-group_1"].tail(1)
    credit_dist_start_2 = trajectory_df["credit_score_distribution-group_2"].head(1)
    credit_dist_end_2 = trajectory_df["credit_score_distribution-group_2"].tail(1)

    credit_dist_start_1 = ast.literal_eval(credit_dist_start_1.values[0])
    credit_dist_end_1 = ast.literal_eval(credit_dist_end_1.values[0])
    credit_dist_start_2 = ast.literal_eval(credit_dist_start_2.values[0])
    credit_dist_end_2 = ast.literal_eval(credit_dist_end_2.values[0])

    credit_dist_group_1 = pd.DataFrame({"start": credit_dist_start_1, "end": credit_dist_end_1}, 
                                       index=[*range(len(credit_dist_end_1))])
    credit_dist_group_2 = pd.DataFrame({"start": credit_dist_start_2, "end": credit_dist_end_2}, 
                                       index=[*range(len(credit_dist_start_2))])

    return credit_dist_group_1, credit_dist_group_2

def view_credit_distributions(credit_dist_df, title=""):
    index_list = list(credit_dist_df.index)
    fig = go.Figure(data=[
        go.Bar(name='Start distribution', x=index_list, y=credit_dist_df["start"]),
        go.Bar(name='End distribution', x=index_list, y=credit_dist_df["end"])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', title=title, title_x=0.5, 
                      xaxis_title="Credit group", yaxis_title="Probability")
    return fig

def view_trajectory(trajectories_path, view_bar_charts=False):
    trajectory_df = pd.read_csv(trajectories_path, index_col=[0])
    
    if view_bar_charts:
        credit_dist_group_1, credit_dist_group_2 = extract_distribution_view_data(trajectory_df)
        fig = view_credit_distributions(credit_dist_group_1, title="Credit score distribution of group 1")
        fig.show()
        fig = view_credit_distributions(credit_dist_group_2, title="Credit score distribution of group 2")
        fig.show()
        
    fig = px.line(trajectory_df,
                  x="Timestep", 
                  y=["average_credit_score-group_1", "average_credit_score-group_2"])
    fig.update_layout(title="Average credit score over time", title_x=0.5, 
                      xaxis_title="Time step", yaxis_title="Value")
    fig.show()
    fig = px.line(trajectory_df, x="Timestep", y=["acceptance_rate-group_1", "acceptance_rate-group_2"])
    fig.update_layout(title="Acceptance rate over time", title_x=0.5, 
                      xaxis_title="Time step", yaxis_title="Value")
    fig.show()
    fig = px.line(trajectory_df, x="Timestep", y=["default_rate-group_1", "default_rate-group_2"])
    fig.update_layout(title="Default rate over time", title_x=0.5,
                      xaxis_title="Time step", yaxis_title="Value")
    fig.show()

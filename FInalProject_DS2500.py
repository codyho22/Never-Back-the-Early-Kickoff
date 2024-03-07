"""
DS2500 Final Project
Cody Ho, Matt Doherty
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split


def file_to_df(year1, year2):
    ''' given a starting year and a ending year, returns a dataframe made from
        data files with certain file name from starting year to end year
    '''
    
    data_frames = []
    
    # iterate from start year to end year, read csv file into df and combine
    for year in range(year1, year2):
        filename = f'results_{year}.csv'
        df = pd.read_csv(filename)        
        data_frames.append(df)
          
    results_df = pd.concat(data_frames, ignore_index = True)
    return results_df

def add_average_columns(df):
    ''' given data frame, returns dataframe with three new columns. Iterate
        through all columns names of the odds columns and find if they represent
        home win, draw, or away win, and then find mean for each row among 
        grouped columns
    '''
    # column names for home win, draw, and away win betting values
    ODDS_COLUMNS = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA',
                     'IWH', 'IWD', 'IWA','PSH', 'PSD', 'PSA','VCH', 'VCD', 'VCA',
                     'WHH', 'WHD', 'WHA']

    # iterate through columns names in odds column and fine if Home, Draw, Away
    home_odds = [col for col in ODDS_COLUMNS if col.endswith('H')]
    draw_odds = [col for col in ODDS_COLUMNS if col.endswith('D')]
    away_odds = [col for col in ODDS_COLUMNS if col.endswith('A')]
    
    # create 3 new columns and find mean for each row among grouped columns
    df['AVG_HOME'] = df[home_odds].mean(axis = 1)
    df['AVG_DRAW'] = df[draw_odds].mean(axis = 1)
    df['AVG_AWAY'] = df[away_odds].mean(axis = 1)
    return df

def add_upset_column(df):
    ''' given df, return df with PRED_RESULT, REAL_RESULT, and UPSET column
        Finds if difference between odds of predicted result and odds of real 
        result signify an upset
    '''
    
    # find which column has the lowest value and return that column title
    min_avg_result = df[['AVG_HOME', 'AVG_DRAW', 'AVG_AWAY']].idxmin(axis = 1)
    
    # creates map that ties column title to H, D, or A 
    result_map = {'AVG_HOME': 'H', 'AVG_DRAW': 'D', 'AVG_AWAY': 'A'}
    
    # Creates new column that hold H, D, A depending on what values
    # was the lowest for that row
    df['PRED_RESULT'] = min_avg_result.map(result_map)
    
    
    df['REAL_RESULT'] = None
    
    # adds the odds of the actual result into real_result column.
    # if the full time result was a draw (D), will add the odds of a draw column
    df.loc[df['FTR'] == 'H', 'REAL_RESULT'] = df.loc[df['FTR'] == 
                                                     'H', 'AVG_HOME']
    df.loc[df['FTR'] == 'A', 'REAL_RESULT'] = df.loc[df['FTR'] == 
                                                     'A', 'AVG_AWAY']
    df.loc[df['FTR'] == 'D', 'REAL_RESULT'] = df.loc[df['FTR'] == 
                                                     'D', 'AVG_DRAW']
    
    
    # find which column has the lowest value among 3 and return that value
    lowest_avg = df[['AVG_HOME', 'AVG_AWAY', 'AVG_DRAW']].min(axis=1)
    
    df['UPSET'] = 0
    
    # if the odds of the real result in REAL_RESULT is not equal to the odds of
    # the most likely result in lowest_avg AND the difference is greater than 3,
    # then add a 1 to signify an upset for that row
    df.loc[(df['REAL_RESULT'] != lowest_avg) & ((df['REAL_RESULT'] - 
                                                 lowest_avg) > 3), 'UPSET'] = 1
    
    return df

def upset_frequency (df):
    ''' given a filtered df with the entries interested in finding the frequency
    for, returns the frequency number of games per upset
    ex) returns 2.14, meaning that an upset occurs every 2.14 games 
    '''
   
    total_games = len(df)
    num_upsets = df['UPSET'].sum()
    upset_freq = 1 / (num_upsets / total_games)
    return upset_freq


def add_month_column(df):
    ''' given df, returns df with Date column in datetime format and adds
        column for Month that holds int for month
    '''
    # dates given in british format, so day is first
    # format dates in date column to datetime to get month value
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True, errors='coerce')
    df['Month'] = df['Date'].dt.month

    return df


def upset_per_kot(df):
    ''' given df, return average number of games player per upset for 3 
        specific time groups, by grouping rows and summing number of upsets
    '''
    
    # groups rows where time of ko fall in between certain times
    early_kickoff = df[(df['Time'] >= '00:01') & (df['Time'] <= '12:30')]
    regular_kickoff = df[(df['Time'] >= '12:31') & (df['Time'] <= '17:29')]
    late_kickoff = df[(df['Time'] >= '17:30') & (df['Time'] <= '23:59')]

    # sum number of upsets found in each time group
    upsets_early = early_kickoff['UPSET'].sum()
    upsets_regular = regular_kickoff['UPSET'].sum()
    upsets_late = late_kickoff['UPSET'].sum()

    # find total number of games played in each time group
    total_early = len(early_kickoff)
    total_regular = len(regular_kickoff)
    total_late = len(late_kickoff)

    # calculate number of games player for each upset in each time group
    upsets = [1 / (upsets_early / total_early), 
              1 / (upsets_regular / total_regular), 
              1 / (upsets_late / total_late)]
    return upsets

def upset_kot_bargraph_v1(upsets):
    ''' create bargraph to show average number of games player per upset at
        3 different kick off times
    '''
    
    categories = ['Early Kick-off', 'Regular Kick-off', 'Late Kick-off']
    colors = ['darkred', 'orangered', 'sandybrown']
    
    plt.bar(categories, upsets, color=colors, edgecolor='black', linewidth=2)
    plt.title('Upsets per Number of Games Played by Kick-off Time')
    plt.xlabel('Kick-off Time')
    plt.ylabel('Upsets per Game Played')
    plt.legend()
    plt.show()

def upset_month_bargraph_v2(df):
    ''' given df, return bargraph showing avg number of upsets for each month
    '''
    lst_frequencies = []
    
    # starting from august, find avg frequency of upset for each month
    for month in range(8,13):
       month_df = df[df['Month'] == month]    
       lst_frequencies.append(upset_frequency(month_df))
    for month in range(1,6):
       month_df = df[df['Month'] == month]    
       lst_frequencies.append(upset_frequency(month_df))

    months = [
        'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
   
    # find average number of upsets, then create line showing total avg
    avg_upsets = sum(lst_frequencies) / len(lst_frequencies)
    plt.axhline(y = avg_upsets, color = 'darkorange', linestyle = '--', 
                label = 'Total Avg Games per Upsets')
    
    # plot bar chart 
    plt.bar(months, lst_frequencies, color='mediumslateblue', edgecolor='black')
    plt.xlabel('Month')
    plt.ylabel('Avg Number of Games per Upsets')
    plt.title('Number of Games per Upset from August to May (10 Years)')
    plt.legend()
    plt.show()
    

def linregress_month_v3(df):
    '''given a month name, creates a plot where frequency of upsets for that
    month is plotted against the last 10 years to create a linear regression plot
    '''
    
    # start with august to december, since season begins in Aug
    # append the upset frequency for year to the list
    lst_frequencies = []
    for month in range(8,13): 
        month_df = df[df['Month'] == month]    
        lst_frequencies.append(upset_frequency(month_df))
    for month in range(1,6):
        month_df = df[df['Month'] == month]    
        lst_frequencies.append(upset_frequency(month_df))
        
    #create plot
    sns.regplot(x = list(range(0,10)), y = lst_frequencies, color = "green")
    plt.xlabel('Month')
    plt.ylabel('Avg Num Games per Upset')
    plt.xticks([0,1,2,3,4,5,6,7,8,9], 
               ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
                'Apr', 'May'])
    plt.yticks(list(range(9, 20)))
    plt.title('Months v Frequency Upsets over Last 10 Years')
    plt.show()

    
def linregress_ko_v4(df):
    ''' given a df, creates a plot where frequency of upsets for that
       Kick Off times is plotted against the last 4 years to create a linear 
       regression plot''' 
    
    # convert values in Time column to datetime then add just hour value to Hour
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    
    lst_frequencies = []
    x_ticks = [12, 14, 15, 16, 17, 18, 19, 20]
    
    # create new df for each hour, and then find AVG upsets for each hour
    for hour in x_ticks:
        hour_df = df[df['Hour'] == hour]    
        lst_frequencies.append(upset_frequency(hour_df))
    
    x_values = list(range(len(x_ticks)))
    
    sns.regplot(x = x_values, y = lst_frequencies, color='hotpink', 
                line_kws={'color': 'darkviolet'})
    plt.xlabel('KO Hour (24 hour)')
    plt.ylabel('Avg Num Games per Upset')
    plt.xticks(x_values, x_ticks)
    plt.yticks(list(range(6, 20)))
    plt.title('KO Hour v Frequency Upsets over Last 4 Years')
    plt.show()


def feature_importance_v5(df):
    ''' given df, predict if upset would occur from selected features. create 
        bar chart showing what features had largest impact in predicting if an 
        upset would occur
    '''
    # featured column for predicting UPSET column will be month, hour, and avg
    # odds of gaming be over 2.5 goals and under 2.5
    columns = ["Month", "Hour", "UPSET", "Avg>2.5", "Avg<2.5"]
    
    df = df[columns]
    
    # pull out features, seperate from the labels and then train the dataset
    X = df.drop(["UPSET"], axis = 1)
    y = df[["UPSET"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)  
    
    # find each features importance in clf, then return indexes of each features
    # importance from most important to least, then get column for each index
    importances = clf.feature_importances_
    sorted_index_lst = importances.argsort()[::-1]
    ordered_col_names = [X.columns[i] for i in sorted_index_lst]
    

    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[sorted_index_lst], 
            align='center', color='orange', edgecolor='black')
    plt.xticks(range(len(importances)), ordered_col_names)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

def avg_goalsKO_v6(df):
   ''' given df, creates bar chart showing average number of goals at each
       different kick off hour, showing avg for home goals and away goals
   '''
   # each kick off hour, excluding 13 because no games has kicked off then 
   x_ticks = [12, 14, 15, 16, 17, 18, 19, 20]
   home_goal_avg = []
   away_goal_avg = []
   
   # create new df for each hour, and then find avg goals for Home and Away
   for hour in x_ticks:
       hour_df = df[df['Hour'] == hour]    
       h_goals = hour_df['FTHG'].sum()
       a_goals = hour_df['FTAG'].sum()
       num_games = len(hour_df)
       avg_h_goals = h_goals / num_games
       avg_a_goals = a_goals / num_games
       home_goal_avg.append(avg_h_goals)
       away_goal_avg.append(avg_a_goals)

   # create list of 0 - 7 for number of different ko times
   bar_width = 0.35
   x_val = np.array(range(len(x_ticks)))
   
   # plot avg home goals, then away goals .35 distance next to home bar
   plt.bar(x_val, home_goal_avg, bar_width, label='Average Home Goals', 
           color='royalblue', edgecolor='black')
   plt.bar(x_val + bar_width, away_goal_avg, bar_width, 
           label='Average Away Goals', color='orange', edgecolor='black')
    
   # find average number of goals per game, then create line showing total avg
   avg_goals = sum(home_goal_avg + away_goal_avg) / len(
       home_goal_avg + away_goal_avg)
   plt.axhline(y = avg_goals, color = 'red', linestyle = '--', 
               label = 'Average Goals per Game')

   plt.xlabel('Kick-off Time')
   plt.ylabel('Average Goals')
   plt.title('Average Home and Away Goals per Kick-off Time')
   plt.ylim(0, 2.25)
   plt.xticks([i + bar_width / 2 for i in x_val], x_ticks)
   plt.legend()
   plt.show()
        

def goals_upset_graph_v7_8(df):
    #select column names needed
    goal_odd_cols = ['Hour', 'FTHG', 'FTAG', 'B365>2.5', 'B365<2.5', 
                     'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5']
    #create Hour column, replace Time
    #filter in given df to only include selected column names
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df = df[goal_odd_cols].copy() 
    
    #create Average Less Than and Greater Than columns. 
    # Data in each row will be the average betting odds 
    # for predicting game over 2.5 goals (GT) and predicting under 2.5 (LT)
    df['Avg_LT'] = df[[col for col in df.columns if '<' in col]].mean(axis=1)
    df['Avg_GT'] = df[[col for col in df.columns if '>' in col]].mean(axis=1)
    
    #Create Game_Goals column, total goals in a given game.
    df['Game_Goals'] = df['FTHG'] + df['FTAG']
    
    #create column to identify if there is a significant gap in goal betting 
    # odds that shows a clear favorite.
    df['Goal_bet_gap'] = 0
    df.loc[(abs(df['Avg_LT'] - df['Avg_GT']) >= 0.8), 'Goal_bet_gap'] = 1 
    
    #creating time classes
    early_ko_df = df[df['Hour'] < 13]
    midday_ko_df = df[(df['Hour'] >= 13) & (df['Hour'] < 17)]
    late_ko_df = df[df['Hour'] >= 17]

    # Get the number of times each case occurs: When a goal betting upset 
    # occurs, and whether
    # or not it was an overscoring game or underscoring game compared to the 
    # favorite odds
    # Lower number for goal betting means pred higher chance. All 3 cases must
    # be true to add 1 to the count of rows that pass all 3 cond
    early_overgoals = ((early_ko_df['Avg_LT'] < early_ko_df['Avg_GT']) 
                       & (early_ko_df['Game_Goals'] > 2.5) 
                       & (early_ko_df['Goal_bet_gap'] == 1)).sum()
    early_undergoals = ((early_ko_df['Avg_LT'] > early_ko_df['Avg_GT']) 
                        & (early_ko_df['Game_Goals'] < 2.5) 
                        & (early_ko_df['Goal_bet_gap'] == 1)).sum()

    midday_overgoals = ((midday_ko_df['Avg_LT'] < midday_ko_df['Avg_GT']) 
                        & (midday_ko_df['Game_Goals'] > 2.5) 
                        & (midday_ko_df['Goal_bet_gap'] == 1)).sum()
    midday_undergoals = ((midday_ko_df['Avg_LT'] > midday_ko_df['Avg_GT']) 
                         & (midday_ko_df['Game_Goals'] < 2.5) 
                         & (midday_ko_df['Goal_bet_gap'] == 1)).sum()

    late_overgoals = ((late_ko_df['Avg_LT'] < late_ko_df['Avg_GT']) 
                      & (late_ko_df['Game_Goals'] > 2.5) 
                      & (late_ko_df['Goal_bet_gap'] == 1)).sum()
    late_undergoals = ((late_ko_df['Avg_LT'] > late_ko_df['Avg_GT']) 
                       & (late_ko_df['Game_Goals'] < 2.5) 
                       & (late_ko_df['Goal_bet_gap'] == 1)).sum()

    
    #BEGIN PLOTTING
    #First, create 6 bars, 2 for each kickoff time catagory. One bar will show
    # the raw num of undergoals, and other will show num overgoals
    sns.barplot(x=['Early', 'Midday', 'Late', 'Early', 'Midday', 'Late'],
                y=[early_overgoals, midday_overgoals, late_overgoals, 
                   early_undergoals, midday_undergoals, late_undergoals],
                hue=['Over', 'Over', 'Over', 'Under', 'Under', 'Under'], 
                edgecolor='black')

    plt.title('Number of Goal Upsets by Kickoff Time')
    plt.xlabel('Kickoff Time Category')
    plt.ylabel('Number of Occurrences')
    plt.legend(title='Outcome')
    plt.show()
    
    #plot frequency of a goal upset per kickoff time category
    early_gu_freq = 1 / ((early_overgoals + early_undergoals) / 
                         (len(early_ko_df)))
    midday_gu_freq = 1 / ((midday_overgoals + midday_undergoals) / 
                          (len(midday_ko_df)))
    late_gu_freq = 1/ ((late_overgoals + late_undergoals) / (len(late_ko_df)))
    
    sns.barplot(x=['Early', 'Midday', 'Late'], y=[early_gu_freq, midday_gu_freq, 
                                                  late_gu_freq], 
                                                edgecolor='black')
    plt.title('Frequency of Goal Upsets by Kickoff Time')
    plt.xlabel('Kickoff Time Category')
    plt.ylabel('Upset Frequency')
    plt.show()

def num_games_hour_v9(df):
    ''' given df, return bar chart showing number of games player per ko time
    '''
    
    hours = [12, 14, 15, 16, 17, 18, 19, 20]
    num_games = []
    
    # count number of occurence of each hour number
    for hour in hours:
        count = df[df['Hour'] == hour].shape[0]
        num_games.append(count)
    
    # create labels for pie chart
    labels = [f'{hour}h' for hour in hours]
    
    plt.pie(num_games, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Games Played Across Hours')
    plt.show()
    
def main():
   
    # create df with different start years for different purposes based on what 
    # year certain data started to be added to data files
    kot_df = file_to_df(2019, 2022)
    total_df = file_to_df(2012, 2022)
    goals_df = file_to_df(2020, 2022)

    kot_df = add_average_columns(kot_df)
    total_df = add_average_columns(total_df)
    
    kot_df = add_upset_column(kot_df)
    total_df = add_upset_column(total_df)
    
    upset_by_kot = upset_per_kot(kot_df)
    
    kot_df = add_month_column(kot_df)
    total_df = add_month_column(total_df)
    
    # viz 1
    upset_kot_bargraph_v1(upset_by_kot)
    
    # viz 2
    upset_month_bargraph_v2(total_df)
    
    # viz 3
    linregress_month_v3(total_df)
    
    # viz 4
    linregress_ko_v4(kot_df)

    # viz 5
    feature_importance_v5(kot_df)
    
    # viz 6
    avg_goalsKO_v6(kot_df)
    
    # viz 7/8
    goals_upset_graph_v7_8(goals_df)
   
    # viz 9
    num_games_hour_v9(kot_df)
    
        
if __name__ == "__main__":
    main()


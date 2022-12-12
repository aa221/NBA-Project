#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nba_api')
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

import pandas as pd
import numpy as np


# In[2]:


get_ipython().system('pip install --upgrade nba_api')


# ## Below i'm doing the process for one team (celtics) but this will be generalized over all teams
# 
# ### Don't run this.. This is just for tutorial purpose

# In[3]:


"""from nba_api.stats.static import teams

nba_teams = teams.get_teams()
# Select the dictionary for the Celtics, which contains their team ID
celtics = [team for team in nba_teams if team['abbreviation'] == 'NOP'][0]
celtics_id = celtics['id']"""


# In[4]:


"""from nba_api.stats.endpoints import leaguegamefinder

# Query for games where the Celtics were playing
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=celtics_id)
# The first DataFrame of those returned is what we want.
games = gamefinder.get_data_frames()[0]
games"""


# ## Want games over the last 10 years excluding 2022

# ### So this is NBA games over the last 10 years for the boston celtics

# ## Below is a function to keep only home or away games.

# In[6]:


def keep_games(df, keep_method='home'):
    '''Combine a TEAM_ID-GAME_ID unique table into rows by game. Slow.

        Parameters
        ----------
        df : Input DataFrame.
        keep_method : {'home', 'away', 'winner', 'loser', ``None``}, default 'home'
            - 'home' : Keep rows where TEAM_A is the home team.
            - 'away' : Keep rows where TEAM_A is the away team.
            - 'winner' : Keep rows where TEAM_A is the losing team.
            - 'loser' : Keep rows where TEAM_A is the winning team.
            - ``None`` : Keep all rows. Will result in an output DataFrame the same
                length as the input DataFrame.
                
        Returns
        -------
        result : DataFrame
    '''
    if keep_method =='home':
        result = df[df["MATCHUP"].str.contains(" vs. ")]

        
    elif keep_method =="away":
        result = df[df["MATCHUP"].str.contains(" @ ")]
        
        
    
    return result
    


# ## Below is a function that translates L to 0 and W to 1 -- this helps us calculate the actual number of losses and wins. 

# In[11]:


def clean_wins(df):
    df["win"] = [1 if i=="W" else 0 for i in df["WL"]]
    return df


# In[12]:


sf = "2019"
sf[:4]


# ## Next is a function that groups the games by season_id so we can get season averages.

# In[13]:


def grouper(df):
    return df.groupby("SEASON_ID").agg(np.sum)


# In[14]:


games_10.groupby("SEASON_ID").agg(np.sum)


# ## Last function is to only look at regular season

# In[15]:


def reg(df):
    at_least_20 = pd.DataFrame(df.groupby("SEASON_ID").size()>20).reset_index()
    the_df = df.merge(at_least_20,on="SEASON_ID").rename(columns={0:"Regular_season"})
    
    return the_df[the_df["Regular_season"]==True]
        
    


# ## Overall pipeline

# ## Now we have a streamlined process to generate these tables for each team.

# ### Below we generate a table for each team

# In[22]:


nba_teams = teams.get_teams()
the_abreviations = [nba_teams[i]["abbreviation"] for i in range(len(nba_teams))]
nba_teams


# In[23]:


import time


# In[24]:


def get_games(abbreviation_lst):
    nba_teams = teams.get_teams()
    the_tracker = {}
    for i in abbreviation_lst:
        team = [team for team in nba_teams if team['abbreviation'] == i][0]
        team_id = team['id']
        ## get all the games
        time.sleep(2)
        game_finder_two = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        ## Storing games in the following: {team_1:[home_df,away_df],team2:[home_df,away_df]}
        games_two = game_finder_two.get_data_frames()[0]
        isolated_dates = games_two[(games_two.GAME_DATE >= '2011-08-16') & (games_two.GAME_DATE < '2022-06-16')]
        away_games = keep_games(isolated_dates,keep_method="away")
        home_games = keep_games(isolated_dates,keep_method="home")
        for_away = clean_wins(away_games)
        for_home = clean_wins(home_games)
        reg_away = reg(for_away)
        reg_home = reg(for_home)
        grouped_home = grouper(reg_home)
        grouped_away = grouper(reg_away)
        the_tracker[i]= [grouped_home,grouped_away]
        print(i)
    return the_tracker


        
        
        
        
        
        


# In[25]:


the_process = get_games(the_abreviations)


# ## EDA part 

# In[26]:


def combine_dfs(dic,home=True):
    empty_data_frame = pd.DataFrame()
    for i in dic:
        if home==True:
            empty_data_frame = pd.concat([empty_data_frame,dic[i][0]], ignore_index=True)
        else:
            empty_data_frame = pd.concat([empty_data_frame,dic[i][1]], ignore_index=True)
            
    return empty_data_frame
        


# In[27]:


home_combined_df = combine_dfs(the_process)


# In[28]:


away_combined_df = combine_dfs(the_process,False)


# In[29]:


away_combined_df


# In[30]:


home_combined_df[home_combined_df["win"]==min(home_combined_df["win"])]


# ## Downloading the combined files so we'll have one CSV for home games, and one for away

# In[31]:


away_combined_df.to_csv("away_table")


# In[32]:


import matplotlib.pyplot as plt
import numpy as np

def getRand(n):
    return np.random.normal(scale=10, size=n)

f = plt.figure(figsize=(20, 20))
f, axes = plt.subplots(nrows = 3, ncols = 3, sharex=False, sharey = True,figsize=(20, 20))
a, b = np.polyfit(home_combined_df["PTS"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)
PPG = axes[0][0].scatter(home_combined_df["PTS"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[0][0].plot(home_combined_df["PTS"]/home_combined_df["Regular_season"], a*(home_combined_df["PTS"]/home_combined_df["Regular_season"])+b, color='Red', linewidth=2)

d, r = np.polyfit(home_combined_df["AST"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

APG = axes[0][1].scatter(home_combined_df["AST"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[0][1].plot(home_combined_df["AST"]/home_combined_df["Regular_season"], d*(home_combined_df["AST"]/home_combined_df["Regular_season"])+r, color='Red', linewidth=2)


s, t = np.polyfit(home_combined_df["TOV"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

TPG = axes[1][0].scatter(home_combined_df["TOV"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[1][0].plot(home_combined_df["TOV"]/home_combined_df["Regular_season"], s*(home_combined_df["TOV"]/home_combined_df["Regular_season"])+t, color='Red', linewidth=2)


n, d = np.polyfit(home_combined_df["REB"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

RPG = axes[1][1].scatter(home_combined_df["REB"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[1][1].plot(home_combined_df["REB"]/home_combined_df["Regular_season"], n*(home_combined_df["REB"]/home_combined_df["Regular_season"])+d, color='Red', linewidth=2)



l, q = np.polyfit(home_combined_df["FTM"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

FTM = axes[2][0].scatter(home_combined_df["FTM"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[2][0].plot(home_combined_df["FTM"]/home_combined_df["Regular_season"], l*(home_combined_df["FTM"]/home_combined_df["Regular_season"])+q, color='Red', linewidth=2)



w, t = np.polyfit(home_combined_df["BLK"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

BLK = axes[2][1].scatter(home_combined_df["BLK"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[2][1].plot(home_combined_df["BLK"]/home_combined_df["Regular_season"], w*(home_combined_df["BLK"]/home_combined_df["Regular_season"])+t, color='Red', linewidth=2)

ww, tt = np.polyfit(home_combined_df["FG3A"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

FG3A = axes[2][2].scatter(home_combined_df["FG3A"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[2][2].plot(home_combined_df["FG3A"]/home_combined_df["Regular_season"], ww*(home_combined_df["FG3A"]/home_combined_df["Regular_season"])+tt, color='Red', linewidth=2)


www, ttt = np.polyfit(home_combined_df["FTM"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

FTA = axes[1][2].scatter(home_combined_df["FTA"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[1][2].plot(home_combined_df["FTA"]/home_combined_df["Regular_season"], www*(home_combined_df["FTA"]/home_combined_df["Regular_season"])+ttt, color='Red', linewidth=2)

w2, t2 = np.polyfit(home_combined_df["STL"]/home_combined_df["Regular_season"],home_combined_df["win"], 1)

STL = axes[0][2].scatter(home_combined_df["STL"]/home_combined_df["Regular_season"],home_combined_df["win"])
axes[0][2].plot(home_combined_df["STL"]/home_combined_df["Regular_season"], w2*(home_combined_df["STL"]/home_combined_df["Regular_season"])+t2, color='Red', linewidth=2)



axes[0][0].set_xlabel('Points per Game', labelpad = 5)
axes[0][1].set_xlabel('Assists per Game', labelpad = 5)
axes[1][0].set_xlabel('Turnovers per Game', labelpad = 5)
axes[1][1].set_xlabel('Rebounds per Game', labelpad = 5)
axes[2][0].set_xlabel('Free throws made per Game', labelpad = 5)
axes[2][1].set_xlabel('Blocks per Game', labelpad = 5)
axes[1][2].set_xlabel('Free Throws Attempted per game', labelpad = 5)
axes[2][2].set_xlabel('Threes attempted per game', labelpad = 5)
axes[0][2].set_xlabel('Steals per game', labelpad = 5)


axes[0][0].set_ylabel('Home Wins', labelpad = 5)
axes[0][1].set_ylabel('Home Wins', labelpad = 5)
axes[1][0].set_ylabel('Home Wins', labelpad = 5)
axes[1][1].set_ylabel('Home Wins', labelpad = 5)
axes[2][0].set_ylabel('Home Wins', labelpad = 5)
axes[2][1].set_ylabel('Home Wins', labelpad = 5)
axes[1][2].set_ylabel('Home Wins', labelpad = 5)
axes[2][2].set_ylabel('Home Wins', labelpad = 5)
axes[0][2].set_ylabel('Home Wins', labelpad = 5)


plt.suptitle('Home Game Stats Vs Home Wins',fontsize=40)
plt.show()


# In[33]:


home_combined_df.columns


# In[34]:


import matplotlib.pyplot as plt
import numpy as np

def getRand(n):
    return np.random.normal(scale=10, size=n)

f = plt.figure(figsize=(20, 20))
f, axes = plt.subplots(nrows = 3, ncols = 3, sharex=False, sharey = True,figsize=(20, 20))
a, b = np.polyfit(away_combined_df["PTS"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)
PPG = axes[0][0].scatter(away_combined_df["PTS"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[0][0].plot(away_combined_df["PTS"]/away_combined_df["Regular_season"], a*(away_combined_df["PTS"]/away_combined_df["Regular_season"])+b, color='Red', linewidth=2)

d, r = np.polyfit(away_combined_df["AST"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

APG = axes[0][1].scatter(away_combined_df["AST"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[0][1].plot(away_combined_df["AST"]/away_combined_df["Regular_season"], d*(away_combined_df["AST"]/away_combined_df["Regular_season"])+r, color='Red', linewidth=2)


s, t = np.polyfit(away_combined_df["TOV"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

TPG = axes[1][0].scatter(away_combined_df["TOV"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[1][0].plot(away_combined_df["TOV"]/away_combined_df["Regular_season"], s*(away_combined_df["TOV"]/away_combined_df["Regular_season"])+t, color='Red', linewidth=2)


n, d = np.polyfit(away_combined_df["REB"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

RPG = axes[1][1].scatter(away_combined_df["REB"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[1][1].plot(away_combined_df["REB"]/away_combined_df["Regular_season"], n*(away_combined_df["REB"]/away_combined_df["Regular_season"])+d, color='Red', linewidth=2)



l, q = np.polyfit(away_combined_df["FTM"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

FTM = axes[2][0].scatter(away_combined_df["FTM"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[2][0].plot(away_combined_df["FTM"]/away_combined_df["Regular_season"], l*(away_combined_df["FTM"]/away_combined_df["Regular_season"])+q, color='Red', linewidth=2)



w, t = np.polyfit(away_combined_df["BLK"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

BLK = axes[2][1].scatter(away_combined_df["BLK"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[2][1].plot(away_combined_df["BLK"]/away_combined_df["Regular_season"], w*(away_combined_df["BLK"]/away_combined_df["Regular_season"])+t, color='Red', linewidth=2)

ww, tt = np.polyfit(away_combined_df["FG3A"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

FG3A = axes[2][2].scatter(away_combined_df["FG3A"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[2][2].plot(away_combined_df["FG3A"]/away_combined_df["Regular_season"], ww*(away_combined_df["FG3A"]/away_combined_df["Regular_season"])+tt, color='Red', linewidth=2)


www, ttt = np.polyfit(away_combined_df["FTA"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

FTA = axes[1][2].scatter(away_combined_df["FTA"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[1][2].plot(away_combined_df["FTA"]/away_combined_df["Regular_season"], www*(away_combined_df["FTA"]/away_combined_df["Regular_season"])+ttt, color='Red', linewidth=2)

w2, t2 = np.polyfit(away_combined_df["STL"]/away_combined_df["Regular_season"],away_combined_df["win"], 1)

STL = axes[0][2].scatter(away_combined_df["STL"]/away_combined_df["Regular_season"],away_combined_df["win"])
axes[0][2].plot(away_combined_df["STL"]/away_combined_df["Regular_season"], w2*(away_combined_df["STL"]/away_combined_df["Regular_season"])+t2, color='Red', linewidth=2)



axes[0][0].set_xlabel('Points per Game', labelpad = 5)
axes[0][1].set_xlabel('Assists per Game', labelpad = 5)
axes[1][0].set_xlabel('Turnovers per Game', labelpad = 5)
axes[1][1].set_xlabel('Rebounds per Game', labelpad = 5)
axes[2][0].set_xlabel('Free throws made per Game', labelpad = 5)
axes[2][1].set_xlabel('Blocks per Game', labelpad = 5)
axes[1][2].set_xlabel('Free Throws Attempted per game', labelpad = 5)
axes[2][2].set_xlabel('Threes attempted per game', labelpad = 5)
axes[0][2].set_xlabel('Steals per game', labelpad = 5)


axes[0][0].set_ylabel('Away Wins', labelpad = 5)
axes[0][1].set_ylabel('Away Wins', labelpad = 5)
axes[1][0].set_ylabel('Away Wins', labelpad = 5)
axes[1][1].set_ylabel('Away Wins', labelpad = 5)
axes[2][0].set_ylabel('Away Wins', labelpad = 5)
axes[2][1].set_ylabel('Away Wins', labelpad = 5)
axes[1][2].set_ylabel('Away Wins', labelpad = 5)
axes[2][2].set_ylabel('Away Wins', labelpad = 5)
axes[0][2].set_ylabel('Away Wins', labelpad = 5)


plt.suptitle('Away Game Stats Vs Away Wins',fontsize=40)
plt.show()


# In[35]:


home_combined_df["diff"] = home_combined_df["win"]/home_combined_df["Regular_season"]


# In[36]:


away_combined_df["diff"] = away_combined_df["win"]/away_combined_df["Regular_season"]


# In[37]:


val_home = np.mean(home_combined_df["diff"])
val_away = np.mean(away_combined_df["diff"])
plt.bar(x=["Home Games","Away Games"], height=[val_home,val_away])
plt.ylabel("Percentage of games that are wins")
plt.title("Win Percentage at Home vs Away")


# In[38]:


home_combined_df["Home Win Percentage"] = home_combined_df["diff"]
away_combined_df["Away Win Percentage"] = away_combined_df["diff"]


# In[39]:


import seaborn as sns
ax = sns.boxplot(data=[home_combined_df["Home Win Percentage"], away_combined_df["Away Win Percentage"]])
ax.set_xticklabels(["Home Games","Away Games"])
ax.set_ylabel("Percent of Wins")
plt.title("Win Percentage Distribution")


# In[40]:


home_combined_df.boxplot(column="Home Win Percentage")
away_combined_df.boxplot(column="Away Win Percentage")


# ## Explination:
# ### Above are the characteristics that relate to teams winning games. Therefore if we want to create a GLM, we must keep these features in mind.

# In[41]:


plt.scatter(home_combined_df["PTS"]/home_combined_df["Regular_season"],home_combined_df["win"])


# In[42]:


plt.scatter(home_combined_df["TOV"]/home_combined_df["Regular_season"],home_combined_df["win"]/home_combined_df["Regular_season"])


# ## Below I create the distribution that we will be bootstrapping from

# ### Basically just take the home and away win percentages and combine them into the same distribution. This work because our null is that there is no difference between 

# In[43]:


combined_populaton = pd.DataFrame(columns = (["Home Win","Away Win"]))
combined_populaton


# In[44]:


combined_populaton["Home Win"]= home_combined_df["Home Win Percentage"]
combined_populaton["Away Win"] = away_combined_df["Away Win Percentage"]


# In[45]:


combined_populaton


# ### For simplicity I stack them

# In[46]:


empty = []
combine_array = pd.concat([combined_populaton["Home Win"], combined_populaton["Away Win"]], axis = 0)


# In[47]:


the_stacked_df = pd.DataFrame(combine_array)


# In[48]:


plt.hist(the_stacked_df)
plt.title("Pooled Distribution of Home and Away Games")
plt.xlabel("Proportion of games won")


# ## Above is the distribution we will be sampling from in order to mimic the distribution of the null Hypothesis. 

# #### The process:
# 1) Take two samples of equal sizes  from this population above (where the population contains the percentage of away wins and the percentage of home wins) and calculate the test statistic (percentage of wins at home vs percentage of wins way)
# 
# 2) Repeat this many times
# 
# 3) Now we have a histogram with the test statistics. 
# 

# The parameters:
# 1)  330 size sample representing home wins. 330 Size Sample representing away wins. Total is 660 points like the population.
#     
#     
#    - This is because this was the size of our accessible population, so if we have the entire data set we can just    bootstrap it (sample with replacement).
# 
# 2) We'll repeat this 10k times to have a sufficiently large data set. 
# 
# 3) Test statistic is the average difference in proportion of wins for home and away games
# 

# In[49]:


def the_bootstrap(the_pop,replications=10000):
    tracker = []
    for i in range(replications):
        away_represented = the_pop.sample(330,replace=True)
        home_represented = the_pop.sample(330,replace=True)
        test_statistic = np.mean(home_represented - away_represented)[0]
        tracker.append(test_statistic)
    return tracker


# In[50]:


the_strap = the_bootstrap(the_stacked_df)


# ### Now I collect the observed statistic for each team.

# In[51]:


the_process["ATL"][1]


# In[52]:


### A function to subtract the mean proportion of wins at home minus the mean proportion of wins away

def find_per(df_home,df_away):
    mean_home = np.mean(df_home["win"]/df_home["Regular_season"])
    mean_away = np.mean(df_away["win"]/df_away["Regular_season"])
    return mean_home - mean_away

observed = {}
for i in the_process:
    ### store it as {the_team:the_diff,the_team:the_diff}
    observed[i]=find_per(the_process[i][0],the_process[i][1])
    

    


# #### Visualizing our observed statistics vs our bootstrap distribution

# In[53]:


get_ipython().system('pip install distinctipy')


# In[54]:


from distinctipy import distinctipy

# number of colours to generate
N = 30

# generate N visually distinct colours
colors = distinctipy.get_colors(N)

# display the colours
import random


# In[55]:


plt.figure(figsize=(15, 15), dpi=80)

plt.hist(the_strap,density=True)
plt.xlabel("Average Proportion of wins at home - Average Proportion of wins away")
used = []
key_list = [i for i in observed.keys()]
for i in range(len(observed)):
    color = random.randint(0,26)
    plt.axvline(x = observed[key_list[i]], ymax = 1,label = key_list[i],c =colors[i],alpha=.7)
plt.legend()

counter=0
for x,y in zip(observed.values(),np.linspace(25,39,30)):
    label = key_list[counter]
    counter+=1

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',rotation=90) # horizontal alignment can be left, right or center
    
plt.show()


# In[40]:


significant ={}
key_list = [i for i in observed.keys()]
for i in key_list:
    number_more = [w for w in the_strap if w>observed[i]]
    number_less = [w for w in the_strap if w<observed[i]]
    if len(number_more)/len(the_strap) < .05 or len(number_less)/len(the_strap) < .05:
        significant[i]= str(len(number_more)/len(the_strap))
    else:
        significant[i]='Not significant'
        
significant   


# ### Immediately it's apparent that every single NBA team has a statistically significant result.

# We can conclude that the mean difference between the proportion of wins at home and away are statisatically significant for each team, with a P-value of 0. Because neither the bonferonni nor the B-H procedure will give us a p-value cutoff below 0, we know that every single result will be significant (with a p-value below cuttoff).

# ### Statistically signifcant factors.

# Since we're not controlling for which players are playing, we cannot directly create a cause and effect relationship between certain game factors and wins.
# 
# What we can do is analyze which factors differ between home and away at a statistically signifcant level in order to gain some context as to why home games might be won more. 
# 
# For example if it is found that the difference between steals at home and steals away is not due to chance, then this may have some association with why teams tend to win more at home. 
# 
# * This doesnt imply causation but adds more context and a shortlist of variables that could be linked through causation.
#     
# * This also gives us an idea about what variables we can use to predict home and away win proportions.

# ### Shortlisting features
# 
# The Variables we'll be looking at come from the association charts above:
# 
# 1) Points per game.
# 
# 2) Assists per game. 
# 
# 3) Steals per game. 
# 
# 4) Rebounds per game.
# 
# 5) Turnovers per game.
# 
# 6) Threes attempted per game.
# 
# 7) Free throws made per game. 

# ### Defining automating functions

# In[41]:


## A function that generates our pop

def create_pop(the_stat):
    combined_populaton = pd.DataFrame(columns = (["Home"+" "+the_stat,"Away"+" "+the_stat]))
    combined_populaton["Home"+" "+the_stat]= home_combined_df[the_stat]/home_combined_df["Regular_season"]
    combined_populaton["Away"+" "+the_stat] = away_combined_df[the_stat]/away_combined_df["Regular_season"]
    empty = []
    combine_array = pd.concat([combined_populaton["Home"+" "+the_stat], combined_populaton["Away"+" "+the_stat]], axis = 0)
    the_stacked_df = pd.DataFrame(combine_array)
    return the_stacked_df


# In[42]:


## A function to bootstrap 
def the_bootstrap(the_pop,replications=10000):
    tracker = []
    for i in range(replications):
        away_represented = np.mean(the_pop.sample(330,replace=True))
        home_represented = np.mean(the_pop.sample(330,replace=True))
        test_statistic = home_represented[0] - away_represented[0]
        tracker.append(test_statistic)
    return tracker


# In[43]:


## A function that finds the observed per for any variable

def find_per(the_stat):
    observed = {}
    for i in the_process:
    ### store it as {the_team:the_diff,the_team:the_diff}
        mean_home = np.mean(the_process[i][0][the_stat]/the_process[i][0]["Regular_season"])
        mean_away = np.mean(the_process[i][1][the_stat]/the_process[i][1]["Regular_season"])
        observed[i]=mean_home-mean_away
    
    return observed



    
the_points = find_per("PTS") 


# In[44]:


### Visualizing results
def visualize(observed_stats,booter,stat_name):

    plt.figure(figsize=(15, 15), dpi=80)

    plt.hist(booter,density=True)
    plt.xlabel("Average "+stat_name+" at home"+" vs away")
    used = []
    key_list = [i for i in observed_stats.keys()]
    for i in range(len(observed_stats)):
        color = random.randint(0,26)
        plt.axvline(x = observed_stats[key_list[i]], ymax = 1,label = key_list[i],c =colors[i],alpha=.7)
    plt.legend(loc='upper right')

    counter=0
    for x,y in zip(observed_stats.values(),np.linspace(25,39,30)):
        label = key_list[counter]
        counter+=1

        plt.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center',rotation=90) # horizontal alignment can be left, right or center

    return plt.show()


# In[45]:


def signif(our_strap,the_observed): 
    significant ={}
    numeri = {}
    key_list = [i for i in the_observed.keys()]
    for i in key_list:
        number_more = [w for w in our_strap if w>=the_observed[i]]
        number_less = [w for w in our_strap if w<=the_observed[i]]
        if len(number_more)/len(our_strap) < .05:
            significant[i]= str(len(number_more)/len(our_strap))
            numeri[i]=len(number_more)/len(our_strap)
            
        elif len(number_less)/len(our_strap) < .05:
            significant[i]= str(len(number_less)/len(our_strap))
            numeri[i]=len(number_less)/len(our_strap)
        else:
            if the_observed[i]>=0:
                significant[i]=str(len(number_more)/len(our_strap))+' Not significant'
                numeri[i]=len(number_more)/len(our_strap)
            else:
    
                significant[i]=str(len(number_less)/len(our_strap))+' Not significant'
                numeri[i]=len(number_less)/len(our_strap)
                

    return [significant,numeri]   


# ### Points per game

# In[46]:


create_pop("PTS")


# In[47]:


points_pop = create_pop("PTS")
ballsohard = the_bootstrap(points_pop)


# In[48]:


visualize(the_points,ballsohard, "Points")


# In[49]:


the_p_values = signif(ballsohard,the_points)[0]
the_p_values


# Before using correction methods, only Chicago and brooklyn are not significant, makes sense because they were one of the teams closest to having non significant results for home and away wins. 
# 
# - If most other teams score more at home than away, and the bulls score the same at home and away.
# 
# - Interestingly enough, Minnesota scores less points per game at home than away on average. Since they win more at home, then their defense at home must be better than away!

# ## Bonferonni and B-H for Points per game

# In[50]:


def benjamini_hochberg(p_values, alpha):
    """
    Returns decisions on p-values using Benjamini-Hochberg.
    
    Inputs:
        p_values: array of p-values
        alpha: desired FDR (FDR = E[# false positives / # positives])
    
    Returns:
        decisions: binary array of same length as p-values, where `decisions[i]` is 1
        if `p_values[i]` is deemed significant, and 0 otherwise
    """

    the_data = pd.DataFrame({"p_vals":p_values})
    the_data["k"] = the_data["p_vals"].rank()
    the_amount = the_data["k"]*alpha/ len(the_data["p_vals"])
    the_data["the_amount"]=the_amount
    filtered = max(the_data[the_data["the_amount"]>the_data["p_vals"]]["k"])
    new_amount = filtered*alpha/ len(the_data["k"])
    decisions = [the_data["p_vals"][i] <=new_amount for i in np.arange(len(p_values) )] 
    return decisions


# In[51]:


def viz_bh(the_data_p_val):
    
    the_data_one = [i for i in the_data_p_val]

    the_data = pd.DataFrame(the_data_one).rename(columns={0:"p_val"})
    the_data["k"] = the_data["p_val"].rank(method='first')
    the_amount = the_data["k"]*.05/ len(the_data["p_val"])
    the_data["the_amount"]=the_amount
    new_amount = the_data["k"]*.05/ len(the_data["k"])

    sns.scatterplot(x=the_data["k"], y=the_data["p_val"],color = 'black');
    plt.plot(the_data["k"],new_amount, label='B-H guide', color='red')
    plt.title("B-H")
    plt.legend();
    return the_data.head()
    


# In[52]:


viz_bh(signif(ballsohard,the_points)[1].values())


# In[53]:


"Even after B-H we make "+ str(sum(benjamini_hochberg(signif(ballsohard,the_points)[1].values(),.05)))+" Discoveries"


# Under Bonferonni we'll make only 26 discoveries

# ### Bonferonni 

# We make the same number of discoveries even under the Bonferonni threshold of .0016 

# ### Assists per game 

# In[54]:


ass_pop = create_pop("AST")
ballsohard_as = the_bootstrap(ass_pop)


# In[55]:


obs_ast = find_per("AST")
visualize(obs_ast,ballsohard_as, "Assists")


# In[56]:


signif(ballsohard_as,obs_ast)[0]


# In[57]:


viz_bh(signif(ballsohard_as,obs_ast)[1].values())


# In[58]:


signif(ballsohard,obs_ast)[0]


# Bonferonni threshold of 0.001666666667 , will mean we make one less discovery with DAL being above the threshold. 
# 
# This controls the probability of making one false positive across all teams. -- More strict than BH
# 
# B-H controls false discovery rate, which is the probability of a non discovery out of the discovery points. 

# ### Rebounds 

# In[59]:


reb_pop = create_pop("REB")
ballsohard = the_bootstrap(reb_pop)
obs_reb = find_per("REB")
visualize(obs_reb,ballsohard, "Rebounds")


# In[60]:


signif(ballsohard,obs_reb)[0]


# In[61]:


viz_bh(signif(ballsohard,obs_reb)[1].values())


# After bonferonni we make 3 less discoveries.

# ## Overall for the 'big three' offensive stats:
# 
# 1) Points do differ between home and away teams for all but two teams. 
# * This is probably the biggest driver as to why home teams win more. 
# 
# 2) For about 5 teams assists per  do not differ between home and away. 
# * The least important in predicting the number of home wins a team has. 
# 
# 3) All but 4 team have differing home and away rebounds. 

# ## Some defensive stats

# ### Steals per game

# In[73]:


stl_pop = create_pop("STL")
ballsohard = the_bootstrap(stl_pop)
obs_stl = find_per("STL")
visualize(obs_stl,ballsohard, "STL")


# In[75]:


viz_bh(signif(ballsohard,obs_stl)[1].values())


# In[76]:


signif(ballsohard,obs_stl)[0]


# With BH we make 15 discoveries. With bonferonni we make 9. 

# ### Blocks per game

# In[77]:


blcks_pop = create_pop("BLK")
ballsohard = the_bootstrap(blcks_pop)
obs_blk = find_per("BLK")
visualize(obs_blk,ballsohard, "BLK")


# In[79]:


viz_bh(signif(ballsohard,obs_blk)[1].values())


# In[81]:


signif(ballsohard,obs_blk)[0]


# We make 28 discoveries under the B-H and 27 under the Bonferonni.

# ## Suspected largest difference makers

# ### Freethrows made per game

# In[82]:


FTM_pop = create_pop("FTM")
ballsohard = the_bootstrap(FTM_pop)
obs_ftm = find_per("FTM")
visualize(obs_blk,ballsohard, "FTM")


# In[83]:


signif(ballsohard,obs_ftm)[0]


# In[84]:


viz_bh(signif(ballsohard,obs_ftm)[1].values())


# 21 Discoveries under the B-H, 18 under bonferonni.

# ### Plus_Minus made per game

# In[85]:


pls_pop = create_pop("PLUS_MINUS")
ballsohard = the_bootstrap(pls_pop)
obs_plus = find_per("PLUS_MINUS")
visualize(obs_plus,ballsohard, "PLUS_MINUS")


# ### Like points this is highly variable between home and away games... This is the average of plus minus per player on the team divided by the number of games that season.
# 
# ALl statistically significant

# ## Overall:
# 
# The following are statistically significant across almost all teams—meaning they differ between home and away not due to chance. These are variables that may have the most impact on proportion of games won at home versus away. 
# 
# 
# 1) Rebounds Per game 
# 
# 2) Points Per game
# 
# 3) Plus Minus Per game 
# 
# 4) Blocks per game
# 
# 
# The following are statistically significant across most teams, however they are statistically significant on less teams.
# 
# 1) Assists Per game
# 
# 
# These are the variables that are statistically significant across the least number of teams. 
# 
# 1) Free throws made per game
# 
# 2) Steals per game
# 
# 
# 

# ### Association between plus minus per game vs proportion of wins at home and away

# In[92]:


away_combined_df.columns


# In[98]:


away_stats = away_combined_df[["Away Win Percentage","PLUS_MINUS"]]
home_stats = home_combined_df[["Home Win Percentage","PLUS_MINUS"]]


# In[103]:


f = plt.figure(figsize=(10, 10))
plt.scatter(home_stats["PLUS_MINUS"]/home_combined_df["Regular_season"],home_stats["Home Win Percentage"])
plt.title("Average plus minus per game for home games vs Proportion of wins at home")
plt.xlabel("Average plus minus per game")
plt.ylabel("Proportion of wins at home")


# In[105]:


f = plt.figure(figsize=(10, 10))
plt.scatter(away_stats["PLUS_MINUS"]/away_combined_df["Regular_season"],away_stats["Away Win Percentage"])
plt.title("Average plus minus per game for away games vs Proportion of wins  away")
plt.xlabel("Average plus minus per game")
plt.ylabel("Proportion of wins away")


# In[ ]:





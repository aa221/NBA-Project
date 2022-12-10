# NBA-Project

Exploring Home vs Away win proportion in the NBA


## Context: 
Analysts commonly agree on a phenomenon known as home-court advantage. It is thought that the psychological impact of playing in front of one’s own team, gives the players of the home team a significant advantage—largely because of the comfort associated with one’s own home-town.
The first part of our question is: do NBA teams perform better at home than away?
We’ll be answering this question using a hypothesis test, repeated for each of the 30 NBA teams. While the distribution we are using will remain the same , the observed statistic for each team will vary.
At a higher level, performance will be evaluated based on wins. Specifically, the number of games at home that result in wins will be compared to the number of games away that result in wins (percent of home games that are wins vs percent of away games that are wins).

The second focus of our research will be to utilize various derived metrics in order to predict the proportion of wins at home minus the proportion of wins away.
These utilized metrics will be derived from the EDA process within the notebook, with some examples potentially including: field goals made, turnovers, rebounds, and assists per game (amongst others).

## Project outline: 
- Conduct a repeated (for each team) hypothesis tet to assess whether teamsm truly perform better at home than away. 
- Conduct this same hypothesis test for various game pertaining statistics in order to see if they vary between home and away. 
- Use the features that vary most between home and away across al teams to predict the difference between home and away win proportion. 
- The logic is that if a given metric (lets say points per game) varies across all teams between home and away without due to chance, then using this feature in our prediction model will help differentiate between which teams may have a higher win proportion at home than away. 


## Hypothesis testing visualized





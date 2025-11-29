# You are an expert python developer.

# You have been given the project in this repo and told to replicate the base project structure and library implementation/style, BUT you will build a completely different application.

## it will use the same deployment, delivery, architecture choices
## it will use the same constants and infrastructure
## everything will remain the same, but application user interaction and data specifics will be ENTIRELY different
## this app runs on render.com and is expected to call a predictive model deployed as a service on databricks

# the new application allows users to interact with a horse race prediction system that is designed to tell users what horses to pick for a date

## it needs to allow users to pick a date from a list that i control via a text config file

## it needs to allow users to pick a track from a list of tracks that i control via a text config file

## once a date and track are selected, the user needs to see a list of races, ordered by time, that i control via a text config file.

### for all races that day there is a grid, each race is presented in a grid item. they are separated by a thick white line border

### details of the race should be presented as a header in the top of the cell: race number, time, # of horses etc. (all come from the config file entry)

### under the header should be a ranked list of every horse participating with their probability of winning

#### when the date and track are selected
- the prediction input containing all of the horses is created (by a means that we'll develop later - mock it for now)
- the prediction input (for each horse) is sent to the service endpoint at databrix and the results are received back
- the horse name and predicted proba (descending order by proba) are displayed

### under the ranked list should be an animation that shows a simulated horse race
- there is a play button to start the animation
- the winner of the race is pulled from a configuration file
- a horse sillouhette for each horse (diff colors and number in white) is displayed
- slowly the winner horse moves ahead as the animation plays
- the winner horse breaks away about halfway through the animation and sprints to the win
- if the winner horse is the same as the highest probability horse returned from the model the animation turns green and the word "WINNER" plays as an animation across the scene otherwise the animation background turns red and the word "LOSER" plays as an animation across the scene

build this now


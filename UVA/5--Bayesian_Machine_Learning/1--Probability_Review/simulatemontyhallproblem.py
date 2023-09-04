'''
We create a simulation of the Monty Hall problem with randomly selected doors, run the simulation many times, and compute the probability of the different outcomes given each strategy.
'''

doors = [0, 1, 2]
number_of_iterations = 10000
number_of_stays = 0
number_of_wins_when_staying = 0
number_of_switches = 0
number_of_wins_when_switching = 0
import numpy as np
for i in range(0, number_of_iterations):
    strategy = np.random.choice(['switch', 'stay'])
    car_location = np.random.choice([0, 1, 2])
    contestant_choice = np.random.choice([0, 1, 2])
    doors_from_which_host_may_pick = [door for door in doors if door not in (car_location, contestant_choice)]
    host_door = np.random.choice(doors_from_which_host_may_pick)
    if strategy == 'switch':
        contestant_choice = [door for door in doors if door not in (contestant_choice, host_door)]
        number_of_switches += 1
    else:
        number_of_stays += 1
    if contestant_choice == car_location:
        #print('Strategy: ' + strategy + ' and you win!')
        if strategy == 'switch':
            number_of_wins_when_switching += 1
        else:
            number_of_wins_when_staying += 1
    #else:
        #print('Strategy: ' + strategy + ' and you get a goat.')
probability_of_winning_when_staying = number_of_wins_when_staying / number_of_stays
probability_of_winning_when_switching = number_of_wins_when_switching / number_of_switches
print(f"Probability of winning when staying: {probability_of_winning_when_staying}")
print(f"Probability of winning when switching: {probability_of_winning_when_switching}")
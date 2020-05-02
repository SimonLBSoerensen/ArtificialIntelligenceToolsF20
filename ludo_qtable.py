from ludopy import Game
from ludpyhelper.RL.QLearning.qtable import NQTable
from ludpyhelper.RL.helpers.reward_tracker import RTrack
from ludpyhelper.gamehelper import cal_state, cal_game_events, cal_reward_and_endgame
from tqdm import tqdm
import random
from ludpyhelper.mics import ThreadHandler
from ludpyhelper.mics.functions import ramp
import matplotlib.pyplot as plt
import numpy as np
import threading


epsilon=1.0
game_to_run = 800


def epsilon_update(ep):
    return ramp(ep, 1.0, game_to_run//4*3, game_to_run//4)

agent = NQTable(4,2,100, 10000, n_old=75, k=20, epsilon=1.0, epsilon_update=epsilon_update)

rt = RTrack()

game_done = False

epsilon_hist = []
memory_hist = []


def run_game_in_runs(master_game):
    g = Game()
    for _ in range(game_to_run):
        g.reset()
        game_done = False
        while not game_done:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, game_done), player_i = g.get_observation()

            if len(move_pieces):
                state = cal_state(player_pieces, enemy_pieces, dice)
                state = tuple(state)

                action = agent.act(state)

                if action not in move_pieces:
                    action = random.choice(move_pieces)

            else:
                action = -1

            (_, _, player_pieces_after, enemy_pieces_after, player_is_a_winner_after, game_done_after) = g.answer_observation(action)

            if len(move_pieces):
                game_event = cal_game_events(player_pieces, enemy_pieces, player_pieces_after, enemy_pieces_after)
                reward, end_game = cal_reward_and_endgame(game_event)

                new_states = [tuple(cal_state(player_pieces, enemy_pieces, d)) for d in range(1,6+1)]

                agent.update(state, action, new_states, reward, terminal_state=game_done_after, save_experience=True, new_state_list=True, auto_update_episode=False)
                rt.add(reward)

        agent.train_on_memory(64)

        if master_game:
            agent.update_episode()
            epsilon_hist.append(agent.epsilon)
            memory_hist.append(len(agent.aer))


th = ThreadHandler()
th.make_threads(None, fcpu=3/4, kill_when_done=True)
th.start_threads()

th.put_job(run_game_in_runs, {"master_game": True})
for _ in range(len(th.hreads) - 1):
    th.put_job(run_game_in_runs, {"master_game": False})

th.progress_bar()
th.join()

agent.save_table("ludo_nq.pickel")

rt.plot()

plt.figure()
plt.title("Epsilon")
plt.plot(epsilon_hist)
plt.show()

plt.figure()
plt.title("Memory hist")
plt.plot(memory_hist)
plt.show()

pass
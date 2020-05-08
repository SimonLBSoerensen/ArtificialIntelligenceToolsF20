from ludopy import Game
from ludpyhelper.RL.QLearning.qtable import NQTable
from ludpyhelper.RL.helpers.reward_tracker import RTrack
from ludpyhelper.gamehelper import cal_state, cal_game_events, cal_reward_and_endgame
from tqdm import tqdm
import random
from ludpyhelper.mics.functions import ramp
from ludpyhelper.mics.mics import save_json
from ludpyhelper.mics.sthreading import ThreadHandler
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os
import shutil


def make_run(config):
    run_id = str(uuid.uuid4())
    out_folder = os.path.join(config["main_out_folder"], run_id)
    os.makedirs(out_folder, exist_ok=True)

    shutil.copy2(os.path.realpath(__file__), os.path.join(out_folder, "run_script.txt"))

    if config["run_random"]:
        print(f"""{config["print_id"]}: This is a random run!!!!""")

    def epsilon_update(ep):
        return 0.3 #ramp(ep, 1.0, config["game_to_run"]//4*3, config["game_to_run"]//4)

    agent = NQTable(action_space=4, n_q_tabels=config["NQ"], initial_memory_size=config["initial_memory_size"],
                    max_memory_size=config["max_memory_size"], n_old=config["n_old"], k=config["k"],
                    epsilon=config["start_epsilon"], epsilon_update=epsilon_update if config["use_epsilon_update"] else None,
                    learning_rate=config["learning_rate"], discount_factor=config["discount_factor"])

    rt = RTrack()

    epsilon_hist = []
    memory_hist = []

    g = Game()

    for run_i in range(config["game_to_run"]):
        if run_i % config["print_every"] == 0:
            print(f"""{config["print_id"]}: {run_i}/{config["game_to_run"]}""")

        g.reset()
        end_game = False

        run_reward = 0
        while not end_game:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, game_done), player_i = g.get_observation()

            if player_i == 0:
                if len(move_pieces):
                    state = cal_state(player_pieces, enemy_pieces, dice)
                    state = tuple(state)

                    if not config["run_random"]:
                        action = agent.act(state)

                        if action not in move_pieces:
                            action = random.choice(move_pieces)
                    else:
                        action = random.choice(move_pieces)

                else:
                    action = -1
            else:
                if len(move_pieces):
                    action = random.choice(move_pieces)
                else:
                    action = -1

            (_, _, player_pieces_after, enemy_pieces_after, player_is_a_winner_after, game_done_after) = g.answer_observation(action)

            if player_i == 0:
                if len(move_pieces):
                    game_event = cal_game_events(player_pieces, enemy_pieces, player_pieces_after, enemy_pieces_after)
                    reward, end_game = cal_reward_and_endgame(game_event)
                    run_reward += reward

                    if not config["run_random"]:
                        new_states = [tuple(cal_state(player_pieces, enemy_pieces, d)) for d in range(1, 6+1)]

                        agent.update(state, action, new_states, reward, terminal_state=end_game, save_experience=True,
                                     new_state_list=True, auto_update_episode=False)

        rt.add(run_reward)

        if config["use_replay_buffer"] and not config["run_random"]:
            agent.train_on_memory(128, flat_sample=config["flat_sample"])

        if not config["run_random"]:
            agent.update_episode()
            epsilon_hist.append(agent.epsilon)
            memory_hist.append(len(agent.aer))

    # Saving info from run --------------------------------------------------------------------------------------------
    print(f"""{config["print_id"]}: Saving info from run""")
    agent.save_table(os.path.join(out_folder, "ludo_nq.pickel"))
    rt.save(os.path.join(out_folder, "rewards.npy"))
    np.save(os.path.join(out_folder, "epsilon_hist.npy"), epsilon_hist)
    np.save(os.path.join(out_folder, "memory_hist.npy"), memory_hist)

    rt.plot(save=out_folder, close_plots=config["close_plots"])

    plt.figure()
    plt.title("Epsilon")
    plt.plot(epsilon_hist)
    plt.savefig(os.path.join(out_folder, "epsilon.svg"))
    if config["close_plots"]:
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Memory hist")
    plt.plot(memory_hist)
    plt.savefig(os.path.join(out_folder, "memory.svg"))
    if config["close_plots"]:
        plt.close()
    else:
        plt.show()

    save_json(os.path.join(out_folder, "config.json"), config)
    print(f"""{config["print_id"]}: Done""")

base_config = {
    "start_epsilon": 1.0,
    "game_to_run": 2000,
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "NQ": 2,
    "initial_memory_size": 100,
    "max_memory_size": 10000,
    "n_old": 75,
    "k": 20,
    "use_epsilon_update": True,
    "use_replay_buffer": True,
    "flat_sample": False,
    "main_out_folder": "Runs/LTest_New_s",
    "run_random": False,
    "close_plots": False,
    "print_id": 0,
    "print_every": 100,
}

th = ThreadHandler()
th.make_threads(None, 1/2, kill_when_done=True)

configs = []

run_config = base_config.copy()
run_config["print_id"] = 0
configs.append(run_config)
run_config = base_config.copy()
run_config["print_id"] = 1
configs.append(run_config)
run_config = base_config.copy()
run_config["print_id"] = 2
configs.append(run_config)
run_config = base_config.copy()
run_config["print_id"] = 3
configs.append(run_config)

for r_config in configs:
    th.put_job(make_run, {"config":r_config})

th.start_threads()
th.join()

pass
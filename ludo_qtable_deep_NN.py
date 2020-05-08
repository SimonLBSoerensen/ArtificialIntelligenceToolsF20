from ludopy import Game
from ludpyhelper.RL.Deep.deepq import DQ
from ludpyhelper.RL.helpers.reward_tracker import RTrack
from ludpyhelper.gamehelper import cal_state, cal_game_events, cal_reward_and_endgame
from tqdm import tqdm
import random
from ludpyhelper.mics.functions import ramp
from ludpyhelper.mics.mics import save_json
#from ludpyhelper.mics.sthreading import ThreadHandler
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os
import shutil
from tensorflow import keras


def make_run(config, creat_model_func):
    run_id = str(uuid.uuid4())
    out_folder = os.path.join(config["main_out_folder"], run_id)
    os.makedirs(out_folder, exist_ok=True)

    shutil.copy2(os.path.realpath(__file__), os.path.join(out_folder, "run_script.txt"))

    if config["run_random"]:
        print(f"""{config["print_id"]}: This is a random run!!!!""")

    def epsilon_update(ep):
        return 0.3 #ramp(ep, 1.0, config["game_to_run"]//4*3, config["game_to_run"]//4)

    agent = DQ(creat_model_func, log_dir=out_folder, update_target_every=config["update_target_every"],
               initial_memory_size=config["initial_memory_size"],
               max_memory_size=config["max_memory_size"], n_old=config["n_old"], k=config["k"],
               epsilon_update=epsilon_update if config["use_epsilon_update"] else None,
               discount_factor=config["discount_factor"])

    rt = RTrack()

    epsilon_hist = []
    memory_hist = []

    g = Game()

    for run_i in range(config["game_to_run"]):
        if run_i % config["print_every"] == 0:
            print(f"""{config["print_id"]}: {run_i}/{config["game_to_run"]}""")

        g.reset()
        game_done_after = False

        run_reward = 0
        while not game_done_after:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, game_done), player_i = g.get_observation()

            if player_i == 0:
                if len(move_pieces):
                    state = cal_state(player_pieces, enemy_pieces, dice)
                    #state = tuple(state)

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
                        new_states = [cal_state(player_pieces, enemy_pieces, d) for d in range(1, 6+1)]
                        agent.add_to_pre_memory(state, action, new_states, reward, end_game)

        agent.update_memory(new_state_are_in_list=True)
        rt.add(run_reward)

        if config["use_replay_buffer"] and not config["run_random"]:
            agent.train_on_memory(128, 128, True, flat_sample=config["flat_sample"], new_state_are_in_list=True)

        if not config["run_random"]:
            epsilon_hist.append(agent.epsilon)
            memory_hist.append(len(agent.aer))

    # Saving info from run --------------------------------------------------------------------------------------------
    print(f"""{config["print_id"]}: Saving info from run""")
    agent.update_target_model()
    agent.target_model.save(os.path.join(out_folder, "dq.h5"))
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

config = {
    "start_epsilon": 1.0,
    "game_to_run": 2000,
    "discount_factor": 0.99,
    "initial_memory_size": 100,
    "max_memory_size": 10000,
    "n_old": 75,
    "k": 20,
    "use_epsilon_update": True,
    "use_replay_buffer": True,
    "flat_sample": False,
    "main_out_folder": "Runs_NN/First_test",
    "run_random": False,
    "close_plots": False,
    "print_id": 0,
    "print_every": 100,
    "update_target_every":10,
}


def make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, input_shape=(36,), activation="relu"))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(4, activation=None))

    model.compile(
        loss="mse",
        optimizer="Adadelta",
        metrics=["mae"]
    )
    return model


make_run(config, make_model)


pass
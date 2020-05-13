from ludopy import Game
from ludpyhelper.RL.QLearning.qtable import NQTable
from ludpyhelper.RL.helpers.reward_tracker import RTrack
from ludpyhelper.gamehelper import cal_state, cal_game_events, cal_reward_and_endgame, cal_move
from tqdm import tqdm
import random
from ludpyhelper.mics.functions import ramp
from ludpyhelper.mics.mics import save_json
from ludpyhelper.mics.sthreading import ThreadHandler
from ludpyhelper.hardcode_players.fifo import FIFOPlayer
from ludpyhelper.hardcode_players.semismart import SemiSmartPlayer
from ludpyhelper.hardcode_players.emil import GANNIndividual
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os
import shutil
import time

fifo_p = FIFOPlayer()
semis_p = SemiSmartPlayer()
emil_p = GANNIndividual()
emil_p.load_chromosome(np.load("gen311.npy"))

def use_emil(move_pieces, player_pieces, enemy_pieces, dice):
    next_piceces = []
    for pice in [0, 1, 2, 3]:
        if pice not in move_pieces:
            next_piceces.append(False)
        else:
            new_player_pieces, new_enemy_pieces = cal_move(player_pieces, enemy_pieces,
                                                           dice, pice)
            next_piceces.append([list(new_player_pieces)] + list(new_enemy_pieces))
    action = emil_p.play([list(player_pieces)] + list(enemy_pieces), dice, next_piceces)
    return action

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_run(config):
    run_id = str(uuid.uuid4())
    out_folder = os.path.join(config["main_out_folder"], run_id)
    os.makedirs(out_folder, exist_ok=True)
    tabs = " " * config["print_id"]
    shutil.copy2(os.path.realpath(__file__), os.path.join(out_folder, "run_script.txt"))
    shutil.copy2("ludpyhelper/gamehelper/mics.py", os.path.join(out_folder, "r_s.txt"))
    shutil.copy2("ludpyhelper/RL/QLearning/qtable.py", os.path.join(out_folder, "q.txt"))
    with open( os.path.join(out_folder, f"""{config["run_name"]}.txt"""), "w") as f:
        f.write(config["run_name"])

    if config["use_full_state"]:
        print(f"""{config["print_id"]}: This is a full state run!!!!""")
    if config["other_player_type"] != "":
        print(f"""{config["print_id"]}: This is a other player type!!!!""")

    player_is_the_winner_arr = []
    player_is_the_winner_ext = []
    def epsilon_update(ep):
        if not config["use_ramp"]:
            if config["game_to_run"]//4*3 < ep:
                return 0.0
            else:
                return config["start_epsilon"]
        else:
            return ramp(ep, 1.0, config["game_to_run"]//4*3, config["game_to_run"]//4)

    def run_cal_state(player_pieces, enemy_pieces, dice, used_piece_states, used_action_states):
        if config["use_full_state"]:
            dice_one_hot = [0]*6
            dice_one_hot[dice - 1] = 1
            state = list(player_pieces) + flatten(enemy_pieces) + dice_one_hot
        else:
            state = cal_state(player_pieces, enemy_pieces, dice, used_piece_states, used_action_states)
        return state

    agent = NQTable(action_space=4, n_q_tabels=config["NQ"], initial_memory_size=config["initial_memory_size"],
                    max_memory_size=config["max_memory_size"], n_old=config["n_old"], k=config["k"],
                    epsilon=config["start_epsilon"],
                    epsilon_update=epsilon_update if config["use_epsilon_update"] else None,
                    learning_rate=config["learning_rate"], discount_factor=config["discount_factor"])

    rt = RTrack(os.path.join(out_folder, "reward.csv"))

    epsilon_hist = []
    memory_hist = []

    g = Game()

    start_time = time.perf_counter()

    for run_i in range(config["game_to_run"]):
        if run_i % config["print_every"] == 0:
            print(f"""{config["run_name"]} - {config["print_id"]}:{tabs} {run_i}/{config["game_to_run"]}""")

        g.reset()
        end_game = False

        run_reward = 0
        while not end_game:
            (dice, move_pieces, player_pieces, enemy_pieces,
             player_is_a_winner, game_done), player_i = g.get_observation()

            if player_i == 0 or config["use_to_players"] and player_i == 1:
                if len(move_pieces):
                    if config["other_player_type"] == "":
                        state = run_cal_state(player_pieces, enemy_pieces, dice, config["used_piece_states"],
                                              config["used_action_states"])
                        state = tuple(state)
                        action = agent.act(state)

                        if action not in move_pieces:
                            action = random.choice(move_pieces)
                    else:
                        if config["other_player_type"] == "random":
                            action = random.choice(move_pieces)
                        if config["other_player_type"] == "FIFO":
                            action = fifo_p.act(move_pieces, player_pieces, enemy_pieces, dice)
                        if config["other_player_type"] == "SemiSmart":
                            action = semis_p.act(move_pieces, player_pieces, enemy_pieces, dice)
                        if config["other_player_type"] == "GA NN":
                            action = use_emil(move_pieces, player_pieces, enemy_pieces, dice)
                else:
                    action = -1
            elif (player_i == 2 or player_i == 3) and config["other_player_enemy"] != "":
                if len(move_pieces):
                    if config["other_player_enemy"] == "random":
                        action = random.choice(move_pieces)
                    if config["other_player_enemy"] == "FIFO":
                        action = fifo_p.act(move_pieces, player_pieces, enemy_pieces, dice)
                    if config["other_player_enemy"] == "SemiSmart":
                        action = semis_p.act(move_pieces, player_pieces, enemy_pieces, dice)
                    if config["other_player_enemy"] == "GA NN":
                        action = use_emil(move_pieces, player_pieces, enemy_pieces, dice)
                else:
                    action = -1
            else:
                if len(move_pieces):
                    action = random.choice(move_pieces)
                else:
                    action = -1

            (_, _, player_pieces_after, enemy_pieces_after,
                   player_is_a_winner_after, game_done_after) = g.answer_observation(action)

            if game_done_after:
                player_is_the_winner_ext.append(player_i)

            if player_i == 0 or config["use_to_players"] and player_i == 1:
                if len(move_pieces):
                    game_event = cal_game_events(player_pieces, enemy_pieces, player_pieces_after, enemy_pieces_after)
                    reward, end_game = cal_reward_and_endgame(game_event, config["reward_tabel"])
                    run_reward += reward

                    if end_game:
                        if player_is_a_winner_after:
                            player_is_the_winner_arr.append(1)
                        else:
                            player_is_the_winner_arr.append(0)

                    if config["other_player_type"] == "":
                        new_states = [tuple(run_cal_state(player_pieces, enemy_pieces, d, config["used_piece_states"],
                                                          config["used_action_states"])) for d in range(1, 6+1)]

                        agent.update(state, action, new_states, reward, terminal_state=end_game, save_experience=True,
                                     new_state_list=True, auto_update_episode=False)

        rt.add(run_reward)

        if config["use_replay_buffer"] and config["other_player_type"] == "":
            agent.train_on_memory(128, flat_sample=config["flat_sample"])

        if config["other_player_type"] == "":
            agent.update_episode()
            epsilon_hist.append(agent.epsilon)
            memory_hist.append(len(agent.aer))

    end_time = time.perf_counter()
    used_time = end_time - start_time
    config["used_time"] = used_time

    # Saving info from run --------------------------------------------------------------------------------------------

    print(f"""{config["print_id"]}:{tabs} Saving info from run""")
    agent.save_table(os.path.join(out_folder, "ludo_nq.pickel"))
    rt.save(os.path.join(out_folder, "rewards.npy"))
    np.save(os.path.join(out_folder, "epsilon_hist.npy"), epsilon_hist)
    np.save(os.path.join(out_folder, "memory_hist.npy"), memory_hist)
    np.save(os.path.join(out_folder, "player_winner_hist.npy"), player_is_the_winner_arr)
    np.save(os.path.join(out_folder, "player_winner_ext.npy"), player_is_the_winner_ext)

    rt.plot(save=out_folder, close_plots=config["close_plots"], pre_fix=config["pre_fix"], alpha=0)

    plt.figure()
    plt.title(config["pre_fix"]+"Epsilon")
    plt.plot(epsilon_hist)
    plt.savefig(os.path.join(out_folder, "epsilon.svg"))
    if config["close_plots"]:
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title(config["pre_fix"]+"Memory hist")
    plt.plot(memory_hist)
    plt.savefig(os.path.join(out_folder, "memory.svg"))
    if config["close_plots"]:
        plt.close()
    else:
        plt.show()

    save_json(os.path.join(out_folder, "config.json"), config)
    print(f"""{config["print_id"]}:{tabs} Done""")


reward_tabel = {
    "enemy_hit_home": [True, 1],
    "player_hit_home": [True, -1],
    "player_left_home": [True, 1],
    "player_piece_got_to_goal": [True, 4],
    "player_is_a_winner": [True, 10],
    "other_player_is_a_winner": [True, -10],
}

used_piece_states = ['home_areal', "safe_round_1"]
# Actions States
used_action_states = [
    'moved_out_of_home',
    {'new_pos_events': [
        'goal',
        'glob',
        'star',
        'home_areal',
    ]},
    'enemy_hit_home',
    "got_hit_home",
]

base_config = {
    "reward_tabel": reward_tabel,
    "used_piece_states":used_piece_states,
    "used_action_states":used_action_states,
    "start_epsilon": 1.0,
    "use_ramp": True,
    "game_to_run": 2000,
    "learning_rate": 0.05,
    "discount_factor": 0.8,
    "NQ": 2,
    "initial_memory_size": 100,
    "max_memory_size": 10000,
    "n_old": 75,
    "k": 20,
    "use_epsilon_update": True,
    "use_replay_buffer": True,
    "flat_sample": False,
    "main_out_folder": "Runs/VS",
    "run_random": False,
    "close_plots": True,
    "print_id": 0,
    "print_every": 100,
    "pre_fix":"",
    "use_full_state": False,
    "other_player_type": "",
    "run_name": "",
    "use_to_players": False,
    "other_player_enemy":"",
}
configs = []
id_count = 0
n_runs = 5

to_test = ["FIFO", "SemiSmart", "random", "GA NN"]
t_ele = to_test[1]

run_name = f"GA NN vs Rand"
for i in range(n_runs):
    run_config = base_config.copy()
    run_config["run_name"] = run_name
    run_config["main_out_folder"] = os.path.join(run_config["main_out_folder"], run_config["run_name"])
    run_config["print_id"] = id_count
    id_count += 1
    run_config["pre_fix"] = run_name + f"""({run_config["print_id"]}) : """

    # TODO Husk at være sikker på alt er sat
    run_config["other_player_type"] = "GA NN"
    run_config["use_to_players"] = True
    run_config["other_player_enemy"] = "random"

    configs.append(run_config)

th = ThreadHandler()
th.make_threads(len(configs), kill_when_done=True)

for r_config in configs:
    th.put_job(make_run, {"config":r_config})

th.start_threads()
th.join()

pass
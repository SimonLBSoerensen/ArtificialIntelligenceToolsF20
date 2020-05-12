"""
Made with inspreation from David Johan Christensen Java code
"""
from ludpyhelper.gamehelper.mics import get_action_states, get_pos_state
import numpy as np

class SemiSmartPlayer:
    def act(self, move_pieces, player_pieces, enemy_pieces, dice):
        scores = self.analyze(move_pieces, player_pieces, enemy_pieces, dice)
        idx = np.argmax(scores)
        return move_pieces[idx]

    def analyze(self, move_pieces, player_pieces, enemy_pieces, dice):
        scores = []
        for piece_i in move_pieces:
            pos = player_pieces[piece_i]
            action_states = get_action_states(piece_i, dice, player_pieces, enemy_pieces)
            #pos_states = get_pos_state(pos, player_pieces, enemy_pieces)

            moved_out_of_home = action_states["moved_out_of_home"]
            enemy_hit_home = action_states["enemy_hit_home"]
            got_hit_home = action_states["got_hit_home"]
            moved_to_goal = action_states["new_pos_events"]["goal"]
            new_pos_star = action_states["new_pos_events"]["star"]

            score = 0
            if enemy_hit_home:
                score = 5+np.random.rand()
            if got_hit_home:
                score = 0.1
            if new_pos_star:
                score = 2+np.random.rand()
            if moved_out_of_home:
                score = 3+np.random.rand()
            if moved_to_goal:
                score = 2+np.random.rand()
            else:
                score = 1+np.random.rand()
            scores.append(score)
        return scores
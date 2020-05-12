import ludopy


class FIFOPlayer:
    def __init__(self):
        self.in_home = [0, 1, 2, 3]
        self.current_piece = 0

    def act(self, move_pieces, player_pieces, enemy_pieces, dice):
        if dice == 6 and len(self.in_home):
            piece_to_move = self.in_home.pop(0)
        else:
            if player_pieces[self.current_piece] == ludopy.player.GOAL_INDEX:
                self.current_piece += 1
                if self.current_piece > 3:
                    self.current_piece = 0  # Skulle ikke ske men nu vil den bare loope
            piece_to_move = self.current_piece
        return piece_to_move
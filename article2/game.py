import numba as nb
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

spec = [
        ('score', nb.int32),
        ('moves', nb.int32),
        ('n_actions', nb.uint8),
        ('possible_actions', nb.uint8[:]),
        ('arr', nb.uint8[:]),
        ('actions', nb.uint8[:]),
        ('preparing', nb.b1),
        ('action', nb.uint8),
        ('board', nb.int32[:, :]),
        ('lost', nb.b1),
        ('valid', nb.b1),
        ('changed', nb.b1),
        ('new_board', nb.int32[:, :]),
        ('rotated_board', nb.int32[:, :]),
        ('after', nb.int32[:, :]),
        ('new_col', nb.int32[:]),
        ('rot_board', nb.int32[:, :]),
        ('result', nb.int32),
        ('col', nb.int32[:]),
        ('j', nb.uint8),
        ('i', nb.uint8),
        ('val', nb.int32),
        ('options', nb.b1),
        ('done', nb.b1),
        ('k', nb.uint8),
        ('previous', nb.int32),
        ('ep_return', nb.int64),
        ('potential', nb.int64[:]),
        ('reward', nb.int64),
        ('is_game_over', nb.b1),
        ]
        
@nb.experimental.jitclass(spec)
class NumbaBoard:

    def __init__(self, n_actions=4):
        self.score = 0
        self.moves = 0
        self.n_actions = n_actions
        self.actions = np.arange(n_actions).astype(np.uint8)
        self.possible_actions = self.actions.copy()
        # self.num_direction = {'L' : 0, 'U' : 1, 'R' : 2, 'D' : 3}

        preparing = True
        while preparing:
            self.board = np.zeros((4, 4), dtype=np.int32)
            for _ in range(2):
                self.fill_cell()
            if self.board.sum() == 4:
                preparing = False
        
    def fill_cell(self):
        i, j = (self.board == 0).nonzero()
        if i.size != 0:
            high = i.size - 1 if i.size > 1 else 1
            rnd = np.random.randint(0, high)
            self.board[i[rnd], j[rnd]] = 2 * ((np.random.random() > .9) + 1)
            
    @staticmethod
    def move_left(col):
        new_col = np.zeros(4, dtype=np.int32)
        j = 0
        previous = -1
        result = 0
        for i in range(4):
            val = col[i]
            if val != 0: # number different from zero
                if previous == -1:
                    previous = val
                else:
                    if previous == val:
                        new_col[j] = 2 * val
                        result += new_col[j]
                        j += 1
                        previous = -1
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = val
        if previous != -1:
            new_col[j] = previous
        return new_col, result
        
    def move(self, k):
        rotated_board = np.rot90(self.board, k)
        score = 0
        new_board = np.zeros((4,4), dtype=np.int32)
        for i,col in enumerate(rotated_board):
            new_col, result = self.move_left(col)
            score += result
            new_board[i] = new_col
        rot_board = np.rot90(new_board, -k)
        return rot_board, score

    def check_options(self):
        """
        Check if playable moves remain

        Returns
        -------
        options : Boolean
            Whether playable moves remain.

        """
        options = False
        before = self.board.copy()
        for k in range(self.n_actions):
            after, score = self.move(k)
            
            # if before and after are different,
            # return True, meaning there are playable actions
            if not np.array_equal(before, after):
                options = True
                break
        return options

    def evaluate_action(self, k):
        """
        Evaluate how the move k affects the state of the board

        Parameters
        ----------
        k : INTEGER
            index of the move.

        Returns
        -------
        score : INTEGER
            Increase in player score.
        lost : BOOLEAN
            Whether the player has lost the game with this move.

        """
        
        lost = False
        valid = False
        after, score = self.move(k)
        if not np.array_equal(self.board, after):
            valid = True
            self.board = after
            self.score += score
            self.fill_cell()
            self.moves += 1
            
            # check if options available
            options = self.check_options()
            if not options:
                lost = True
        return score, lost, valid
                
    def evaluate_next_actions(self):
        """
        Evaluate whether there are playable moves left

        Returns
        -------
        potential : LIST of integers
            Possible scores resulting from the playable moves.

        """
        potential = []
        arr = []
        for a in self.actions:
            rot_board, score = self.move(a)
            changed = not np.array_equal(rot_board, self.board)
            if changed:
                arr.append(a)
            potential.append(score)
        self.possible_actions = np.array(arr).astype(np.uint8)
        self.is_game_over = len(arr) > 0
        return potential
                
    # reset and step functions required by OpenAI Gym
    def reset(self):
        self.__init__()
        self.ep_return  = 0
        return self.board
    
    def step(self, action):
        # call the action and get the score and outcome of that action
        score, done, valid = self.evaluate_action(action)
        
        if valid:
            # look one step ahead at each possible option
            reward_list = self.evaluate_next_actions()
            reward = max(reward_list)
        else:
            reward = 0
        
        # Increment the episodic return
        self.ep_return += 1
        return self.board, reward, valid, done
    
# All method calls are working as expected
def test_nb_board():
    board = NumbaBoard()
    board.evaluate_action(0)
    board.check_options()
    board.evaluate_next_actions()
    board.reset()
    board.step(0)


class Board:

    def __init__(self, show=True):
        self.score = 0
        self.moves = 0
        self.lost = False
        self.show = show
        self.num_direction = {'L' : 0, 'U' : 1, 'R' : 2, 'D' : 3}

        preparing = True
        while preparing:
            self.board = np.zeros((4, 4), dtype=np.int32)
            for _ in range(2):
                self.fill_cell()
            if self.board.sum() == 4:
                preparing = False
        if show:
            self.show_board()
        
    def show_board(self):
        # clear axes
        plt.close('all')
        plt.imshow(self.board, cmap='plasma')

        # adapted from: https://stackoverflow.com/questions/33828780/matplotlib-display-array-values-with-imshow
        positions = np.arange(0, 4)
        for y_index, y in enumerate(positions):
            for x_index, x in enumerate(positions):
                label = self.board[y_index, x_index]
                if label != 0:
                    plt.text(x, y, label, color='black', ha='center', va='center')

        # turn off axis ticks and labels
        plt.axis('off')
        plt.show()

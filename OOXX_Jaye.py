import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Player:
    # name为player的代号，learning_rate为学习率, epsilon为做出随机选择的概率
    def __init__(self, name, learning_rate, epsilon, value_estimate=None, accu_win=0):
        if value_estimate is None:
            value_estimate = {}
            # for i in itertools.product([0, 1, 2], repeat=9):
            #     value_estimate.update({i: 0})  # 3^9种棋盘情况，每一种情况作为Dict里的一个Key值，0作为Value，即奖励，初始化这一Dict
        self.name = name
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.value_estimate = value_estimate  # 这是落子后对这一棋盘的价值估计,是估计从这一棋盘状态开始到最终能够得到的奖励之和
        self.accu_win = accu_win

    def get_value(self, board):  # 获取当前棋盘的预估价值
        if tuple(board) not in self.value_estimate.keys():
            return 0
        else:
            return self.value_estimate[tuple(board)]

    def get_next_max_value_and_pos(self, board):  # 获取最大价值及其对应的pos，如果有多个则随机选择一个
        max_value = float('-inf')
        max_pos_list = []
        for i in range(len(board)):
            if board[i] == 0:
                temp = board[:]
                temp[i] = self.name
                v = self.get_value(temp)
                if v > max_value:
                    max_value = v
                    max_pos_list.clear()
                    max_pos_list.append(i)
                elif v == max_value:
                    max_pos_list.append(i)

        return max_value, max_pos_list[np.random.randint(0, len(max_pos_list))]

    def get_next_total_value(self, board):  # 获取此时情况即将落子的所有可能性的价值总和
        total_value = 0
        for i in range(len(board)):
            if board[i] == 0:
                temp = board[:]
                temp[i] = self.name
                v = self.get_value(temp)
                total_value += v
        return total_value

    def update_value_learning(self, board, reward):     # 通过学习方法更新价值
        old_estimate = self.get_value(board)
        new_estimate = \
            old_estimate + self.learning_rate * (reward + self.get_next_total_value(board) - old_estimate)
        self.update_value(board, new_estimate)

    def update_value(self, board, value):   # 修改对应board价值为value
        self.value_estimate.update({tuple(board): value})


def choose_pos_randomly(board):  # 随机返回可落子的一个空地
    temp = []
    for i in range(len(board)):
        if board[i] == 0:
            temp.append(i)
    pos = temp[np.random.randint(0, len(temp))]
    return pos


def print_board(board):  # 打印棋盘
    for i in range(len(board)):
        if (i + 1) % 3 == 0:
            print(f'\033[33m {board[i]}\033[33m')
        else:
            print(f'\033[33m {board[i]}\033[33m', end="")
    print('--------')


def is_win(board):  # 判断有没有人赢
    win_case = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for (a, b, c) in win_case:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return True
    return False


def random_pick(some_list, probabilities):  # 根据设定概率从some_list中选出一个item
    p = random.uniform(0, 1)
    cumulative_probability = 0.0
    item = some_list[0]
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if p < cumulative_probability:
            break
    return item


def value_feed_back(path, winner, loser, last_board):  # 通过回溯来设置各“后果”的预估价值
    temp = last_board[:]
    while len(path) > 0:
        res = path.pop()
        temp[res[0]] = 0
        if res[1] is winner:
            loser.update_value_learning(temp, reward_lose)
        else:
            winner.update_value_learning(temp, reward_win)


# 调参数的地方除了这里，还有 update_value_learning 方法内更新 value_estimate 的数学公式
# Player(name, learning-rate, epsilon)
player1 = Player(1, 0.1, 0.1)
player2 = Player(2, 0.1, 0.1)
reward_win = 2
reward_lose = -2
reward_draw = 0
x = []
y1 = []
y2 = []
y3 = []


def train():  # 训练过程
    draw_count = 0
    print('训练中')
    epochs = 100000
    for episode in tqdm(range(1, epochs + 1)):
        # if episode > 2000:
        #     player1.epsilon = 0
        #     player2.epsilon = 0
        if episode % 10 == 0:
            x.append(episode / 10)
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 初始化棋盘
        players = (player1, player2)  # player1先手
        is_game_end = False
        path = []  # 下棋落子顺序记录，这样可以进行复盘而求出价值估计
        while 0 in board:  # 循环直到棋盘下满
            if is_game_end:  # 本局游戏结束则退出循环
                break
            for now_player in players:
                if is_game_end:
                    break
                pick_res = random_pick([0, 1], [now_player.epsilon, 1 - now_player.epsilon])
                if pick_res == 0:
                    pos = choose_pos_randomly(board)  # 随机选择一个棋盘上的空位
                else:
                    max_value_and_pos = now_player.get_next_max_value_and_pos(board)  # 选择预计价值最高的那个空位，若有多个则随机选取一个
                    pos = max_value_and_pos[1]
                board[pos] = now_player.name    # 当前玩家落子
                path.append((pos, now_player))  # 记录落子路径
                if is_win(board):  # 当前玩家胜利
                    now_player.update_value_learning(board, reward_win)  # 赢了获得奖励
                    now_player.accu_win += 1    # 赢的局数统计
                    if now_player is player1:
                        loser = player2
                    else:
                        loser = player1
                    loser.update_value_learning(board, reward_lose)  # 输了获得奖励
                    is_game_end = True
                    value_feed_back(path, now_player, loser, board)  # 此时进行进度回溯去设置所有Player的value_estimate
                    continue
                if 0 not in board:  # 棋盘满了
                    now_player.update_value_learning(board, reward_draw)  # 平局获得奖励
                    draw_count += 1  # 平局局数统计
                    is_game_end = True
                    continue
                #  游戏还在继续，则更新另外一个player的value，因为下一回合是另外一个player进行落子
                if now_player is player1:
                    updating_player = player2
                else:
                    updating_player = player1
                updating_player.update_value_learning(board, 0)     # 游戏没有结束获得奖励0
        # 下完一局之后统计胜率
        if episode % 10 == 0:
            y1.append(player1.accu_win / episode)
            y2.append(player2.accu_win / episode)
            y3.append(draw_count / episode)


def computer_down(board, player):   # 电脑落子，返回游戏是否结束
    max_value_and_pos = player.get_next_max_value_and_pos(board)  # 电脑已经选择了最大价值的地方下棋
    pos = max_value_and_pos[1]
    board[pos] = player.name  # 将棋盘的该位置标记为电脑已下棋
    print_board(board)  # 打印棋盘
    if is_win(board):  # 如果电脑胜利
        print('\033[31m电脑胜利了，游戏结束，最终棋盘：\033[31m')
        return True
    if 0 not in board:
        print('\033[31m平局，游戏结束，最终棋盘：\033[31m')
        return True
    return False


def human_down(board, turn):   # 玩家落子，返回游戏是否结束
    while True:
        print(f'\033[34m你的棋子为 {turn} \033[34m')
        k = int(input("\033[36m输入你想要下棋的位置,范围是0到8:\033[36m"))  # k为玩家下棋的位置下标
        if board[k] != 0 or k not in range(0, 9):
            print('\033[31m输入非法，请重新选择位置!\033[31m')
            continue
        board[k] = turn  # 玩家落子
        print_board(board)  # 打印棋盘
        if is_win(board):  # 如果玩家胜利
            print('\033[31m你赢了！游戏结束，最终棋盘：\033[31m')
            return True
        if 0 not in board:
            print('\033[31m平局，游戏结束，最终棋盘：\033[31m')
            return True
        break
    return False


def begin_play(turn):   # 开始游戏
    if turn == 1:   # 玩家先手
        player = player2
    elif turn == 2:     # 电脑先手
        player = player1
    else:
        print('\033[31m Error! \033[31m')
        return
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    while 0 in board:
        if turn == 1:   # 玩家先手
            if human_down(board, turn):
                break
            if computer_down(board, player):
                break
        else:   # 电脑先手
            if computer_down(board, player):
                break
            if human_down(board, turn):
                break
    print_board(board)  # 打印最后的棋盘


def draw_pic():
    plt.plot(x, y1, label='player1')
    plt.plot(x, y2, label='player2')
    plt.plot(x, y3, label='draw')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
    draw_pic()

    # 开始n局游戏
    while True:
        user_input = input("\033[32m游戏开始，输入1你先手，输入2电脑先手，输入其它数字终止游戏:\033[31m")
        if user_input.isnumeric():
            game = int(user_input)
            if game != 1 and game != 2:
                break
            begin_play(game)
        else:
            print('\033[31m输入非法！请重新输入！\033[31m')

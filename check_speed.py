import random, time
from mini_game import State
from SpuriousSearch import spurious_search_action
from cy_mini_game import State as CyState
from cy_spurious_search import spurious_search_action as cy_spurious_search_action


# paramを使った評価テーブル（ミニ）の作成
def create_ev_table(params):
    ev_table = [0] * 20
    for row in range(5):
        i = row * 4
        h = row * 2
        ev_table[i] = params[h]
        ev_table[i + 1] = params[h + 1]
        ev_table[i + 2] = params[h + 1]
        ev_table[i + 3] = params[h]
    return ev_table


# 評価関数を生成する関数
def create_ev_func(params):
    # テーブルの次数はここで調整
    blue_ev_table = create_ev_table(params[0])  # 青駒を評価
    red_ev_table = create_ev_table(params[1])  # 赤駒を評価

    def ev_func(ii_state):
        value = 0
        # 自分の駒をテーブルに従って評価
        for index, piece in enumerate(ii_state.pieces):
            if piece == 1:
                value += blue_ev_table[index]
            elif piece == 2:
                value += red_ev_table[index]
        # 敵のコマがどれだけ自分のゴールに近いか（ここ諸説あるが、全部青駒として考える）
        for index, piece in enumerate(ii_state.pieces):
            if piece == -1:
                value -= blue_ev_table[19 - index]
        if ii_state.my_turn:
            return value
        else:
            return -value

    return ev_func


# 動作確認
if __name__ == "__main__":
    # # 状態の生成
    # random.seed(314)
    # state = State()
    # random.seed(314)
    # cy_state = CyState()

    # params = [
    #     [9, 5, 5, 3, 3, 2, 2, 1, 1, 0],
    #     [3, 5, 5, 9, 3, 5, 2, 3, 1, 2],
    # ]
    # ev_func = create_ev_func(params)
    # # print(alpha_beta_action(state, ev_func, 2))
    # est_val_and_coo = [
    #     [[0.5, 0.5, 0.5, 0.5], [13, 14, 17, 18]],
    #     [[0.5, 0.5, 0.5, 0.5], [1, 2, 5, 6]],
    # ]
    # time_sta, time_end = 0.0, 0.0
    # random.seed(314)
    # time_sta = time.perf_counter()
    # for i in range(100):
    #     spurious_search_action(state, ev_func, est_val_and_coo, 7)
    # time_end = time.perf_counter()
    # print("純Python:", time_end - time_sta)

    # random.seed(314)
    # time_sta = time.perf_counter()
    # for i in range(100):
    #     cy_spurious_search_action(cy_state, ev_func, est_val_and_coo, 7)
    # time_end = time.perf_counter()
    # print("Cython:", time_end - time_sta)

    from spurious_mcts import *

    state = State()
    est_val_and_coo = [
        [[1.0, 0.2, 0.3, 0.5], [13, 14, 17, 18]],
        [[0.5, 0.7, 0.3, 0.5], [1, 2, 5, 6]],
    ]
    time_sta = time.perf_counter()
    spurious_mcts_action(state, est_val_and_coo, 300)
    time_end = time.perf_counter()
    print("sp_mcts:", time_end - time_sta)

    test_enemy_params = [
        [9, 5, 5, 3, 3, 2, 2, 1, 1, 0],
        [3, 5, 5, 9, 3, 5, 2, 3, 1, 2],
        [0.2, 0.4, 0.05, 0.4, -0.1],
    ]
    enemy_ev_func = create_ev_func(test_enemy_params)
    time_sta = time.perf_counter()
    action = spurious_search_action(state, enemy_ev_func, est_val_and_coo, 7)
    time_end = time.perf_counter()
    print("sp_ab:", time_end - time_sta)

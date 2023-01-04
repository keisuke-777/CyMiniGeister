# SpuriousMCTS、永久、進化計算
import random, datetime, copy, csv, os, sys, itertools, pickle, ast
from mini_game import State
from SpuriousSearch import (
    calc_coo_of_destination_from_action,
    create_ev_func,
    random_action,
)
from rl_mcts import *
from typing import Callable
import pandas as pd

# from cy_spurious_mcts import *

# 他のマスとの距離を示すmap10*10
DISTANCE_MAP = [
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],  # 0
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],  # 1
    [1, 2, 0, 1, 1, 2, 2, 3, 3, 4],  # 2
    [2, 1, 1, 0, 2, 1, 3, 2, 4, 3],  # 3
    [2, 3, 1, 2, 0, 1, 1, 2, 2, 3],  # 4
    [3, 2, 2, 1, 1, 0, 2, 1, 3, 2],  # 5
    [3, 4, 2, 3, 1, 2, 0, 1, 1, 2],  # 6
    [4, 3, 3, 2, 2, 1, 1, 0, 2, 1],  # 7
    [4, 5, 3, 4, 2, 3, 1, 2, 0, 1],  # 8
    [5, 4, 4, 3, 3, 2, 2, 1, 1, 0],  # 9
]

# 距離が少ない順に格納されたindex
DISTANCE_ORDER_INDEX_MAP = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8],
    [2, 0, 3, 4, 1, 5, 6, 7, 8, 9],
    [3, 1, 2, 5, 0, 4, 7, 6, 9, 8],
    [4, 2, 5, 6, 0, 3, 7, 8, 1, 9],
    [5, 3, 4, 7, 1, 2, 6, 9, 0, 8],
    [6, 4, 7, 8, 2, 5, 9, 0, 3, 1],
    [7, 5, 6, 9, 3, 4, 8, 1, 2, 0],
    [8, 6, 9, 4, 7, 2, 5, 0, 3, 1],
    [9, 7, 8, 5, 6, 3, 4, 1, 2, 0],
]

EST_EVAL_RANGE = 0.5
MIN = 1
MAX = 9
WIDTH = 3


def dead_color_check(action, state):
    aft_coo = calc_coo_of_destination_from_action(action)
    if aft_coo < 0 or 19 < aft_coo:
        return 0
    aft_coo = 19 - aft_coo  # aft_cooはmy視点の座標なのでenemy視点に修正
    # 青->1, 赤->2, 死駒無し->0
    return state.enemy_pieces[aft_coo]


# 行動番号の視点を入れ替える
def switch_action_view(action):
    coo = int(action / 4)
    dir = action % 4
    new_coo = 19 - coo
    if dir == 0:
        new_dir = 2
    elif dir == 1:
        new_dir = 3
    elif dir == 2:
        new_dir = 0
    else:  # dir == 3
        new_dir = 1
    return new_coo * 4 + new_dir


# 周囲の敵駒の数をカウント
def culc_around_enemy_num(piece_coo, est_val_and_coo, is_my_piece):
    around_enemy_num = 0
    around_list = get_around_list(piece_coo)  # piece_cooの周囲のマスを洗い出す
    if is_my_piece:  # piece_cooが自駒の座標を指している
        for enemy_coo in est_val_and_coo[1][1]:
            if enemy_coo in around_list:  # 周囲に敵の駒が存在
                around_enemy_num += 1
    else:  # piece_cooが敵駒の座標を指している
        for enemy_coo in est_val_and_coo[0][1]:
            if enemy_coo in around_list:  # 周囲に自分の駒が存在
                around_enemy_num += 1
    return around_enemy_num


# 離れている->-1、近づいている->1、どちらでもない->0
def judge_approaching_goal(action, is_my_action):
    if is_my_action:
        symbol = 1
    else:
        symbol = -1
    direction = action % 4
    bef_coo = int(action / 4)
    bef_x = bef_coo % 4
    if direction == 2:  # 上に進んでいる場合、敵のゴールに近づいている
        return 1 * symbol
    elif direction == 0:  # 下に進んでいる場合、敵のゴールから遠ざかっている
        return -1 * symbol
    elif direction == 1:  # 左に進んでいる場合は場合わけ
        if bef_x == 1:  # 0は左に進めないので1から
            return 1 * symbol
        elif bef_x == 2:
            return 0
        else:
            return -1 * symbol
    else:  # direction == 3
        if bef_x == 2:
            return 1 * symbol
        elif bef_x == 1:
            return 0
        else:
            return -1 * symbol


# 敵駒に隣接している自駒のリストを返す
def check_adjacent_pieces(my_pieces, enemy_pieces):
    adjacent_piece_list = []
    for my_p in my_pieces:
        around_my_p = get_around_list(my_p)
        if len(set(around_my_p) & set(enemy_pieces)) > 0:
            adjacent_piece_list.append(my_p)
    return adjacent_piece_list


# 与えられた座標の周囲のマスをリストにして返す
def get_around_list(piece_coo):
    around_list = []
    x = piece_coo % 4
    y = int(piece_coo / 4)
    if y != 4:  # 下
        around_list.append(piece_coo + 4)
    if x != 0:  # 左
        around_list.append(piece_coo - 1)
    if y != 0:  # 上
        around_list.append(piece_coo - 4)
    if x != 3:  # 右
        around_list.append(piece_coo + 1)
    return around_list


class Estimate:
    def __init__(self, fst_est_params: list, sec_est_params: list) -> None:
        self.fst_est_params = fst_est_params
        self.sec_est_params = sec_est_params
        self.fst_est_val_and_coo = [
            [[0.5, 0.5, 0.5, 0.5], [13, 14, 17, 18]],
            [[0.5, 0.5, 0.5, 0.5], [1, 2, 5, 6]],
        ]
        self.sec_est_val_and_coo = [
            [[0.5, 0.5, 0.5, 0.5], [13, 14, 17, 18]],
            [[0.5, 0.5, 0.5, 0.5], [1, 2, 5, 6]],
        ]

    # actionからmyとenemyをどっちも更新
    def double_update_from_action(self, action, state, is_my_action):
        dead_piece_color = dead_color_check(action, state)
        self.fst_est_val_and_coo = self.update_est_val_and_coo(
            self.fst_est_params,
            self.fst_est_val_and_coo,
            action,
            dead_piece_color,
            is_my_action,
        )
        self.sec_est_val_and_coo = self.update_est_val_and_coo(
            self.sec_est_params,
            self.sec_est_val_and_coo,
            action,
            dead_piece_color,
            not is_my_action,
        )

    # selfを使えば引数を少なくできるけど流用しやすいようにこうしています...
    def update_est_val_and_coo(
        self, est_params, est_val_and_coo, action, dead_piece_color, is_my_action,
    ):
        if is_my_action:
            my = 0
            enemy = 1
        else:
            action = switch_action_view(action)
            my = 1
            enemy = 0

        cp_est_val_and_coo = copy.deepcopy(est_val_and_coo)  # est_val_and_cooのコピーを生成
        bef_coo = int(action / 4)
        aft_coo = calc_coo_of_destination_from_action(action)

        bef_around_enemy_num = culc_around_enemy_num(
            bef_coo, cp_est_val_and_coo, is_my_action
        )
        aft_around_enemy_num = culc_around_enemy_num(
            aft_coo, cp_est_val_and_coo, is_my_action
        )
        update_est_index = cp_est_val_and_coo[my][1].index(bef_coo)  # ここでは座標は未更新
        edit_val = cp_est_val_and_coo[my][0][update_est_index]  # 変更の対象となる値
        if edit_val != 0 and edit_val != 1:  # est_valが1か0なら編集しない（値が固定される）
            if is_my_action:  # 自分の駒に対する推測
                # actionで隣接した(0)
                if bef_around_enemy_num == 0 and aft_around_enemy_num > 0:
                    cp_est_val_and_coo[my][0][update_est_index] += est_params[0]
                # actionで逃げた(1) (敵駒をとって隣接を減らした場合も該当)
                if aft_around_enemy_num == 0 and bef_around_enemy_num > 0:
                    cp_est_val_and_coo[my][0][update_est_index] += est_params[1]
                # actionで脱出口に近づいた(2)
                is_close = judge_approaching_goal(action, is_my_action)
                if is_close == 1:
                    cp_est_val_and_coo[my][0][update_est_index] += est_params[2]
            else:  # 対戦相手の駒に対する推測
                # actionで隣接した(3)
                if bef_around_enemy_num == 0 and aft_around_enemy_num > 0:
                    cp_est_val_and_coo[my][0][update_est_index] += est_params[3]
                # actionで逃げた(4) (敵駒をとって隣接を減らした場合も該当)
                if aft_around_enemy_num == 0 and bef_around_enemy_num > 0:
                    cp_est_val_and_coo[my][0][update_est_index] += est_params[4]
                # actionで脱出口に近づいた(5)
                is_close = judge_approaching_goal(action, is_my_action)
                if is_close == 1:
                    cp_est_val_and_coo[my][0][update_est_index] += est_params[5]

        # 敵駒と隣接している駒のリスト
        adjacent_pieces_list = check_adjacent_pieces(
            cp_est_val_and_coo[my][1], cp_est_val_and_coo[enemy][1]
        )
        # 行動した駒については既に上で処理しているのでbef_cooを消す
        if bef_coo in adjacent_pieces_list:
            adjacent_pieces_list.remove(bef_coo)

        # ゴールできるのにしなかった（評価関数に関係なく確定で赤駒）
        # if 0 in cp_est_val_and_coo[my][1] and action != 2:
        #     goal_i = cp_est_val_and_coo[my][1].index(0)
        #     cp_est_val_and_coo[my][0][goal_i] = 0
        # if 3 in cp_est_val_and_coo[my][1] and action != 14:
        #     goal_i = cp_est_val_and_coo[my][1].index(3)
        #     cp_est_val_and_coo[my][0][goal_i] = 0

        # 座標の更新
        cp_est_val_and_coo[my][1][update_est_index] = aft_coo
        if dead_piece_color != 0:
            dead_index = cp_est_val_and_coo[enemy][1].index(aft_coo)
            cp_est_val_and_coo[enemy][1][dead_index] = -1
            if dead_piece_color == 1:
                cp_est_val_and_coo[enemy][0][dead_index] = 0
            elif dead_piece_color == 2:
                cp_est_val_and_coo[enemy][0][dead_index] = 1
            else:
                print("不明なエラー")

        # 最後に正規化処理
        # まず固定値の数を数える
        zero_quantity, one_quantity = 0, 0
        zero_index, one_index = [], []
        for index, est_val in enumerate(cp_est_val_and_coo[my][0]):
            if est_val == 0:
                zero_index.append(index)
                zero_quantity += 1
            elif est_val == 1:
                one_index.append(index)
                one_quantity += 1
        # 死駒は何があっても固定（色を変化させない）
        dead_index = []
        for index, est_coo in enumerate(cp_est_val_and_coo[my][1]):
            if est_coo == -1:
                dead_index.append(index)
        # 駒色の個数の帳尻を合わせる（3を超えている場合おかしい）
        if zero_quantity >= 3 or one_quantity >= 3:
            if zero_quantity >= 3:
                nondead_zero_index = list(set(zero_index) - set(dead_index))
                if len(nondead_zero_index) >= zero_quantity - 2:  # バグで4つ死駒がくる時がある
                    # バグで4つ死駒がくる時があるのでそれを避ける（1,1,1,0かつ全て死駒でエラー）
                    be_one = random.sample(nondead_zero_index, zero_quantity - 2)
                    for index in be_one:
                        cp_est_val_and_coo[my][0][index] = 1
            if one_quantity >= 3:
                nondead_one_index = list(set(one_index) - set(dead_index))
                if len(nondead_one_index) >= one_quantity - 2:  # バグで4つ死駒がくる時がある
                    be_zero = random.sample(nondead_one_index, one_quantity - 2)
                    for index in be_zero:
                        cp_est_val_and_coo[my][0][index] = 0

        # 駒色が確定してる（推測が外れていた場合に限り、死駒が出た際に変化する可能性がある）
        if zero_quantity == 2 or one_quantity == 2:  # 3以上のパターンは上で弾いている
            all_index = {0, 1, 2, 3}
            if zero_quantity == 2:
                be_one = all_index - set(zero_index)
                for index in be_one:
                    cp_est_val_and_coo[my][0][index] = 1
            if one_quantity == 2:
                be_zero = all_index - set(one_index)
                for index in be_zero:
                    cp_est_val_and_coo[my][0][index] = 0

        # はみ出ている値を0か1に修正（0や1にしてしまうと固定されるため0.001だけズラす）
        for index, est_val in enumerate(cp_est_val_and_coo[my][0]):
            if est_val < 0:
                cp_est_val_and_coo[my][0][index] = 0.001
            if est_val > 1:
                cp_est_val_and_coo[my][0][index] = 0.999

        # 固定されていないest_valを正規化する
        not_fix_est_sum, total_est, not_fix_index_list = 0, 2, []
        for index, est_val in enumerate(cp_est_val_and_coo[my][0]):
            if est_val >= 0.999:  # nonfixな青駒がいる場合はestの合計を1下げる
                total_est -= 1
            if 0.001 < est_val and est_val < 0.999:  # 固定されていないest
                not_fix_est_sum += est_val
                not_fix_index_list.append(index)
        if not_fix_est_sum != 0:
            for nf_index in not_fix_index_list:
                # FIXME:これ1超える時ある？？？（0, 0.1, 0.1, 0.9とか）
                cp_est_val_and_coo[my][0][nf_index] = (
                    cp_est_val_and_coo[my][0][nf_index] * total_est / not_fix_est_sum
                )

        return cp_est_val_and_coo


BLUE_TABLE = [9, 5, 5, 3, 3, 2, 2, 1, 1, 0]
RED_TABLE = [3, 5, 5, 9, 3, 5, 2, 3, 1, 2]

# iwasaki_evolからの変更点
# 制約を満たすようなパラメータを生成
def generate_rdm_params() -> list:
    rdm_params = []
    rdm_params.append(BLUE_TABLE)  # 青駒テーブル
    rdm_params.append(RED_TABLE)  # 赤駒テーブル
    # 隣接してきた駒,逃げた駒,ゴールに近づいてきた駒 -> 0,1,2
    # 敵（隣接してきた駒,逃げた駒,ゴールに近づいてきた駒）-> 3,4,5
    est_table = [
        random.uniform(-EST_EVAL_RANGE, EST_EVAL_RANGE) for _ in range(ESTIMATE_LEN)
    ]
    rdm_params.append(est_table)  # 推測値
    return rdm_params


# 対戦で使う駒配置を決めておき、その配置のリストを返す
def create_buttle_piece_lists() -> list:
    biased_selection = [1, 2, 1, 2]
    no_biased_selection = [1, 2, 2, 1]
    return [biased_selection, no_biased_selection]


# piece_listを左右反転させる
def invert_piece_list(piece_list: list) -> list:
    inv_piece_list = piece_list
    inv_piece_list[0], inv_piece_list[1] = inv_piece_list[1], inv_piece_list[0]
    inv_piece_list[2], inv_piece_list[3] = inv_piece_list[3], inv_piece_list[2]
    return inv_piece_list


# 配列の並びからStateで使うpiecesを生成
def create_pieces_matrix(piece_list: list) -> list:
    # piece_listの中身は青か赤かを判別する配列
    # ex) [1, 1, 2, 2]
    pieces = [0] * 20
    pieces[13] = piece_list[0]
    pieces[14] = piece_list[1]
    pieces[17] = piece_list[2]
    pieces[18] = piece_list[3]
    return pieces


# paramが最低限の強さを持つか確認する
# ランダムエージェントと対戦して勝率が高いものを合格とする
def is_exceeds_standard(params: list, standard_winning_rate: float) -> bool:
    return True  # 実行時間を早めるためにランダムエージェントと対戦させて間引くのをやめる

    ev_func = create_ev_func(params)
    buttle_piece_lists = create_buttle_piece_lists()
    range_num = 1
    num_of_wins = 0.0
    num_of_matches = 16.0 * range_num  # range(3)で48
    # n/48 > 0.9 -> n > 0.9*48
    # 負けが許される回数 = 48 - n
    allowable_limit = num_of_matches - (standard_winning_rate * num_of_matches)
    lose_point = 0.0

    for _ in range(range_num):
        for i in buttle_piece_lists:
            mirror_i = invert_piece_list(i)
            for k in [i, mirror_i]:
                for j in buttle_piece_lists:
                    state = State(create_pieces_matrix(k), create_pieces_matrix(j))
                    estimate = Estimate(params[2], params[2])  # 推測を司る
                    while True:
                        if state.is_done():  # ゲーム終了時
                            if state.is_lose():
                                if state.depth % 2 == 1:
                                    num_of_wins += 1  # 先手勝ち
                                else:
                                    lose_point += 1  # 先手負け
                            else:
                                num_of_wins += 0.5
                                lose_point += 0.5  # 引き分け
                            break
                        if state.is_first_player():
                            action = spurious_mcts_action(
                                state, estimate.fst_est_val_and_coo, 300
                            )
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = random_action(state)  # 対戦相手はランダムエージェント
                            # action = alpha_beta_action(state, 1)  # 完全情報αβ
                            estimate.double_update_from_action(action, state, False)
                        state = state.next(action)
                    # 先手後手を入れ替えて対戦（盤面はそのまま）
                    state = State(create_pieces_matrix(k), create_pieces_matrix(j))
                    estimate = Estimate(params[2], params[2])  # 推測を司る
                    while True:
                        if state.is_done():
                            if state.is_lose():
                                if state.depth % 2 == 0:
                                    num_of_wins += 1  # 後手勝ち
                                else:
                                    lose_point += 1  # 後手負け
                            else:
                                num_of_wins += 0.5
                                lose_point += 0.5  # 引き分け
                            break
                        if state.is_first_player():
                            action = random_action(state)  # 対戦相手はランダムエージェント
                            # action = alpha_beta_action(state, 1)  # 完全情報αβ
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = spurious_mcts_action(
                                state, estimate.sec_est_val_and_coo, 300
                            )
                            estimate.double_update_from_action(action, state, False)
                        state = state.next(action)

                    if lose_point > allowable_limit:
                        return False
    # winning_rate = num_of_wins / num_of_matches
    # 敗北回数が十分に少なければここに到達
    return True


# 制約を満たし強さが保証されている評価関数をn個生成
def create_many_eval_func(n: int, standard_winning_rate: float) -> list:
    eval_func_list = []
    for _ in range(n):
        print("\r初期生成:" + str(len(eval_func_list)) + "/" + str(n), end="")
        while True:
            rdm_params = generate_rdm_params()
            if is_exceeds_standard(rdm_params, standard_winning_rate):
                eval_func_list.append(rdm_params)
                break
    print("\r初期生成:完了", end="")
    print()
    return eval_func_list


# エリートを対戦により評価する
def evaluate_elite(
    params: list,
    enemy_params: list,
    sp_alphazero_action: Callable[[State, list], int],
    enemy_agent=None,
) -> float:
    # return random.uniform(0, 1)  # テスト用
    # paramsから評価関数を作成
    ev_func = create_ev_func(params)
    # 対戦相手の評価関数を生成
    enemy_ev_func = create_ev_func(enemy_params)

    buttle_piece_lists = create_buttle_piece_lists()

    num_of_wins = 0.0
    num_of_matches = 0.0

    # kとjの部分：buttle_piece_listsに入っているパターンから重複を許して2つ選ぶ
    for k in buttle_piece_lists:
        # 通常のiと、iを左右反転させるパターンを用意する
        for j in buttle_piece_lists:
            state = State(create_pieces_matrix(k), create_pieces_matrix(j))
            estimate = Estimate(params[2], enemy_params[2])  # 推測を司る
            while True:
                # ゲーム終了時
                if state.is_done():
                    num_of_matches += 1
                    if state.is_lose():
                        if state.depth % 2 == 1:
                            num_of_wins += 1  # 先手勝ち
                    else:  # 引き分け
                        num_of_wins += 0.5
                    break

                # 行動の取得
                if state.is_first_player():
                    action = sp_alphazero_action(
                        state, estimate.fst_est_val_and_coo, 300
                    )
                    estimate.double_update_from_action(action, state, True)
                else:
                    if enemy_agent == None:
                        action = sp_alphazero_action(
                            state, estimate.sec_est_val_and_coo, 300
                        )
                    else:
                        action = enemy_agent(state)
                    estimate.double_update_from_action(action, state, False)
                state = state.next(action)

            # 先手後手を入れ替えて対戦（盤面はそのまま）
            state = State(create_pieces_matrix(k), create_pieces_matrix(j))
            estimate = Estimate(enemy_params[2], params[2])  # 推測を司る
            while True:
                if state.is_done():
                    num_of_matches += 1
                    if state.is_lose():
                        if state.depth % 2 == 0:
                            num_of_wins += 1  # 後手勝ち
                    else:
                        num_of_wins += 0.5
                    break

                # 行動の取得
                if state.is_first_player():
                    if enemy_agent == None:
                        action = sp_alphazero_action(
                            state, estimate.fst_est_val_and_coo, 300
                        )
                    else:
                        action = enemy_agent(state)
                    estimate.double_update_from_action(action, state, True)
                else:
                    action = sp_alphazero_action(
                        state, estimate.sec_est_val_and_coo, 300
                    )
                    estimate.double_update_from_action(action, state, False)
                state = state.next(action)

    winning_rate = num_of_wins / num_of_matches
    return winning_rate


# 1回だけ対戦する
def buttle_single(
    params: list,
    enemy_params: list,
    sp_alphazero_action: Callable[[State, list], int],
    enemy_agent=None,
) -> float:
    state = State()
    estimate = Estimate(params[2], enemy_params[2])  # 推測を司る
    num_of_wins = 0.0
    while True:
        # ゲーム終了時
        if state.is_done():
            if state.is_lose():
                if state.depth % 2 == 1:
                    num_of_wins += 1  # 先手勝ち
            else:  # 引き分け
                num_of_wins += 0.5
            break

        # 行動の取得
        if state.is_first_player():
            action = sp_alphazero_action(state, estimate.fst_est_val_and_coo)
            estimate.double_update_from_action(action, state, True)
        else:
            if enemy_agent == None:
                action = sp_alphazero_action(state, estimate.sec_est_val_and_coo)
            else:
                action = enemy_agent(state)
            estimate.double_update_from_action(action, state, False)
        state = state.next(action)
    return num_of_wins


# def league_match(
#     params_list: list, sp_alphazero_action: Callable[[State, list], int]
# ) -> list:
#     length = len(params_list)
#     length_list = list(range(length))
#     vs_result = np.zeros((length, length))
#     vs_tuples = itertools.combinations(length_list, 2)
#     for vs_tp in vs_tuples:
#         result = buttle_double(
#             params_list[vs_tp[0]], params_list[vs_tp[1]], sp_alphazero_action
#         )
#         vs_result[vs_tp[0]][vs_tp[1]] = result
#         vs_result[vs_tp[1]][vs_tp[0]] = 2 - result

#     score_list = [0] * 100
#     for i in range(length):
#         # 勝率（％）
#         score_list[i] = int(np.sum(vs_result[i])) / 2
#     return score_list


def small_league(
    small_params_list: list,
    sp_alphazero_action: Callable[[State, list], int],
    group_size: int,
):
    length = len(small_params_list)
    length_list = list(range(length))
    vs_result = np.zeros((length, length))
    vs_tuples = itertools.combinations(length_list, 2)
    for vs_tp in vs_tuples:
        result = buttle_single(
            small_params_list[vs_tp[0]],
            small_params_list[vs_tp[1]],
            sp_alphazero_action,
        )
        vs_result[vs_tp[0]][vs_tp[1]] = result
        vs_result[vs_tp[1]][vs_tp[0]] = 1 - result

    sm_score_list = [0] * group_size
    for i in range(length):
        # 勝率（％）
        sm_score_list[i] = int(np.sum(vs_result[i])) / 2
    return sm_score_list


# n個のグループに分けてリーグ戦をする（対戦数の少ないリーグ戦）
def league_match(
    params_list: list,
    sp_alphazero_action: Callable[[State, list], int],
    split_num: int = 10,
) -> list:
    length = len(params_list)
    index_list = list(range(100))
    random.shuffle(index_list)
    group_size = length // split_num
    score_list = [0] * 100
    for i in range(split_num):
        small_index = index_list[i * 10 : (i + 1) * 10]
        small_params_list = []
        for si in small_index:
            small_params_list.append(params_list[si])
        sm_score_list = small_league(small_params_list, sp_alphazero_action, group_size)
        for index, si in enumerate(small_index):
            score_list[si] = sm_score_list[index]

    # 勝率を返す
    return score_list


###################
##### k-means #####
###################
import numpy as np
import matplotlib.pyplot as plt

# 重心の初期化. XからランダムにK個抽出する
def kmeans_init_centroids(X, K):
    randidx = np.array(random.sample(range(len(X)), K))
    centroids = X[randidx, :]
    return centroids


# 各データ点を色分けしてプロット
def plot_data_points(X, idx, K):
    x = X[:, 0]
    y = X[:, 1]
    cmap = plt.get_cmap("hsv")
    for i in range(K):
        # 重心iのインデックスの要素番号を抽出
        c = np.where(idx == i)
        plt.scatter(x[c[0]], y[c[0]], marker="o", color=cmap(float(i) / K), alpha=0.2)


# クラスタ毎のデータ点の平均を計算し重心を求める
def compute_centroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for i in range(K):
        c = np.where(idx == i)
        centroids[i, :] = np.mean(X[c[0], :], axis=0)
    return centroids


def run_kmeans(X, init_centroids, max_iters, K, plot_progress):
    centroids = init_centroids
    prev_centroids = init_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plot_progress_kmeans(X, centroids, prev_centroids, idx, K, i)
        prev_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    if plot_progress:
        plt.show()
    return centroids


# 重心の前回点からの移動(線分)をプロット
def draw_line(p1, p2):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1)


# クラスタと重心の変化をプロットする. 描画は単純に重ねていくだけ.
def plot_progress_kmeans(X, centroids, previous, idx, K, i):
    plot_data_points(X, idx, K)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c="black", linewidth=2)
    for j in range(K):
        draw_line(centroids[j, :], previous[j, :])
    plt.title("Iteration number " + str(i + 1))
    plt.pause(0.5)  # 0.5秒sleep


# 各データ点から最も距離が近い重心のインデックスを割り当てる
def find_closest_centroids(X, centroids):
    K = len(centroids)
    m = len(X)
    idx = np.zeros((m, 1))
    for i in range(len(X)):
        min = sys.maxsize
        for j in range(K):
            # データiと重心jの距離を計算
            distance = np.linalg.norm(X[i, :] - centroids[j, :])
            if distance < min:
                min = distance
                idx[i, :] = j
    return idx


######################
##### 5:変異と交叉 #####
######################

ESTIMATE_LEN = 6

# 突然変異（6要素のみ）
def mutation(selected_ev_func: list) -> list:
    mutate_ev_func = []
    mutate_ev_func.append(BLUE_TABLE)  # 青
    mutate_ev_func.append(RED_TABLE)  # 赤
    # 推測の評価関数
    mutate_ev_func.append([])
    for i in range(ESTIMATE_LEN):
        if random.choice([0, 1]) < 1:
            # 変異
            mutate_ev_func[2].append(random.uniform(-EST_EVAL_RANGE, EST_EVAL_RANGE))
        else:
            # 変異なし
            mutate_ev_func[2].append(selected_ev_func[2][i])
    return mutate_ev_func


# 交叉（6要素のみを交叉）
def cross_inheritance(selected_ev_funcs: list) -> list:
    cro_inh_ev_func = pickle.loads(pickle.dumps(selected_ev_funcs[0], -1))
    # 推測の評価関数を処理
    for i in range(ESTIMATE_LEN):
        select = random.choice([0, 1])
        cro_inh_ev_func[2][i] = selected_ev_funcs[select][2][i]

    # 全く同じ個体が生成されていないか確認
    if (
        selected_ev_funcs[0] == cro_inh_ev_func
        or selected_ev_funcs[1] == cro_inh_ev_func
    ):
        # 全く同じ個体が生成されている場合、与えられた2つのev_funcが同一の場合がある
        # その場合は何度交叉しても結果が同じになってしまうため、ここで弾く
        if selected_ev_funcs[0] == selected_ev_funcs[1]:
            # FIXME:ここの処理適当すぎんか？？
            return mutation(selected_ev_funcs[0])
        else:
            # 与えられた2つのev_funcが同一でなくても同じ個体しか生成されない場合がある
            # ex）[a,b,x]と[a,b,y]の場合
            # ここでは似た個体が存在する場合、多様性を増やすためにランダム生成をする
            while True:
                rdm_params = generate_rdm_params()
                if is_exceeds_standard(rdm_params, STANDARD_WINNING_RATE):
                    return rdm_params
            # return cross_inheritance(selected_ev_funcs)
    return cro_inh_ev_func


def create_new_generations_from_elites(
    surviving_evals: list, standard_winning_rate: float, mutation_probability: float,
) -> list:
    new_generations = []
    for eval_func in surviving_evals:
        new_generations.append(eval_func)
    quota = NUM_OF_SAVES - len(surviving_evals)

    # quota（ノルマ）が0になるまで突然変異と交叉を繰り返す
    while True:
        if quota <= 0:
            break
        # mutation_probability（mp）に従って交叉か突然変異を選択
        roll_the_dice = random.random()
        if roll_the_dice < mutation_probability:
            # 突然変異
            selected_ev_func = random.choice(surviving_evals)  # 突然変異させる個体を選択
            new_eval_func = mutation(selected_ev_func)
        else:
            # 交叉
            selected_ev_funcs = random.sample(surviving_evals, 2)  # 交叉させる個体を選択
            new_eval_func = cross_inheritance(selected_ev_funcs)
        # 最低限の性能を有しているか確認
        if is_exceeds_standard(new_eval_func, standard_winning_rate):
            new_generations.append(new_eval_func)
            quota -= 1

    # # 余った分はランダムに個体を生成する
    # rdm_gene_num = num_of_saves - retained_by_cluster_num * num_of_cluster
    # for _ in range(rdm_gene_num):
    #     while True:
    #         rdm_params = generate_rdm_params(MIN, MAX, WIDTH)
    #         if is_exceeds_standard(rdm_params, standard_winning_rate):
    #             new_generations.append(rdm_params)
    #             break

    return new_generations


NUM_OF_SAVES = 100
NUM_OF_ELITES = 30
STANDARD_WINNING_RATE = 0.9  # 強さの閾値（ランダム個体へ保証される勝率）
MUTATION_PROBABILITY = 0.2

# 読み込み
def read_parent(cycle: int, file_pass: str):
    folder_pass = file_pass + "/" + str(cycle) + "/eval_func_list.csv"
    str_sv_evals = pd.read_csv(folder_pass).values.tolist()
    fitness_list, eval_lists = [], []
    for index, _ in enumerate(str_sv_evals):
        fitness_list.append(str_sv_evals[index][0])
        eval_lists.append(ast.literal_eval(str_sv_evals[index][1]))
    return fitness_list, eval_lists


# 書き込み
def output_eval_list(cycle: int, file_pass: str, fitness_list: list, eval_lists: list):
    default_pass = file_pass + str(cycle) + "/"
    os.makedirs(default_pass)  # ディレクトリの生成
    with open(default_pass + "eval_func_list.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["fitness", "eval_func"])
        for i in range(len(eval_lists)):
            writer.writerow([fitness_list[i], eval_lists[i]])


# estの6要素->評価関数
def create_ev_func_from_six_est(six_est: list) -> list:
    ev_func = [BLUE_TABLE, RED_TABLE, six_est]
    return ev_func


# 評価関数->estの6要素
def create_six_est_from_ev_func(ev_func: list) -> list:
    six_est = ev_func[2]
    return six_est


def kei_neat(
    cycle: int, file_pass: str, sp_alphazero_action: Callable[[State, list], int]
) -> None:
    # 適応度の高い個体をn個体取得
    n = NUM_OF_ELITES
    _, eval_lists = read_parent(cycle, file_pass)
    six_est_eval_lists = eval_lists[:n]

    # csvに保存されているのはestの6要素だけなので評価関数を復元する
    surviving_evals = [0] * len(six_est_eval_lists)
    for i, six_est in enumerate(six_est_eval_lists):
        surviving_evals[i] = create_ev_func_from_six_est(six_est)

    # 残った個体から突然変異+交叉で100個体生成
    new_eval_lists = create_new_generations_from_elites(
        surviving_evals, STANDARD_WINNING_RATE, MUTATION_PROBABILITY
    )

    # 総当たりによって生成した個体を評価
    new_fitness_list = league_match(new_eval_lists, sp_alphazero_action)

    # csvに出力する情報を削減するため、評価関数をestの6要素に圧縮
    new_six_ests = [0] * len(new_eval_lists)
    for i, ev_func in enumerate(new_eval_lists):
        new_six_ests[i] = create_six_est_from_ev_func(ev_func)

    # fitnessが高い順にソート
    zip_lists = zip(new_fitness_list, new_six_ests)
    zip_sort = sorted(zip_lists, reverse=True)
    sorted_fit, sorted_three_ests = zip(*zip_sort)

    # cycle+1番目の個体としてcsvに出力
    output_eval_list(cycle + 1, file_pass, sorted_fit, sorted_three_ests)


# 動作確認
if __name__ == "__main__":
    # データを保存するパスを作成
    file_pass = "csv_data/dim6/AlphaZero/" + str(datetime.date.today()) + "/"

    # モデルの読み込み
    path = sorted(Path("./model").glob("*.h5"))[-1]
    model = load_model(str(path))

    # 指し手を決める関数のインスタンスを作成
    sp_alphazero_action = spurious_alphazero_action(model, 15)

    # 1.評価関数をランダムに100個生成
    eval_lists = create_many_eval_func(NUM_OF_SAVES, STANDARD_WINNING_RATE)

    # 2.生成した評価関数でリーグ戦をさせて勝率を計測（これを適応度とする）
    fitness_list = league_match(eval_lists, sp_alphazero_action)

    # csvに出力する情報を削減するため、評価関数をestの3要素に圧縮
    six_ests = [0] * len(eval_lists)
    for i, ev_func in enumerate(eval_lists):
        six_ests[i] = create_six_est_from_ev_func(ev_func)

    # fitnessが高い順にソート
    zip_lists = zip(fitness_list, six_ests)
    zip_sort = sorted(zip_lists, reverse=True)
    sorted_fit, sorted_three_ests = zip(*zip_sort)

    # 0番目の個体としてcsvに出力
    output_eval_list(0, file_pass, sorted_fit, sorted_three_ests)

    # ひたすらkei_neatを実行
    for i in range(10000):
        print("\r第" + str(i) + "世代", end="")
        kei_neat(i, file_pass, sp_alphazero_action)

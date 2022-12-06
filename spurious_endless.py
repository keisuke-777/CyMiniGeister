# SpuriousSearch、永久、0.75、勝利のみ

import random, datetime, copy, csv, os, sys
from mini_game import State
from SpuriousSearch import *

x_min = 0
x_max = 1
x_div_num = 20
x_assign_num = (x_max - x_min) / x_div_num

y_min = 0
y_max = 3
y_div_num = 20
y_assign_num = (y_max - y_min) / y_div_num


class ElitesMap:
    def __init__(self, enemy_params, create_num=15):
        # 適応度を格納するマップとパラメータを格納するマップを分離して保存
        # [y][x]の順番で参照する（printした時の見た目を重視）

        # 適応度を格納するマップ
        self.fitness_map = [[0 for _ in range(x_div_num)] for _ in range(y_div_num)]

        # パラメータを格納するマップ
        self.param_map = [[None for _ in range(x_div_num)] for _ in range(y_div_num)]

        # 格納したセルを記憶しておく
        self.cells_with_elite = []

        # 初期個体
        self.enemy_params = enemy_params

        # 最初のエリートをランダム生成
        for i in range(create_num):
            print(i + 1, "つ目の初期生成")
            params = []
            while True:
                # ランダムにパラメータの組を生成
                params = generate_rdm_params()
                # ランダムエージェントに対して勝率が9割に満たないエージェントを排除
                if is_exceeds_standard(params, 0.9):
                    break
            # 生成したパラメータに従って行動するエージェントをMAPに入れる
            self.store_elite(params)

    # map-elitesで10回探索する
    def search_elites(self):
        for _ in range(10):
            self.mutation()
        self.cross_inheritance()

    # エリートをMAPに格納する
    def store_elite(self, params):
        elite_data = evaluate_elite(self, params)
        fitness = elite_data[0]
        x, y = conv_xy(elite_data[1], elite_data[2])
        # 既存のエリートよりも適応度が上なら書き換える
        if self.fitness_map[y][x] < fitness:
            if self.fitness_map[y][x] == 0:
                self.cells_with_elite.append(x * y_div_num + y)  # 探索済のセルに追加
            self.fitness_map[y][x] = fitness
            self.param_map[y][x] = params

    # 突然変異
    def mutation(self):
        params = []
        while True:
            # マップから無作為にエリートを1つ選ぶ
            elite_xy = random.choice(self.cells_with_elite)
            x = int(elite_xy // y_div_num)
            y = int(elite_xy % y_div_num)
            # エリートを突然変異させる
            params = copy.deepcopy(self.param_map[y][x])
            for eval_func_index, ev_func_arr in enumerate(params):
                # 配列の要素をいくつ変更するか決める
                mutate_num = random.randint(1, len(ev_func_arr))
                # 実際に変更する配列の要素を選ぶ
                change_list = random.sample(range(len(ev_func_arr)), k=mutate_num)
                for change_tile_index in change_list:
                    # 配列の要素をランダム値に変更
                    if eval_func_index == 2:
                        params[eval_func_index][change_tile_index] = random.uniform(
                            -0.5, 0.5
                        )
                    else:
                        params[eval_func_index][change_tile_index] = random.randint(
                            0, 49
                        )
            # ランダムエージェントに対して勝率が9割に満たないエージェントを排除
            if is_exceeds_standard(params, 0.9):
                break

        # 突然変異したパラメータをMAPに格納
        self.store_elite(params)

    # 交叉遺伝
    def cross_inheritance(self):
        # マップから無作為にエリートを2つ選ぶ
        if len(self.cells_with_elite) >= 2:
            elite_xy_list = random.sample(self.cells_with_elite, 2)
            x0 = int(elite_xy_list[0] // y_div_num)
            y0 = int(elite_xy_list[0] % y_div_num)
            x1 = int(elite_xy_list[1] // y_div_num)
            y1 = int(elite_xy_list[1] % y_div_num)
            base_params = copy.deepcopy(self.param_map[y0][x0])
            sec_params = self.param_map[y1][x1]
            for coo, ev_func_arr in enumerate(base_params):
                # base_paramsをベースに何枚かタイルを変更する
                # 変更するタイルを選ぶ
                change_list = random.sample(
                    range(len(ev_func_arr)), k=random.randint(1, len(ev_func_arr))
                )
                for tile_index in change_list:
                    base_params[coo][tile_index] = sec_params[coo][tile_index]
            # 交叉後のパラメータをMAPに格納
            self.store_elite(base_params)


# ランダムにパラメータを生成して、そのリストを返す
def generate_rdm_params(min=0, max=49):
    rdm_params = []
    rdm_params.append([random.randint(min, max) for _ in range(10)])  # 青駒テーブル
    rdm_params.append([random.randint(min, max) for _ in range(10)])  # 赤駒テーブル
    rdm_params.append([random.uniform(-0.5, 0.5) for _ in range(5)])  # 推測値
    return rdm_params


# 対戦結果をxyの整数値に直す
def conv_xy(elite_data_1, elite_data_2):
    conv_x = int((elite_data_1 - x_min) // x_assign_num)
    conv_y = int((elite_data_2 - y_min) // y_assign_num)
    if conv_x < 0 or x_div_num - 1 < conv_x:
        print("elite_data_1(x) is out of range:", elite_data_1)
        if conv_x < 0:
            conv_x = 0
        elif x_div_num - 1 < conv_x:
            conv_x = x_div_num - 1
    if conv_y < 0 or y_div_num - 1 < conv_y:
        print("elite_data_2(y) is out of range:", elite_data_2)
        if conv_y < 0:
            conv_y = 0
        elif y_div_num - 1 < conv_y:
            conv_y = y_div_num - 1
    return conv_x, conv_y


# エリートを対戦により評価する
#  -> float, int, int
def evaluate_elite(elites_map, params):
    # return [random.uniform(0, 1), random.uniform(20, 160), random.uniform(0, 6)]  # テスト用
    # paramsから評価関数を作成
    ev_func = create_ev_func(params)
    # 対戦相手の評価関数を生成
    enemy_ev_func = create_ev_func(elites_map.enemy_params)

    buttle_piece_lists = create_buttle_piece_lists()

    num_of_wins = 0.0
    num_of_matches = 0.0
    num_of_turns = 0.0  # 決着までにかかったターン数（MAP要素）
    num_of_kill_pieces = 0.0  # 決着までに取った相手の駒の数
    num_of_blue_move_turns = 0.0  # 青駒が動いた数

    # iとjの部分：buttle_piece_listsに入っているパターンから重複を許して2つ選ぶ
    for i in buttle_piece_lists:
        # 通常のiと、iを左右反転させるパターンを用意する
        # （反転i vs 反転jは、i vs jと等価なのでjを反転させる必要はない）
        mirror_i = invert_piece_list(i)
        for k in [i, mirror_i]:
            for j in buttle_piece_lists:
                state = State(create_pieces_matrix(k), create_pieces_matrix(j))
                estimate = Estimate(params[2], elites_map.enemy_params[2])  # 推測を司る
                while True:
                    # ゲーム終了時
                    if state.is_done():
                        num_of_matches += 1
                        num_of_turns += state.depth
                        num_of_kill_pieces += check_num_of_killed_enemy_pieces(
                            state, True
                        )
                        if state.is_lose():
                            if state.depth % 2 == 1:
                                num_of_wins += 1  # 先手勝ち
                        break

                    # 行動の取得
                    if state.is_first_player():
                        action = spurious_search_action(
                            state, ev_func, estimate.fst_est_val_and_coo, 3
                        )
                        num_of_blue_move_turns += judge_move_piece_color(
                            action, state.pieces
                        )
                        estimate.double_update_from_action(action, state, True)
                    else:
                        action = spurious_search_action(
                            state, enemy_ev_func, estimate.sec_est_val_and_coo, 3,
                        )
                        estimate.double_update_from_action(action, state, False)
                    state = state.next(action)

                # 先手後手を入れ替えて対戦（盤面はそのまま）
                state = State(create_pieces_matrix(k), create_pieces_matrix(j))
                estimate = Estimate(elites_map.enemy_params[2], params[2])  # 推測を司る
                while True:
                    if state.is_done():
                        num_of_matches += 1
                        num_of_turns += state.depth
                        num_of_kill_pieces += check_num_of_killed_enemy_pieces(
                            state, False
                        )
                        if state.is_lose():
                            if state.depth % 2 == 0:
                                num_of_wins += 1  # 後手勝ち
                        break

                    # 行動の取得
                    if state.is_first_player():
                        action = spurious_search_action(
                            state, enemy_ev_func, estimate.fst_est_val_and_coo, 3,
                        )
                        estimate.double_update_from_action(action, state, True)
                    else:
                        action = spurious_search_action(
                            state, ev_func, estimate.sec_est_val_and_coo, 3
                        )
                        estimate.double_update_from_action(action, state, False)
                        num_of_blue_move_turns += judge_move_piece_color(
                            action, state.pieces
                        )
                    state = state.next(action)

    winning_rate = num_of_wins / num_of_matches
    avg_num_of_turns = num_of_turns / num_of_matches
    avg_num_of_kill_pieces = num_of_kill_pieces / num_of_matches
    avg_ratio_of_move_blue_piece = (2 * num_of_blue_move_turns) / num_of_turns
    return [winning_rate, avg_ratio_of_move_blue_piece, avg_num_of_kill_pieces]


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


# 対戦で使う駒配置を決めておき、その配置のリストを返す（人力でしか無理）
def create_buttle_piece_lists():
    biased_selection = [1, 2, 1, 2]
    no_biased_selection = [1, 2, 2, 1]
    return [biased_selection, no_biased_selection]


# piece_listを左右反転させる
def invert_piece_list(piece_list):
    inv_piece_list = piece_list
    inv_piece_list[0], inv_piece_list[1] = inv_piece_list[1], inv_piece_list[0]
    inv_piece_list[2], inv_piece_list[3] = inv_piece_list[3], inv_piece_list[2]
    return inv_piece_list


# 配列の並びからStateで使うpiecesを生成
def create_pieces_matrix(piece_list):
    # piece_listの中身は青か赤かを判別する配列
    # ex) [1, 1, 2, 2]
    pieces = [0] * 20
    pieces[13] = piece_list[0]
    pieces[14] = piece_list[1]
    pieces[17] = piece_list[2]
    pieces[18] = piece_list[3]
    return pieces


# すでに取った敵駒の数をカウント
def check_num_of_killed_enemy_pieces(state, first_is_elites=True):
    if first_is_elites:  # 先手がエリート
        if state.is_first_player():  # 今はエリートのターン
            enemy_pieces = state.enemy_pieces
        else:  # 今は敵のターン
            enemy_pieces = state.pieces
    else:
        if state.is_first_player():  # 今は敵のターン
            enemy_pieces = state.pieces
        else:  # 今はエリートのターン
            enemy_pieces = state.enemy_pieces
    num_of_enemy_pieces = 0
    for color in enemy_pieces:
        if color == 1 or color == 2:
            num_of_enemy_pieces += 1
    return 4 - num_of_enemy_pieces


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


# my or enemy or zero
def moeoz(piece):
    if piece > 0:  # my
        return 0
    elif piece < 0:  # enemy
        return 1
    else:  # zero
        return 2


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


# 動いた駒が青なら1、赤なら0を返す
def judge_move_piece_color(action, pieces):
    pos = int(action / 4)
    if pieces[pos] == 1:
        return 1
    elif pieces[pos] == 2:
        return 0
    else:
        print("不当な行動を検知", pieces, action)
        return 0


def output_csv(elites_map: ElitesMap, folder_name: str, cycle: int):
    default_pass = "./csv_data/mini_MAP_Elites/" + folder_name + "/" + str(cycle) + "/"
    os.makedirs(default_pass)  # ディレクトリの生成
    with open(default_pass + "fitness_map.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(elites_map.fitness_map)
    with open(default_pass + "param_map.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(elites_map.param_map)


# 勝率判定
# 勝率が閾値を超えている個体をリストに入れて返す
def winning_rate_judgment(elites_map, winning_rate):
    # [勝率, fst_index, sec_index]
    passing_params = []
    for f, fit_list in enumerate(elites_map.fitness_map):
        for s, fitness in enumerate(fit_list):
            if fitness >= winning_rate:
                passing_params.append([fitness, f, s])
    # 勝率が高い順にソート
    passing_params.sort(reverse=True)
    return passing_params


# 閾値を超えた個体をカウンター個体としてcsvに出力する
# （この関数が呼ばれる時にはディレクトリは既に生成されている）
def output_counter_agent_csv(
    elites_map: ElitesMap, folder_name: str, cycle: int, passing_params: list
):
    default_pass = "./csv_data/mini_MAP_Elites/" + folder_name + "/"
    with open(default_pass + str(cycle) + "_counter_agent.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows([["適応度", "パラメータ"]])
        for params in passing_params:
            f, s = params[1], params[2]
            writer.writerows([[params[0], elites_map.param_map[f][s]]])


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


class Estimate:
    def __init__(self, fst_est_params, sec_est_params):
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
            # actionで隣接した(0)
            if bef_around_enemy_num == 0 and aft_around_enemy_num > 0:
                cp_est_val_and_coo[my][0][update_est_index] += est_params[0]
            # actionで逃げた(1) (敵駒をとって隣接を減らした場合も該当)
            if aft_around_enemy_num == 0 and bef_around_enemy_num > 0:
                cp_est_val_and_coo[my][0][update_est_index] += est_params[1]

            is_close = judge_approaching_goal(action, is_my_action)
            # actionで脱出口に近づいた(2)
            if is_close == 1:
                cp_est_val_and_coo[my][0][update_est_index] += est_params[2]
            # actionで脱出口から離れた(3)
            if is_close == -1:
                cp_est_val_and_coo[my][0][update_est_index] += est_params[3]

        # 敵駒と隣接している駒のリスト
        adjacent_pieces_list = check_adjacent_pieces(
            cp_est_val_and_coo[my][1], cp_est_val_and_coo[enemy][1]
        )
        # 行動した駒については既に上で処理しているのでbef_cooを消す
        if bef_coo in adjacent_pieces_list:
            adjacent_pieces_list.remove(bef_coo)

        # 敵駒に隣接されているのに動かない(4)
        if len(adjacent_pieces_list) > 0:
            for adj_piece in adjacent_pieces_list:
                adj_index = cp_est_val_and_coo[my][1].index(adj_piece)
                cp_est_val_and_coo[my][0][adj_index] += est_params[4]

        # ゴールできるのにしなかった（評価関数に関係なく確定で赤駒）
        if 0 in cp_est_val_and_coo[my][1] and action != 2:
            goal_i = cp_est_val_and_coo[my][1].index(0)
            cp_est_val_and_coo[my][0][goal_i] = 0
        if 3 in cp_est_val_and_coo[my][1] and action != 14:
            goal_i = cp_est_val_and_coo[my][1].index(3)
            cp_est_val_and_coo[my][0][goal_i] = 0

        # 座標の更新
        cp_est_val_and_coo[my][1][update_est_index] = aft_coo
        if dead_piece_color != 0:
            # print(state)
            # print(action, "enemy", cp_est_val_and_coo[enemy][1], is_my_action)
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

        # はみ出ている値を0か1に修正
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


# 敵駒に隣接している自駒のリストを返す
def check_adjacent_pieces(my_pieces, enemy_pieces):
    adjacent_piece_list = []
    for my_p in my_pieces:
        around_my_p = get_around_list(my_p)
        if len(set(around_my_p) & set(enemy_pieces)) > 0:
            adjacent_piece_list.append(my_p)
    return adjacent_piece_list


def dead_color_check(action, state):
    aft_coo = calc_coo_of_destination_from_action(action)
    if aft_coo < 0 or 19 < aft_coo:
        return 0
    aft_coo = 19 - aft_coo  # aft_cooはmy視点の座標なのでenemy視点に修正
    # 青->1, 赤->2, 死駒無し->0
    return state.enemy_pieces[aft_coo]


# paramが最低限の強さを持つか確認する
# ランダムエージェントと対戦して勝率が高いものを合格とする
def is_exceeds_standard(params, standard_winning_rate):
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
                            action = spurious_search_action(
                                state, ev_func, estimate.fst_est_val_and_coo, 3
                            )
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = random_action(state)  # 対戦相手はランダムエージェント
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
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = spurious_search_action(
                                state, ev_func, estimate.sec_est_val_and_coo, 3
                            )
                            estimate.double_update_from_action(action, state, False)
                        state = state.next(action)

                    if lose_point > allowable_limit:
                        return False
    # winning_rate = num_of_wins / num_of_matches
    # 敗北回数が十分に少なければここに到達
    return True


# 永久にメタを回す
def endless_cycle(folder_name, enemy_params):
    print("フォルダ名：", folder_name)
    print("対戦相手のパラメータ：", enemy_params)
    elites_map = ElitesMap(enemy_params, 50)
    next_enemy_params = []
    for i in range(10000):
        print(folder_name + "mini:サイクル数：", i)
        print("時刻等", datetime.datetime.utcnow() + datetime.timedelta(hours=9))
        elites_map.search_elites()
        print("〜fitness〜")
        print(elites_map.fitness_map)
        if i % 10 == 0:
            output_csv(elites_map, folder_name, i)
            # 勝率判定
            check = winning_rate_judgment(elites_map, 0.8)
            if len(check) > 0:
                # 新世代
                output_counter_agent_csv(elites_map, folder_name, i, check)
                next_enemy_params = elites_map.param_map[check[0][1]][check[0][2]]
                break
    endless_cycle(folder_name + "+", next_enemy_params)


# 動作確認
if __name__ == "__main__":
    folder_name = "SpuriousSearch/エンドレス/勝利のみ交叉初期強" + str(datetime.date.today())
    init_enemy_params = [
        [9, 5, 5, 3, 3, 2, 2, 1, 1, 0],
        [3, 5, 5, 9, 3, 5, 2, 3, 1, 2],
        [0.2, 0.4, 0.05, 0.4, -0.1],
    ]
    endless_cycle(folder_name, init_enemy_params)

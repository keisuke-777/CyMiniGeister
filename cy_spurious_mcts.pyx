# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
import numpy as np
import math, random
from cy_mini_game import State, random_action
from libcpp cimport bool

# モンテカルロ木探索のノード
cdef class node:
    cdef public int w
    cdef public int n
    cdef public int real_w
    cdef public int real_n
    cdef public list child_nodes
    cdef public int action
    cdef public bool is_my_turn

    # 初期化
    def __init__(self, action, is_my_turn):
        # self.state = state  # 状態
        self.w = 0  # 累計価値
        self.n = 0  # 試行回数
        self.real_w = 0
        self.real_n = 0
        self.child_nodes = None  # 子ノード群
        self.action = action
        self.is_my_turn = is_my_turn

    # 評価
    def evaluate(self, state, is_real):
        # ゲーム終了時
        if state.is_done():
            # 勝敗結果で価値を取得
            value = -1 if state.is_lose() else 0  # 負けは-1、引き分けは0

            # 累計価値と試行回数の更新
            self.w += value
            self.n += 1
            return value

        # 子ノードが存在しない時
        if not self.child_nodes:
            # プレイアウトで価値を取得
            value = playout(state)

            # 累計価値と試行回数の更新
            self.w += value
            self.n += 1

            # 子ノードの展開
            if self.n == 10:
                self.expand(state)
            return value

        # 子ノードが存在する時
        else:  # ノード選択
            # UCB1が最大の子ノードの評価で価値を取得
            next_child = self.next_child_node()
            # ノード選択の際には実際にstate（インスタンス）を更新する
            value = -next_child.evaluate(state.next(next_child.action), is_real)

            # 累計価値と試行回数の更新
            if is_real:  # realの場合のみreal値を更新
                self.real_w += value
                self.real_n += 1
            self.w += value
            self.n += 1
            return value

    # 子ノードの展開
    def expand(self, state):
        legal_actions = state.legal_actions()
        self.child_nodes = []
        for action in legal_actions:
            # 行動番号を子ノードに与える
            self.child_nodes.append(node(action, not self.is_my_turn))

    # UCB1が最大の子ノードを取得
    def next_child_node(self):
        if self.is_my_turn:  # 自分のターン（real値によってノードを選択）
            # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.real_n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.real_n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(
                    -child_node.real_w / child_node.real_n
                    + 2 * (2 * math.log(t) / child_node.real_n) ** 0.5
                )

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]
        else:  # 相手のターン（通常値によってノードを選択）
            # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(
                    -child_node.w / child_node.n
                    + 2 * (2 * math.log(t) / child_node.n) ** 0.5
                )

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]


cpdef spurious_mcts_action(state, est_val_and_coo, search_num=300):
    # ルートノードの生成
    cdef node root_node
    cdef list state_pattern
    cdef list correct_pattern_list
    cdef list board_pattern_prob
    root_node = node(-1, True)
    root_node.expand(state)

    # ルートノードを評価 (rangeを変化させると評価回数を変化させられる)
    # あり得る盤面を決める
    state_pattern, correct_pattern_list, board_pattern_prob = create_all_pattern_state(
        state, est_val_and_coo
    )
    # # 誤差や死駒の反映が遅れることによって、合計が1から大きくずれることがあるため正規化する
    # board_pattern_prob = np.array(board_pattern_prob, dtype=np.float64) / np.sum(
    #     board_pattern_prob
    # )

    # 遅くて良いのでindexの配列をあらかじめ作っておく
    # cdef int choice_indexes[30000]
    choice_indexes = random.choices(range(36), k=search_num, weights=board_pattern_prob)

    cdef int i
    for i in range(search_num):
        # 確率に従って探索で使う盤面を選ぶ
        rdm_state = state_pattern[choice_indexes[i]]
        is_real = correct_pattern_list[choice_indexes[i]]
        root_node.evaluate(rdm_state, is_real)

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.cy_legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)

    # 最後に指し手を決める際にはreal_nでなくnを参照する
    # これは、real_nを参照すると乱数による影響が大きくなることと、
    # real_nのvalueはnにも間接的に関わってくるため
    # （違う？UCB1は新規性も求めるのでこれだとまずい？？）
    return legal_actions[argmax(n_list)]

    # w_list = []
    # real_n_list = []
    # real_w_list = []
    # for c in root_node.child_nodes:
    #     w_list.append(c.w)
    #     real_n_list.append(c.real_n)
    #     real_w_list.append(c.real_w)
    # print("n", n_list)
    # print("w", w_list)
    # print("real_n", real_n_list)
    # print("real_w", real_w_list)
    # return legal_actions[argmax(real_n_list)]


# 最大値のインデックスを返す
def argmax(collection, key=None):
    return collection.index(max(collection))


# ゲームの終端までシミュレート
cdef playout(state):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    return -playout(state.next(random_action(state)))


# 推測値からあり得る全ての盤面を作成しそれらの確率を計算する
def create_all_pattern_state(state, est_val_and_coo):
    # 自分の駒に着目した確率
    # 最悪のパワー実装
    my_pattern_set = [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    ]
    my_pattern_prob = np.zeros(6)
    my_est_val_and_coo = est_val_and_coo[0][0]
    for i in range(6):
        my_pattern_prob[i] = (
            my_pattern_set[i][0] * my_est_val_and_coo[0]
            + my_pattern_set[i][1] * my_est_val_and_coo[1]
            + my_pattern_set[i][2] * my_est_val_and_coo[2]
            + my_pattern_set[i][3] * my_est_val_and_coo[3]
        ) / 6.0

    en_pattern_set = [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    ]
    en_pattern_prob = np.zeros(6)
    en_est_val_and_coo = est_val_and_coo[1][0]
    for i in range(6):
        en_pattern_prob[i] = (
            en_pattern_set[i][0] * en_est_val_and_coo[0]
            + en_pattern_set[i][1] * en_est_val_and_coo[1]
            + en_pattern_set[i][2] * en_est_val_and_coo[2]
            + en_pattern_set[i][3] * en_est_val_and_coo[3]
        ) / 6.0

    # 実際に正しい盤面のパターンをstateから確認する
    correct_pattern = [-1, -1, -1, -1]
    for i in range(4):
        if est_val_and_coo[0][1][i] >= 20:
            correct_pattern[i] = -1
        elif state.pieces[est_val_and_coo[0][1][i]] == 2:
            correct_pattern[i] = 0
        elif state.pieces[est_val_and_coo[0][1][i]] == 1:
            correct_pattern[i] = 1

    # 0~3->自分の駒、4~7->相手の駒、8->その盤面があり得るか否か（あり得るなら1）
    # board_pattern = np.zeros((36, 9))
    board_pattern = [0] * 36
    correct_pattern_list = [0] * 36
    board_pattern_prob = [0] * 36

    for i in range(6):
        for j in range(6):
            board_pattern[i * 6 + j] = my_pattern_set[i] + en_pattern_set[j]
            # 対応するパターンがあり得るかを保存する
            correct_pattern_list[i * 6 + j] = is_pattern_correct(
                my_pattern_set[i], correct_pattern
            )
            # その盤面に対応する確率を計算
            board_pattern_prob[i * 6 + j] = my_pattern_prob[i] * en_pattern_prob[j]

    # board_patternをstateに変換
    state_pattern = edit_state_with_board_pat(board_pattern, state, est_val_and_coo)

    return state_pattern, correct_pattern_list, board_pattern_prob


# board_patternをstateに変換
def edit_state_with_board_pat(board_pattern, state, est_val_and_coo):
    state_pattern = [0] * 36
    my_coo = est_val_and_coo[0][1]
    en_coo = est_val_and_coo[1][1]  # これ反転させないと使えない
    for i in range(36):
        my_pieces = state.pieces.copy()
        for j in range(4):
            if my_coo[j] != -1:  # 死駒でない
                if board_pattern[i][j] == 0:
                    my_pieces[my_coo[j]] = 2  # 赤
                else:
                    my_pieces[my_coo[j]] = 1  # 青
        en_pieces = state.enemy_pieces.copy()
        for j in range(4):
            if en_coo[j] != -1:  # 死駒でない
                if board_pattern[i][4 + j] == 0:
                    # 座標の反転処理
                    en_pieces[19 - en_coo[j]] = 2
                else:
                    en_pieces[19 - en_coo[j]] = 1
        edit_state = State(my_pieces, en_pieces, state.depth)
        state_pattern[i] = edit_state
    return state_pattern


# Trueで1、Falseで0を返す（boolを返さない）
def is_pattern_correct(check_pat, correct_pat):
    # length = len(check_pat)
    length = 4
    for i in range(length):
        if correct_pat[i] != -1 and check_pat[i] != correct_pat[i]:
            return 0
    return 1


# 動作確認
if __name__ == "__main__":
    # 状態の生成
    state = State()
    est_val_and_coo = [
        [[1.0, 0.2, 0.3, 0.5], [13, 14, 17, 18]],
        [[0.5, 0.7, 0.3, 0.5], [1, 2, 5, 6]],
    ]

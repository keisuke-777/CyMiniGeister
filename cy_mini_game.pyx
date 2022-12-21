# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

import random, math
cimport numpy as np
from libcpp cimport bool

# ゲームの状態
cdef class State:
    cdef public int pieces[20]
    cdef public int enemy_pieces[20]
    cdef public int depth
    cdef public bool is_goal

    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        self.is_goal = False

        # 駒の配置
        if pieces != None:
            self.pieces = pieces
        else:
            self.pieces = [0] * 20

        if enemy_pieces != None:
            self.enemy_pieces = enemy_pieces
        else:
            self.enemy_pieces = [0] * 20

        # ターンの深さ(ターン数)
        self.depth = depth

        # 駒の初期配置
        if pieces == None or enemy_pieces == None:
            # 青2赤2
            piece_list = [1, 1, 2, 2]

            random.shuffle(piece_list)
            self.pieces[13] = piece_list[0]
            self.pieces[14] = piece_list[1]
            self.pieces[17] = piece_list[2]
            self.pieces[18] = piece_list[3]

            random.shuffle(piece_list)
            self.enemy_pieces[13] = piece_list[0]
            self.enemy_pieces[14] = piece_list[1]
            self.enemy_pieces[17] = piece_list[2]
            self.enemy_pieces[18] = piece_list[3]

    # 負けかどうか
    def is_lose(self):
        if not any(elem == 1 for elem in self.pieces):  # 自分の青駒が存在しないなら負け
            # print("青喰い")
            return True
        if not any(elem == 2 for elem in self.enemy_pieces):  # 敵の赤駒が存在しない(全部取っちゃった)なら負け
            # print("赤喰い")
            return True
        # 前の手でゴールされてたらis_goalがTrueになってる(はず)
        if self.is_goal:
            # print("ゴール")
            return True
        return False

    # 引き分けかどうか
    def is_draw(self):
        return self.depth >= 100  # 200手

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # デュアルネットワークの入力の2次元配列の取得
    def pieces_array(self):
        # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
        def pieces_array_of(pieces):
            table_list = []
            # 青駒(1)→赤駒(2)の順に取得
            for j in range(1, 3):
                table = [0] * 20
                table_list.append(table)
                # appendは参照渡しっぽいのでtable書き換えればtable_listも書き換わる
                for i in range(20):
                    if pieces[i] == j:
                        table[i] = 1

            return table_list

        # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
        return [pieces_array_of(self.pieces), pieces_array_of(self.enemy_pieces)]

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    # 行動を駒の移動元と移動方向に変換
    def action_to_position(self, action):
        return (int(action / 4), action % 4)  # position,direction

    # 合法手のリストの取得
    def legal_actions(self):
        return self.cy_legal_actions()

    cdef cy_legal_actions(self):
        cdef list actions = []
        cdef int p
        for p in range(20):
            # 駒の移動時
            if self.pieces[p] != 0:
                # 移動前の駒の位置を渡す
                actions.extend(self.legal_actions_pos(p))
        # 青駒のゴール行動は例外的に合法手リストに追加
        if self.pieces[0] == 1:
            actions.extend([2])  # 0*4 + 2
        if self.pieces[3] == 1:
            actions.extend([14])  # 3*4 + 2
        return actions

    # 駒の移動時の合法手のリストの取得
    cdef legal_actions_pos(self, int position):
        cdef list actions = []
        cdef int x = position % 4
        cdef int y = position // 4
        if y != 4 and self.pieces[position + 4] == 0:  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] == 0:  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 4] == 0:  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 3 and self.pieces[position + 1] == 0:  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動は例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 次の状態の取得
    def next(self, action):
        # 次の状態の作成
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth + 1)

        # 行動を(移動元, 移動方向)に変換
        position_bef, direction = self.action_to_position(action)

        # 合法手がくると仮定
        # 駒の移動（後ろに動く頻度が少ない + if文そんなに踏ませたくないと思ったので判定を左右下上の順番にしてる）
        if direction == 1:
            position_aft = position_bef - 1
        elif direction == 3:
            position_aft = position_bef + 1
        elif direction == 0:
            position_aft = position_bef + 4
        elif direction == 2:
            if position_bef == 0 or position_bef == 3:
                # ゴールした場合は特殊処理（駒を移動させない）
                state.is_goal = True
                tmp = state.pieces
                state.pieces = state.enemy_pieces
                state.enemy_pieces = tmp
                return state
            else:
                position_aft = position_bef - 4
        else:
            print("error関数名:next")

        # 実際に駒移動
        state.pieces[position_aft] = state.pieces[position_bef]
        state.pieces[position_bef] = 0

        # 移動先に敵駒が存在した場合は取る
        if state.enemy_pieces[19 - position_aft] != 0:
            state.enemy_pieces[19 - position_aft] = 0

        # 駒の交代
        tmp = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = tmp
        return state

    # 先手かどうか
    def is_first_player(self):
        return self.depth % 2 == 0

    # 文字列表示
    def __str__(self):
        row = "|{}|{}|{}|{}|"
        hr = "\n---------------------\n"
        board = [0] * 20
        if self.depth % 2 == 0:
            my_p = self.pieces.copy()
            rev_ep = list(reversed(self.enemy_pieces))
            for i in range(20):
                board[i] = my_p[i] - rev_ep[i]
        else:
            my_p = list(reversed(self.pieces))
            rev_ep = self.enemy_pieces.copy()
            for i in range(20):
                board[i] = rev_ep[i] - my_p[i]

        board_essence = []
        for i in board:
            if i == 1:
                board_essence.append("自青")
            elif i == 2:
                board_essence.append("自赤")
            elif i == -1:
                board_essence.append("敵青")
            elif i == -2:
                board_essence.append("敵赤")
            else:
                board_essence.append("　　")

        str = (hr + row + hr + row + hr + row + hr + row + hr + row + hr).format(
            *board_essence
        )
        return str


# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# 人間に行動を選択させる
def human_player_action(state):
    # 盤面を表示
    print(state)

    # 入力を待つ(受ける)
    before_move_place = int(input("Please enter to move piece (左上~右下にかけて0~19) : "))
    direction = int(input("direction (下0 左1 上2 右3) : "))
    move = state.position_to_action(before_move_place, direction)

    # 合法手か確認
    legal_actions = state.legal_actions()
    if any(elem == move for elem in legal_actions):
        return move

    # エラー処理(デバッグでしか使わんから適当)
    print("よくわからんけど非合法手を選んだのでランダム手を選択しました")
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# モンテカルロ木探索の行動選択
def mcts_action(state):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 評価
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0  # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
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

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    # ルートノードを評価 (rangeを変化させると評価回数を変化させられる)
    for _ in range(100):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]


# 最大値のインデックスを返す
def argmax(collection, key=None):
    return collection.index(max(collection))


# ゲームの終端までシミュレート
def playout(state):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    return -playout(state.next(random_action(state)))


from mini_ii_game import AccessableState

INFINITY = 10000

# 評価関数を直接受けるアルファベータ
def alpha_beta(
    ii_state, alpha, beta, search_depth, max_depth, ev_func,
):
    if ii_state.is_lose():
        if ii_state.my_turn:
            return -INFINITY + search_depth
        else:
            return INFINITY - search_depth
    if ii_state.is_win():
        if ii_state.my_turn:
            return INFINITY - search_depth
        else:
            return -INFINITY + search_depth

    # 規定の深さにきたら盤面を評価
    if search_depth == max_depth:
        return ev_func(ii_state)

    for action in ii_state.legal_actions():
        score = -alpha_beta(
            ii_state.next(action), -beta, -alpha, search_depth + 1, max_depth, ev_func,
        )
        if score > alpha:
            alpha = score
        if alpha >= beta:
            return alpha
    return alpha


# 評価関数を直接受けるアルファベータ探索
def alpha_beta_action(state, ev_func, max_depth=5):
    ii_state = AccessableState()
    ii_state.create_ii_state_from_state(state)

    # 合法手の状態価値の計算
    best_action = 0
    alpha = -INFINITY
    for action in ii_state.legal_actions():
        # betaとalphaを符号反転させて入れ替える
        score = -alpha_beta(
            ii_state.next(action), -INFINITY, -alpha, 0, max_depth, ev_func,
        )
        if score > alpha:
            best_action = action
            alpha = score
    return best_action


# 動作確認
if __name__ == "__main__":
    # 状態の生成
    state = State()
    all_depth = 0

    for _ in range(10000):
        # ゲーム終了までのループ
        while True:
            # ゲーム終了時
            if state.is_done():
                # print(state.depth)
                all_depth += state.depth
                state = State()
                break

            # 次の状態の取得
            state = state.next(random_action(state))

    print(all_depth / 10000)

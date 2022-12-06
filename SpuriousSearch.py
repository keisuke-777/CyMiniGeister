import random, copy, math, mojimoji, pickle
from mini_game import State, random_action, human_player_action
from mini_ii_game import AccessableState

INFINITY = 1000

# 駒の移動元と移動方向を行動に変換
def position_to_action(position, direction):
    return position * 4 + direction


# 行動を駒の移動元と移動方向に変換
def action_to_position(action):
    return (int(action / 4), action % 4)  # position,direction


# 合法手のリストの取得
def use_pieces_legal_actions(pieces):
    actions = []
    for p in range(20):
        # 駒の移動時
        if pieces[p] != 0:
            # 移動前の駒の位置を渡す
            actions.extend(use_pieces_legal_actions_pos(pieces, p))
    # 青駒のゴール行動は例外的に合法手リストに追加
    if pieces[0] == 1:
        actions.extend([2])  # 0*4 + 2
    if pieces[3] == 1:
        actions.extend([14])  # 3*4 + 2
    return actions


# 駒の移動時の合法手のリストの取得
def use_pieces_legal_actions_pos(pieces, position):
    actions = []
    x = position % 4
    y = int(position / 4)
    if y != 4 and pieces[position + 4] == 0:  # 下端でない and 下に自分の駒がいない
        actions.append(position_to_action(position, 0))
    if x != 0 and pieces[position - 1] == 0:  # 左端でない and 左に自分の駒がいない
        actions.append(position_to_action(position, 1))
    if y != 0 and pieces[position - 4] == 0:  # 上端でない and 上に自分の駒がいない
        actions.append(position_to_action(position, 2))
    if x != 3 and pieces[position + 1] == 0:  # 右端でない and 右に自分の駒がいない
        actions.append(position_to_action(position, 3))
    # 青駒のゴール行動は例外的にuse_pieces_legal_actions側で処理する(ここでは処理しない)
    return actions


# 行動番号から移動先の座標を計算
def calc_coo_of_destination_from_action(action):
    start_coo = int(action / 4)
    direction = action % 4
    if direction == 0:
        return start_coo + 4
    elif direction == 1:
        return start_coo - 1
    elif direction == 2:
        return start_coo - 4
    elif direction == 3:
        return start_coo + 1


# チャンスノードに遷移するような行動を探す
# （チャンスノードには駒を取る or ゴールをする際に遷移）
# ※ 仕様上は駒をとる行動 + ゴール行動を追加するような行動（ゴール前に駒を運ぶ行動）が
#   対象になっているが、実装の簡略化のためにゴール行動だけ別処理にする
def search_actions_to_transition_chance_node(pieces, my_turn):
    chance_action_list = []  # チャンスノードに遷移する行動
    if my_turn:
        my_type = [1, 2]
        enemy_type = [-1, -2]
    else:
        my_type = [-1, -2]
        enemy_type = [1, 2]

    # 隣接する駒を確認
    for index, piece in enumerate(pieces):
        x = index % 4
        y = int(index / 4)
        if piece in my_type:
            if y != 4 and pieces[index + 4] in enemy_type:  # 下端でない and 下に敵の駒がある
                chance_action_list.append(position_to_action(index, 0))  # 下に移動する行動
            if x != 0 and pieces[index - 1] in enemy_type:  # 左端でない and 左に敵の駒がある
                chance_action_list.append(position_to_action(index, 1))  # 左に移動する行動
            if y != 0 and pieces[index - 4] in enemy_type:  # 上端でない and 上に敵の駒がある
                chance_action_list.append(position_to_action(index, 2))  # 上に移動する行動
            if x != 3 and pieces[index + 1] in enemy_type:  # 右端でない and 右に敵の駒がある
                chance_action_list.append(position_to_action(index, 3))  # 右に移動する行動

    # ゴール前に駒がいるか
    if my_turn:
        if pieces[0] in [1]:
            chance_action_list.append(2)  # 0*4 + 2
        if pieces[3] in [1]:
            chance_action_list.append(14)  # 3*4 + 2
    else:
        if pieces[16] in [-1, -2]:
            chance_action_list.append(64)  # 16*4 + 0
        if pieces[19] in [-1, -2]:
            chance_action_list.append(76)  # 19*4 + 0

    return chance_action_list


# 受け取ったタプルの符号をひっくり返したタプルを返す
def nega(pos):
    nega_list = [0, 0]
    nega_list[0], nega_list[1] = -pos[0], -pos[1]
    return tuple(nega_list)


def spurious_search(
    ii_state,
    real_alpha,
    spurious_alpha,
    real_beta,
    spurious_beta,
    search_depth,
    max_depth,
    ev_func,
    est_value_and_coo,
):
    if ii_state.is_lose():
        if ii_state.my_turn:
            return -INFINITY + search_depth, -INFINITY + search_depth
        else:
            return INFINITY - search_depth, INFINITY - search_depth
    if ii_state.is_win():
        if ii_state.my_turn:
            return INFINITY - search_depth, INFINITY - search_depth
        else:
            return -INFINITY + search_depth, -INFINITY + search_depth
    # 規定の深さにきたら盤面を評価
    if search_depth == max_depth:
        return ev_func(ii_state), ev_func(ii_state)

    # チャンスノードに遷移する行動をリストアップ
    chance_actions = search_actions_to_transition_chance_node(
        ii_state.pieces, ii_state.my_turn
    )
    legal_actions = list(set(ii_state.legal_actions()) - set(chance_actions))

    # 先にチャンスノードを処理
    for action in chance_actions:
        target_piece_coo = -1
        # まず行動番号から注目すべき駒を確定させる
        is_goal = False
        if action in [2, 14, 64, 76]:
            # ゴール行動（ゴールする駒がtarget）
            is_goal = True
            if action == 2:
                target_piece_coo = 0
            if action == 14:
                target_piece_coo = 3
            if action == 64:
                target_piece_coo = 16
            if action == 76:
                target_piece_coo = 19

            est_index = est_value_and_coo[0][1].index(target_piece_coo)
            est_val = est_value_and_coo[0][0][est_index]
        else:
            # ゴール以外（取られる駒がtarget）
            target_piece_coo = calc_coo_of_destination_from_action(action)

            # deepcopyと同じ処理
            cp_est_val_and_coo = pickle.loads(pickle.dumps(est_value_and_coo, -1))

            # target_pieceの推測値を取り出す
            est_index = cp_est_val_and_coo[1][1].index(target_piece_coo)
            est_val = cp_est_val_and_coo[1][0][est_index]
            cp_est_val_and_coo[1][0][est_index] = -1  # targetは死亡

            # 攻撃者の座標を更新
            attacker_piece_coo = int(action / 4)
            attacker_index = cp_est_val_and_coo[0][1].index(attacker_piece_coo)
            cp_est_val_and_coo[0][1][attacker_index] = target_piece_coo

        if is_goal:
            if est_val != 0:  # ゴール行動があり得る（この場合、厳密には親のノードがチャンスノード）
                real_blue_score, spurious_blue_score = INFINITY, INFINITY  # 青なら即ゴール
                # 赤の場合ゴール行動は取れないので、それ以外の行動を探索し直す
                # deepcopy
                red_est_val_and_coo = pickle.loads(pickle.dumps(est_value_and_coo, -1))

                red_est_val_and_coo[0][0][est_index] = 0  # targetが青の可能性を排除
                real_red_score, spurious_red_score = spurious_search(
                    ii_state,
                    real_alpha,
                    spurious_alpha,
                    real_beta,
                    spurious_beta,
                    search_depth,
                    max_depth,
                    ev_func,
                    red_est_val_and_coo,
                )
            else:  # ゴール行動があり得ない（est_valが0 -> ゴール前は赤駒）
                real_blue_score, spurious_blue_score = 0, 0  # 青の値は何でも良い
                real_red_score, spurious_red_score = -INFINITY, -INFINITY  # 赤のゴールは非合法手
        else:
            real_blue_score, spurious_blue_score = nega(
                spurious_search(
                    ii_state.specify_color_next(action, target_piece_coo, 1),
                    -real_beta,
                    -spurious_beta,
                    -real_alpha,
                    -spurious_alpha,
                    search_depth + 1,
                    max_depth,
                    ev_func,
                    [cp_est_val_and_coo[1], cp_est_val_and_coo[0]],
                )
            )
            real_red_score, spurious_red_score = nega(
                spurious_search(
                    ii_state.specify_color_next(action, target_piece_coo, 2),
                    -real_beta,
                    -spurious_beta,
                    -real_alpha,
                    -spurious_alpha,
                    search_depth + 1,
                    max_depth,
                    ev_func,
                    [cp_est_val_and_coo[1], cp_est_val_and_coo[0]],
                )
            )

        # target_piece_cooが自分の駒であるか確認する
        # ii_stateから確認できるのは自分の駒の色だけなのでどちらの手番でもOKなはず?
        target_piece_color = ii_state.pieces[target_piece_coo]
        if target_piece_color == 1:  # 青駒であることを知っている
            real_score = real_blue_score
            spurious_score = (
                est_val * spurious_blue_score + (1 - est_val) * spurious_red_score
            )
        elif target_piece_color == 2:  # 赤駒であることを知っている
            real_score = real_red_score
            spurious_score = (
                est_val * spurious_blue_score + (1 - est_val) * spurious_red_score
            )
        else:  # 敵の駒（本当の色は知らない）
            # est_valをかけて期待値を算出
            real_score = est_val * real_blue_score + (1 - est_val) * real_red_score
            spurious_score = (
                est_val * spurious_blue_score + (1 - est_val) * spurious_red_score
            )

        # 自分はreal_alphaが高くなるように
        # 相手はspurious_alphaが高くなるように選ぶ
        if ii_state.my_turn:
            if real_score > real_alpha:
                real_alpha, spurious_alpha = real_score, spurious_score
            if real_alpha >= real_beta:
                return real_alpha, spurious_alpha
        else:
            if spurious_score > spurious_alpha:
                real_alpha, spurious_alpha = real_score, spurious_score
            if spurious_alpha >= spurious_beta:
                return real_alpha, spurious_alpha

    for action in legal_actions:
        # 行動者の座標を更新
        # FIXME:当時のこと何も覚えてないけどこれTrueで固定してていいのか？？
        cp_est_val_and_coo = update_est_coo(action, est_value_and_coo, True)

        # 通常の処理
        real_score, spurious_score = nega(
            spurious_search(
                ii_state.next(action),
                -real_beta,
                -spurious_beta,
                -real_alpha,
                -spurious_alpha,
                search_depth + 1,
                max_depth,
                ev_func,
                [cp_est_val_and_coo[1], cp_est_val_and_coo[0]],
            )
        )

        # 自分はreal_alphaが高くなるように
        # 相手はspurious_alphaが高くなるように選ぶ
        if ii_state.my_turn:
            if real_score > real_alpha:
                real_alpha, spurious_alpha = real_score, spurious_score
            if real_alpha >= real_beta:
                return real_alpha, spurious_alpha
        else:
            if spurious_score > spurious_alpha:
                real_alpha, spurious_alpha = real_score, spurious_score
            if spurious_alpha >= spurious_beta:
                return real_alpha, spurious_alpha
    return real_alpha, spurious_alpha


def spurious_search_action(state, ev_func, est_value_and_coo, max_depth=5):
    ii_state = AccessableState()
    ii_state.create_ii_state_from_state(state)

    # 自分のゴール行動がある場合、即座にそれを返す
    if 2 in ii_state.legal_actions():
        return 2
    if 14 in ii_state.legal_actions():
        return 14

    # チャンスノードに遷移する行動をリストアップ
    chance_actions = search_actions_to_transition_chance_node(
        ii_state.pieces, ii_state.my_turn
    )
    legal_actions = list(set(ii_state.legal_actions()) - set(chance_actions))

    # 合法手の状態価値の計算
    best_action = -1
    real_alpha = -INFINITY
    spurious_alpha = -INFINITY

    # チャンスノードの処理
    # この関数が呼ばれるのは自分（駒の色が確定している側）のターンのみなので、ゴール行動はありえない
    for action in chance_actions:
        # まず行動番号から注目すべき駒を確定させる
        target_piece_coo = calc_coo_of_destination_from_action(action)

        # deepcopyと同じ処理
        cp_est_val_and_coo = pickle.loads(pickle.dumps(est_value_and_coo, -1))

        # target_pieceの推測値を取り出す
        est_index = cp_est_val_and_coo[1][1].index(target_piece_coo)
        est_val = cp_est_val_and_coo[1][0][est_index]
        cp_est_val_and_coo[1][0][est_index] = -1  # targetは死亡

        # 攻撃者の座標を更新
        attacker_piece_coo = int(action / 4)
        attacker_index = cp_est_val_and_coo[0][1].index(attacker_piece_coo)
        cp_est_val_and_coo[0][1][attacker_index] = target_piece_coo

        real_blue_score, _ = nega(
            spurious_search(
                ii_state.specify_color_next(action, target_piece_coo, 1),
                -INFINITY,
                -INFINITY,
                -real_alpha,
                -spurious_alpha,
                0,
                max_depth,
                ev_func,
                [cp_est_val_and_coo[1], cp_est_val_and_coo[0]],
            )
        )
        real_red_score, _ = nega(
            spurious_search(
                ii_state.specify_color_next(action, target_piece_coo, 2),
                -INFINITY,
                -INFINITY,
                -real_alpha,
                -spurious_alpha,
                0,
                max_depth,
                ev_func,
                [cp_est_val_and_coo[1], cp_est_val_and_coo[0]],
            )
        )

        # spurious_scoreについては自分のノード選択に関係ないのでdepth=0では計算する必要がない
        real_score = est_val * real_blue_score + (1 - est_val) * real_red_score

        if real_score > real_alpha:
            best_action = action
            real_alpha = real_score

    # 通常の処理
    for action in legal_actions:
        # 攻撃者の座標を更新
        target_piece_coo = calc_coo_of_destination_from_action(action)

        # deepcopyと同じ処理
        cp_est_val_and_coo = pickle.loads(pickle.dumps(est_value_and_coo, -1))

        attacker_piece_coo = int(action / 4)
        attacker_index = cp_est_val_and_coo[0][1].index(attacker_piece_coo)
        cp_est_val_and_coo[0][1][attacker_index] = target_piece_coo

        # betaとalphaを符号反転させて入れ替える
        real_score, spurious_score = nega(
            spurious_search(
                ii_state.next(action),
                -INFINITY,
                -INFINITY,
                -real_alpha,
                -spurious_alpha,
                0,
                max_depth,
                ev_func,
                [cp_est_val_and_coo[1], cp_est_val_and_coo[0]],
            )
        )
        if real_score > real_alpha:
            best_action = action
            real_alpha, spurious_alpha = real_score, spurious_score

    if best_action == -1:  # 全ての評価が-INFだった場合ランダムに行動（バグの握り潰しなので良くない）
        legal_actions = ii_state.legal_actions()
        return legal_actions[random.randint(0, len(legal_actions) - 1)]
    return best_action


# est_val_and_cooの視点を切り替える
def change_view(est_val_and_coo):
    change_view_evac = [
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    ]
    for i, player in enumerate(est_val_and_coo):
        for j, val_coo in enumerate(player):
            for k, _ in enumerate(val_coo):
                if j == 0:
                    change_view_evac[i][j][k] = est_val_and_coo[i][j][k]
                elif j == 1:
                    change_view_evac[i][j][k] = 19 - est_val_and_coo[i][j][k]
                else:
                    print("終わりや")
    return change_view_evac


# estの座標だけ変更する（推測値は変更しない）
def update_est_coo(action, est_val_and_coo, is_my_action):

    # deepcopyと同じ処理
    cp_est_val_and_coo = pickle.loads(pickle.dumps(est_val_and_coo, -1))

    bef_coo = int(action / 4)
    aft_coo = calc_coo_of_destination_from_action(action)
    if is_my_action:
        my = 0
        enemy = 1
    else:
        my = 1
        enemy = 0

    update_index = cp_est_val_and_coo[my][1].index(bef_coo)
    # 座標の更新
    cp_est_val_and_coo[my][1][update_index] = aft_coo
    if aft_coo in cp_est_val_and_coo[enemy][1]:  # 相手の駒を取っている場合
        dead_index = cp_est_val_and_coo[enemy][1].index(aft_coo)
        cp_est_val_and_coo[enemy][1][dead_index] = -1
    return cp_est_val_and_coo


##### 以下テスト用 #####

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
    # 状態の生成
    state = State()
    params = [
        [9, 5, 5, 3, 3, 2, 2, 1, 1, 0],
        [3, 5, 5, 9, 3, 5, 2, 3, 1, 2],
    ]
    ev_func = create_ev_func(params)
    # print(alpha_beta_action(state, ev_func, 2))
    est_val_and_coo = [
        [[0.5, 0.5, 0.5, 0.5], [13, 14, 17, 18]],
        [[0.5, 0.5, 0.5, 0.5], [1, 2, 5, 6]],
    ]
    random.seed(314)
    for i in range(100):
        spurious_search_action(state, ev_func, est_val_and_coo)

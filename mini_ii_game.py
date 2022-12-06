# Stateからアクセスできる情報を抽出したState（毎ターンstateから生成する）
# 用途はアルファベータ探索で使用するのみにとどめ、状態管理などはstateで行う
class AccessableState:
    def __init__(self):
        self.is_goal = False
        self.enemy_is_goal = False
        self.enemy_left_blue_piece = 0
        self.enemy_left_red_piece = 0
        self.my_left_blue_piece = 0
        self.my_left_red_piece = 0
        self.pieces = [0] * 20
        self.my_turn = True
        self.depth = 0

    def create_ii_state_from_state(self, state):
        self.is_goal = False
        self.enemy_is_goal = False

        # 残っている駒の情報（これは公開されている情報）
        self.enemy_left_blue_piece = 0
        self.enemy_left_red_piece = 0
        self.my_left_blue_piece = 0
        self.my_left_red_piece = 0

        # 駒の情報は1つのボードに集める
        self.pieces = [0] * 20
        for i, piece in enumerate(state.pieces):
            if piece != 0:
                if piece == 1:
                    self.my_left_blue_piece += 1
                elif piece == 2:
                    self.my_left_red_piece += 1
                self.pieces[i] = state.pieces[i]
        for i, piece in enumerate(state.enemy_pieces):
            if piece != 0:
                if piece == 1:
                    self.enemy_left_blue_piece += 1
                elif piece == 2:
                    self.enemy_left_red_piece += 1
                # 表示上は全て青駒とする
                self.pieces[19 - i] = -1
        self.my_turn = True
        self.depth = state.depth

    # 座標をトラッキングする都合で盤面をひっくり返す
    # （stateのnextに合わせて座標を反転させる）
    def create_star_ii_state_from_state(self, state, is_fsp_action):
        self.is_goal = False
        self.enemy_is_goal = False

        # 残っている駒の情報（これは公開されている情報）
        self.enemy_left_blue_piece = 0
        self.enemy_left_red_piece = 0
        self.my_left_blue_piece = 0
        self.my_left_red_piece = 0

        if is_fsp_action:
            my_piece = state.pieces
            enemy_pieces = state.enemy_pieces
            self.my_turn = True
        else:
            my_piece = state.enemy_pieces  # 常にmyにプレイヤ側の駒が入るようにする
            enemy_pieces = state.pieces
            self.my_turn = False

        # 駒の情報は1つのボードに集める
        self.pieces = [0] * 20
        for i, piece in enumerate(my_piece):
            if piece != 0:
                if piece == 1:
                    self.my_left_blue_piece += 1
                elif piece == 2:
                    self.my_left_red_piece += 1
                self.pieces[i] = my_piece[i]
        for i, piece in enumerate(enemy_pieces):
            if piece != 0:
                if piece == 1:
                    self.enemy_left_blue_piece += 1
                elif piece == 2:
                    self.enemy_left_red_piece += 1
                # 表示上は全て青駒とする
                self.pieces[19 - i] = -1
        self.depth = state.depth

    def overwrite_from_ii_state(self, ii_state):
        self.is_goal = ii_state.is_goal
        self.enemy_is_goal = ii_state.enemy_is_goal
        self.enemy_left_blue_piece = ii_state.enemy_left_blue_piece
        self.enemy_left_red_piece = ii_state.enemy_left_red_piece
        self.my_left_blue_piece = ii_state.my_left_blue_piece
        self.my_left_red_piece = ii_state.my_left_red_piece
        self.pieces = ii_state.pieces.copy()
        self.my_turn = ii_state.my_turn
        self.depth = ii_state.depth

    # 敗北
    def is_lose(self):
        return (
            self.my_left_blue_piece <= 0
            or self.enemy_left_red_piece <= 0
            or self.enemy_is_goal
        )

    # 勝利
    def is_win(self):
        return (
            self.enemy_left_blue_piece <= 0
            or self.my_left_red_piece <= 0
            or self.is_goal
        )

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    # 行動を駒の移動元と移動方向に変換
    def action_to_position(self, action):
        return (int(action / 4), action % 4)  # position,direction

    def legal_actions(self):
        actions = []
        if self.my_turn:  # 自分のターン
            for p in range(20):
                if self.pieces[p] in (1, 2):
                    # 自分の駒が存在するなら駒の位置を渡して、その駒の取れる行動をactionsに追加
                    actions.extend(self.legal_actions_pos(p))
            # 青駒のゴール行動は例外的に合法手リストに追加
            if self.pieces[0] == 1:
                actions.extend([2])  # 0*4 + 2
            if self.pieces[3] == 1:
                actions.extend([14])  # 3*4 + 2
        else:
            for p in range(20):
                if self.pieces[p] in (-1, -2):
                    actions.extend(self.enemy_legal_actions_pos(p))
            # ゴール行動は例外的に合法手リストに追加（相手の駒は全てゴール可能とする）
            if self.pieces[16] == -1:
                actions.extend([64])  # 16*4 + 0
            if self.pieces[19] == -1:
                actions.extend([76])  # 19*4 + 0
        return actions

    # 駒ごと(駒1つに着目した)の合法手のリストの取得
    def legal_actions_pos(self, position):
        actions = []
        x = position % 4
        y = int(position / 4)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        if y != 4 and self.pieces[position + 4] not in (1, 2):  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] not in (1, 2):  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 4] not in (1, 2):  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 3 and self.pieces[position + 1] not in (1, 2):  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 敵視点でのlegal_actions_pos
    def enemy_legal_actions_pos(self, position):
        actions = []
        x = position % 4
        y = int(position / 4)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        if y != 4 and self.pieces[position + 4] not in (-1, -2):  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] not in (-1, -2):  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 4] not in (-1, -2):  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 3 and self.pieces[position + 1] not in (-1, -2):  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 次の状態の取得
    def next(self, action):
        ii_state = AccessableState()
        ii_state.overwrite_from_ii_state(self)
        if ii_state.my_turn:  # 自分のターン
            # position_bef->移動前の駒の位置、position_aft->移動後の駒の位置
            # 行動を(移動元, 移動方向)に変換
            position_bef, direction = ii_state.action_to_position(action)

            # 合法手がくると仮定
            # 駒の移動(後ろに動くことは少ないかな？ + if文そんなに踏ませたくないな と思ったので判定を左右下上の順番にしてるけど意味あるのかは不明)
            if direction == 1:  # 左
                position_aft = position_bef - 1
            elif direction == 3:  # 右
                position_aft = position_bef + 1
            elif direction == 0:  # 下
                position_aft = position_bef + 4
            elif direction == 2:  # 上
                if position_bef == 0 or position_bef == 3:  # 0と5の上行動はゴール処理なので先に弾く
                    # ゴールした場合は特殊処理（駒を移動させない）
                    ii_state.is_goal = True
                    ii_state.my_turn = False
                    ii_state.depth += 1
                    return ii_state
                else:
                    position_aft = position_bef - 4
            else:
                print("error関数名:next")

            # 倒した駒を反映（倒した駒は全て赤駒として扱う）
            if ii_state.pieces[position_aft] != 0:
                ii_state.enemy_left_red_piece -= 1

            # 実際に駒移動
            ii_state.pieces[position_aft] = ii_state.pieces[position_bef]
            ii_state.pieces[position_bef] = 0

            ii_state.my_turn = False
            ii_state.depth += 1
            return ii_state

        else:  # 敵のターン
            position_bef, direction = ii_state.action_to_position(action)
            if direction == 1:  # 左
                position_aft = position_bef - 1
            elif direction == 3:  # 右
                position_aft = position_bef + 1
            elif direction == 0:  # 下
                if position_bef == 16 or position_bef == 19:  # 16と19の下行動はゴール処理なので先に弾く
                    # ゴールした場合は特殊処理（駒を移動させない）
                    ii_state.enemy_is_goal = True
                    ii_state.my_turn = True
                    ii_state.depth += 1
                    return ii_state
                else:
                    position_aft = position_bef + 4
            elif direction == 2:  # 上
                position_aft = position_bef - 4
            else:
                print("error関数名:next")

            # 倒した駒を反映
            if ii_state.pieces[position_aft] != 0:
                if ii_state.pieces[position_aft] == 1:
                    ii_state.my_left_blue_piece -= 1
                elif ii_state.pieces[position_aft] == 2:
                    ii_state.my_left_red_piece -= 1

            # 実際に駒移動
            ii_state.pieces[position_aft] = ii_state.pieces[position_bef]
            ii_state.pieces[position_bef] = 0

            ii_state.my_turn = True
            ii_state.depth += 1
            return ii_state

    # star_alpha_betaのチャンスノードにおいて、駒色を指定できるようにしたnext
    # ここではゴール行動の場合、赤駒かどうかを確認しない
    def specify_color_next(self, action, target_piece_coo, piece_color):
        ii_state = AccessableState()
        ii_state.overwrite_from_ii_state(self)
        if ii_state.my_turn:  # 自分のターン
            # position_bef->移動前の駒の位置、position_aft->移動後の駒の位置
            # 行動を(移動元, 移動方向)に変換
            position_bef, direction = ii_state.action_to_position(action)

            # 合法手がくると仮定
            # 駒の移動
            if direction == 1:  # 左
                position_aft = position_bef - 1
            elif direction == 3:  # 右
                position_aft = position_bef + 1
            elif direction == 0:  # 下
                position_aft = position_bef + 4
            elif direction == 2:  # 上 # ゴール処理はここに入らない
                position_aft = position_bef - 4
            else:
                print("error関数名:next")

            # 倒した駒を反映（色はpiece_colorを参照）
            if ii_state.pieces[position_aft] != 0:
                if position_aft == target_piece_coo:
                    if piece_color == 1:
                        ii_state.enemy_left_blue_piece -= 1
                    else:
                        ii_state.enemy_left_red_piece -= 1
                else:
                    print("very おかしいよ")

        else:  # 敵のターン
            position_bef, direction = ii_state.action_to_position(action)
            if direction == 1:  # 左
                position_aft = position_bef - 1
            elif direction == 3:  # 右
                position_aft = position_bef + 1
            elif direction == 0:  # 下 # ゴール処理はここに入らない
                position_aft = position_bef + 4
            elif direction == 2:  # 上
                position_aft = position_bef - 4
            else:
                print("error関数名:next")

            # 倒した駒を反映（色はpiece_colorを参照）
            if ii_state.pieces[position_aft] != 0:
                if position_aft == target_piece_coo:
                    if piece_color == 1:
                        ii_state.my_left_blue_piece -= 1
                    else:
                        ii_state.my_left_red_piece -= 1
                else:
                    print("very おかしいよ")

        # 実際に駒移動
        ii_state.pieces[position_aft] = ii_state.pieces[position_bef]
        ii_state.pieces[position_bef] = 0

        ii_state.depth += 1
        if ii_state.my_turn:
            ii_state.my_turn = False
        else:
            ii_state.my_turn = True
        return ii_state

    # 文字列表示
    def __str__(self):
        row = "|{}|{}|{}|{}|"
        hr = "\n---------------------\n"
        board_essence = []
        for i in self.pieces:
            if i == 1:
                board_essence.append("自青")
            elif i == 2:
                board_essence.append("自赤")
            elif i == -1:
                board_essence.append("敵駒")
            elif i == -2:
                board_essence.append("敵駒")
            else:
                board_essence.append("　　")

        str = (hr + row + hr + row + hr + row + hr + row + hr + row + hr).format(
            *board_essence
        )
        return str

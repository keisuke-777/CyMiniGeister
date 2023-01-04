# 生成した個体の優秀なものを、まとめてクラスタリングする
import numpy as np
import pandas as pd
import itertools, pyclustering, pylab, ast, random
from pyclustering.cluster import xmeans, kmeans
from matplotlib import pyplot as plt
from est_xmeans import *

# 読み込み
def read_parent(cycle: int, file_pass: str):
    folder_pass = file_pass + "/" + str(cycle) + "/eval_func_list.csv"
    str_sv_evals = pd.read_csv(folder_pass).values.tolist()
    fitness_list, eval_lists = [], []
    for index, _ in enumerate(str_sv_evals):
        fitness_list.append(str_sv_evals[index][0])
        eval_lists.append(ast.literal_eval(str_sv_evals[index][1]))
    return fitness_list, eval_lists


# 各世代の上位s個体をcsvから持ってくる
def read_s_eval(s: int, cycle_num: int, file_pass: str):
    elites = []
    for cycle in range(cycle_num):
        _, eval_lists = read_parent(cycle, file_pass)
        elites.append(eval_lists[:s])
    elites = list(itertools.chain.from_iterable(elites))
    return elites


# x-meansでクラスタリング
def by_xmeans(data_list: list):
    # pyclustering.utils.color.color.TITLES = ["blue", "orange", "green", "red", "purple"] # これしても色変わらん
    data = np.array(data_list)

    init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(
        data, CLUSTER_MIN
    ).initialize()
    xm = pyclustering.cluster.xmeans.xmeans(data, init_center, ccore=False)  # xmeans
    xm.process()
    clusters = xm.get_clusters()
    return clusters


# k-meansでクラスタリング
def by_kmeans(data_list: list):
    # pyclustering.utils.color.color.TITLES = ["blue", "orange", "green", "red", "purple"] # これしても色変わらん
    data = np.array(data_list)

    init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(
        data, CLUSTER_MIN
    ).initialize()
    xm = pyclustering.cluster.kmeans.kmeans(data, init_center, ccore=False)  # kmeans
    xm = pyclustering.cluster.kmeans.kmeans(data, init_center, ccore=False)  # kmeans
    xm = pyclustering.cluster.kmeans.kmeans(data, init_center, ccore=False)  # kmeans
    xm.process()
    clusters = xm.get_clusters()
    return clusters


def print_cluster(data_list, clusters):
    data = np.array(data_list)
    pyclustering.utils.draw_clusters(data, clusters, display_result=False)
    pylab.show()
    return clusters


# グラフの色をpyclusteringに揃える
# 参照:https://pyclustering.github.io/docs/0.8.2/html/d7/d96/utils_2____init_____8py_source.html#l00793
COLOR_LIST = [
    "red",
    "blue",
    "darkgreen",
    "brown",
    "violet",
    "deepskyblue",
    "darkgrey",
    "lightsalmon",
    "deeppink",
    "yellow",
    "black",
    "mediumspringgreen",
    "orange",
    "darkviolet",
    "darkblue",
    "silver",
    "lime",
    "pink",
    "gold",
    "bisque",
]

# 比率の遷移を計算して積み上げ折れ線でプロットする
def stackplot_ratio_transition(
    cluster_num: int, x: list, y: list, is_view_fig: bool, save_fig_pass: str = None
):
    plt.stackplot(x, y)  # 積上げ折れ線グラフの描画
    legend = ["cluster"] * cluster_num
    for i in range(cluster_num):
        legend[i] += str(i)
    plt.legend(legend)  # 凡例を設定
    if is_view_fig:
        plt.show()
    if save_fig_pass != None:
        plt.savefig(save_fig_pass)
        plt.close()


# 比率の遷移を計算して折れ線でプロットする
def plot_ratio_transition(
    cluster_num: int, x: list, y: list, is_view_fig: bool, save_fig_pass: str = None
):
    alpha_list = ["A", "B", "C", "D", "E"]
    for i in range(len(y)):
        plt.plot(x, y[i], color=COLOR_LIST[i], label="Cluster " + alpha_list[i], lw=3)
    legend = ["Cluster "] * cluster_num
    for i in range(cluster_num):
        legend[i] += alpha_list[i]
    plt.legend(
        legend, fontsize=28, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=3
    )  # 凡例を設定
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlabel("Number of agent generations", fontsize=26)
    plt.ylabel("Number of elite agents", fontsize=26)
    if is_view_fig:
        plt.show()
    if save_fig_pass != None:
        plt.savefig(save_fig_pass)
        plt.close()


# クラスタの比率の遷移をlistで表現する
def stack_ratio_transition(clusters: list, elites_num: int, cycle_num: int):
    x = list(range(cycle_num))
    y = [[0 for k in range(cycle_num)] for j in range(len(clusters))]

    for clus_id, order_num_list in enumerate(clusters):
        for order_num in order_num_list:
            hoge = int(order_num / elites_num)
            y[clus_id][hoge] += 1
    return x, y


# エリートを対戦により評価する
def buttle_double(params: list, enemy_params: list, enemy_agent=None) -> float:
    # return random.uniform(0, 1)  # テスト用
    # paramsから評価関数を作成
    ev_func = create_ev_func(params)
    # 対戦相手の評価関数を生成
    enemy_ev_func = create_ev_func(enemy_params)
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
            action = spurious_search_action(
                state, ev_func, estimate.fst_est_val_and_coo, 3
            )
            estimate.double_update_from_action(action, state, True)
        else:
            if enemy_agent == None:
                action = spurious_search_action(
                    state, enemy_ev_func, estimate.sec_est_val_and_coo, 3,
                )
            else:
                action = enemy_agent(state)
            estimate.double_update_from_action(action, state, False)
        state = state.next(action)

    # 先手後手を入れ替えて対戦（盤面はそのまま）
    # FIXME:これだと盤面そのままじゃない
    state = State()
    estimate = Estimate(enemy_params[2], params[2])  # 推測を司る
    while True:
        if state.is_done():
            if state.is_lose():
                if state.depth % 2 == 0:
                    num_of_wins += 1  # 後手勝ち
            else:
                num_of_wins += 0.5
            break

        # 行動の取得
        if state.is_first_player():
            if enemy_agent == None:
                action = spurious_search_action(
                    state, enemy_ev_func, estimate.fst_est_val_and_coo, 3,
                )
            else:
                action = enemy_agent(state)
            estimate.double_update_from_action(action, state, True)
        else:
            action = spurious_search_action(
                state, ev_func, estimate.sec_est_val_and_coo, 3
            )
            estimate.double_update_from_action(action, state, False)
        state = state.next(action)

    return num_of_wins


# estの3要素->評価関数
def create_ev_func_from_three_est(three_est: list) -> list:
    est_table = three_est + [0, 0]
    ev_func = [BLUE_TABLE, RED_TABLE, est_table]
    return ev_func


def eval_tuple(elites: list, ev_index_1: list, ev_index_2: list):
    winning_rate = 0.0
    for i in range(500):
        winning_rate += buttle_double(
            create_ev_func_from_three_est(elites[random.choice(ev_index_1)]),
            create_ev_func_from_three_est(elites[random.choice(ev_index_2)]),
        )
    return winning_rate


# クラスター同士で対戦して結果を出力
def buttle_cluster_against_cluster(clusters: list, elites: list):
    l = list(range(len(clusters)))
    vs_result = np.zeros((len(l), len(l)))
    vs_tuples = itertools.combinations(l, 2)
    for vs_tp in vs_tuples:
        result = eval_tuple(elites, clusters[vs_tp[0]], clusters[vs_tp[1]])
        vs_result[vs_tp[0]][vs_tp[1]] = result
        vs_result[vs_tp[1]][vs_tp[0]] = 1000 - result
    return vs_result


from spurious_mcts import spurious_mcts_action


def evaluate_elite(
    params: list, enemy_params: list, search_func_name="alpha-beta"
) -> float:
    # return random.uniform(0, 1)  # テスト用
    # paramsから評価関数を作成
    ev_func = create_ev_func(params)
    # 対戦相手の評価関数を生成
    enemy_ev_func = create_ev_func(enemy_params)

    buttle_piece_lists = create_buttle_piece_lists()

    num_of_wins = 0.0
    num_of_matches = 0.0

    # if enemy_name == "alpha-beta":

    # kとjの部分：buttle_piece_listsに入っているパターンから重複を許して2つ選ぶ
    for i in buttle_piece_lists:
        mirror_i = invert_piece_list(i)
        # 通常のiと、iを左右反転させるパターンを用意する
        for k in [i, mirror_i]:
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
                    if search_func_name == "alpha-beta":
                        if state.is_first_player():
                            action = spurious_search_action(
                                state, ev_func, estimate.fst_est_val_and_coo, 3
                            )
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = spurious_search_action(
                                state, enemy_ev_func, estimate.sec_est_val_and_coo, 3,
                            )
                            estimate.double_update_from_action(action, state, False)
                    elif search_func_name == "mcts":
                        if state.is_first_player():
                            action = spurious_mcts_action(
                                state, estimate.fst_est_val_and_coo, 30000
                            )
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = spurious_mcts_action(
                                state, estimate.fst_est_val_and_coo, 30000
                            )
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
                    if search_func_name == "alpha-beta":
                        if state.is_first_player():
                            action = spurious_search_action(
                                state, ev_func, estimate.fst_est_val_and_coo, 3
                            )
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = spurious_search_action(
                                state, enemy_ev_func, estimate.sec_est_val_and_coo, 3,
                            )
                            estimate.double_update_from_action(action, state, False)
                    elif search_func_name == "mcts":
                        if state.is_first_player():
                            action = spurious_mcts_action(
                                state, estimate.fst_est_val_and_coo, 30000
                            )
                            estimate.double_update_from_action(action, state, True)
                        else:
                            action = spurious_mcts_action(
                                state, estimate.fst_est_val_and_coo, 30000
                            )
                            estimate.double_update_from_action(action, state, False)
                    state = state.next(action)

    winning_rate = num_of_wins / num_of_matches
    return winning_rate


# パラメータの平均をとる
def culc_ave_cluster(clusters: list, elites: list) -> list:
    # パラメータ群を1つの配列に集める
    clparam_lists = [[] for _ in range(len(clusters))]
    for i, cluster in enumerate(clusters):
        for cl_index in cluster:
            clparam_lists[i].append(elites[cl_index])

    # パラメータの平均をとる（どうせ１回しか使わないので乱暴）
    ave_params = []
    for clparam_list in clparam_lists:
        sum = [0.0] * 6
        for clparam in clparam_list:
            sum[0] += clparam[0]
            sum[1] += clparam[1]
            sum[2] += clparam[2]
            sum[3] += clparam[3]
            sum[4] += clparam[4]
            sum[5] += clparam[5]
        sum[0] /= len(clparam_list)
        sum[1] /= len(clparam_list)
        sum[2] /= len(clparam_list)
        sum[3] /= len(clparam_list)
        sum[4] /= len(clparam_list)
        sum[5] /= len(clparam_list)
        ave_params.append(create_ev_func_from_three_est(sum))

    print("ave:", ave_params)
    return ave_params


# クラスター内でパラメータの平均をとって、クラスター間の対戦を実施
def buttle_avecl_against_avecl_with_ab(
    clusters: list, elites: list, search_func_name: str
):
    # パラメータの平均を計算
    ave_params = culc_ave_cluster(clusters, elites)

    # 対戦
    l = list(range(len(ave_params)))
    vs_result = np.zeros((len(l), len(l)))
    vs_tuples = itertools.combinations(l, 2)
    for vs_tp in vs_tuples:
        result = evaluate_elite(
            ave_params[vs_tp[0]], ave_params[vs_tp[1]], search_func_name
        )
        vs_result[vs_tp[0]][vs_tp[1]] = result
        vs_result[vs_tp[1]][vs_tp[0]] = 1 - result
    return vs_result


# クラスター内でパラメータの平均をとって、クラスター間の対戦を実施
def buttle_avecl_against_avecl_with_mcts(
    clusters: list, elites: list, search_func_name: str
):
    # パラメータの平均を計算
    ave_params = culc_ave_cluster(clusters, elites)

    # 対戦
    l = list(range(len(ave_params)))
    vs_result = np.zeros((len(l), len(l)))
    vs_tuples = itertools.combinations(l, 2)
    for vs_tp in vs_tuples:
        result = evaluate_elite(
            ave_params[vs_tp[0]], ave_params[vs_tp[1]], search_func_name
        )
        vs_result[vs_tp[0]][vs_tp[1]] = result
        vs_result[vs_tp[1]][vs_tp[0]] = 1 - result
    return vs_result


CLUSTER_MIN = 5

# 動作確認
if __name__ == "__main__":

    seed = int(random.random() * 10000)
    print(seed)
    random.seed(seed)
    np.random.seed(seed)

    folder_name = "2022-12-21"
    # file_pass = "csv_data/dim6/SpuriousMCTS/" + folder_name + "/"
    file_pass = "csv_data/dim6/SpuriousSearch/" + folder_name + "/"
    xmeans_save_fig_pass = file_pass + str(seed) + "xmeans.png"
    kmeans_save_fig_pass = file_pass + str(seed) + "kmeans.png"
    ratio_save_fig_pass = file_pass + str(seed) + "ratio.png"
    ELITES_NUM = 30
    CYCLE_NUM = 50

    elites = read_s_eval(ELITES_NUM, CYCLE_NUM, file_pass)
    clusters = by_kmeans(elites)
    # clusters = by_xmeans(elites)

    print(elites, clusters)

    # print(
    #     "clusters:\n",
    #     clusters,
    #     "\nelites:\n",
    #     elites,
    #     file=codecs.open(file_pass + str(seed) + "data.txt", "w", "utf-8"),
    # )

    x, y = stack_ratio_transition(clusters, ELITES_NUM, CYCLE_NUM)
    plot_ratio_transition(len(clusters), x, y, True)
    # plot_ratio_transition(len(clusters), x, y, False, ratio_save_fig_pass)

    # 対戦が必要な場合
    # vs_result = buttle_cluster_against_cluster(clusters, elites)
    # vs_result = buttle_avecl_against_avecl_with_ab(clusters, elites, "alpha-beta")
    vs_result = buttle_avecl_against_avecl_with_mcts(clusters, elites, "mcts")
    print(vs_result)

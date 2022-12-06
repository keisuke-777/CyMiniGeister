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
def by_xmeans(data_list: list, is_view_fig: bool = False, save_fig_pass: bool = None):
    # pyclustering.utils.color.color.TITLES = ["blue", "orange", "green", "red", "purple"] # これしても色変わらん
    data = np.array(data_list)

    init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(
        data, CLUSTER_MIN
    ).initialize()
    xm = pyclustering.cluster.xmeans.xmeans(data, init_center, ccore=False)  # xmeans
    xm.process()
    clusters = xm.get_clusters()
    pyclustering.utils.draw_clusters(data, clusters, display_result=False)
    if is_view_fig:
        pylab.show()
    if save_fig_pass != None:
        pylab.savefig(save_fig_pass)
        plt.close()
    return clusters


# k-meansでクラスタリング
def by_kmeans(data_list: list, is_view_fig: bool = False, save_fig_pass: bool = None):
    # pyclustering.utils.color.color.TITLES = ["blue", "orange", "green", "red", "purple"] # これしても色変わらん
    data = np.array(data_list)

    init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(
        data, CLUSTER_MIN
    ).initialize()
    xm = pyclustering.cluster.kmeans.kmeans(data, init_center, ccore=False)  # kmeans
    xm.process()
    clusters = xm.get_clusters()
    pyclustering.utils.draw_clusters(data, clusters, display_result=False)
    if is_view_fig:
        pylab.show()
    if save_fig_pass != None:
        pylab.savefig(save_fig_pass)
        plt.close()
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


def evaluate_elite(params: list, enemy_params: list, enemy_agent=None) -> float:
    # return random.uniform(0, 1)  # テスト用
    # paramsから評価関数を作成
    ev_func = create_ev_func(params)
    # 対戦相手の評価関数を生成
    enemy_ev_func = create_ev_func(enemy_params)

    buttle_piece_lists = create_buttle_piece_lists()

    num_of_wins = 0.0
    num_of_matches = 0.0

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

    winning_rate = num_of_wins / num_of_matches
    return winning_rate


# クラスター内でパラメータの平均をとって、クラスター間の対戦を実施
def buttle_avecl_against_avecl(clusters: list, elites: list):
    # パラメータ群を1つの配列に集める
    clparam_lists = [[] for _ in range(len(clusters))]
    for i, cluster in enumerate(clusters):
        for cl_index in cluster:
            clparam_lists[i].append(elites[cl_index])

    # パラメータの平均をとる（どうせ１回しか使わないので乱暴）
    ave_params = []
    for clparam_list in clparam_lists:
        sum = [0.0] * 3
        for clparam in clparam_list:
            sum[0] += clparam[0]
            sum[1] += clparam[1]
            sum[2] += clparam[2]
        sum[0] /= len(clparam_list)
        sum[1] /= len(clparam_list)
        sum[2] /= len(clparam_list)
        ave_params.append(create_ev_func_from_three_est(sum))

    print("ave:", ave_params)

    # 対戦
    l = list(range(len(ave_params)))
    vs_result = np.zeros((len(l), len(l)))
    vs_tuples = itertools.combinations(l, 2)
    for vs_tp in vs_tuples:
        result = evaluate_elite(ave_params[vs_tp[0]], ave_params[vs_tp[1]])
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

    folder_name = "xNEAT2022-11-03"
    file_pass = "csv_data/SpuriousSearch/" + folder_name + "/"
    xmeans_save_fig_pass = file_pass + str(seed) + "xmeans.png"
    kmeans_save_fig_pass = file_pass + str(seed) + "kmeans.png"
    ratio_save_fig_pass = file_pass + str(seed) + "ratio.png"
    ELITES_NUM = 30
    CYCLE_NUM = 50

    elites = read_s_eval(ELITES_NUM, CYCLE_NUM, file_pass)
    clusters = by_kmeans(elites, True)
    # clusters = by_kmeans(elites, False, kmeans_save_fig_pass)
    # clusters = by_xmeans(elites, True)
    # clusters = by_xmeans(elites, False, xmeans_save_fig_pass)

    clusters = [
        [
            1,
            2,
            6,
            7,
            14,
            16,
            21,
            31,
            35,
            40,
            42,
            43,
            44,
            60,
            67,
            75,
            76,
            78,
            81,
            82,
            85,
            86,
            94,
            100,
            108,
            114,
            118,
            119,
            127,
            136,
            141,
            150,
            163,
            165,
            184,
            193,
            196,
            204,
            205,
            209,
            211,
            216,
            220,
            238,
            243,
            251,
            252,
            261,
            288,
            293,
            304,
            319,
            320,
            325,
            332,
            333,
            334,
            349,
            360,
            381,
            386,
            387,
            388,
            389,
            390,
            391,
            398,
            419,
            435,
            444,
            445,
            446,
            447,
            494,
            498,
            503,
            510,
            511,
            513,
            539,
            540,
            541,
            542,
            543,
            545,
            546,
            547,
            551,
            553,
            556,
            557,
            565,
            571,
            572,
            576,
            577,
            578,
            580,
            581,
            586,
            587,
            588,
            589,
            594,
            595,
            601,
            616,
            617,
            619,
            620,
            629,
            690,
            692,
            707,
            723,
            728,
            729,
            752,
            769,
            770,
            772,
            773,
            782,
            789,
            807,
            809,
            822,
            867,
            871,
            872,
            875,
            876,
            877,
            878,
            903,
            907,
            917,
            918,
            920,
            922,
            923,
            929,
            934,
            935,
            957,
            958,
            959,
            985,
            990,
            1030,
            1110,
            1111,
            1114,
            1126,
            1128,
            1145,
            1150,
            1187,
            1191,
            1193,
            1200,
            1205,
            1213,
            1246,
            1247,
            1320,
            1321,
            1323,
            1325,
            1353,
            1355,
            1356,
            1366,
            1367,
            1368,
            1371,
            1372,
            1373,
            1376,
            1399,
            1400,
            1401,
            1402,
            1404,
            1405,
            1406,
            1407,
            1408,
            1409,
            1417,
            1418,
            1419,
            1420,
            1421,
            1422,
            1432,
            1436,
            1437,
            1438,
            1439,
            1440,
            1441,
            1442,
            1443,
            1444,
            1445,
            1446,
            1450,
            1451,
            1452,
            1453,
            1454,
            1455,
            1456,
            1457,
            1458,
            1473,
            1474,
            1496,
            1497,
            1499,
        ],
        [
            0,
            3,
            4,
            5,
            34,
            39,
            59,
            62,
            63,
            66,
            70,
            90,
            91,
            92,
            93,
            97,
            99,
            103,
            120,
            121,
            123,
            125,
            126,
            135,
            137,
            152,
            153,
            155,
            164,
            180,
            183,
            186,
            202,
            203,
            212,
            213,
            214,
            217,
            219,
            229,
            240,
            241,
            242,
            253,
            260,
            270,
            271,
            273,
            279,
            281,
            282,
            290,
            301,
            302,
            303,
            317,
            318,
            338,
            339,
            353,
            354,
            368,
            369,
            378,
            379,
            383,
            392,
            397,
            400,
            401,
            402,
            416,
            417,
            418,
            420,
            421,
            422,
            423,
            432,
            441,
            442,
            443,
            448,
            453,
            470,
            471,
            472,
            478,
            479,
            484,
            487,
            489,
            496,
            514,
            544,
            550,
            564,
            570,
            573,
            574,
            575,
            590,
            598,
            599,
            600,
            602,
            603,
            604,
            606,
            607,
            608,
            609,
            610,
            611,
            630,
            632,
            633,
            636,
            637,
            638,
            639,
            640,
            641,
            654,
            655,
            658,
            661,
            672,
            673,
            674,
            685,
            686,
            687,
            689,
            691,
            693,
            709,
            721,
            722,
            730,
            731,
            732,
            744,
            751,
            753,
            758,
            759,
            760,
            768,
            771,
            780,
            781,
            783,
            786,
            787,
            788,
            806,
            808,
            810,
            811,
            814,
            816,
            817,
            818,
            819,
            820,
            821,
            831,
            833,
            843,
            844,
            845,
            857,
            858,
            859,
            881,
            882,
            883,
            884,
            885,
            887,
            900,
            901,
            902,
            904,
            905,
            908,
            919,
            930,
            931,
            932,
            933,
            940,
            941,
            942,
            943,
            944,
            945,
            946,
            956,
            960,
            961,
            964,
            965,
            966,
            967,
            979,
            980,
            981,
            982,
            983,
            984,
            991,
            992,
            993,
            998,
            999,
            1000,
            1001,
            1002,
            1020,
            1021,
            1022,
            1023,
            1024,
            1025,
            1026,
            1029,
            1032,
            1055,
            1056,
            1057,
            1058,
            1072,
            1074,
            1075,
            1076,
            1083,
            1086,
            1087,
            1113,
            1115,
            1116,
            1117,
            1119,
            1120,
            1121,
            1122,
            1138,
            1140,
            1141,
            1142,
            1143,
            1144,
            1146,
            1147,
            1148,
            1149,
            1151,
            1153,
            1154,
            1157,
            1160,
            1163,
            1170,
            1172,
            1173,
            1174,
            1175,
            1176,
            1177,
            1184,
            1185,
            1186,
            1188,
            1189,
            1192,
            1201,
            1202,
            1203,
            1204,
            1206,
            1212,
            1225,
            1226,
            1227,
            1228,
            1229,
            1230,
            1231,
            1232,
            1233,
            1234,
            1235,
            1243,
            1244,
            1245,
            1260,
            1261,
            1262,
            1270,
            1271,
            1272,
            1273,
            1280,
            1281,
            1282,
            1292,
            1295,
            1296,
            1297,
            1315,
            1316,
            1317,
            1319,
            1324,
            1347,
            1348,
            1350,
            1351,
            1352,
            1354,
            1364,
            1365,
            1380,
            1381,
            1383,
            1389,
            1390,
            1415,
            1416,
            1431,
            1434,
            1435,
            1493,
            1494,
            1495,
            1498,
        ],
        [
            8,
            9,
            11,
            13,
            15,
            18,
            19,
            20,
            22,
            24,
            25,
            26,
            27,
            28,
            30,
            32,
            33,
            36,
            37,
            38,
            41,
            45,
            46,
            47,
            48,
            51,
            56,
            57,
            61,
            64,
            65,
            68,
            69,
            74,
            77,
            79,
            80,
            84,
            88,
            95,
            96,
            98,
            102,
            104,
            105,
            106,
            107,
            109,
            115,
            124,
            129,
            130,
            131,
            138,
            139,
            140,
            142,
            147,
            148,
            154,
            169,
            172,
            173,
            181,
            195,
            208,
            210,
            215,
            218,
            222,
            227,
            228,
            231,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            255,
            257,
            258,
            262,
            263,
            264,
            280,
            284,
            285,
            286,
            287,
            291,
            294,
            297,
            305,
            306,
            322,
            323,
            327,
            328,
            345,
            346,
            347,
            365,
            366,
            367,
            380,
            385,
            393,
            413,
            431,
            449,
            450,
            451,
            452,
            481,
            492,
            495,
            502,
            538,
            548,
            549,
            552,
            579,
            583,
            591,
            593,
            596,
            597,
            605,
            614,
            618,
            622,
            625,
            626,
            628,
            631,
            635,
            649,
            650,
            651,
            656,
            657,
            659,
            666,
            671,
            678,
            679,
            680,
            681,
            688,
            708,
            715,
            873,
            879,
            906,
            915,
            916,
            1073,
            1084,
            1085,
            1088,
            1112,
            1118,
            1123,
            1134,
            1156,
            1161,
            1166,
            1169,
            1196,
            1198,
            1252,
            1318,
            1370,
            1378,
            1427,
            1447,
            1449,
            1463,
            1464,
            1470,
            1471,
            1480,
            1481,
            1482,
            1483,
            1484,
            1485,
            1490,
            1491,
            1492,
        ],
        [
            10,
            12,
            17,
            23,
            49,
            50,
            52,
            53,
            55,
            71,
            72,
            73,
            83,
            101,
            111,
            112,
            113,
            128,
            143,
            144,
            145,
            146,
            156,
            157,
            158,
            159,
            166,
            167,
            168,
            170,
            171,
            175,
            187,
            188,
            189,
            190,
            191,
            192,
            194,
            206,
            207,
            221,
            230,
            232,
            233,
            234,
            235,
            236,
            237,
            239,
            254,
            256,
            265,
            266,
            267,
            268,
            269,
            274,
            275,
            276,
            277,
            278,
            283,
            289,
            292,
            295,
            296,
            299,
            307,
            308,
            309,
            310,
            311,
            312,
            313,
            314,
            315,
            321,
            324,
            329,
            340,
            341,
            342,
            343,
            344,
            361,
            362,
            363,
            364,
            370,
            371,
            372,
            373,
            374,
            375,
            376,
            377,
            382,
            384,
            394,
            395,
            396,
            399,
            403,
            404,
            405,
            406,
            407,
            408,
            409,
            410,
            411,
            412,
            424,
            425,
            426,
            427,
            428,
            429,
            433,
            434,
            436,
            437,
            438,
            439,
            454,
            455,
            456,
            457,
            458,
            459,
            460,
            461,
            462,
            463,
            464,
            465,
            466,
            467,
            468,
            469,
            473,
            474,
            475,
            476,
            480,
            485,
            490,
            491,
            497,
            499,
            500,
            504,
            505,
            506,
            507,
            508,
            509,
            512,
            515,
            516,
            517,
            518,
            519,
            520,
            521,
            522,
            523,
            524,
            525,
            526,
            527,
            528,
            529,
            530,
            531,
            532,
            533,
            534,
            535,
            536,
            537,
            554,
            555,
            558,
            559,
            560,
            561,
            562,
            566,
            567,
            568,
            569,
            582,
            584,
            592,
            612,
            613,
            621,
            623,
            624,
            634,
            642,
            643,
            644,
            645,
            646,
            647,
            648,
            662,
            663,
            664,
            665,
            667,
            675,
            676,
            677,
            694,
            695,
            696,
            697,
            698,
            699,
            700,
            701,
            702,
            703,
            704,
            710,
            711,
            712,
            713,
            714,
            716,
            717,
            718,
            719,
            720,
            724,
            733,
            734,
            735,
            736,
            737,
            738,
            739,
            740,
            741,
            742,
            745,
            746,
            747,
            748,
            749,
            761,
            762,
            763,
            764,
            765,
            766,
            774,
            775,
            776,
            777,
            778,
            779,
            790,
            791,
            792,
            793,
            794,
            795,
            796,
            797,
            798,
            799,
            823,
            824,
            834,
            835,
            836,
            837,
            838,
            839,
            846,
            847,
            860,
            861,
            862,
            863,
            864,
            865,
            866,
            868,
            886,
            888,
            889,
            890,
            891,
            892,
            893,
            894,
            895,
            896,
            897,
            898,
            899,
            909,
            910,
            911,
            912,
            913,
            914,
            921,
            924,
            925,
            926,
            927,
            928,
            936,
            937,
            947,
            948,
            949,
            950,
            951,
            952,
            953,
            954,
            955,
            962,
            963,
            968,
            969,
            970,
            971,
            972,
            973,
            974,
            975,
            976,
            986,
            987,
            988,
            989,
            994,
            995,
            996,
            997,
            1003,
            1004,
            1005,
            1006,
            1007,
            1008,
            1009,
            1010,
            1011,
            1012,
            1013,
            1014,
            1015,
            1016,
            1017,
            1018,
            1019,
            1027,
            1031,
            1033,
            1034,
            1035,
            1036,
            1038,
            1039,
            1040,
            1041,
            1042,
            1043,
            1044,
            1045,
            1046,
            1047,
            1048,
            1049,
            1051,
            1052,
            1053,
            1059,
            1060,
            1061,
            1062,
            1063,
            1064,
            1065,
            1066,
            1067,
            1068,
            1069,
            1070,
            1071,
            1077,
            1078,
            1079,
            1080,
            1081,
            1082,
            1089,
            1090,
            1091,
            1092,
            1093,
            1094,
            1095,
            1096,
            1097,
            1098,
            1099,
            1100,
            1101,
            1102,
            1103,
            1104,
            1105,
            1106,
            1107,
            1108,
            1109,
            1124,
            1125,
            1127,
            1129,
            1130,
            1131,
            1139,
            1152,
            1155,
            1162,
            1164,
            1165,
            1178,
            1179,
            1180,
            1190,
            1194,
            1195,
            1197,
            1199,
            1207,
            1208,
            1209,
            1210,
            1214,
            1215,
            1216,
            1217,
            1218,
            1219,
            1220,
            1236,
            1237,
            1238,
            1239,
            1240,
            1241,
            1248,
            1249,
            1250,
            1251,
            1253,
            1254,
            1263,
            1264,
            1274,
            1275,
            1276,
            1283,
            1284,
            1285,
            1286,
            1287,
            1288,
            1289,
            1293,
            1294,
            1298,
            1299,
            1300,
            1301,
            1302,
            1303,
            1304,
            1305,
            1306,
            1307,
            1326,
            1327,
            1328,
            1329,
            1330,
            1331,
            1332,
            1349,
            1357,
            1358,
            1359,
            1360,
            1361,
            1369,
            1374,
            1375,
            1377,
            1384,
            1385,
            1386,
            1387,
            1391,
            1392,
            1393,
            1394,
            1395,
            1396,
            1397,
            1403,
            1423,
            1424,
            1425,
            1426,
            1459,
            1460,
            1461,
            1462,
            1472,
            1475,
            1476,
            1477,
            1478,
            1479,
            1486,
            1487,
            1488,
            1489,
        ],
        [
            29,
            54,
            58,
            87,
            89,
            110,
            116,
            117,
            122,
            132,
            133,
            134,
            149,
            151,
            160,
            161,
            162,
            174,
            176,
            177,
            178,
            179,
            182,
            185,
            197,
            198,
            199,
            200,
            201,
            223,
            224,
            225,
            226,
            259,
            272,
            298,
            300,
            316,
            326,
            330,
            331,
            335,
            336,
            337,
            348,
            350,
            351,
            352,
            355,
            356,
            357,
            358,
            359,
            414,
            415,
            430,
            440,
            477,
            482,
            483,
            486,
            488,
            493,
            501,
            563,
            585,
            615,
            627,
            652,
            653,
            660,
            668,
            669,
            670,
            682,
            683,
            684,
            705,
            706,
            725,
            726,
            727,
            743,
            750,
            754,
            755,
            756,
            757,
            767,
            784,
            785,
            800,
            801,
            802,
            803,
            804,
            805,
            812,
            813,
            815,
            825,
            826,
            827,
            828,
            829,
            830,
            832,
            840,
            841,
            842,
            848,
            849,
            850,
            851,
            852,
            853,
            854,
            855,
            856,
            869,
            870,
            874,
            880,
            938,
            939,
            977,
            978,
            1028,
            1037,
            1050,
            1054,
            1132,
            1133,
            1135,
            1136,
            1137,
            1158,
            1159,
            1167,
            1168,
            1171,
            1181,
            1182,
            1183,
            1211,
            1221,
            1222,
            1223,
            1224,
            1242,
            1255,
            1256,
            1257,
            1258,
            1259,
            1265,
            1266,
            1267,
            1268,
            1269,
            1277,
            1278,
            1279,
            1290,
            1291,
            1308,
            1309,
            1310,
            1311,
            1312,
            1313,
            1314,
            1322,
            1333,
            1334,
            1335,
            1336,
            1337,
            1338,
            1339,
            1340,
            1341,
            1342,
            1343,
            1344,
            1345,
            1346,
            1362,
            1363,
            1379,
            1382,
            1388,
            1398,
            1410,
            1411,
            1412,
            1413,
            1414,
            1428,
            1429,
            1430,
            1433,
            1448,
            1465,
            1466,
            1467,
            1468,
            1469,
        ],
    ]

    print_cluster(elites, clusters)

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
    # vs_result = buttle_avecl_against_avecl(clusters, elites)
    # print(vs_result)

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys


def year2gen(years, gtime):
    return [[s / gtime, e / gtime, N] for s, e, N in years]


def year2gen_sel(years, gtime):
    return [[ts / gtime, te / gtime, s, h] for ts, te, s, h in years]


def gen2unit4N(demography_in_gen, n0):
    return [[s / 4 / n0, e / 4 / n0, n / n0] for s, e, n in demography_in_gen]


def gen2unit4Nsel(sel_in_gen, n0):
    return [[ts / 4 / n0, te / 4 / n0, s, h] for ts, te, s, h in sel_in_gen]


def check_demography(demography):
    ts = 0
    prev = []
    for i in demography:

        # check if demography is continuous
        if i[0] != ts:
            print('Error: demographic phases should be continuous and seamless!')
            print('t_end(i) must be equal to t_start(i+1)')
            print('\t', prev)
            print('\t', i)
            sys.exit(0)
        ts = i[1]

        prev = i
        # print(i)

    # append the last (oldest) phase
    if demography[-1][1] < 999:
        s, e, n = demography[-1]
        demography.append([e, 999., n])

    # print('seems OK')
    return demography


def check_sel_history(history):
    ts = 0
    prev = []
    for i in history:

        # check if demography is continuous
        if i[0] != ts:
            print('Error: demographic phases should be continuous and seamless!')
            print('t_end(i) must be equal to t_start(i+1)')
            print('\t', prev)
            print('\t', i)
            sys.exit(0)
        ts = i[1]

        prev = i
        # print(i)

    # print('seems OK')
    return history


def split_history(demography, ts):
    # time when selection starts
    selected = 1

    # initialize
    under_sel = []
    under_neu = []

    for s, e, n in demography:

        # trajectory under selection
        if selected == 1:
            if ts < e:
                under_sel.append([s, ts, n])
                under_neu.append([ts, e, n])
                selected = 0
            else:
                under_sel.append([s, e, n])

        # neutral trajectory
        else:
            under_neu.append([s, e, n])

    return (under_sel, under_neu)


def freq_change(q, s=0, h=0.5, N=1000, N0=1000, resolution=0.01):
    """ Simulate frequency change forward in time
    -------------
    AA   Aa    aa
     1  1+2hs 1+2s
    p^2  2pq   q^2
    -------------

    Args:
        q (float): freq of derived allele
        s (float): selection coefficient
        h (float): dominant coefficient
        N (int): diploid population size (N)
        N0 (int): reference population size
        resolution: range to record the freq change
    Return:
        delta_t (float): time until freq changes
        freq (float): freq of the derived allele after delta_t
    """

    # initial conditions
    q0 = q
    t = 0

    # simulate until a specific amount of frequency change is observed
    # 頻度がresolutionの幅を超えたら記録する
    while int(q / resolution) == int(q0 / resolution):

        # calc freq
        p = 1 - q

        # expected freq change of 'a'
        # 頻度の増分の期待値
        delta_q = 2 * s * p * q * (q + h * (p - q))

        # when expected freq is less than 0 or more than 1,
        #  adjust next q, q=0 or 1
        if q + delta_q < 0:
            q_exp = 0

        elif q + delta_q < 1:
            q_exp = q + delta_q
        else:
            q_exp = 1

        # number of derived alleles in the next generation
        # 次世代のderivedアリル数
        derived = np.random.binomial(n=(2 * N), p=q_exp)

        # frequency of the mutant in the next generation
        # 次世代の頻度
        q = derived / (2 * N)
        t += 1

        # when fixed or lost, stop simulation
        if derived == 0 or derived == 2 * N:
            break

    # 頻度変化がresolution幅を超えるまでの時間と
    # 超えたあとの頻度を返す
    return t / 4 / N0, q


def backward_trajectory(demography, t, p, N0, resolution):
    # current time
    tc = t
    p0 = p
    pp = p0
    trajectory = []
    common_ancestor = 0

    # 時間をさかのぼるので、単一の変異に由来するはず
    # 頻度が 0に吸収される場合だけを考える
    # 頻度が 1になった場合は祖先型由来を意味するので、やりなおし
    while not common_ancestor:

        # phaseの情報を取得する
        for ts, te, f in demography:

            # 既に祖先にたどり着いている場合は頻度0
            # この場合はphase情報を頻度0でそのまま記録
            if common_ancestor:
                trajectory.append([ts, te, f, pp])

                # 次のphaseへ移動
                tc = te

            # phase内で頻度変化をシミュレートする
            while tc < te:

                # シミュレーション
                # delta_t はresolution以上の頻度変化を起こすまでの時間
                # p は変化したあとの頻度。　pp -> p
                delta_t, p = freq_change(pp, 0, 0.5, f * N0, N0, resolution)
                #print(p)
                # 祖先形由来になった場合はやり直し
                if 1 <= p:
                    trajectory = []
                    tc = t
                    pp = p0
                    break

                # 頻度変化が phase内で完結していれば記録する
                if tc + delta_t < te:

                    trajectory.append([tc, tc + delta_t, f, pp])
                    tc += delta_t
                    pp = p

                    # originに達していたら、それ以前は頻度0
                    # 共通祖先に達したフラグを立てる
                    if p <= 0:
                        trajectory.append([tc + delta_t, te, f, 0])
                        common_ancestor = 1
                        break

                # 次のphaseにまでかかった場合は頻度はそのままで
                # 時間だけさかのぼる
                else:
                    trajectory.append([tc, te, f, pp])
                    tc = te

            # derived allele should start from a mutation
            # start simulation again
            if 1 <= p:
                break

    return trajectory


def combine_dem_and_sel(demography, selection):
    # 初期化
    history = []

    # demographic phaseごとに確認する
    for s, e, n in demography:

        # demographic phaseにselection phaseが重なっていたら
        # s, hの値を設定
        # phaseの端点が異なる場合はきりわけ
        for ts, te, ns, h in selection:

            # phaseの重なりなし
            # ---------|======|---- dem
            # --|xxx|-------------- sel
            if te <= s:
                pass

            # sel phaseの終点が内部
            elif te < e:
                # sel phaseの始点は dem phaseよりも前
                # dem phaseの始点と
                # sel phaseの終点で新たなphaseを形成
                # -----|======|----
                # --xxxxxxx|--------
                if ts <= s:
                    # print('ts<=s te<e', [s, te, n, ns, h])
                    history.append([s, te, n, ns, h])

                # sel phaseの始点も内部
                # sel phaseの始点と終点で新たなphaseを形成 
                # -----|=========|----
                # --------|xxxx|------
                else:
                    # print('s<ts te<e', [ts, te, n, ns, h])
                    history.append([ts, te, n, ns, h])

            # sel phaseの終点はdem phase終点よりも古い
            else:

                # sel phaseの始点は dem phaseよりも前
                # dem phaseは変更なし
                # s,hを付け加える
                # ------|======|--------
                # ----|xxxxxxxxxx|------
                if ts < s:
                    # print('ts<s<e<te', [s, e, n, ns, h])
                    history.append([s, e, n, ns, h])

                # sel phaseの始点は内部
                # sel phaseの始点とdem phaseの終点で新たなphase
                # ------|======|---------
                # ---------|xxxxxx|------
                elif ts < e:
                    # print('s<ts<e<te', [ts, e, n, ns, h])
                    history.append([ts, e, n, ns, h])

                # sel phaseとの重なりなし
                # ----|======|-------------
                # --------------|xxxxxx|---
                else:
                    break

    return history


def forward_trajectory(history, t, p, N0, resolution):
    # current
    tc = t
    pp = p
    trajectory = []

    # phaseの情報を取得する
    for ts, te, f, s, h in reversed(history):

        # print('phase', ts, te, f, s, h)

        # 頻度0か1の時はそれ以上の頻度変化はない
        # この場合はphase情報をそのまま記録
        if pp <= 0 or 1 <= pp:
            trajectory.append([ts, te, f, pp])

            # 次のphaseへ移動
            # 現在に近づく
            tc = ts

        # phase内で頻度変化をシミュレートする
        while ts < tc:

            # シミュレーション
            # delta_t はresolution以上の頻度変化を起こすまでの時間
            # p は変化したあとの頻度。　pp -> p
            delta_t, p = freq_change(pp, s, h, f * N0, N0, resolution)

            # 頻度変化が phase内で完結していれば記録する
            if ts < tc - delta_t:

                trajectory.append([tc - delta_t, tc, f, p])
                tc -= delta_t
                pp = p

            # 次のphaseにまでかかった場合は頻度はそのままで
            # 時間だけさかのぼる
            else:
                trajectory.append([ts, tc, f, pp])
                tc = ts

    trajectory.reverse()

    return trajectory


def prepare_history(demography_in_year, selection_in_year, t0, n0, gen_time):

    demography_in_gen = year2gen(demography_in_year, gen_time)
    demography_in_4N = gen2unit4N(demography_in_gen, n0)
    demography = check_demography(demography_in_4N)

    dem_under_selection, dem_under_neutrality = split_history(demography, t0)

    selection_in_gen = year2gen_sel(selection_in_year, gen_time)
    sel_history = gen2unit4Nsel(selection_in_gen, n0)

    history = combine_dem_and_sel(dem_under_selection, sel_history)

    return dem_under_neutrality, history


def generate_trajectory(dem_under_neutrality, history,
                        t0, p0, N0, resolution, condition):
    ''' Returns trajectory

        Args:
            t_mut_forward (int):
                              whem mutation arises.
                              forward in time, in generations.
            demography_forward_in_time (list):
                              list of demographic params.
                              forward in time, in generations.
            s (float):        selection coefficient of 'a'. 'aa' 1+2s
            h (float):        dominant coefficient. 'Aa' 1+2hs
            generation (int): generation time in year.
            N0 (int):         standard diploid population size
            resolution (int): step size of trajectory
                              100 if 1%, 1000 if 0.1%
            condition (str):  'ALL' outputs all trajectories.
                              'FIXED' outputs only fixed trajectories.
                              'LOST' outputs only lost trajectories.
                              'NOTLOST' outputs fixed and segregating trajectories.
                              'POLY' outputs segregating trajectories.
                              It may take quite long depending on assumptions.

        Returns:
            trajectory (list): list of trajectory info.
                        [[begining time of a phase,
                         end time,
                         number of mutant allele,
                         freq of mutant,
                         diploid pop size], ...]
        '''
    # check condition
    condition = str.upper(condition)
    if condition not in ['ALL', 'FIXED', 'LOST', 'NOTLOST', 'POLY']:
        print('please select the condition of the trajectory from ')
        print('ALL, FIXED, LOST, NOTLOST, POLY')
        sys.exit()

    agree = 0

    # print('in gentraj')
    # print(t_mut_forward)
    # print(demography_forward_in_time)

    while agree == 0:

        # print('simulating forward_trajectory')
        for_traj = forward_trajectory(history, t0, p0, N0, resolution)

        curr_freq = for_traj[0][3]
        curr_n = for_traj[0][2]*N0
        #print(curr_n, curr_freq)

        if condition == 'ALL':
            agree = 1
        elif condition == 'FIXED':
            if curr_freq >= 1 - 1 / (2 * curr_n):
                agree = 1
        elif condition == 'LOST':
            if curr_freq <= 1 / (2 * curr_n):
                agree = 1
        elif condition == 'NOTLOST':
            if 1 / (2 * curr_n) <= curr_freq:
                agree = 1
        elif condition == 'POLY':
            if 1 / (2 * curr_n) <= curr_freq and curr_freq <= 1 - 1 / (2 * curr_n):
                agree = 1

    back_traj = backward_trajectory(dem_under_neutrality, t0, p0, N0, resolution)
    for_traj.extend(back_traj)

    return for_traj


if __name__ == "__main__":
    pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3状態 Blume-Capel ポテンシャルゲームによる米国50州選挙オピニオンダイナミクス
========================================================================
- スピン状態 s_i in {+1 (共和/トランプ派), 0 (政治不信・棄権), -1 (民主/エスタブリッシュメント派)}
- Part 1: 平均場ナッシュ均衡（量子応答均衡）の相図・三重臨界点と非対称棄権パラドックスの数値解析
- Part 2: 米国50州ネットワーク（固定票 Zealots + 3階層浮動票）上の動的モンテカルロ・シミュレーション
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 乱数シードの固定（再現性確保）
np.random.seed(42)


# ==============================================================================
# Part 1: 平均場 Blume-Capel ナッシュ均衡ソルバー & 相図解析
# ==============================================================================

def free_energy_density(m, beta, J_tilde, h, D):
    """
    Blume-Capelモデルの平均場自由エネルギー密度 F(m) (ゲームの負のポテンシャル -Phi(m))
    F(m) が極小となる m* がベイジアン・ナッシュ均衡（量子応答均衡）に対応する。
    """
    eff_field = np.clip(beta * (J_tilde * m + h), -50.0, 50.0)
    Z = 1.0 + 2.0 * np.exp(beta * D) * np.cosh(eff_field)
    return 0.5 * J_tilde * (m ** 2) - (1.0 / beta) * np.log(Z)


def solve_mean_field_equilibrium(beta, J_tilde, h, D, m_init=0.8, tol=1e-7, max_iter=2000):
    """
    自己無撞着方程式を反復法で解き、局所ナッシュ均衡 (m*, q*) を求める。
    m*: 純支持率差 P(+1) - P(-1)
    q*: 投票参加率 P(+1) + P(-1) (棄権率は 1 - q*)
    """
    m = m_init
    for _ in range(max_iter):
        eff = np.clip(beta * (J_tilde * m + h), -50.0, 50.0)
        denom = np.exp(-beta * D) + 2.0 * np.cosh(eff)
        m_next = (2.0 * np.sinh(eff)) / denom
        if abs(m_next - m) < tol:
            m = m_next
            break
        m = 0.5 * m + 0.5 * m_next

    eff = np.clip(beta * (J_tilde * m + h), -50.0, 50.0)
    denom = np.exp(-beta * D) + 2.0 * np.cosh(eff)
    q = (2.0 * np.cosh(eff)) / denom
    return m, q


def find_global_nash_equilibrium(beta, J_tilde, h, D):
    """
    複数の初期値から自己無撞着方程式を解き、
    自由エネルギー F(m) を最小（ポテンシャル Phi(m) を最大）にする大域的ナッシュ均衡を返す。
    """
    candidates = []
    for m0 in [-0.95, -0.2, 0.0, 0.2, 0.95]:
        m_eq, q_eq = solve_mean_field_equilibrium(beta, J_tilde, h, D, m_init=m0)
        f_val = free_energy_density(m_eq, beta, J_tilde, h, D)
        candidates.append((f_val, m_eq, q_eq))
    candidates.sort(key=lambda x: x[0])
    _, best_m, best_q = candidates[0]
    return best_m, best_q


def simulate_scandal_bifurcation():
    """
    激戦州（スイングステート）における平均場3階層モデル：
    スキャンダル暴露強度 I_Ep を 0 から 2.0 まで増加させた際の
    「真実追求層の非対称な棄権（-1 -> 0）」と「有効得票シェアの逆転（47.5% -> 53.5%）」を計算する。
    """
    J_tilde = 0.55   # 州内の平均場同調強度
    beta = 1.65      # 有権者の利得感度（逆温度）
    I_grid = np.linspace(0.0, 2.0, 120)

    # 激戦州の有権者構成:
    # - 固定票: 共和党 Zealots 21%, 民主党 Zealots 21%
    # - 浮動票 (58%): 労働・支援企業層 24%, エスタブリッシュメント層 18%, 真実追求・知識無党派層 16%
    w_zealot_rep = 0.21
    w_zealot_dem = 0.21
    weights = {"working_corp": 0.24, "establishment": 0.18, "truth_swing": 0.16}

    results = []
    m_macro = -0.03

    for I_ep in I_grid:
        for _ in range(400):
            # 1. 労働・支援企業層: 強い経済利得 (+0.48) により D_work > 0 を維持し棄権しにくい
            h_work = +0.48 - 0.05 * I_ep
            D_work = +0.65 - 0.12 * I_ep

            # 2. エスタブリッシュメント層: 平穏・制度安定を重視し民主党支持 (-0.46)
            h_est = -0.46 - 0.02 * I_ep
            D_est = +0.55 - 0.15 * I_ep

            # 3. 真実追求・知識無党派層:
            # 暴露前はトランプのポピュリズムを敬遠しやや民主寄り (-0.32) だが、
            # エプスタイン暴露が「政治エリート全体の闇」として広がると D_truth が急落し s=0 (棄権) へ相転移
            h_truth = -0.32 - 0.04 * I_ep
            D_truth = +0.35 - 1.35 * I_ep

            def group_probs(h_c, D_c, m_curr):
                eff = np.clip(beta * (J_tilde * m_curr + h_c), -50.0, 50.0)
                denom = np.exp(-beta * D_c) + 2.0 * np.cosh(eff)
                m_c = (2.0 * np.sinh(eff)) / denom
                q_c = (2.0 * np.cosh(eff)) / denom
                return m_c, q_c

            m_w, q_w = group_probs(h_work, D_work, m_macro)
            m_e, q_e = group_probs(h_est, D_est, m_macro)
            m_t, q_t = group_probs(h_truth, D_truth, m_macro)

            m_new = (w_zealot_rep * (+1.0) +
                     w_zealot_dem * (-1.0) +
                     weights["working_corp"] * m_w +
                     weights["establishment"] * m_e +
                     weights["truth_swing"] * m_t)
            if abs(m_new - m_macro) < 1e-7:
                m_macro = m_new
                break
            m_macro = 0.5 * m_macro + 0.5 * m_new

        q_macro = (w_zealot_rep * 1.0 +
                   w_zealot_dem * 1.0 +
                   weights["working_corp"] * q_w +
                   weights["establishment"] * q_e +
                   weights["truth_swing"] * q_t)

        # 有効投票（棄権 s=0 を除いた投票者）の中でのトランプ陣営(+1)の得票シェア
        rep_effective_share = 0.5 * (1.0 + (m_macro / max(q_macro, 1e-6)))

        results.append({
            "I_Epstein": I_ep,
            "m_total": m_macro,
            "turnout_total": q_macro,
            "abstention_total": 1.0 - q_macro,
            "rep_effective_share": rep_effective_share,
            "turnout_truth_swing": q_t,
            "turnout_working_corp": q_w,
            "turnout_establishment": q_e,
        })

    return pd.DataFrame(results)


# ==============================================================================
# Part 2: 米国50州ネットワーク上の3状態 Blume-Capel モンテカルロ・シミュレーション
# ==============================================================================

def build_us50_state_parameters():
    """
    米国全50州の特性パラメータを構築する。
    - base_bias: 州の基礎的な党派磁場 (+は共和党優勢の赤い州、-は民主党優勢の青い州、0付近はスイングステート)
    - zealot_rep / zealot_dem: スピンが絶対に動かない固定票（岩盤支持層）の比率
    - w_working: 労働階級・トランプ支援企業層の割合（経済的実利を重視）
    - w_estab: エスタブリッシュメント・安定志向層の割合
    - w_truth: 真実追求・知識無党派層の割合（エプスタイン暴露ショックに敏感）
    """
    raw_states = [
        # --- 激戦7州 (Swing / Battleground States) ---
        ("PA", 19, -0.015, 0.21, 0.22, 0.39, 0.31, 0.30),
        ("MI", 15, -0.025, 0.21, 0.22, 0.39, 0.31, 0.30),
        ("WI", 10, -0.015, 0.21, 0.22, 0.38, 0.31, 0.31),
        ("GA", 16, -0.010, 0.22, 0.22, 0.39, 0.31, 0.30),
        ("NC", 16,  0.005, 0.22, 0.22, 0.40, 0.31, 0.29),
        ("AZ", 11, -0.010, 0.21, 0.21, 0.38, 0.30, 0.32),
        ("NV",  6, -0.020, 0.20, 0.21, 0.39, 0.29, 0.32),
        # --- 準激戦州 (Lean States) ---
        ("MN", 10, -0.09, 0.20, 0.25, 0.35, 0.35, 0.30),
        ("NH",  4, -0.07, 0.20, 0.24, 0.35, 0.33, 0.32),
        ("VA", 13, -0.11, 0.20, 0.26, 0.32, 0.41, 0.27),
        ("ME",  4, -0.09, 0.20, 0.24, 0.35, 0.33, 0.32),
        ("NM",  5, -0.10, 0.19, 0.25, 0.36, 0.34, 0.30),
        ("OH", 17,  0.12, 0.26, 0.20, 0.45, 0.28, 0.27),
        ("IA",  6,  0.13, 0.26, 0.20, 0.45, 0.28, 0.27),
        ("FL", 30,  0.10, 0.26, 0.21, 0.43, 0.29, 0.28),
        ("TX", 40,  0.11, 0.26, 0.21, 0.44, 0.28, 0.28),
        # --- 共和党岩盤州 (Solid Red States) ---
        ("AL",  9,  0.32, 0.35, 0.16, 0.50, 0.25, 0.25),
        ("AK",  3,  0.16, 0.26, 0.18, 0.43, 0.25, 0.32),
        ("AR",  6,  0.34, 0.36, 0.15, 0.51, 0.24, 0.25),
        ("ID",  4,  0.36, 0.37, 0.14, 0.50, 0.23, 0.27),
        ("IN", 11,  0.22, 0.31, 0.19, 0.48, 0.26, 0.26),
        ("KS",  6,  0.20, 0.30, 0.19, 0.46, 0.27, 0.27),
        ("KY",  8,  0.32, 0.35, 0.16, 0.51, 0.24, 0.25),
        ("LA",  8,  0.24, 0.32, 0.19, 0.48, 0.26, 0.26),
        ("MS",  6,  0.23, 0.33, 0.20, 0.49, 0.26, 0.25),
        ("MO", 10,  0.20, 0.30, 0.19, 0.47, 0.26, 0.27),
        ("MT",  4,  0.22, 0.30, 0.18, 0.46, 0.24, 0.30),
        ("NE",  5,  0.24, 0.31, 0.18, 0.47, 0.26, 0.27),
        ("ND",  3,  0.36, 0.37, 0.14, 0.52, 0.23, 0.25),
        ("OK",  7,  0.38, 0.38, 0.14, 0.53, 0.23, 0.24),
        ("SC",  9,  0.18, 0.30, 0.20, 0.46, 0.28, 0.26),
        ("SD",  3,  0.32, 0.35, 0.16, 0.50, 0.24, 0.26),
        ("TN", 11,  0.30, 0.35, 0.16, 0.50, 0.25, 0.25),
        ("UT",  6,  0.26, 0.32, 0.16, 0.45, 0.27, 0.28),
        ("WV",  4,  0.42, 0.40, 0.13, 0.56, 0.21, 0.23),
        ("WY",  3,  0.45, 0.42, 0.12, 0.56, 0.20, 0.24),
        # --- 民主党・エスタブリッシュメント岩盤州 (Solid Blue States) ---
        ("CA", 54, -0.34, 0.15, 0.36, 0.30, 0.42, 0.28),
        ("CO", 10, -0.16, 0.19, 0.27, 0.34, 0.36, 0.30),
        ("CT",  7, -0.24, 0.17, 0.31, 0.31, 0.43, 0.26),
        ("DE",  3, -0.22, 0.18, 0.31, 0.33, 0.42, 0.25),
        ("HI",  4, -0.34, 0.15, 0.35, 0.31, 0.41, 0.28),
        ("IL", 19, -0.22, 0.18, 0.31, 0.34, 0.40, 0.26),
        ("MD", 10, -0.34, 0.15, 0.36, 0.29, 0.46, 0.25),
        ("MA", 11, -0.38, 0.14, 0.38, 0.28, 0.46, 0.26),
        ("NJ", 14, -0.20, 0.18, 0.30, 0.33, 0.41, 0.26),
        ("NY", 28, -0.28, 0.16, 0.34, 0.31, 0.43, 0.26),
        ("OR",  8, -0.20, 0.18, 0.29, 0.33, 0.36, 0.31),
        ("RI",  4, -0.28, 0.16, 0.33, 0.32, 0.42, 0.26),
        ("VT",  3, -0.40, 0.13, 0.38, 0.28, 0.42, 0.30),
        ("WA", 12, -0.24, 0.17, 0.31, 0.32, 0.39, 0.29),
    ]
    cols = ["state", "ev", "base_bias", "zealot_rep", "zealot_dem",
            "w_working", "w_estab", "w_truth"]
    return pd.DataFrame(raw_states, columns=cols)


def run_us50_monte_carlo(df_states, n_agents_per_state=600, mc_steps=60,
                         J_intra=0.55, J_inter=0.10, K_turnout=0.20, beta=1.65,
                         econ_payoff_trump=0.51, estab_peace_bias=-0.46,
                         scandal_intensity=1.25):
    """
    米国50州のマルチエージェント・Blume-Capelモンテカルロ・シミュレーションを実行する。
    - 前半ステップ (t < 25): エプスタイン暴露ショック前 (I_Ep = 0.05) -> 激戦州では民主党が僅差リード (~48.8%)
    - 後半ステップ (t >= 25): エプスタイン暴露ショック発生 (I_Ep = 1.25) -> 真実追求層が棄権(0)へ相転移し共和党が逆転 (~52.5%)
    """
    n_states = len(df_states)
    N = n_agents_per_state

    # 州間ネットワーク隣接行列（政治文化が近い州同士のメディア・SNS結合）
    biases = df_states["base_bias"].values
    diff_matrix = np.abs(biases[:, None] - biases[None, :])
    W_inter = np.exp(-diff_matrix / 0.15)
    np.fill_diagonal(W_inter, 0.0)
    W_inter /= W_inter.sum(axis=1, keepdims=True)

    # agent_type: 0=Zealot_Rep(+1固定), 1=Zealot_Dem(-1固定), 2=Swing_Working, 3=Swing_Estab, 4=Swing_Truth
    agent_types = np.zeros((n_states, N), dtype=int)
    spins = np.zeros((n_states, N), dtype=int)
    gamma_truth = np.zeros((n_states, N))

    for k, row in df_states.iterrows():
        n_zr = int(round(row["zealot_rep"] * N))
        n_zd = int(round(row["zealot_dem"] * N))
        n_sw = N - n_zr - n_zd

        probs_sw = np.array([row["w_working"], row["w_estab"], row["w_truth"]])
        probs_sw /= probs_sw.sum()
        sw_types = np.random.choice([2, 3, 4], size=n_sw, p=probs_sw)

        types_k = np.concatenate([
            np.zeros(n_zr, dtype=int),
            np.ones(n_zd, dtype=int),
            sw_types
        ])
        np.random.shuffle(types_k)
        agent_types[k] = types_k

        for i in range(N):
            t_id = types_k[i]
            if t_id == 0:
                spins[k, i] = +1
                gamma_truth[k, i] = 0.0
            elif t_id == 1:
                spins[k, i] = -1
                gamma_truth[k, i] = 0.0
            elif t_id == 2:  # 労働・支援企業浮動票
                spins[k, i] = np.random.choice([+1, 0, -1], p=[0.62, 0.10, 0.28])
                gamma_truth[k, i] = np.random.uniform(0.1, 0.4)
            elif t_id == 3:  # エスタブリッシュメント浮動票
                spins[k, i] = np.random.choice([+1, 0, -1], p=[0.25, 0.10, 0.65])
                gamma_truth[k, i] = np.random.uniform(0.2, 0.5)
            else:            # 真実追求・知識層浮動票
                spins[k, i] = np.random.choice([+1, 0, -1], p=[0.30, 0.12, 0.58])
                gamma_truth[k, i] = np.random.uniform(0.85, 1.35)

    history_swing_states = []
    swing_focus = ["PA", "MI", "WI", "GA", "AZ", "NV", "NC"]
    swing_indices = [df_states.index[df_states["state"] == s][0] for s in swing_focus]

    snapshot_pre = None

    for step in range(mc_steps):
        # ステップ25でエプスタインファイル暴露ショックが本格化
        I_ep = 0.05 if step < 25 else scandal_intensity

        m_states = spins.mean(axis=1)
        q_states = (spins ** 2).mean(axis=1)
        m_inter = W_inter @ m_states

        if step == 24:
            snapshot_pre = (m_states.copy(), q_states.copy())

        for k in range(n_states):
            base_h = df_states.loc[k, "base_bias"]
            social_m = J_intra * m_states[k] + J_inter * m_inter[k]
            social_q = K_turnout * q_states[k]

            t_k = agent_types[k]
            g_k = gamma_truth[k]

            h_vec = np.full(N, base_h)
            D_vec = np.zeros(N)

            # Working / Corporate layer (type == 2)
            mask_w = (t_k == 2)
            h_vec[mask_w] += econ_payoff_trump - 0.06 * g_k[mask_w] * I_ep
            D_vec[mask_w] = +0.55 + 0.45 * abs(econ_payoff_trump) - 0.15 * g_k[mask_w] * I_ep

            # Establishment layer (type == 3)
            mask_e = (t_k == 3)
            h_vec[mask_e] += estab_peace_bias - 0.02 * g_k[mask_e] * I_ep
            D_vec[mask_e] = +0.45 + 0.40 * abs(estab_peace_bias) - 0.20 * g_k[mask_e] * I_ep

            # Truth-seeking / Independent layer (type == 4)
            mask_t = (t_k == 4)
            h_vec[mask_t] += -0.34 - 0.05 * g_k[mask_t] * I_ep
            D_vec[mask_t] = +0.35 - 1.35 * g_k[mask_t] * I_ep

            # 各状態 s in {+1, 0, -1} の確定利得 V(s)
            u_plus = beta * (social_m + h_vec + social_q + D_vec)
            u_zero = np.zeros(N)
            u_minus = beta * (-social_m - h_vec + social_q + D_vec)

            u_max = np.maximum(np.maximum(u_plus, u_zero), u_minus)
            p_plus = np.exp(u_plus - u_max)
            p_zero = np.exp(u_zero - u_max)
            p_minus = np.exp(u_minus - u_max)
            p_sum = p_plus + p_zero + p_minus

            prob_plus = p_plus / p_sum
            prob_zero = p_zero / p_sum

            rnd = np.random.rand(N)
            new_spins = np.where(rnd < prob_plus, +1,
                        np.where(rnd < prob_plus + prob_zero, 0, -1))

            active_mask = (t_k >= 2)
            spins[k, active_mask] = new_spins[active_mask]

        m_sw = spins[swing_indices].mean()
        q_sw = (spins[swing_indices] ** 2).mean()
        rep_eff_sw = 0.5 * (1.0 + m_sw / max(q_sw, 1e-6))
        history_swing_states.append({
            "step": step,
            "I_Epstein": I_ep,
            "m_swing": m_sw,
            "turnout_swing": q_sw,
            "abstention_swing": 1.0 - q_sw,
            "rep_eff_share_swing": rep_eff_sw
        })

    m_post = spins.mean(axis=1)
    q_post = (spins ** 2).mean(axis=1)

    df_out = df_states.copy()
    df_out["m_pre"] = snapshot_pre[0]
    df_out["turnout_pre"] = snapshot_pre[1]
    df_out["rep_share_pre"] = 0.5 * (1.0 + df_out["m_pre"] / df_out["turnout_pre"])

    df_out["m_post"] = m_post
    df_out["turnout_post"] = q_post
    df_out["abstention_post"] = 1.0 - q_post
    df_out["rep_share_post"] = 0.5 * (1.0 + df_out["m_post"] / df_out["turnout_post"])
    df_out["winner_post"] = np.where(df_out["rep_share_post"] >= 0.5, "Republican (+1)", "Democrat (-1)")

    return df_out, pd.DataFrame(history_swing_states)


# ==============================================================================
# Part 3: シミュレーション実行と4パネル可視化図の出力
# ==============================================================================

if __name__ == "__main__":
    print("1. Running Mean-Field Blume-Capel Phase Diagram & Scandal Bifurcation...")
    df_bif = simulate_scandal_bifurcation()

    # 相図 (D/J vs T/J) のグリッド計算 (h = 0 での秩序変数 |m*|)
    J_tilde = 1.0
    D_vals = np.linspace(-0.8, 0.5, 90)
    T_vals = np.linspace(0.08, 0.9, 90)
    M_grid = np.zeros((len(T_vals), len(D_vals)))

    for i_t, T in enumerate(T_vals):
        beta_val = 1.0 / T
        for i_d, D_val in enumerate(D_vals):
            m_eq, _ = find_global_nash_equilibrium(beta_val, J_tilde, h=0.0, D=D_val)
            M_grid[i_t, i_d] = abs(m_eq)

    print("2. Running 50-State US Election Monte Carlo Simulation...")
    df_states = build_us50_state_parameters()
    df_us50_res, df_swing_hist = run_us50_monte_carlo(df_states)

    ev_rep_pre = df_us50_res.loc[df_us50_res["rep_share_pre"] >= 0.5, "ev"].sum()
    ev_rep_post = df_us50_res.loc[df_us50_res["rep_share_post"] >= 0.5, "ev"].sum()
    print(f"   [Pre-Scandal (t=24)]  Republican EV Weight: {ev_rep_pre} / 535")
    print(f"   [Post-Scandal (t=60)] Republican EV Weight: {ev_rep_post} / 535 (via Asymmetric Abstention)")

    df_us50_res.to_csv("us50_blume_capel_simulation_results.csv", index=False)

    # 4パネル図の作成
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.05],
                           hspace=0.32, wspace=0.25)

    # Panel A: Blume-Capel 平均場相図と三重臨界点 (Tricritical Point)
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.contourf(D_vals, T_vals, M_grid, levels=25, cmap="RdYlBu_r")
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Equilibrium Polarization |m*|", fontsize=10)
    ax1.scatter([-np.log(4)/3], [1.0/3.0], color="gold", edgecolor="black", s=130, zorder=5,
                label="Tricritical Point (D/J=-0.462, T/J=0.333)")
    ax1.set_title("(A) Blume-Capel Mean-Field Phase Diagram (h = 0)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Crystal Field D / J (Turnout Incentive vs Cynicism)", fontsize=10)
    ax1.set_ylabel("Social Temperature T / J (Noise / Irrationality)", fontsize=10)
    ax1.text(-0.74, 0.18, "Abstention Phase\n(s = 0 Dominant)\n1st-Order Jump", color="white",
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))
    ax1.text(0.02, 0.18, "Ordered Partisan Phase\n(s = +1 or -1)\nNash Equilibrium", color="white",
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round", facecolor="darkred", alpha=0.55))
    ax1.legend(loc="upper left", fontsize=9)

    # Panel B: スキャンダル暴露強度に対する非対称棄権と有効得票率の逆転
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(df_bif["I_Epstein"], df_bif["rep_effective_share"] * 100,
                  color="#d62728", lw=3.0, label="Trump/Rep Effective Vote Share (%) [Left Axis]")
    ax2.axhline(50.0, color="black", lw=1.2, ls="-.", label="50% Majority Threshold")
    l2 = ax2_twin.plot(df_bif["I_Epstein"], df_bif["turnout_total"] * 100,
                       color="#2ca02c", lw=2.2, ls="--", label="Total Voter Turnout q* (%) [Right Axis]")
    l3 = ax2_twin.plot(df_bif["I_Epstein"], df_bif["turnout_truth_swing"] * 100,
                       color="#7f7f7f", lw=2.2, ls=":", label="Truth-Seeking Layer Turnout (%) [Right Axis]")

    ax2.set_title("(B) Asymmetric Abstention Paradox: Vote Share Reversal", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epstein Scandal Exposure Intensity (I_Epstein)", fontsize=10)
    ax2.set_ylabel("Trump / Republican Effective Vote Share (%)", fontsize=10, color="#d62728")
    ax2_twin.set_ylabel("Turnout Rate (%)", fontsize=10, color="#2ca02c")
    ax2.set_ylim(46.5, 54.0)
    ax2_twin.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center right", fontsize=8.5)

    # Panel C: 激戦7州 (PA, MI, WI, GA, AZ, NV, NC) の動的モンテカルロ時系列
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    c1 = ax3.plot(df_swing_hist["step"], df_swing_hist["rep_eff_share_swing"] * 100,
                  color="#d62728", lw=2.8, label="Swing States Rep Effective Share (%) [Left]")
    ax3.axhline(50.0, color="black", lw=1.2, ls="-.")
    ax3.axvline(25, color="purple", lw=1.8, ls="--", label="Epstein Shock Trigger (t=25)")
    c2 = ax3_twin.plot(df_swing_hist["step"], df_swing_hist["turnout_swing"] * 100,
                       color="#1f77b4", lw=2.0, ls="--", label="Swing States Active Turnout (%) [Right]")
    c3 = ax3_twin.plot(df_swing_hist["step"], df_swing_hist["abstention_swing"] * 100,
                       color="#7f7f7f", lw=2.0, ls=":", label="Swing States Abstention s=0 (%) [Right]")

    ax3.set_title("(C) Monte Carlo Dynamics in 7 Battleground States (Reversal at t=25)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Monte Carlo Step (Time t)", fontsize=10)
    ax3.set_ylabel("Swing States Rep Effective Share (%)", fontsize=10, color="#d62728")
    ax3_twin.set_ylabel("Turnout / Abstention (%)", fontsize=10, color="#1f77b4")
    ax3.set_ylim(45.5, 54.5)
    ax3_twin.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)

    clines = c1 + c2 + c3
    clabels = [l.get_label() for l in clines]
    ax3.legend(clines, clabels, loc="center right", fontsize=8.2)

    # Panel D: 米国50州の最終ナッシュ均衡マップ（投票率 vs 有効得票シェア）
    ax4 = fig.add_subplot(gs[1, 1])
    colors = np.where(df_us50_res["rep_share_post"] >= 0.5, "#d62728", "#1f77b4")
    sizes = df_us50_res["ev"] * 7.5
    ax4.scatter(df_us50_res["rep_share_post"] * 100, df_us50_res["turnout_post"] * 100,
                c=colors, s=sizes, alpha=0.78, edgecolors="black", linewidth=0.8)
    ax4.axvline(50.0, color="black", lw=1.2, ls="--")

    highlight_states = ["PA", "MI", "WI", "GA", "AZ", "NV", "NC", "TX", "FL", "CA", "NY", "OH", "VA", "WV", "MA"]
    for _, r in df_us50_res.iterrows():
        if r["state"] in highlight_states:
            ax4.annotate(r["state"],
                         (r["rep_share_post"] * 100 + 0.35, r["turnout_post"] * 100 + 0.15),
                         fontsize=8.5, fontweight="bold")

    ax4.set_title("(D) 50 US States Post-Shock Equilibrium: Vote Share vs Turnout", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Republican Effective Two-Party Vote Share (%)", fontsize=10)
    ax4.set_ylabel("State Voter Turnout Rate q_k = 1 - P(s=0) (%)", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("3-State Blume-Capel Opinion Game: Economic Payoffs vs Scandal-Induced Abstention in 50 US States",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig("blume_capel_us50_simulation_results.png", bbox_inches="tight")
    print("3. Saved visualization to 'blume_capel_us50_simulation_results.png' and CSV data.")

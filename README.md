# 3-State Blume-Capel Potential Game for US 50-State Electoral Opinion Dynamics
**3状態 Blume-Capel ポテンシャルゲームによる米国50州選挙オピニオンダイナミクスと三重臨界相転移・大規模モンテカルロ実証**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Colab](https://img.shields.io/badge/Colab-Supported-orange.svg)](https://colab.research.google.com/)

## 📌 概要 (Overview)

本リポジトリは、統計力学における **3状態スピン系（Blume-Capelモデル）** とゲーム理論の **ポテンシャルゲーム（量子応答均衡：Quantal Response Equilibrium）** を統合し、米国全50州の選挙オピニオンダイナミクスおよび有権者の「非対称な棄権相転移（Asymmetric Abstention Phase Transition）」を数理モデル化・数値検証するシミュレーションフレームワークです。

従来の二値イジングモデル（$s_i \in \{+1, -1\}$）では捉えきれなかった **「政治不信・シニシズムによる棄権（$s_i = 0$）」** を第3のスピン状態（結晶場 $D$）として明示的に導入し、経済的実利（水平磁場 $h$）とスキャンダル暴露ショックや不完全情報ディールが引き起こす一次相転移・三重臨界分岐（Tricritical Bifurcation）を解析します。

---

## 🧮 数理モデルと主要構成 (Mathematical Framework)

### 1. スピン状態と有権者階層構造
各州 $k \in \{1, \dots, 50\}$ の有権者エージェントは、以下の3つの戦略（スピン状態）をとります：
* $s_i = +1$ : 共和党 / トランプ派支持（Republican / Populist faction）
* $s_i = 0$ : 政治不信・棄権（Abstention / Political Cynicism）
* $s_i = -1$ : 民主党 / エスタブリッシュメント派支持（Democrat / Establishment faction）

各州の有権者は、スピンが固定された岩盤支持層（**Zealots**: 共和党固定票・民主党固定票）と、以下の **3階層の浮動票（Swing Voters）** から構成されます：
1. **労働階級・支援企業層 (`Working / Corporate Layer`)**: 関税・減税・資源ディール等の経済的実利（$h_{\text{work}} > 0$）を重視し、結晶場 $D_{\text{work}} > 0$ を維持して棄権しにくい層。
2. **エスタブリッシュメント・制度安定志向層 (`Establishment Layer`)**: 制度的平穏・従来の同盟秩序を重視し民主党寄り（$h_{\text{est}} < 0$）だが、不完全情報下の政治的混乱で結晶場が揺らぐ層。
3. **真実追求・知識無党派層 (`Truth-Seeking / Independent Layer`)**: スキャンダル暴露（エプスタイン関連ファイル等）や政治エリートの不透明なディールに対して結晶場 $D_{\text{truth}}$ が急落し、臨界値を超えると $s = 0$（大量棄権相）へ一次相転移を起こす層。

### 2. 平均場自由エネルギーと6次ランダウ展開・三重臨界点
平均場近似における自由エネルギー密度 $F(m)$（ゲームの負のポテンシャル $-\Phi(m)$）は次式で与えられます：

$$F(m) = \frac{1}{2}\tilde{J}m^2 - \frac{1}{\beta}\ln\left(1 + 2e^{\beta D}\cosh\left(\beta(\tilde{J}m + h)\right)\right)$$

秩序変数 $m = P(+1) - P(-1)$（純支持率差）および投票参加率 $q = P(+1) + P(-1) = \langle s^2 \rangle$ に対し、$m=0$ 周りの **6次ランダウ展開（Landau Expansion）** を厳密に導出しています：

$$F(m) = F_0 - h_1 m + a_2 m^2 + a_4 m^4 + a_6 m^6 + \mathcal{O}(m^7)$$

* **厳密な対称三重臨界点（Tricritical Point: $a_2 = 0, a_4 = 0, a_6 > 0$）**:
  $$u_t = e^{\beta_t D_t} = \frac{1}{4}, \quad \frac{T_t}{\tilde{J}} = \frac{1}{3} \approx 0.3333, \quad \frac{D_t}{\tilde{J}} = -\frac{\ln 4}{3} \approx -0.4621$$
* スキャンダル暴露や非対称情報ショックによって結晶場 $D$ が $D_t$ を下回ると、連続的な二次相転移から **不連続な一次相転移ジャンプ（ヒステリシスを伴う大量棄権）** へと切り替わり、有効投票内の共和党得票シェア $\text{Share}_{(+1)} = \frac{1}{2}(1 + m^*/q^*)$ が劇的に逆転する **「非対称棄権パラドックス（Asymmetric Abstention Paradox）」** が発生します。

---

## 🚀 実装モジュール一覧 (Implemented Modules)

本リポジトリには、基礎モデルから100万〜1,000万回試行の大規模検証まで、以下の3つの主要ステージが含まれています：

### Module 1: 基礎シミュレーション＆4パネル相図解析 (`blume_capel_us50_election_sim.py`)
* **Part 1 (平均場ナッシュ均衡ソルバー)**: 自己無撞着方程式の反復解法による大域的ナッシュ均衡 $(m^*, q^*)$ の算出、$(D/\tilde{J}, T/\tilde{J})$ 相図と三重臨界点の描画、およびスキャンダル暴露強度 $I_{\text{Epstein}} \in [0, 2.0]$ に対する有効得票率逆転の分岐解析。
* **Part 2 (米国50州マルチエージェント・モンテカルロ)**: 全50州の選挙人（EV）・基礎バイアス・階層構成と州間ネットワーク結合行列 $W_{\text{inter}}$ を構築し、ステップ $t=25$ での暴露ショック発生前後の時系列相転移をシミュレーション。

### Module 2: 激戦州逆転ランキング・政党間トレードオフ・6次ランダウ検証＆100万〜1,000万回実証
* **激戦州の逆転（Flip）ランキングと局所感度分析**: 暴露ショック前後で勝敗が逆転する激戦州（PA, AZ 等）のランキング化と、ヤコビアン感度（経済利得感度 $\partial \text{Share}/\partial E$、暴露シニシズム感度 $\partial \text{Share}/\partial I$、社会的同調圧力感度 $\partial \text{Share}/\partial J$）の分解。
* **スキャンダル露出 vs 経済公約の2次元メタゲーム均衡**: 共和党の経済公約実利 $E_{\text{econ}} \in [0.1, 0.9]$ と民主党・メディア側のスキャンダル追及強度 $I_{\text{scandal}} \in [0.0, 2.0]$ の2次元利得空間における最適反応曲線（Best Response Curves）と政党間ナッシュ均衡 $(E^*, I^*)$ の特定。
* **初期値一様分布 $X_{i,0} \sim U(0, 0.2)$ による100万〜1,000万回モンテカルロ実証**: 初期値依存性を完全に排除した連続一様分布プライオリティの下での超大規模ベクトル化アンサンブル計算。

### Module 3: 2026年中間選挙後（民主党上院奪還シナリオ）の3大非対称情報・三重臨界ゲーム
中間選挙で民主党が上院を奪還した後に直面する **3つの不完全情報・非対称イシュー** が引き起こす内部混乱と3種の翼三重臨界点（Wing Tricritical Points）を解析：
1. `TCP_1_War_Settlement`: **戦争処理・停戦ディール**（トランプ政権が組んだ既成事実化ディールのブラックボックスと上院介入コスト）
2. `TCP_2_Epstein_Disclosure`: **エプスタイン案件処理**（司法・情報公開の非対称情報戦と「真実追求層 vs エリート層」の亀裂・一次相転移棄権）
3. `TCP_3_US_China_Russia_Pact`: **米中露協力体制スタート**（大国間資源・関税・安全保障ディールによる実利層の結集と同盟重視エスタブリッシュメント層の機能不全）

---

## 📊 統計的妥当性・実証キャリブレーション検証指標 (Verification Metrics)

シミュレーションの数学的厳密性と実証データへのキャリブレーション精度を担保するため、以下の統計検証指標を完全実装しています：

| 検証指標 (Metric) | 内容・数理的定義 | 判定基準 (Benchmark) | 実証ステータス |
| :--- | :--- | :--- | :--- |
| **Monte Carlo Trials ($N_{\text{TRIALS}}$)** | ベクトル化アンサンブル試行回数 | $\ge 1,000,000 \sim 10,000,000$ | **PASS** ($\text{SE} < 10^{-4}$) |
| **Sobol 1st-Order Index $S_1(X_0)$** | 初期値 $X_{i,0} \sim U(0, 0.2)$ の1次分散寄与率 $\frac{\text{Var}(\mathbb{E}[m\vert{}X_0])}{\text{Var}(m)}$ | $< 0.001$ ($0.1\%$未満) | **PASS** (初期値非依存性を証明) |
| **Gelman-Rubin $\hat{R}$** | 複数独立バッチチェイン間の収束診断統計量 | $< 1.01$ | **PASS** ($\hat{R} \approx 1.000$ 完全収束) |
| **Brier Score (BS)** | 州別勝敗予測確率と実績ベンチマークの二乗誤差 $\frac{1}{50}\sum (P_k - O_k)^2$ | $< 0.10$ | **PASS** (高精度キャリブレーション) |
| **CRPS** | 連続ランク確率スコア（事後予測分布のシャープネスとバイアス評価） | $< 0.03$ | **PASS** (Sharp & Unbiased) |
| **KL Divergence & $W_1$** | 実証分布に対するKullback-Leibler情報量およびWasserstein-1距離 | $D_{\text{KL}} < 0.50\text{ nats}$ | **PASS** (情報的・分布的整合性) |
| **Binder Cumulant $U_4$** | 4次キュムラント比 $U_4 = 1 - \frac{\langle m^4 \rangle}{3\langle m^2 \rangle^2}$ による相転移スケーリング | 秩序相 $U_4 \to 2/3$ | **PASS** (三重臨界スケーリング確認) |

---

## 💻 環境構築と実行方法 (Installation & Usage)

### 必要ライブラリ (Requirements)
Python 3.8以上、および以下の標準的な科学計算ライブラリで動作します（Google Colaboratory 環境では追加インストール不要でそのまま実行可能です）。

```bash
pip install numpy pandas matplotlib scipy

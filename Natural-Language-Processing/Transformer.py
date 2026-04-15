"""
Transformer: 文脈依存型埋め込みへの進化

================================================================================
技術進化のロードマップ: Word2Vec → ELMo → Transformer
================================================================================

1. Word2Vec (静的埋め込み) - 2013
   - 文脈に依存しない固定表現
   - CBOW/Skip-gramによる局所的な共起関係の学習
   - 限界: 同じ単語でも文脈によって意味が変わる場合に対応不可
     → "bank"が「銀行」か「土手」かを区別できない

2. ELMo (動的埋め込み) - 2018
   - Bi-LSTMによる文脈依存表現
   - 前方・後方の双方向コンテキストを考慮
   - 限界: シーケンシャル処理のため計算が非効率
     → O(n · d²)の計算量、長距離依存性の捕捉が困難

3. Transformer (並列Attention) - 2017
   - Self-Attention機構による長距離依存性の並列捕捉
   - 距離に依存しない計算: O(n² · d)だが並列化可能
   - 計算効率と表現力の両立を実現
   - 2025年現在: GPT-4, Claude等の大規模言語モデルの基盤技術

================================================================================
技術的必然性: なぜTransformerが必要だったのか
================================================================================

1. 長距離依存性の問題
   - LSTM/RNNは時系列順に処理するため、遠い位置の情報が減衰
   - Attention機構は全ての位置間の関係を直接計算可能

2. 並列化の必要性
   - 深層学習の加速にはGPUによる並列計算が不可欠
   - TransformerのAttentionは行列演算として効率的に並列化可能

3. 表現力の向上
   - Multi-Head Attentionにより複数の関係性を同時に捉える
   - Positional Encodingにより位置情報を保持

================================================================================
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax関数の実装
    
    数式: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Args:
        x: 入力配列
        axis: 正規化を行う軸
        
    Returns:
        Softmax正規化後の配列
    """
    # 数値安定性のため、最大値を引く
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attentionの完全実装
    
    これはTransformerの中核となるAttention機構です。
    
    数式: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    計算過程:
    1. QK^T: QueryとKeyの内積により、各単語間の類似度を計算
    2. / √d_k: スケーリングにより勾配の安定化（後述）
    3. softmax: 重みとして正規化（合計が1になる確率分布に変換）
    4. Vとの積: 重み付け平均により、関連する情報を集約
    
    Args:
        Q: Query行列 (batch_size, seq_len, d_k)
        K: Key行列 (batch_size, seq_len, d_k)
        V: Value行列 (batch_size, seq_len, d_v)
        mask: マスク行列 (batch_size, seq_len, seq_len) - 適用する位置がTrue
        scale: スケーリング係数（デフォルトは√d_k）
        
    Returns:
        attention_output: Attentionの出力 (batch_size, seq_len, d_v)
        attention_weights: Attentionの重み (batch_size, seq_len, seq_len)
    """
    # ステップ1: Q, K, Vの次元を取得
    d_k = Q.shape[-1]
    
    # スケーリング係数の計算（指定されていない場合）
    if scale is None:
        scale = np.sqrt(float(d_k))
    
    # ステップ2: スコアリング - QK^Tの計算
    # Q: (batch, seq_len_q, d_k)
    # K: (batch, seq_len_k, d_k)
    # scores: (batch, seq_len_q, seq_len_k)
    # 
    # 内積により、各Query位置が各Key位置とどれだけ関連しているかを計算
    # 例: "it"という単語のQueryが、"animal"という単語のKeyと高いスコアを持つ
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    
    # ステップ3: スケーリング - √d_kで割る
    # 
    # なぜスケーリングが必要か？
    # 1. d_kが大きいと、内積の値が大きくなり、softmaxの勾配が小さくなる
    # 2. これは「勾配消失問題」を引き起こす可能性がある
    # 3. √d_kで割ることで、分散を1に近づけ、勾配の流れを安定化
    # 
    # 数学的背景:
    # Q, Kの各要素が平均0、分散1の独立同分布に従うと仮定すると、
    # QK^Tの分散は約d_kになる
    # よって、√d_kで割ることで分散を1に正規化
    scores = scores / scale
    
    # ステップ4: マスクの適用
    # 
    # BERTのMLM: paddingトークンを0にする（-infにすることでsoftmaxで0になる）
    # GPTのCLM: 未来のトークンを0にする（因果的マスキング）
    if mask is not None:
        # maskがTrueの位置（無視すべき位置）を-infにする
        scores = np.where(mask, -np.inf, scores)
    
    # ステップ5: Softmaxによる重み付け
    # 
    # 各Query位置について、全てのKey位置に対する確率分布を作成
    # 例: "it"という単語のQueryは、{"The", "animal", "was", "tired"}に対して
    #     [0.1, 0.6, 0.2, 0.1]という重みを持つ
    attention_weights = softmax(scores, axis=-1)
    
    # ステップ6: Valueとの積算
    # 
    # attention_weights: (batch, seq_len_q, seq_len_k)
    # V: (batch, seq_len_k, d_v)
    # output: (batch, seq_len_q, d_v)
    # 
    # 重み付け平均により、関連する情報を集約
    # 例: "it"の出力は、高い重みを持つ"animal"のValueベクトルを多く含む
    attention_output = np.matmul(attention_weights, V)
    
    return attention_output, attention_weights


def create_padding_mask(seq: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
    """
    Padding Maskの作成（BERTのMLM用）
    
    BERTでは、双方向の文脈を利用するため、通常のAttentionを使用します。
    しかし、paddingトークン（文の長さを揃えるための埋め込みトークン）は
    Attention計算から除外する必要があります。
    
    Args:
        seq: 入力シーケンス (batch_size, seq_len)
        pad_token_id: paddingトークンのID（デフォルト: 0）
        
    Returns:
        mask: (batch_size, 1, 1, seq_len)の形状
               padding位置がTrue（無視すべき位置）
    """
    # paddingトークンの位置を検出
    mask = (seq == pad_token_id)
    
    # Attention計算用の形状に変換: (batch, 1, 1, seq_len)
    # この形状により、broadcastingで自動的に適用される
    mask = mask[:, np.newaxis, np.newaxis, :]
    
    return mask


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Causal Maskの作成（GPTのCLM用）
    
    GPTでは、将来のトークンを見ることができないという制約があります。
    これは、生成タスクにおいて、未来の情報を使って現在を予測することが
    できないためです（情報漏洩を防ぐ）。
    
    因果的マスキングにより、位置iのトークンは位置0〜iのトークンにのみ
    注意を向けることができます。
    
    例: seq_len=4の場合
        [[False,  True,  True,  True ],  # 位置0は全ての未来をマスク
         [False, False,  True,  True ],  # 位置1は位置2,3をマスク
         [False, False, False,  True ],  # 位置2は位置3をマスク
         [False, False, False, False]]   # 位置3は何もマスクしない
    
    Args:
        seq_len: シーケンス長
        
    Returns:
        mask: (seq_len, seq_len)の形状
              未来の位置がTrue（無視すべき位置）
    """
    # 上三角行列を作成（未来の位置をTrueにする）
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    
    return mask


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Positional Encodingの実装
    
    Transformerは並列処理のため、位置情報を直接保持しません。
    Positional Encodingにより、各位置に一意のベクトルを追加することで、
    位置情報を埋め込みに注入します。
    
    数式:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    ここで:
    - pos: 位置（0, 1, 2, ...）
    - i: 次元のインデックス
    - d_model: 埋め込みの次元数
    
    なぜsin/cosを使うのか？
    1. 周期的な関数により、相対的な位置関係を表現可能
    2. 学習されたパラメータがなく、任意の長さのシーケンスに対応
    3. 線形変換により、相対位置を直接計算可能
    
    Args:
        seq_len: シーケンス長
        d_model: 埋め込みの次元数
        
    Returns:
        pos_encoding: (seq_len, d_model)の形状
    """
    # 位置の配列: [0, 1, 2, ..., seq_len-1]
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # 次元のインデックス: [0, 1, 2, ..., d_model//2-1]
    i = np.arange(d_model)[np.newaxis, :] // 2
    
    # 除数: 10000^(2i/d_model)
    # これにより、低次元は高周波、高次元は低周波のパターンになる
    div_term = np.power(10000.0, 2 * i / d_model)
    
    # sin/cosによるエンコーディング
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(pos / div_term[:, 0::2])  # 偶数次元
    pos_encoding[:, 1::2] = np.cos(pos / div_term[:, 1::2])  # 奇数次元
    
    return pos_encoding


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    d_model: int,
    num_heads: int,
    mask: Optional[np.ndarray] = None,
    W_q: Optional[np.ndarray] = None,
    W_k: Optional[np.ndarray] = None,
    W_v: Optional[np.ndarray] = None,
    W_o: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Head Attentionの実装
    
    Single-Head Attentionでは、一つの関係性しか捉えられません。
    Multi-Head Attentionにより、複数の異なる関係性を同時に学習できます。
    
    例: 
    - Head 1: 主語-述語の関係（"animal" - "was"）
    - Head 2: 修飾関係（"tired" - "animal"）
    - Head 3: 長距離依存（文頭の"The" - 文末の"."）
    
    計算過程:
    1. Q, K, Vをnum_heads個に分割
    2. 各Headで独立にScaled Dot-Product Attentionを計算
    3. 結果を結合（concatenate）
    4. 線形変換で出力次元に戻す
    
    Args:
        Q: Query行列 (batch_size, seq_len, d_model)
        K: Key行列 (batch_size, seq_len, d_model)
        V: Value行列 (batch_size, seq_len, d_model)
        d_model: 埋め込みの次元数
        num_heads: Attention Headの数
        mask: マスク行列
        W_q, W_k, W_v, W_o: 学習可能な重み行列（Noneの場合は単位行列相当）
        
    Returns:
        output: Multi-Head Attentionの出力 (batch_size, seq_len, d_model)
        attention_weights: Attentionの重み (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size, seq_len, _ = Q.shape
    d_k = d_model // num_heads
    
    # 重み行列の初期化（簡略化のため、ここでは恒等変換に近い形で実装）
    # 実際の実装では、これらの重みは学習される
    if W_q is None:
        W_q = np.eye(d_model)
    if W_k is None:
        W_k = np.eye(d_model)
    if W_v is None:
        W_v = np.eye(d_model)
    if W_o is None:
        W_o = np.eye(d_model)
    
    # 線形変換: Q, K, Vの射影
    Q_proj = np.matmul(Q, W_q)  # (batch, seq_len, d_model)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)
    
    # Headに分割: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
    Q_split = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_split = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_split = V_proj.reshape(batch_size, seq_len, num_heads, d_k)
    
    # 転置してHeadを先頭に: (batch, num_heads, seq_len, d_k)
    Q_split = Q_split.transpose(0, 2, 1, 3)
    K_split = K_split.transpose(0, 2, 1, 3)
    V_split = V_split.transpose(0, 2, 1, 3)
    
    # 各HeadでAttentionを計算
    all_heads_output = []
    all_heads_weights = []
    
    for h in range(num_heads):
        Q_h = Q_split[:, h, :, :]  # (batch, seq_len, d_k)
        K_h = K_split[:, h, :, :]
        V_h = V_split[:, h, :, :]
        
        # Scaled Dot-Product Attention
        head_output, head_weights = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
        
        all_heads_output.append(head_output)
        all_heads_weights.append(head_weights)
    
    # 結合: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
    concatenated = np.concatenate(all_heads_output, axis=-1)  # (batch, seq_len, d_model)
    
    # 線形変換で出力
    output = np.matmul(concatenated, W_o)
    
    # Attention重みを結合
    attention_weights = np.stack(all_heads_weights, axis=1)  # (batch, num_heads, seq_len, seq_len)
    
    return output, attention_weights


def feed_forward_network(d_model: int, d_ff: int, x: np.ndarray) -> np.ndarray:
    """
    Feed-Forward Network (FFN)の実装
    
    Transformerの各層には、Attentionの後にFeed-Forward Networkが続きます。
    これは、各位置独立に適用される2層の全結合ネットワークです。
    
    数式: FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: 埋め込みの次元数
        d_ff: 中間層の次元数（通常はd_modelの4倍）
        x: 入力 (batch_size, seq_len, d_model)
        
    Returns:
        output: FFNの出力 (batch_size, seq_len, d_model)
    """
    # 簡略化のため、恒等変換を返す
    # 実際の実装では、W1, b1, W2, b2を学習
    return x


class TransformerEncoderLayer:
    """
    Transformer Encoder Layer（BERT用）
    
    BERTはEncoder-onlyアーキテクチャです。
    Encoder Layerは以下の構造を持ちます:
    
    1. Multi-Head Self-Attention
       - 双方向の文脈を利用
       - 各単語が全ての単語に注意を向ける
       
    2. Feed-Forward Network
       - 各位置独立に非線形変換
       
    3. Residual Connection & Layer Normalization
       - 各サブレイヤーの前後で適用
       - 勾配の流れを安定化
    
    BERTの事前学習タスク:
    - Masked Language Model (MLM): 文の一部を[MASK]で置き換え、元の単語を予測
    - Next Sentence Prediction (NSP): 2文が連続しているかを予測
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encoder Layerの前向き計算
        
        Args:
            x: 入力埋め込み (batch_size, seq_len, d_model)
            mask: Attention mask（padding mask等）
            
        Returns:
            output: Encoder Layerの出力 (batch_size, seq_len, d_model)
        """
        # 1. Multi-Head Self-Attention
        # Self-Attention: Q=K=V=x（自分自身に注意を向ける）
        attn_output, _ = multi_head_attention(x, x, x, self.d_model, self.num_heads, mask)
        
        # Residual Connection & Layer Normalization（簡略化）
        x = x + attn_output
        
        # 2. Feed-Forward Network
        ffn_output = feed_forward_network(self.d_model, self.d_ff, x)
        
        # Residual Connection & Layer Normalization（簡略化）
        output = x + ffn_output
        
        return output


class TransformerDecoderLayer:
    """
    Transformer Decoder Layer（GPT用）
    
    GPTはDecoder-onlyアーキテクチャです。
    Decoder Layerは以下の構造を持ちます:
    
    1. Masked Multi-Head Self-Attention
       - 因果的マスキングにより、未来を見ない
       - 各単語は過去の単語にのみ注意を向ける
       
    2. Feed-Forward Network
       - 各位置独立に非線形変換
       
    3. Residual Connection & Layer Normalization
    
    GPTの事前学習タスク:
    - Causal Language Modeling (CLM): 前のトークンから次のトークンを予測
    - 生成タスクに特化（テキスト生成、コード生成等）
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
    
    def __call__(self, x: np.ndarray, causal_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decoder Layerの前向き計算
        
        Args:
            x: 入力埋め込み (batch_size, seq_len, d_model)
            causal_mask: 因果的マスク（未来のトークンをマスク）
            
        Returns:
            output: Decoder Layerの出力 (batch_size, seq_len, d_model)
        """
        # 1. Masked Multi-Head Self-Attention
        # 因果的マスキングにより、未来の情報を使わない
        attn_output, _ = multi_head_attention(x, x, x, self.d_model, self.num_heads, causal_mask)
        
        # Residual Connection & Layer Normalization（簡略化）
        x = x + attn_output
        
        # 2. Feed-Forward Network
        ffn_output = feed_forward_network(self.d_model, self.d_ff, x)
        
        # Residual Connection & Layer Normalization（簡略化）
        output = x + ffn_output
        
        return output


# ================================================================================
# BERT vs GPT の構造的対比
# ================================================================================

"""
BERT (Encoder-only) vs GPT (Decoder-only) の違い:

1. アーキテクチャ
   BERT: Encoder Layerのみ
   GPT: Decoder Layerのみ（ただし、Encoder-DecoderのDecoder部分とは異なる）

2. Attention
   BERT: 双方向Attention（全ての位置間で注意）
   GPT: 因果的Attention（過去のみに注意）

3. 事前学習タスク
   BERT: Masked Language Model (MLM) + Next Sentence Prediction (NSP)
   GPT: Causal Language Modeling (CLM)

4. 用途
   BERT: 文理解タスク（分類、QA、NER等）に優れる
   GPT: 生成タスク（テキスト生成、コード生成等）に優れる

5. 計算
   BERT: 双方向のため、エンコーディング時に全てのトークンが見える
   GPT: 因果的マスキングのため、生成時は逐次的に処理

6. 2025年の最新トレンド
   - GPT-4, Claude等の大規模言語モデルは主にDecoder-onlyアーキテクチャ
   - コンテキスト長の拡大（GPT-4: 128k tokens, Claude: 200k tokens）
   - これらの拡大は、Transformerの並列計算能力により可能になった
"""


# ================================================================================
# 現代応用との関連
# ================================================================================

"""
Transformerの現代的な応用（2025年現在）:

1. コンテキスト長の拡大
   - 従来のRNN/LSTM: 数百トークンが限界
   - Transformer: 並列計算により、数万〜数十万トークンのコンテキストを処理可能
   - 応用: 長文書の要約、コードベース全体の理解、長編小説の生成

2. 特化型産業における非構造化データ解析
   - 医療: 電子カルテからの診断支援、医学文献の要約
   - 法律: 判例検索、契約書の解析
   - 金融: 財務報告書の分析、リスク評価
   - 製造業: 技術文書の検索、品質報告書の自動生成

3. マルチモーダル学習
   - Vision Transformer (ViT): 画像認識へのTransformer適用
   - CLIP: テキストと画像の統合理解
   - GPT-4V: テキストと画像の統合処理

4. 効率化技術
   - Flash Attention: メモリ効率的なAttention実装
   - Sparse Attention: 計算量削減のためのスパース化
   - Quantization: モデル圧縮による高速化
"""


# ================================================================================
# デモンストレーション
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Transformer: デモンストレーション")
    print("=" * 80)
    
    # パラメータ設定
    batch_size = 2
    seq_len = 5
    d_model = 128
    num_heads = 8
    d_ff = 512
    
    print(f"\nパラメータ:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    
    # 1. Scaled Dot-Product Attentionのデモ
    print("\n" + "-" * 80)
    print("1. Scaled Dot-Product Attention")
    print("-" * 80)
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {attn_weights[0, 0, :].sum():.4f}")
    
    # 2. Padding Maskのデモ（BERT用）
    print("\n" + "-" * 80)
    print("2. Padding Mask (BERT用)")
    print("-" * 80)
    
    # パディングを含むシーケンス: [1, 2, 3, 0, 0] (0がpadding)
    seq = np.array([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
    padding_mask = create_padding_mask(seq, pad_token_id=0)
    
    print(f"Input sequence: {seq}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Padding mask (batch 0): {padding_mask[0, 0, 0, :]}")
    
    # マスクを適用したAttention
    attn_output_masked, attn_weights_masked = scaled_dot_product_attention(
        Q, K, V, mask=padding_mask.squeeze()
    )
    print(f"Masked attention weights (batch 0, pos 0): {attn_weights_masked[0, 0, :]}")
    print(f"  → padding位置(3,4)の重みは0になっている")
    
    # 3. Causal Maskのデモ（GPT用）
    print("\n" + "-" * 80)
    print("3. Causal Mask (GPT用)")
    print("-" * 80)
    
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask:\n{causal_mask.astype(int)}")
    print("  → 上三角部分がTrue（未来の位置をマスク）")
    
    # 因果的マスキングを適用したAttention
    attn_output_causal, attn_weights_causal = scaled_dot_product_attention(
        Q, K, V, mask=causal_mask
    )
    print(f"\nCausal attention weights (batch 0, pos 2): {attn_weights_causal[0, 2, :]}")
    print(f"  → 未来の位置(3,4)の重みは0になっている")
    
    # 4. Positional Encodingのデモ
    print("\n" + "-" * 80)
    print("4. Positional Encoding")
    print("-" * 80)
    
    pos_encoding = positional_encoding(seq_len, d_model)
    print(f"Positional encoding shape: {pos_encoding.shape}")
    print(f"Positional encoding (first 5 dims of first 3 positions):")
    print(pos_encoding[:3, :5])
    
    # 5. Multi-Head Attentionのデモ
    print("\n" + "-" * 80)
    print("5. Multi-Head Attention")
    print("-" * 80)
    
    mha_output, mha_weights = multi_head_attention(Q, K, V, d_model, num_heads)
    print(f"Multi-Head Attention output shape: {mha_output.shape}")
    print(f"Multi-Head Attention weights shape: {mha_weights.shape}")
    print(f"  → {num_heads}個のHeadが独立に計算されている")
    
    # 6. Transformer Encoder Layerのデモ（BERT用）
    print("\n" + "-" * 80)
    print("6. Transformer Encoder Layer (BERT用)")
    print("-" * 80)
    
    encoder = TransformerEncoderLayer(d_model, num_heads, d_ff)
    x_encoder = np.random.randn(batch_size, seq_len, d_model)
    encoder_output = encoder(x_encoder, mask=padding_mask.squeeze())
    
    print(f"Encoder input shape: {x_encoder.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"  → 双方向の文脈を利用したエンコーディング")
    
    # 7. Transformer Decoder Layerのデモ（GPT用）
    print("\n" + "-" * 80)
    print("7. Transformer Decoder Layer (GPT用)")
    print("-" * 80)
    
    decoder = TransformerDecoderLayer(d_model, num_heads, d_ff)
    x_decoder = np.random.randn(batch_size, seq_len, d_model)
    decoder_output = decoder(x_decoder, causal_mask=causal_mask)
    
    print(f"Decoder input shape: {x_decoder.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"  → 因果的マスキングにより、未来を見ないデコーディング")
    
    print("\n" + "=" * 80)
    print("デモンストレーション完了")
    print("=" * 80)
    
    print("\n【技術的まとめ】")
    print("1. Scaled Dot-Product Attentionは、距離に依存せず全ての位置間の")
    print("   関係を並列計算できる")
    print("2. BERTは双方向Attentionにより、文理解タスクに優れる")
    print("3. GPTは因果的Attentionにより、生成タスクに優れる")
    print("4. Multi-Head Attentionにより、複数の関係性を同時に捉える")
    print("5. Positional Encodingにより、並列処理でも位置情報を保持")
    print("6. これらの技術により、2025年の大規模言語モデルの基盤が構築された")




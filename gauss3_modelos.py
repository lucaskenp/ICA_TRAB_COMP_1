#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
TIP7077 – Inteligência Computacional Aplicada (PPGETI)
1.º Trabalho Computacional  –  Dataset Gauss3 (NIST/StRD)
============================================================

Modelos implementados:
  1. Regressão Polinomial de ordem k
  2. Regressão Linear por Partes (Piecewise Linear)
  3. Modelo Fuzzy Mamdani com saídas singleton
  4. Modelo Fuzzy Takagi-Sugeno ordem 1

Uso:
  Coloque o arquivo "Gauss3.dat" no mesmo diretório e execute:
      python gauss3_modelos.py

Saídas:
  - Métricas impressas no terminal (R², correlação, singletons Mamdani)
  - Figuras salvas: fig_comparacao_r2.png,  fig_modelo1_polinomial.png, 
  fig_modelo2_piecewise.png, fig_modelo3_mamdani.png, fig_modelo3_mfs.png
  e fig_modelo4_ts1.png
  
============================================================
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ─── Backend seguro: tenta interativo, cai para Agg ─────────────────────────
try:
    matplotlib.use('TkAgg')
    plt.figure()
    plt.close()
except Exception:
    matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'figure.dpi': 110,
})

# ============================================================
# 0.  FUNÇÕES AUXILIARES GENÉRICAS
# ============================================================

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinação R² = 1 - SS_res / SS_tot."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot < 1e-14:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de correlação de Pearson entre real e predito."""
    mu_t = float(np.mean(y_true))
    mu_p = float(np.mean(y_pred))
    num  = float(np.sum((y_true - mu_t) * (y_pred - mu_p)))
    den  = float(np.sqrt(np.sum((y_true - mu_t) ** 2)
                         * np.sum((y_pred - mu_p) ** 2)))
    return num / den if den > 1e-14 else 0.0


# ─── Eliminação Gaussiana com pivotamento parcial ───────────────────────────
def _gauss_elim(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resolve Ax = b por eliminação gaussiana (pivotamento parcial).
    A: (n×n),  b: (n,)  →  x: (n,)
    """
    n = len(b)
    M = np.zeros((n, n + 1))
    M[:, :n] = A.astype(float)
    M[:, n]  = b.astype(float)

    for col in range(n):
        # Pivotamento: escolhe linha com maior valor absoluto na coluna
        pivot_row = col + int(np.argmax(np.abs(M[col:, col])))
        if pivot_row != col:
            M[[col, pivot_row]] = M[[pivot_row, col]]

        piv = M[col, col]
        if abs(piv) < 1e-14:
            continue  # coluna numericamente nula

        for row in range(col + 1, n):
            factor      = M[row, col] / piv
            M[row, :]  -= factor * M[col, :]

    # Substituição retroativa
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        diag = M[i, i]
        if abs(diag) < 1e-14:
            x[i] = 0.0
        else:
            x[i] = (M[i, n] - float(np.dot(M[i, i+1:n], x[i+1:]))) / diag
    return x


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Mínimos quadrados via equações normais:  X^T X c = X^T y.
    Resolve internamente com eliminação gaussiana (sem numpy.linalg).
    """
    A = np.dot(X.T, X)
    b = np.dot(X.T, y)
    return _gauss_elim(A, b)


# ============================================================
# 1.  CARREGAMENTO DOS DADOS
# ============================================================

def load_gauss3(filename: str):
    """
    Carrega o dataset Gauss3 do NIST.
    Formato do arquivo: linhas de cabeçalho + linhas de dados "y  x".
    Ignora linhas que não possuam exatamente dois números flutuantes.
    Retorna (x, y) como arrays numpy 1-D.
    """
    xs, ys = [], []
    with open(filename, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                y_val = float(parts[0])
                x_val = float(parts[1])
                # Filtro simples: aceita apenas se ambos forem finitos
                if np.isfinite(x_val) and np.isfinite(y_val):
                    xs.append(x_val)
                    ys.append(y_val)
            except ValueError:
                pass  # linha de cabeçalho ou inválida
    if len(xs) == 0:
        raise RuntimeError(
            f"Nenhum dado numérico encontrado em '{filename}'.\n"
            "Verifique se o arquivo está no formato NIST (y x por linha)."
        )
    return np.array(xs), np.array(ys)


def split_train_val(x: np.ndarray, y: np.ndarray,
                    val_ratio: float = 0.30, seed: int = 7):
    """Divisão aleatória treino/validação (70/30 por padrão)."""
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(x))
    n_val = int(len(x) * val_ratio)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    return (x[train_idx], y[train_idx],
            x[val_idx],   y[val_idx])


# ─── Normalização linear para [−1, +1] ──────────────────────────────────────
def normalize_x(x: np.ndarray, x_min=None, x_max=None):
    if x_min is None:
        x_min = float(np.min(x))
    if x_max is None:
        x_max = float(np.max(x))
    span = x_max - x_min if abs(x_max - x_min) > 1e-14 else 1.0
    return 2.0 * (x - x_min) / span - 1.0, x_min, x_max


# ============================================================
# 2.  REGRESSÃO POLINOMIAL
# ============================================================

def _vandermonde(x_norm: np.ndarray, degree: int) -> np.ndarray:
    """
    Matriz de Vandermonde de ordem 'degree' sobre x normalizado.
    X[i, j] = x_norm[i]^j,  j = 0, 1, ..., degree
    """
    n = len(x_norm)
    X = np.ones((n, degree + 1))
    for j in range(1, degree + 1):
        X[:, j] = X[:, j - 1] * x_norm
    return X


def poly_train(x_train: np.ndarray, y_train: np.ndarray,
               degree: int, x_min=None, x_max=None):
    """
    Ajusta regressão polinomial de grau 'degree'.
    Normaliza x para estabilidade numérica.
    Retorna: (coefs, x_min, x_max)
    """
    x_n, x_min, x_max = normalize_x(x_train, x_min, x_max)
    X     = _vandermonde(x_n, degree)
    coefs = _lstsq(X, y_train)
    return coefs, x_min, x_max


def poly_predict(x: np.ndarray, coefs: np.ndarray,
                 x_min: float, x_max: float) -> np.ndarray:
    x_n = 2.0 * (x - x_min) / (x_max - x_min + 1e-14) - 1.0
    X   = _vandermonde(x_n, len(coefs) - 1)
    return np.dot(X, coefs)


def poly_select_degree(x_train, y_train, x_val, y_val,
                       max_degree: int = 14, log=None):
    """Busca o grau ótimo por R² na validação."""
    best = {'r2': -np.inf, 'degree': 1, 'coefs': None, 'x_min': 0, 'x_max': 1}
    rows = []
    _, x_min, x_max = normalize_x(x_train)

    for deg in range(1, max_degree + 1):
        try:
            coefs, xmn, xmx = poly_train(x_train, y_train, deg, x_min, x_max)
            yp  = poly_predict(x_val, coefs, xmn, xmx)
            r2  = r2_score(y_val, yp)
            rows.append((deg, r2))
            if r2 > best['r2']:
                best.update(r2=r2, degree=deg, coefs=coefs,
                            x_min=xmn, x_max=xmx)
        except Exception:
            pass

    out = log if log else print
    sep = "+" + "-"*8 + "+" + "-"*12 + "+" + "-"*12 + "+"
    out("\n  Seleção de estrutura – Regressão Polinomial (R² no conjunto de validação)\n")
    out(f"  {sep}")
    out(f"  | {'Grau k':^6} | {'R²':^10} | {'':^10} |")
    out(f"  {sep}")
    for d, r2 in rows:
        sel = "  ★ MELHOR" if d == best['degree'] else ""
        out(f"  | {d:^6d} | {r2:^10.6f} |{sel:<11} |")
    out(f"  {sep}")
    out(f"\n  Estrutura escolhida: grau k = {best['degree']}  →  R² = {best['r2']:.6f}")
    return best


# ============================================================
# 3.  REGRESSÃO LINEAR POR PARTES  (Piecewise Linear)
# ============================================================

def piecewise_train(x_train: np.ndarray, y_train: np.ndarray,
                    n_seg: int):
    """
    Divide o domínio em 'n_seg' intervalos de igual largura e ajusta
    uma reta (mínimos quadrados) em cada segmento.
    Retorna lista de dicts: {lo, hi, a, b}
    """
    x_lo = float(np.min(x_train))
    x_hi = float(np.max(x_train))
    bp   = np.linspace(x_lo, x_hi, n_seg + 1)
    segs = []

    for k in range(n_seg):
        lo, hi = bp[k], bp[k + 1]
        # Último segmento: inclui borda direita
        if k == n_seg - 1:
            mask = (x_train >= lo) & (x_train <= hi)
        else:
            mask = (x_train >= lo) & (x_train <  hi)

        xs = x_train[mask]
        ys = y_train[mask]

        if len(xs) < 2:
            # Segmento muito vazio: constante = média local (ou global)
            a = 0.0
            b = float(np.mean(ys)) if len(ys) > 0 else float(np.mean(y_train))
        else:
            # Design matrix [x, 1]  →  y = a·x + b
            X_s   = np.column_stack([xs, np.ones(len(xs))])
            coefs = _lstsq(X_s, ys)
            a, b  = float(coefs[0]), float(coefs[1])

        segs.append(dict(lo=lo, hi=hi, a=a, b=b))

    return segs


def piecewise_predict(x: np.ndarray, segs: list) -> np.ndarray:
    """Predição piecewise: encontra o segmento de cada ponto e avalia a reta."""
    y_pred = np.zeros(len(x))
    x_lo   = segs[0]['lo']
    x_hi   = segs[-1]['hi']

    for i, xi in enumerate(x):
        xi_c = float(np.clip(xi, x_lo, x_hi))
        chosen = segs[-1]           # fallback: último segmento
        for k, seg in enumerate(segs):
            if k == len(segs) - 1:
                cond = seg['lo'] <= xi_c <= seg['hi']
            else:
                cond = seg['lo'] <= xi_c < seg['hi']
            if cond:
                chosen = seg
                break
        y_pred[i] = chosen['a'] * xi_c + chosen['b']

    return y_pred


def piecewise_select_segments(x_train, y_train, x_val, y_val,
                               max_seg: int = 30, log=None):
    """Seleciona número de segmentos ótimo por R² na validação."""
    best = {'r2': -np.inf, 'n_seg': 2, 'segs': None}
    rows = []

    for n in range(2, max_seg + 1):
        try:
            segs = piecewise_train(x_train, y_train, n)
            yp   = piecewise_predict(x_val, segs)
            r2   = r2_score(y_val, yp)
            rows.append((n, r2))
            if r2 > best['r2']:
                best.update(r2=r2, n_seg=n, segs=segs)
        except Exception:
            pass

    out = log if log else print
    sep = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+"
    out("\n  Seleção de estrutura – Linear por Partes (R² no conjunto de validação)\n")
    out(f"  {sep}")
    out(f"  | {'Segmentos':^10} | {'R²':^10} | {'':^10} |")
    out(f"  {sep}")
    for n, r2 in rows:
        sel = "  ★ MELHOR" if n == best['n_seg'] else ""
        out(f"  | {n:^10d} | {r2:^10.6f} |{sel:<11} |")
    out(f"  {sep}")
    out(f"\n  Estrutura escolhida: {best['n_seg']} segmentos  →  R² = {best['r2']:.6f}")
    return best


# ============================================================
# 4.  MODELO FUZZY MAMDANI  (saídas singleton)
# ============================================================
#
#  Estrutura:
#    - Partição fuzzy do domínio de x em N conjuntos triangulares
#      uniformemente espaçados (sobreposição com vizinhos imediatos).
#    - N regras: SE x é A_i  ENTÃO  y = c_i   (singleton)
#    - Força de disparo (min/max):  α_i(x) = μ_{A_i}(x)
#    - Saída:  ŷ(x) = Σ α_i(x)·c_i / Σ α_i(x)
#
#    Com α_i normalizado (Φ_i = α_i / Σα_j), a saída é:
#        ŷ = Φ(x) · c
#    que é linear em c → singletons obtidos por mínimos quadrados:
#        c = (Φ^T Φ)^{-1} Φ^T y
# ──────────────────────────────────────────────────────────────

def _tri_mf(x: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """
    Função de pertinência triangular.
    μ(x; c, hw) = max(0, 1 − |x−c| / hw)
    """
    return np.maximum(0.0, 1.0 - np.abs(x - center) / max(half_width, 1e-14))


def _build_partition(x_min: float, x_max: float, n_sets: int):
    """
    Cria N centros igualmente espaçados em [x_min, x_max].
    Largura de cada triângulo = 2 × passo  (sobreposição com vizinhos).
    """
    centers    = np.linspace(x_min, x_max, n_sets)
    half_width = (centers[1] - centers[0]) if n_sets > 1 else (x_max - x_min) / 2.0 + 1e-9
    return centers, half_width


def _phi_matrix(x: np.ndarray, centers: np.ndarray,
                half_width: float) -> np.ndarray:
    """
    Calcula a matriz Φ de forças de disparo NORMALIZADAS.
    Φ[i, j] = μ_j(x_i) / Σ_k μ_k(x_i)
    Shape: (len(x), n_sets)
    """
    n, n_sets = len(x), len(centers)
    Phi = np.zeros((n, n_sets))
    for j, c in enumerate(centers):
        Phi[:, j] = _tri_mf(x, c, half_width)

    row_sums = Phi.sum(axis=1, keepdims=True)
    # Pontos fora de todos os suportes → pertinência ao conjunto mais próximo
    zero_rows = (row_sums.ravel() < 1e-14)
    if zero_rows.any():
        nearest = np.argmin(np.abs(x[zero_rows, None] - centers[None, :]),
                            axis=1)
        Phi[zero_rows, :] = 0.0
        for idx, col in zip(np.where(zero_rows)[0], nearest):
            Phi[idx, col] = 1.0
        row_sums = Phi.sum(axis=1, keepdims=True)

    Phi /= (row_sums + 1e-14)
    return Phi


def mamdani_train(x_train: np.ndarray, y_train: np.ndarray,
                  n_sets: int, x_min=None, x_max=None):
    """
    Treina o Mamdani singleton.
    Retorna: centers, half_width, singletons, x_min, x_max
    """
    if x_min is None:
        x_min = float(np.min(x_train))
    if x_max is None:
        x_max = float(np.max(x_train))
    centers, hw  = _build_partition(x_min, x_max, n_sets)
    Phi          = _phi_matrix(x_train, centers, hw)
    singletons   = _lstsq(Phi, y_train)
    return centers, hw, singletons, x_min, x_max


def mamdani_predict(x: np.ndarray, centers: np.ndarray,
                    hw: float, singletons: np.ndarray,
                    x_min: float, x_max: float) -> np.ndarray:
    """
    Defuzzificação por centro de massa (equivalente com Φ normalizado):
        ŷ(x) = Φ(x) @ c
    """
    x_c  = np.clip(x, x_min, x_max)   # clamp fora do domínio de treino
    Phi  = _phi_matrix(x_c, centers, hw)
    return np.dot(Phi, singletons)


def mamdani_select_sets(x_train, y_train, x_val, y_val,
                        max_sets: int = 25, log=None):
    """Seleciona N ótimo de conjuntos fuzzy por R² na validação."""
    x_min = float(np.min(x_train))
    x_max = float(np.max(x_train))
    best  = {'r2': -np.inf, 'n': 3, 'params': None}
    rows  = []

    for n in range(2, max_sets + 1):
        try:
            params = mamdani_train(x_train, y_train, n, x_min, x_max)
            ctrs, hw, sng, xmn, xmx = params
            yp = mamdani_predict(x_val, ctrs, hw, sng, xmn, xmx)
            r2 = r2_score(y_val, yp)
            rows.append((n, r2))
            if r2 > best['r2']:
                best.update(r2=r2, n=n, params=params)
        except Exception:
            pass

    out = log if log else print
    sep = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+"
    out("\n  Seleção de estrutura – Fuzzy Mamdani (R² no conjunto de validação)\n")
    out(f"  {sep}")
    out(f"  | {'Conjuntos N':^10} | {'R²':^10} | {'':^10} |")
    out(f"  {sep}")
    for n, r2 in rows:
        sel = "  ★ MELHOR" if n == best['n'] else ""
        out(f"  | {n:^10d} | {r2:^10.6f} |{sel:<11} |")
    out(f"  {sep}")
    out(f"\n  Estrutura escolhida: N = {best['n']} conjuntos  →  R² = {best['r2']:.6f}")
    return best





# ============================================================
# 5.  MODELO FUZZY TAKAGI-SUGENO  (Ordem 1)
# ============================================================
#
#  Regra i:  SE x é A_i  ENTÃO  y_i = a_i · x + b_i
#  Saída:    ŷ = Σ Φ_i · (a_i·x + b_i)
#           = [Φ_1·x, Φ_1, Φ_2·x, Φ_2, ...] · [a_1,b_1,a_2,b_2,...]
#  → sistema linear em [a_i, b_i], resolvido por mínimos quadrados.
# ──────────────────────────────────────────────────────────────

def ts1_train(x_train: np.ndarray, y_train: np.ndarray,
              n_sets: int, x_min=None, x_max=None):
    """Treina Takagi-Sugeno ordem 1."""
    if x_min is None:
        x_min = float(np.min(x_train))
    if x_max is None:
        x_max = float(np.max(x_train))
    centers, hw = _build_partition(x_min, x_max, n_sets)
    Phi = _phi_matrix(x_train, centers, hw)      # (N, n_sets)

    # Expande: colunas [Φ_1·x, Φ_1, Φ_2·x, Φ_2, ...]
    n = len(x_train)
    X_ts = np.zeros((n, 2 * n_sets))
    for j in range(n_sets):
        X_ts[:, 2 * j]     = Phi[:, j] * x_train   # coef a_j
        X_ts[:, 2 * j + 1] = Phi[:, j]              # coef b_j

    params = _lstsq(X_ts, y_train)
    return centers, hw, params, x_min, x_max


def ts1_predict(x: np.ndarray, centers: np.ndarray,
                hw: float, params: np.ndarray,
                x_min: float, x_max: float) -> np.ndarray:
    n_sets = len(centers)
    x_c    = np.clip(x, x_min, x_max)
    Phi    = _phi_matrix(x_c, centers, hw)
    n      = len(x)
    X_ts   = np.zeros((n, 2 * n_sets))
    for j in range(n_sets):
        X_ts[:, 2 * j]     = Phi[:, j] * x_c
        X_ts[:, 2 * j + 1] = Phi[:, j]
    return np.dot(X_ts, params)


def ts1_select_sets(x_train, y_train, x_val, y_val,
                    max_sets: int = 25, log=None):
    x_min = float(np.min(x_train))
    x_max = float(np.max(x_train))
    best  = {'r2': -np.inf, 'n': 3, 'params': None}
    rows  = []

    for n in range(2, max_sets + 1):
        try:
            res = ts1_train(x_train, y_train, n, x_min, x_max)
            ctrs, hw, prm, xmn, xmx = res
            yp  = ts1_predict(x_val, ctrs, hw, prm, xmn, xmx)
            r2  = r2_score(y_val, yp)
            rows.append((n, r2))
            if r2 > best['r2']:
                best.update(r2=r2, n=n, params=res)
        except Exception:
            pass

    out = log if log else print
    sep = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+"
    out("\n  Seleção de estrutura – Takagi-Sugeno Ord.1 (R² no conjunto de validação)\n")
    out(f"  {sep}")
    out(f"  | {'Conjuntos N':^10} | {'R²':^10} | {'':^10} |")
    out(f"  {sep}")
    for n, r2 in rows:
        sel = "  ★ MELHOR" if n == best['n'] else ""
        out(f"  | {n:^10d} | {r2:^10.6f} |{sel:<11} |")
    out(f"  {sep}")
    out(f"\n  Estrutura escolhida: N = {best['n']} conjuntos  →  R² = {best['r2']:.6f}")
    return best



# ============================================================
# 6.  LOGGER DUPLO  (terminal + arquivo de texto)
# ============================================================

class Logger:
    """
    Escreve simultaneamente no terminal (stdout) e num arquivo .txt.
    Uso: log = Logger('arquivo.txt'); log('mensagem')
    """
    def __init__(self, filepath: str):
        self._fh = open(filepath, 'w', encoding='utf-8')

    def __call__(self, msg: str = ''):
        print(msg)
        self._fh.write(msg + '\n')

    def close(self):
        self._fh.close()


# ============================================================
# 7.  FUNÇÕES DE VISUALIZAÇÃO  (figuras para relatório)
# ============================================================

# Estilo global limpo para todas as figuras
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.35,
    'figure.dpi'       : 150,
})

# Paleta de cores por modelo
_CORES = {
    'ajuste'  : '#C62828',   # vermelho escuro
    'dados'   : '#546E7A',   # cinza-azulado
    'hist'    : '#1565C0',   # azul escuro
    'normal'  : '#C62828',
    'scatter' : '#1565C0',
    'diagonal': '#C62828',
}


def _header(title: str, log=None):
    msg = f"\n{'='*62}\n  {title}\n{'='*62}"
    if log:
        log(msg)
    else:
        print(msg)


def _salvar_fig(fig, nome: str, log):
    fig.savefig(nome, dpi=180, bbox_inches='tight', facecolor='white')
    log(f"  [FIG] {nome}")
    plt.close(fig)


def plot_modelo(x_all, y_all, x_fine, y_fine,
                y_val, y_pred, residuos, r2,
                titulo_modelo: str, nome_arquivo: str, log):
    """
    Gera figura individual de cada modelo com 3 painéis lado a lado:
      Painel A – Ajuste (dados + curva)
      Painel B – Histograma dos resíduos
      Painel C – Dispersão real × predito
    Tamanho adequado para uma página de relatório (A4 paisagem).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(titulo_modelo, fontsize=13, fontweight='bold', y=1.01)

    # ── Painel A: Ajuste ──────────────────────────────────────────────────
    ax = axes[0]
    sort_idx = np.argsort(x_all)
    ax.scatter(x_all[sort_idx], y_all[sort_idx],
               s=7, color=_CORES['dados'], alpha=0.55, label='Dados', zorder=1)
    sort_fine = np.argsort(x_fine)
    ax.plot(x_fine[sort_fine], y_fine[sort_fine],
            color=_CORES['ajuste'], linewidth=2.0, label='Modelo', zorder=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(A) Curva de Ajuste')
    ax.legend()

    # ── Painel B: Histograma dos resíduos ────────────────────────────────
    ax = axes[1]
    ax.hist(residuos, bins=30, color=_CORES['hist'], edgecolor='white',
            alpha=0.80, density=True, label='Resíduos')
    mu  = float(np.mean(residuos))
    sig = float(np.std(residuos)) + 1e-14
    xs  = np.linspace(float(residuos.min()), float(residuos.max()), 300)
    pdf = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))
    ax.plot(xs, pdf, color=_CORES['normal'], linewidth=2.0,
            label=f'Normal\n(μ={mu:.2f}, σ={sig:.2f})')
    ax.axvline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.set_xlabel('Resíduo')
    ax.set_ylabel('Densidade')
    ax.set_title('(B) Histograma dos Resíduos')
    ax.legend()

    # ── Painel C: Real × Predito ──────────────────────────────────────────
    ax = axes[2]
    lo = min(float(np.min(y_val)), float(np.min(y_pred)))
    hi = max(float(np.max(y_val)), float(np.max(y_pred)))
    ax.scatter(y_val, y_pred, s=10, color=_CORES['scatter'], alpha=0.55, zorder=2)
    ax.plot([lo, hi], [lo, hi], color=_CORES['diagonal'],
            linewidth=1.8, linestyle='--', label='Perfeito (y = ŷ)', zorder=1)
    ax.set_xlabel('Valor Real (y)')
    ax.set_ylabel('Valor Predito (ŷ)')
    ax.set_title(f'(C) Real × Predito  |  R² = {r2:.4f}')
    ax.legend()

    fig.tight_layout()
    _salvar_fig(fig, nome_arquivo, log)


def plot_mfs(centers, hw, x_min, x_max, n_sets: int,
             nome_arquivo: str, log):
    """Figura isolada das funções de pertinência Mamdani."""
    fig, ax = plt.subplots(figsize=(12, 4))
    xs     = np.linspace(x_min, x_max, 600)
    colors = plt.cm.tab20(np.linspace(0, 1, n_sets))
    for j, (c, col) in enumerate(zip(centers, colors)):
        mu = _tri_mf(xs, c, hw)
        ax.plot(xs, mu, color=col, linewidth=1.6, label=f'A{j+1} (c={c:.1f})')
    ax.set_xlabel('x')
    ax.set_ylabel('μ(x)')
    ax.set_ylim(-0.05, 1.12)
    ax.set_title(f'Funções de Pertinência Triangulares – Mamdani  ({n_sets} conjuntos)')
    if n_sets <= 15:
        ax.legend(ncol=min(5, n_sets), fontsize=8,
                  loc='upper right', framealpha=0.7)
    fig.tight_layout()
    _salvar_fig(fig, nome_arquivo, log)


def plot_comparacao_r2(summary: dict, nome_arquivo: str, log):
    """Gráfico de barras comparando R² de todos os modelos."""
    nomes   = list(summary.keys())
    r2_vals = [v[0] for v in summary.values()]
    cores   = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(nomes)), r2_vals,
                  color=cores[:len(nomes)], edgecolor='white',
                  linewidth=1.5, width=0.55)
    ax.set_xticks(range(len(nomes)))
    ax.set_xticklabels(nomes, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_title('Comparação de R² – Conjunto de Validação (70/30)', fontsize=13)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    fig.tight_layout()
    _salvar_fig(fig, nome_arquivo, log)


# ============================================================
# 8.  AVALIAÇÃO COM LOG
# ============================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             label: str, log) -> tuple:
    """Calcula e registra R², correlação e resíduos."""
    res  = y_true - y_pred
    r2   = r2_score(y_true, y_pred)
    r    = pearson_r(y_true, y_pred)
    r2_r = r ** 2
    diff = abs(r2 - r2_r)
    ok   = "(≈ 0  ✓)" if diff < 1e-6 else "(diferem!)"

    log('')
    log(f'  Métricas finais – {label}')
    log(f'  {"─"*46}')
    log(f'  {"Métrica":<30} {"Valor":>12}')
    log(f'  {"─"*46}')
    log(f'  {"R²":<30} {r2:>12.6f}')
    log(f'  {"Correlação de Pearson (r)":<30} {r:>12.6f}')
    log(f'  {"r²":<30} {r2_r:>12.6f}')
    log(f'  {"|R² − r²|":<30} {diff:>12.2e}  {ok}')
    log(f'  {"Resíduo médio (bias)":<30} {float(np.mean(res)):>12.4f}')
    log(f'  {"Desvio padrão dos resíduos":<30} {float(np.std(res)):>12.4f}')
    log(f'  {"─"*46}')
    return r2, r, res


# ============================================================
# 9.  FUNÇÃO PRINCIPAL
# ============================================================

def main():
    # ── Inicializa logger ────────────────────────────────────────────────────
    log = Logger('resultados_gauss3.txt')

    log('=' * 62)
    log('  TIP7077 – Inteligência Computacional Aplicada  (PPGETI)')
    log('  1.º Trabalho Computacional  –  Dataset Gauss3  (NIST/StRD)')
    log('=' * 62)

    # ── 9.1  Carregar dados ──────────────────────────────────────────────────
    fname = 'Gauss3.dat'
    if not os.path.exists(fname):
        log(f'\n[ERRO] Arquivo "{fname}" não encontrado.')
        log(f'       Diretório atual: {os.getcwd()}')
        log('       Coloque Gauss3.dat no mesmo diretório do script.')
        sys.exit(1)

    x_all, y_all = load_gauss3(fname)
    sort_idx = np.argsort(x_all)
    x_all, y_all = x_all[sort_idx], y_all[sort_idx]

    log('')
    log('  INFORMAÇÕES DO DATASET')
    log(f'  {"─"*40}')
    log(f'  Arquivo            : {fname}')
    log(f'  Total de amostras  : {len(x_all)}')
    log(f'  Intervalo de x     : [{x_all.min():.4f}, {x_all.max():.4f}]')
    log(f'  Intervalo de y     : [{y_all.min():.4f}, {y_all.max():.4f}]')
    log(f'  Média de y         : {float(np.mean(y_all)):.4f}')
    log(f'  Desvio padrão de y : {float(np.std(y_all)):.4f}')

    # ── 9.2  Divisão treino / validação (70/30) ──────────────────────────────
    x_tr, y_tr, x_val, y_val = split_train_val(x_all, y_all, val_ratio=0.30)

    log('')
    log('  DIVISÃO TREINO / VALIDAÇÃO')
    log(f'  {"─"*40}')
    log(f'  Proporção   : 70% treino  /  30% validação')
    log(f'  Treino      : {len(x_tr)} amostras')
    log(f'  Validação   : {len(x_val)} amostras')
    log(f'  Semente     : 7  (reprodutibilidade)')

    x_fine = np.linspace(float(x_all.min()), float(x_all.max()), 600)
    summary = {}

    # ════════════════════════════════════════════════════════════════════════
    # MODELO 1 – REGRESSÃO POLINOMIAL
    # ════════════════════════════════════════════════════════════════════════
    _header('MODELO 1 – REGRESSÃO POLINOMIAL', log)

    bst_poly   = poly_select_degree(x_tr, y_tr, x_val, y_val, max_degree=14, log=log)
    k          = bst_poly['degree']
    coefs_poly = bst_poly['coefs']
    xmn_poly   = bst_poly['x_min']
    xmx_poly   = bst_poly['x_max']

    y_val_poly  = poly_predict(x_val,  coefs_poly, xmn_poly, xmx_poly)
    y_fine_poly = poly_predict(x_fine, coefs_poly, xmn_poly, xmx_poly)

    lbl_poly = f'Polinomial (k={k})'
    r2_p, r_p, res_p = evaluate(y_val, y_val_poly, lbl_poly, log)
    summary[lbl_poly] = (r2_p, r_p)

    log('')
    log('  Observações sobre os gráficos – Modelo 1:')
    log('  (A) Curva de Ajuste: verifica se o polinômio captura a tendência')
    log('      global dos dados sem oscilações excessivas (sobreajuste).')
    log('  (B) Histograma dos Resíduos: para um bom ajuste espera-se')
    log('      distribuição aproximadamente gaussiana e centrada em zero.')
    log('      Assimetria ou caudas pesadas indicam padrão sistemático não')
    log('      capturado pelo modelo.')
    log('  (C) Real × Predito: pontos sobre a diagonal y=ŷ indicam ajuste')
    log('      perfeito. Desvios sistemáticos revelam viés do modelo.')
    log('  Nota: para regressão linear nos parâmetros, r² = R² exatamente.')
    log(f'  Arquivo de figura: fig_modelo1_polinomial.png')

    plot_modelo(x_all, y_all, x_fine, y_fine_poly,
                y_val, y_val_poly, res_p, r2_p,
                f'Modelo 1 – Regressão Polinomial  (grau k = {k})',
                'fig_modelo1_polinomial.png', log)

    # ════════════════════════════════════════════════════════════════════════
    # MODELO 2 – LINEAR POR PARTES
    # ════════════════════════════════════════════════════════════════════════
    _header('MODELO 2 – REGRESSÃO LINEAR POR PARTES', log)

    bst_pw     = piecewise_select_segments(x_tr, y_tr, x_val, y_val, max_seg=30, log=log)
    n_seg      = bst_pw['n_seg']
    segs_final = bst_pw['segs']

    y_val_pw  = piecewise_predict(x_val,  segs_final)
    y_fine_pw = piecewise_predict(x_fine, segs_final)

    lbl_pw = f'Linear por Partes ({n_seg} seg)'
    r2_pw, r_pw, res_pw = evaluate(y_val, y_val_pw, lbl_pw, log)
    summary[lbl_pw] = (r2_pw, r_pw)

    log('')
    log('  Observações sobre os gráficos – Modelo 2:')
    log('  (A) Curva de Ajuste: a curva é uma sequência de segmentos lineares.')
    log('      Mais segmentos → maior capacidade de aproximação local.')
    log('  (B) Histograma dos Resíduos: com número adequado de segmentos,')
    log('      os resíduos devem se aproximar de gaussiana centrada em zero.')
    log('  (C) Real × Predito: o alinhamento com a diagonal indica qualidade')
    log('      do ajuste nos dados de validação (nunca vistos no treino).')
    log(f'  Arquivo de figura: fig_modelo2_piecewise.png')

    plot_modelo(x_all, y_all, x_fine, y_fine_pw,
                y_val, y_val_pw, res_pw, r2_pw,
                f'Modelo 2 – Regressão Linear por Partes  ({n_seg} segmentos)',
                'fig_modelo2_piecewise.png', log)

    # ════════════════════════════════════════════════════════════════════════
    # MODELO 3 – FUZZY MAMDANI (singleton)
    # ════════════════════════════════════════════════════════════════════════
    _header('MODELO 3 – FUZZY MAMDANI (saídas singleton)', log)

    bst_mam = mamdani_select_sets(x_tr, y_tr, x_val, y_val, max_sets=25, log=log)
    n_mam   = bst_mam['n']
    ctrs_mam, hw_mam, sng_mam, xmn_mam, xmx_mam = bst_mam['params']

    # ── Regras e singletons ──────────────────────────────────────────────────
    LABELS = [
        'muito baixo',   'baixo-moderado', 'médio-baixo',    'médio',
        'médio-alto',    'alto-moderado',  'alto',            'muito alto',
        'nível 9',       'nível 10',       'nível 11',        'nível 12',
        'nível 13',      'nível 14',       'nível 15',        'nível 16',
        'nível 17',      'nível 18',       'nível 19',        'nível 20',
        'nível 21',      'nível 22',       'nível 23',        'nível 24',
        'nível 25',
    ]
    sep66 = '=' * 66
    log('')
    log(sep66)
    log('  REGRAS DO MODELO FUZZY MAMDANI  –  saídas singleton')
    log(sep66)
    log(f'  {"#":>4}  {"Conjunto A_i":<22}  {"Centro (x)":>10}  {"Singleton c_i (y)":>17}')
    log(f'  {"─"*60}')
    for i, (c, s) in enumerate(zip(ctrs_mam, sng_mam)):
        lbl = LABELS[i] if i < len(LABELS) else f'A_{i+1}'
        log(f'  Regra {i+1:>2d}:  SE x é "{lbl:<20}"'
            f'  (centro={c:>8.3f})  →  y = {s:>10.4f}')
    log(sep66)

    y_val_mam  = mamdani_predict(x_val,  ctrs_mam, hw_mam, sng_mam, xmn_mam, xmx_mam)
    y_fine_mam = mamdani_predict(x_fine, ctrs_mam, hw_mam, sng_mam, xmn_mam, xmx_mam)

    lbl_mam = f'Fuzzy Mamdani ({n_mam} conj.)'
    r2_mam, r_mam, res_mam = evaluate(y_val, y_val_mam, lbl_mam, log)
    summary[lbl_mam] = (r2_mam, r_mam)

    log('')
    log('  Observações sobre os gráficos – Modelo 3:')
    log('  (A) Curva de Ajuste: a curva suave resulta da interpolação ponderada')
    log('      dos singletons pelas pertinências triangulares normalizadas.')
    log('  (B) Histograma dos Resíduos: com N suficiente, os resíduos devem')
    log('      aproximar uma gaussiana. N baixo gera resíduos com estrutura.')
    log('  (C) Real × Predito: o alinhamento com y=ŷ valida a capacidade')
    log('      de generalização do modelo fuzzy para dados não vistos.')
    log(f'  Arquivos de figura: fig_modelo3_mamdani.png  /  fig_modelo3_mfs.png')

    plot_modelo(x_all, y_all, x_fine, y_fine_mam,
                y_val, y_val_mam, res_mam, r2_mam,
                f'Modelo 3 – Fuzzy Mamdani  ({n_mam} conjuntos, saídas singleton)',
                'fig_modelo3_mamdani.png', log)

    plot_mfs(ctrs_mam, hw_mam, xmn_mam, xmx_mam, n_mam,
             'fig_modelo3_mfs.png', log)

    # ════════════════════════════════════════════════════════════════════════
    # MODELO 4 – TAKAGI-SUGENO ORDEM 1
    # ════════════════════════════════════════════════════════════════════════
    _header('MODELO 4 – FUZZY TAKAGI-SUGENO ORDEM 1', log)

    bst_ts = ts1_select_sets(x_tr, y_tr, x_val, y_val, max_sets=25, log=log)
    n_ts   = bst_ts['n']
    ctrs_ts, hw_ts, prm_ts, xmn_ts, xmx_ts = bst_ts['params']

    sep70 = '=' * 70
    log('')
    log(sep70)
    log('  REGRAS DO MODELO TAKAGI-SUGENO ORDEM 1')
    log(sep70)
    log(f'  {"Regra":<8}  {"Centro A_i":>12}  {"a_i (coef. x)":>15}  {"b_i (constante)":>15}')
    log(f'  {"─"*56}')
    for i in range(n_ts):
        a_i = prm_ts[2 * i]
        b_i = prm_ts[2 * i + 1]
        log(f'  Regra {i+1:>2d}:  centro = {ctrs_ts[i]:>8.3f}  '
            f'→  y = {a_i:>+10.4f}·x  {b_i:>+10.4f}')
    log(sep70)

    y_val_ts  = ts1_predict(x_val,  ctrs_ts, hw_ts, prm_ts, xmn_ts, xmx_ts)
    y_fine_ts = ts1_predict(x_fine, ctrs_ts, hw_ts, prm_ts, xmn_ts, xmx_ts)

    lbl_ts = f'Takagi-Sugeno Ord.1 ({n_ts} conj.)'
    r2_ts, r_ts, res_ts = evaluate(y_val, y_val_ts, lbl_ts, log)
    summary[lbl_ts] = (r2_ts, r_ts)

    log('')
    log('  Observações sobre os gráficos – Modelo 4:')
    log('  (A) Curva de Ajuste: TS-1 usa funções afins por regra, o que confere')
    log('      maior flexibilidade local do que singletons constantes (TS-0).')
    log('  (B) Histograma dos Resíduos: espera-se gaussiana centrada em zero;')
    log('      resíduos menores que o Mamdani para mesma quantidade de regras.')
    log('  (C) Real × Predito: TS-1 tende a apresentar R² mais alto que')
    log('      Mamdani, pois cada regra possui graus de liberdade adicionais.')
    log(f'  Arquivo de figura: fig_modelo4_ts1.png')

    plot_modelo(x_all, y_all, x_fine, y_fine_ts,
                y_val, y_val_ts, res_ts, r2_ts,
                f'Modelo 4 – Takagi-Sugeno Ordem 1  ({n_ts} conjuntos)',
                'fig_modelo4_ts1.png', log)

    # ════════════════════════════════════════════════════════════════════════
    # RESUMO COMPARATIVO FINAL
    # ════════════════════════════════════════════════════════════════════════
    _header('RESUMO COMPARATIVO – TODOS OS MODELOS', log)

    log('')
    log('  Avaliação no conjunto de VALIDAÇÃO (30% dos dados, nunca vistos')
    log('  durante o treino). Todos os modelos foram selecionados e treinados')
    log('  exclusivamente com os 70% de treino.')
    log('')
    sep_tab = '+' + '-'*44 + '+' + '-'*10 + '+' + '-'*10 + '+' + '-'*10 + '+' + '-'*14 + '+'
    log(sep_tab)
    log(f'| {"Modelo":<42} | {"R²":^8} | {"r":^8} | {"r²":^8} | {"|R²−r²|":^12} |')
    log(sep_tab)
    for nome, (r2, r) in summary.items():
        diff = abs(r2 - r**2)
        log(f'| {nome:<42} | {r2:^8.4f} | {r:^8.4f} | {r**2:^8.4f} | {diff:^12.2e} |')
    log(sep_tab)
    log('')
    log('  Interpretação da tabela:')
    log('  • R²: proporção da variância de y explicada pelo modelo (0–1).')
    log('  • r : correlação de Pearson entre valores reais e preditos.')
    log('  • r²: quadrado da correlação; deve ser igual ao R² para modelos')
    log('        lineares nos parâmetros (MQ). A coluna |R²−r²| verifica isso.')
    log('  • Quanto mais próximo de 1.0, melhor o ajuste.')

    plot_comparacao_r2(summary, 'fig_comparacao_r2.png', log)

    # ── Arquivos gerados ─────────────────────────────────────────────────────
    log('')
    log('  ARQUIVOS GERADOS')
    log(f'  {"─"*50}')
    arquivos = [
        ('fig_modelo1_polinomial.png', 'Ajuste + resíduos + dispersão – Modelo 1'),
        ('fig_modelo2_piecewise.png',  'Ajuste + resíduos + dispersão – Modelo 2'),
        ('fig_modelo3_mamdani.png',    'Ajuste + resíduos + dispersão – Modelo 3'),
        ('fig_modelo3_mfs.png',        'Funções de pertinência – Mamdani'),
        ('fig_modelo4_ts1.png',        'Ajuste + resíduos + dispersão – Modelo 4'),
        ('fig_comparacao_r2.png',      'Comparação de R² entre modelos'),
        ('resultados_gauss3.txt',      'Este arquivo – todos os resultados'),
    ]
    for fname_out, desc in arquivos:
        log(f'  {fname_out:<34} {desc}')

    log('')
    log('  Execução concluída com sucesso.')
    log('=' * 62)

    log.close()

    try:
        plt.show()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()
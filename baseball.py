# app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests
from tensorflow import keras

# =========================================================
# LLM (Qwen / Ollama) Fixed Config
# =========================================================
LLM_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434"

try:
    from llm_qwen import generate_explanation
    LLM_IMPORT_OK = True
    LLM_IMPORT_ERR = ""
except Exception as _e:
    generate_explanation = None
    LLM_IMPORT_OK = False
    LLM_IMPORT_ERR = str(_e)

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="KBO Next Season Predictor (Δ)", layout="wide")
st.title("⚾ KBO 다음 시즌 성적 예측")

# =========================================================
# Paths
# =========================================================
DEFAULT_CSV = r"C:\Users\yusan\OneDrive\Desktop\2025 winter\Notebook\Colab Notebooks\dataset\kbo_batting_stats.csv"

MODEL_DIR = "model_kbo"
MODEL_PATH = os.path.join(MODEL_DIR, "kbo_mlp.keras")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")
X_SCALER_PATH = os.path.join(MODEL_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")
TARGETS_PATH = os.path.join(MODEL_DIR, "targets.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.pkl")

# =========================================================
# Column names
# =========================================================
ID_COL = "Id"
YEAR_COL = "Year"
AGE_COL = "Age"
TEAM_COL = "Team"
PA_COL = "PA"
NAME_CANDIDATES = ["Name", "Player", "player_name", "선수명", "이름"]

# =========================================================
# 정책(필터/규칙)
# =========================================================
FILTER_TEAM_YEAR = 2025
DEBUT_MIN_YEAR = 2000
PA_MIN_PRED = 223  # ✅ base_year PA 기준으로 예측 가능 여부 판단

# =========================================================
# Utils / Load
# =========================================================
@st.cache_resource
def load_bundle():
    needed = [
        MODEL_PATH,
        IMPUTER_PATH,
        X_SCALER_PATH,
        Y_SCALER_PATH,
        FEATURE_COLS_PATH,
        TARGETS_PATH,
        META_PATH,
    ]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "모델 번들 파일이 없습니다. (model_kbo 폴더 확인)\n"
            f"누락: {missing}"
        )

    model = keras.models.load_model(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    display_targets = joblib.load(TARGETS_PATH)
    meta = joblib.load(META_PATH)

    if meta.get("mode") != "delta":
        raise ValueError(f"meta.pkl의 mode가 delta가 아닙니다: {meta.get('mode')}")

    return model, imputer, x_scaler, y_scaler, feature_cols, display_targets, meta


@st.cache_data
def load_and_prepare_csv(csv_path: str, meta: dict, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"wRC+": "wRC_plus"})

    need = [ID_COL, YEAR_COL, AGE_COL, PA_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

    name_col = None
    for c in NAME_CANDIDATES:
        if c in df.columns:
            name_col = c
            break
    if name_col and name_col != "Name":
        df["Name"] = df[name_col]

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df[PA_COL] = pd.to_numeric(df[PA_COL], errors="coerce")

    base_targets = meta.get("targets", [])
    for t in base_targets:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")

    df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    df["Age2"] = df[AGE_COL] ** 2

    if f"{PA_COL}_prev" not in df.columns:
        df[f"{PA_COL}_prev"] = df.groupby(ID_COL)[PA_COL].shift(1)

    for t in base_targets:
        prev_col = f"{t}_prev"
        if prev_col not in df.columns and t in df.columns:
            df[prev_col] = df.groupby(ID_COL)[t].shift(1)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df


def predict_next_from_row(model, imputer, x_scaler, y_scaler, row_df: pd.DataFrame, feature_cols: list[str], meta: dict):
    x_raw = row_df[feature_cols].copy().fillna(0.0)
    x_imp = imputer.transform(x_raw.values.astype("float32"))
    x = x_scaler.transform(x_imp).astype("float32")

    delta_s = model.predict(x, verbose=0)
    delta = y_scaler.inverse_transform(delta_s)[0]

    delta_targets = meta["delta_targets"]
    current_cols = meta["current_cols"]
    next_names = meta["next_names"]

    delta_pred = dict(zip(delta_targets, [float(v) for v in delta]))

    cur = np.array([float(row_df.iloc[0][c]) for c in current_cols], dtype="float32")
    next_pred_arr = cur + delta.astype("float32")
    next_pred_arr[-1] = max(0.0, float(next_pred_arr[-1]))

    next_pred = dict(zip(next_names, [float(v) for v in next_pred_arr]))
    return next_pred, delta_pred


def pretty_table_next(next_pred: dict):
    out_next = pd.DataFrame([{"Target": k, "Predicted": v} for k, v in next_pred.items()])
    round_map = {
        "AVG_next": 3, "OBP_next": 3, "SLG_next": 3, "OPS_next": 3,
        "WAR_next": 3, "wRC_plus_next": 1,
        "HR_next": 1, "H_next": 1, "RBI_next": 1, "SB_next": 1,
        "PA_next": 0,
    }

    def _round(k, v):
        return round(v, round_map.get(k, 3))

    out_next["Predicted"] = out_next.apply(lambda r: _round(r["Target"], r["Predicted"]), axis=1)

    key_order = [t for t in ["HR_next", "H_next", "RBI_next", "AVG_next", "OPS_next", "WAR_next", "wRC_plus_next", "PA_next"]
                 if t in out_next["Target"].values]
    key_out = out_next[out_next["Target"].isin(key_order)].copy()
    rest_out = out_next[~out_next["Target"].isin(key_order)].copy()

    return key_out, pd.concat([key_out, rest_out], ignore_index=True)


def pretty_table_delta(delta_pred: dict):
    out_delta = pd.DataFrame([{"Target": k, "Delta": v} for k, v in delta_pred.items()])
    d_round_map = {
        "AVG_delta": 3, "OBP_delta": 3, "SLG_delta": 3, "OPS_delta": 3,
        "WAR_delta": 2, "wRC_plus_delta": 1,
        "HR_delta": 1, "H_delta": 1, "RBI_delta": 1, "SB_delta": 1,
        "PA_delta": 0,
    }
    out_delta["Delta"] = out_delta.apply(lambda r: round(r["Delta"], d_round_map.get(r["Target"], 3)), axis=1)
    return out_delta


def map_next_delta_to_base(next_pred: dict, delta_pred: dict):
    pred_next = {k.replace("_next", ""): float(v) for k, v in next_pred.items()}
    pred_delta = {k.replace("_delta", ""): float(v) for k, v in delta_pred.items()}
    return pred_next, pred_delta


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("설정")
    csv_path = st.text_input("데이터 CSV 경로", value=DEFAULT_CSV)
    show_raw = st.checkbox("기준년도 원본 스탯(입력 row) 표시", value=True)
    show_delta = st.checkbox("증감(Δ)도 같이 보기", value=False)

    st.divider()
    st.header("필터/정책")
    st.caption(f"✅ 데뷔 {DEBUT_MIN_YEAR}년 이상 선수만 표시")
    st.caption(f"✅ 팀 필터는 {FILTER_TEAM_YEAR}년 시즌 소속 기준")
    st.caption(f"✅ 예측 가능(시즌 기준): base_year PA ≥ {PA_MIN_PRED}")
    st.caption("✅ Δ모델: next = current + delta")


# =========================================================
# Load bundle
# =========================================================
try:
    model, imputer, x_scaler, y_scaler, feature_cols, display_targets, meta = load_bundle()
except Exception as e:
    st.error(str(e))
    st.stop()

TRAIN_MIN_YEAR = int(meta.get("train_min_year", 2015))

# =========================================================
# Load data
# =========================================================
try:
    df = load_and_prepare_csv(csv_path, meta, feature_cols)
except Exception as e:
    st.error(f"CSV 로드/전처리 실패: {e}")
    st.stop()

missing_feat = [c for c in feature_cols if c not in df.columns]
if missing_feat:
    st.error("CSV에 feature 컬럼이 부족합니다:\n" f"{missing_feat}\n\n" f"현재 컬럼: {list(df.columns)}")
    st.stop()

# =========================================================
# Filters: Debut >= 2000
# =========================================================
debut_year_by_player = df.groupby(ID_COL)[YEAR_COL].min().dropna().astype(int)
eligible_ids = set(debut_year_by_player[debut_year_by_player >= int(DEBUT_MIN_YEAR)].index)

# =========================================================
# Team filter based on roster in 2025
# =========================================================
st.subheader("1) 팀 / 선수 / 기준년도 선택")

if TEAM_COL not in df.columns:
    st.error("CSV에 Team 컬럼이 없습니다. 팀 필터를 사용하려면 Team 컬럼이 필요합니다.")
    st.stop()

teams_2025 = (
    df.loc[df[YEAR_COL].astype("Int64") == int(FILTER_TEAM_YEAR), TEAM_COL]
    .dropna().astype(str).sort_values().unique().tolist()
)

if not teams_2025:
    st.error(f"{FILTER_TEAM_YEAR}년 시즌 데이터에서 Team 값을 찾지 못했습니다. (Year/Team 컬럼 확인)")
    st.stop()

selected_team = st.selectbox("팀 선택 (2025년 기준)", options=["(전체)"] + teams_2025, index=0)

if selected_team != "(전체)":
    ids_in_team_2025 = set(
        df.loc[
            (df[YEAR_COL].astype("Int64") == int(FILTER_TEAM_YEAR)) &
            (df[TEAM_COL].astype(str) == str(selected_team)),
            ID_COL
        ].dropna().tolist()
    )
    eligible_ids = eligible_ids.intersection(ids_in_team_2025)

df_filtered = df[df[ID_COL].isin(list(eligible_ids))].copy()

st.caption(
    f"필터: 데뷔>= {DEBUT_MIN_YEAR}"
    + (f", 2025팀={selected_team}" if selected_team != "(전체)" else ", 2025팀=전체")
    + f" | 선수 수: {df_filtered[ID_COL].nunique()}"
)

if df_filtered.empty:
    st.warning("필터 조건에 맞는 선수가 없습니다. 팀을 (전체)로 바꾸거나 조건을 완화하세요.")
    st.stop()

# =========================================================
# Player select
# =========================================================
if "Name" in df_filtered.columns:
    players = df_filtered[[ID_COL, "Name"]].drop_duplicates().copy()
else:
    players = df_filtered[[ID_COL]].drop_duplicates().copy()

players = players.sort_values(["Name"] if "Name" in players.columns else [ID_COL])
player_options = players["Name"].fillna("선수").tolist() if "Name" in players.columns else players[ID_COL].astype(str).tolist()
player_ids = players[ID_COL].tolist()

sel_idx = st.selectbox(
    "선수 선택",
    options=list(range(len(player_options))),
    format_func=lambda i: player_options[i],
)
player_id = player_ids[sel_idx]

# =========================================================
# Base year select
# =========================================================
years = (
    df_filtered.loc[(df_filtered[ID_COL] == player_id) & (df_filtered[YEAR_COL] >= TRAIN_MIN_YEAR), YEAR_COL]
    .dropna().astype(int).sort_values().unique().tolist()
)
if not years:
    st.warning(f"선택한 선수에 대해 {TRAIN_MIN_YEAR}년 이후 Year 데이터가 없습니다.")
    st.stop()

base_year = st.selectbox("기준년도(base_year) 선택", options=years, index=len(years) - 1)
pred_year = int(base_year) + 1

row = df_filtered[(df_filtered[ID_COL] == player_id) & (df_filtered[YEAR_COL].astype(int) == int(base_year))].copy()
if len(row) != 1:
    st.error(f"선택한 선수/연도 행을 1개로 찾지 못했습니다. (개수={len(row)})")
    st.stop()

base_pa = float(row.iloc[0].get(PA_COL, 0.0) or 0.0)
is_predictable = (base_pa >= PA_MIN_PRED)

st.caption(
    f"선택 시즌 정보: base_year={base_year}, PA={base_pa:.0f} | "
    f"예측 가능 조건: PA ≥ {PA_MIN_PRED} => {'✅ 가능' if is_predictable else '❌ 불가'}"
)

# =========================================================
# Predict
# =========================================================
st.subheader("2) 예측 실행")
run = st.button("🚀 다음 시즌 예측하기", use_container_width=True)

if run:
    if not is_predictable:
        st.error(f"예측 불가: {base_year}년 PA < {PA_MIN_PRED} (표본 부족)")
        st.stop()

    next_pred, delta_pred = predict_next_from_row(model, imputer, x_scaler, y_scaler, row, feature_cols, meta)

    st.success(f"예측 완료! (기준 {base_year} → 예측 {pred_year})")

    key_out, full_out = pretty_table_next(next_pred)

    if show_raw:
        st.markdown("### 📌 기준년도 원본 스탯")
        raw_cols = [c for c in ["Name", ID_COL, TEAM_COL, YEAR_COL, PA_COL] + feature_cols if c in row.columns]
        raw_cols = list(dict.fromkeys(raw_cols))
        raw_cols = [c for c in raw_cols if not c.endswith("_prev")]

        st.dataframe(row[raw_cols], use_container_width=True, hide_index=True)


    st.markdown("### ⭐ 핵심 예측 결과")
    st.dataframe(key_out, use_container_width=True, hide_index=True)

    with st.expander("전체 예측 결과 보기"):
        st.dataframe(full_out, use_container_width=True, hide_index=True)

    if show_delta:
        st.markdown("### 🔁 예측된 증감(Δ)")
        st.dataframe(pretty_table_delta(delta_pred), use_container_width=True, hide_index=True)


    # =========================================================
    # ✅ 자동 LLM 해설 (버튼 없이 바로 생성/표시)
    # =========================================================
    st.markdown("### 🧠 예측 이유 (LLM 해설)")

    if not LLM_IMPORT_OK:
        st.info("LLM 기능을 쓰려면 같은 폴더에 llm_qwen.py가 있어야 합니다.")
        with st.expander("import 에러 보기"):
            st.code(LLM_IMPORT_ERR)
    else:
        out_key = f"auto_reason_text_{player_id}_{base_year}"
        ctx_key = f"auto_reason_ctx_{player_id}_{base_year}"
        reason_box = st.empty()

        # 이미 있으면 재사용
        if out_key in st.session_state and st.session_state[out_key]:
            reason_box.markdown(st.session_state[out_key])
        else:
            pred_next_base, pred_delta_base = map_next_delta_to_base(next_pred, delta_pred)
            default_stat_cols = ["AVG", "OBP", "SLG", "OPS", "WAR", "wRC_plus", "HR", "H", "RBI", "SB", "PA"]
            stat_cols = [c for c in default_stat_cols if c in df_filtered.columns]

            with st.spinner("Qwen이 예측 이유를 생성 중..."):
                try:
                    explanation, ctx_used = generate_explanation(
                        df_all=df_filtered,
                        row_df=row,
                        next_pred=pred_next_base,
                        delta_pred=pred_delta_base,
                        player_id=player_id,
                        base_year=int(base_year),
                        pred_year=int(pred_year),
                        id_col=ID_COL,
                        year_col=YEAR_COL,
                        age_col=AGE_COL,
                        team_col=TEAM_COL,
                        pa_col=PA_COL,
                        stat_cols=stat_cols,
                        age_band=1,
                        pa_min=int(PA_MIN_PRED),
                        same_team_only=False,
                        model_name=LLM_MODEL,
                        base_url=OLLAMA_URL,
                    )

                    st.session_state[out_key] = explanation
                    st.session_state[ctx_key] = ctx_used
                    reason_box.markdown(explanation if explanation.strip() else "LLM 응답이 비어 있습니다.")

                except requests.exceptions.ConnectionError:
                    reason_box.error("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
                except Exception as e:
                    reason_box.error(f"LLM 해설 생성 실패: {e}")

        if ctx_key in st.session_state:
            with st.expander("LLM 근거 JSON(디버그) 보기"):
                st.json(st.session_state[ctx_key])

    st.caption("※ 이 예측은 참고용이며 실제 결과를 보장하지 않습니다.")

else:
    st.info("팀/선수/기준년도를 선택한 뒤 '다음 시즌 예측하기'를 누르세요.")

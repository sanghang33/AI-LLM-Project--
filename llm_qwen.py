# llm_qwen.py
# - Ollama(Qwen) 해설 생성 (근거 JSON은 LLM이 참고만, 화면에는 출력 안 하도록 유도)
# - 프롬프트(JSON) 크기 줄이기(핵심 스탯만)로 속도/timeout 개선
# - timeout(read) 기본 300초로 상향
# - 기본 모델: qwen2.5:3b

import json
import requests
import numpy as np
import pandas as pd


# =========================
# Helpers
# =========================
def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _to_py(v):
    if isinstance(v, (np.integer, np.int64)):
        return int(v)
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    return v


def _single_row_dict(row_df: pd.DataFrame) -> dict:
    r = row_df.iloc[0].to_dict()
    return {k: _to_py(v) for k, v in r.items()}


def _pick_name(row_dict: dict) -> str:
    for k in ["Name", "Player", "player_name", "선수명", "이름"]:
        if k in row_dict and row_dict[k] not in [None, ""]:
            return str(row_dict[k])
    return "선수"


def _round_stat_map():
    return {
        "AVG": 3, "OBP": 3, "SLG": 3, "OPS": 3,
        "WAR": 2,
        "wRC_plus": 1, "wRC+": 1,
        "HR": 1, "H": 1, "RBI": 1, "SB": 1,
        "PA": 0,
    }


def _round_dict(d: dict) -> dict:
    rm = _round_stat_map()
    out = {}
    for k, v in (d or {}).items():
        base_k = k.replace("_next", "").replace("_delta", "")
        dec = rm.get(base_k, 3)
        try:
            out[k] = round(float(v), dec)
        except Exception:
            out[k] = v
    return out


# =========================
# Prompt size reducer (핵심: JSON 줄이기)
# =========================
DEFAULT_LLM_STAT_COLS = ["OPS", "WAR", "wRC_plus", "AVG", "HR", "RBI", "PA"]


def _filter_stats_dict(d: dict, keep_cols) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k in keep_cols:
        if k in d:
            out[k] = d[k]
    return out


def _compact_career_series(series: list, keep_cols) -> list:
    if not isinstance(series, list):
        return []
    out = []
    for item in series:
        if not isinstance(item, dict):
            continue
        new_item = {"year": item.get("year")}
        for k in keep_cols:
            if k in item:
                new_item[k] = item[k]
        out.append(new_item)
    return out


def make_compact_context_for_llm(context: dict, keep_stat_cols=None) -> dict:
    """
    LLM에 전달할 context만 작게 만든다.
    - current/pred_next/pred_delta: keep_stat_cols만
    - career_recent: series/yoy도 keep_stat_cols만
    - age_peers: mean/median/peer_yoy_mean_delta도 keep_stat_cols만
    """
    if keep_stat_cols is None:
        keep_stat_cols = DEFAULT_LLM_STAT_COLS

    ctx = json.loads(json.dumps(context, ensure_ascii=False, default=_to_py))

    # current / preds
    ctx["current"] = _filter_stats_dict(ctx.get("current", {}), keep_stat_cols)
    ctx["pred_next"] = _filter_stats_dict(ctx.get("pred_next", {}), keep_stat_cols)
    ctx["pred_delta"] = _filter_stats_dict(ctx.get("pred_delta", {}), keep_stat_cols)

    # career_recent
    cr = ctx.get("career_recent", {}) or {}
    if isinstance(cr, dict):
        cr["yoy_delta_base_vs_prev"] = _filter_stats_dict(cr.get("yoy_delta_base_vs_prev", {}), keep_stat_cols)
        cr["series"] = _compact_career_series(cr.get("series", []), keep_stat_cols)
        ctx["career_recent"] = cr

    # age_peers
    ap = ctx.get("age_peers", {}) or {}
    if isinstance(ap, dict):
        ap["mean"] = _filter_stats_dict(ap.get("mean", {}), keep_stat_cols)
        ap["median"] = _filter_stats_dict(ap.get("median", {}), keep_stat_cols)
        ap["peer_yoy_mean_delta"] = _filter_stats_dict(ap.get("peer_yoy_mean_delta", {}), keep_stat_cols)
        ctx["age_peers"] = ap

    # notes는 길어질 수 있어서 제거(원하면 주석 해제해서 유지 가능)
    ctx.pop("notes", None)

    return ctx


# =========================
# Context builders
# =========================
def build_career_context(
    df_all: pd.DataFrame,
    player_id,
    base_year: int,
    id_col="Id",
    year_col="Year",
    age_col="Age",
    team_col="Team",
    pa_col="PA",
    stat_cols=None,
    lookback=3,
):
    """
    최근 lookback 시즌의 시계열 + base_year vs base_year-1 YoY Δ
    """
    if stat_cols is None:
        stat_cols = ["AVG", "OBP", "SLG", "OPS", "WAR", "wRC_plus", "HR", "H", "RBI", "SB", "PA"]

    me_all = df_all[df_all[id_col] == player_id].copy()
    if me_all.empty:
        return {}

    me_all[year_col] = pd.to_numeric(me_all[year_col], errors="coerce")
    me_all = me_all.dropna(subset=[year_col]).sort_values(year_col)

    me_hist = me_all[me_all[year_col].astype(int) <= int(base_year)].copy()
    if me_hist.empty:
        return {}

    years = sorted(me_hist[year_col].astype(int).unique().tolist())[-lookback:]
    me_hist = me_hist[me_hist[year_col].astype(int).isin(years)].sort_values(year_col)

    keep = []
    for c in ["Name", id_col, team_col, year_col, age_col, pa_col]:
        if c in me_hist.columns:
            keep.append(c)
    for c in stat_cols:
        if c in me_hist.columns and c not in keep:
            keep.append(c)

    me_hist = me_hist[keep].copy()

    for c in stat_cols:
        if c in me_hist.columns:
            me_hist[c] = pd.to_numeric(me_hist[c], errors="coerce")

    series = []
    for _, r in me_hist.iterrows():
        item = {"year": int(r[year_col])}
        for c in stat_cols:
            if c in me_hist.columns:
                item[c] = _to_py(r.get(c, None))
        series.append(item)

    yoy = {}
    prev_year = int(base_year) - 1
    base_row = me_all[me_all[year_col].astype(int) == int(base_year)]
    prev_row = me_all[me_all[year_col].astype(int) == int(prev_year)]
    if (not base_row.empty) and (not prev_row.empty):
        b = base_row.iloc[0]
        p = prev_row.iloc[0]
        for c in stat_cols:
            if c in me_all.columns:
                bv = pd.to_numeric(b.get(c, np.nan), errors="coerce")
                pv = pd.to_numeric(p.get(c, np.nan), errors="coerce")
                if pd.notna(bv) and pd.notna(pv):
                    yoy[c] = float(bv - pv)

    return {
        "available_years": years,
        "series": series,
        "yoy_delta_base_vs_prev": yoy,
    }


def build_age_peer_context(
    df_all: pd.DataFrame,
    player_id,
    base_year: int,
    id_col="Id",
    year_col="Year",
    age_col="Age",
    team_col="Team",
    pa_col="PA",
    stat_cols=None,
    age_band=1,
    pa_min=223,
    same_team_only=False,
    include_peer_yoy=True,
):
    """
    동나이대(±age_band) 집단 요약 (base_year 동일, PA>=pa_min)
    + (옵션) peer_yoy_mean_delta: (base_year - base_year-1) 평균 Δ
    """
    if stat_cols is None:
        stat_cols = ["AVG", "OBP", "SLG", "OPS", "WAR", "wRC_plus", "HR", "H", "RBI", "SB", "PA"]

    df = df_all.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df[pa_col] = pd.to_numeric(df[pa_col], errors="coerce")

    me = df[(df[id_col] == player_id) & (df[year_col].astype(int) == int(base_year))].copy()
    if me.empty:
        return {}

    my_age = _safe_float(me.iloc[0].get(age_col, np.nan), default=np.nan)
    my_team = str(me.iloc[0].get(team_col, ""))

    peers = df[df[year_col].astype(int) == int(base_year)].copy()

    if not np.isnan(my_age):
        peers = peers[(peers[age_col] >= my_age - age_band) & (peers[age_col] <= my_age + age_band)]

    peers = peers[peers[pa_col] >= float(pa_min)]

    if same_team_only and team_col in peers.columns:
        peers = peers[peers[team_col].astype(str) == my_team]

    for c in stat_cols:
        if c in peers.columns:
            peers[c] = pd.to_numeric(peers[c], errors="coerce")

    numeric_cols = [c for c in stat_cols if c in peers.columns]

    out = {
        "definition": f"base_year 동일, 나이±{age_band}세, PA>={pa_min}" + (", 같은 팀" if same_team_only else ""),
        "count": int(peers[id_col].nunique()) if id_col in peers.columns else int(len(peers)),
    }

    if not peers.empty and numeric_cols:
        out["mean"] = peers[numeric_cols].mean(numeric_only=True).to_dict()
        out["median"] = peers[numeric_cols].median(numeric_only=True).to_dict()

    if include_peer_yoy:
        prev_year = int(base_year) - 1
        df_prev = df[df[year_col].astype(int) == prev_year].copy()

        peer_ids = set(peers[id_col].dropna().tolist())
        df_base_peer = df[(df[year_col].astype(int) == int(base_year)) & (df[id_col].isin(peer_ids))].copy()
        df_prev_peer = df_prev[df_prev[id_col].isin(peer_ids)].copy()

        base_keep = [id_col] + [c for c in numeric_cols if c in df_base_peer.columns]
        prev_keep = [id_col] + [c for c in numeric_cols if c in df_prev_peer.columns]
        dfb = df_base_peer[base_keep].copy()
        dfp = df_prev_peer[prev_keep].copy()

        for c in numeric_cols:
            if c in dfb.columns:
                dfb[c] = pd.to_numeric(dfb[c], errors="coerce")
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

        merged = pd.merge(dfb, dfp, on=id_col, suffixes=("_b", "_p"))
        if not merged.empty:
            deltas = {}
            for c in numeric_cols:
                cb = f"{c}_b"
                cp = f"{c}_p"
                if cb in merged.columns and cp in merged.columns:
                    diff = merged[cb] - merged[cp]
                    deltas[c] = float(diff.mean(skipna=True))
            out["peer_yoy_mean_delta"] = deltas

    return out


# =========================
# Prompt
# =========================
def build_explain_prompt(context: dict) -> str:
    # ✅ LLM에 전달하는 근거 context를 "작게" 만든다
    ctx = make_compact_context_for_llm(context, keep_stat_cols=DEFAULT_LLM_STAT_COLS)

    # round for readability
    for key in ["current", "pred_next", "pred_delta"]:
        if key in ctx and isinstance(ctx[key], dict):
            ctx[key] = _round_dict(ctx[key])

    if "career_recent" in ctx and isinstance(ctx["career_recent"], dict):
        if "yoy_delta_base_vs_prev" in ctx["career_recent"]:
            ctx["career_recent"]["yoy_delta_base_vs_prev"] = _round_dict(ctx["career_recent"]["yoy_delta_base_vs_prev"])

    if "age_peers" in ctx and isinstance(ctx["age_peers"], dict):
        if "mean" in ctx["age_peers"]:
            ctx["age_peers"]["mean"] = _round_dict(ctx["age_peers"]["mean"])
        if "median" in ctx["age_peers"]:
            ctx["age_peers"]["median"] = _round_dict(ctx["age_peers"]["median"])
        if "peer_yoy_mean_delta" in ctx["age_peers"]:
            ctx["age_peers"]["peer_yoy_mean_delta"] = _round_dict(ctx["age_peers"]["peer_yoy_mean_delta"])

    # ✅ JSON을 "압축"해서 토큰/시간 절약
    context_json = json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))

    return f"""너는 KBO 타자 성적 예측(Δ모델) 결과를 설명하는 데이터 분석가다.
반드시 [근거 JSON]에 포함된 숫자와 사실만 사용해서 해설해라.
부상/멘탈/코치/트레이드/컨디션/타구질 같은 외부 요인은 근거 JSON에 없으면 절대 추측하지 마라.

🚫 매우 중요(출력 제한):
- 근거 JSON(원문)을 절대로 그대로 출력하지 마라.
- JSON/코드블록/```/괄호 {{}} 형태로 재출력 금지.
- 출력에는 "근거 JSON", "JSON", "context_json" 같은 단어도 쓰지 마라.
- 숫자는 문장 속에 녹여서 설명만 해라.

[출력 형식]
- 한국어 8~12문장
- 문장형 서술로 작성(표/코드/불릿 금지)
- 반드시 포함할 것:
  1) 한 줄 결론: “다음 시즌은 전년 대비 ○○ 방향(상승/하락/유지)으로 예측” (OPS/WAR 중심)
  2) 전년도(base_year)의 현재 스탯과 예측(next), 그리고 Δ(증감) 연결 설명
  3) 커리어 흐름(최근 N년): 최근 2~3년의 변화가 예측에 어떻게 반영됐는지
  4) 동나이대 비교: 동나이대(±1세) 집단 평균/중앙값과 비교해서 “평균 대비 어떤 편인지”
  5) PA(타석) 변화가 예측 해석에 주는 의미(표본/출전기회 관점) — 단, 근거 JSON 수치로만
  6) 마지막 문장에 “참고용” 안내

[해설 규칙]
- OPS, WAR, wRC+를 우선으로 설명하고, HR/RBI/AVG는 보조로 사용
- “동나이대 평균적인 aging trend(증감)”과 “선수 개인 커리어 trend(증감)”이
  같은 방향이면 “추세 일치”, 반대면 “추세 역행/상쇄”로 표현해라.
- 숫자는 AVG/OPS 3자리, WAR 2자리, wRC+ 1자리, PA는 정수로 말해라.

[근거 JSON]
{context_json}
"""


# =========================
# Ollama client
# =========================
def ollama_chat(
    prompt: str,
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.4,
    top_p: float = 0.9,
    timeout: int = 300,  # ✅ read timeout 기본 300초
) -> str:
    """
    - LLM은 근거 JSON을 참고하지만
    - 응답은 스키마(format)로 강제해서 "text"만 돌려받도록 시도
    """
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "너는 야구 데이터 해설을 근거 기반으로 하는 분석가다."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature, "top_p": top_p},
        # ✅ 출력 형식 강제 (환경/버전에 따라 무시될 수도 있음)
        "format": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }

    # ✅ connect=10초, read=timeout초
    r = requests.post(url, json=payload, timeout=(10, timeout))
    r.raise_for_status()
    data = r.json()

    content = (data.get("message", {}) or {}).get("content", "") or ""
    content = content.strip()

    # format이 먹으면 content가 JSON 문자열로 올 가능성 큼 -> text만 뽑기
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "text" in parsed:
            return str(parsed["text"]).strip()
    except Exception:
        pass

    # 보험: 혹시 모델이 근거 블록을 출력하면 잘라버림
    marker = "[근거 JSON]"
    if marker in content:
        content = content.split(marker, 1)[0].strip()

    return content


def generate_explanation(
    df_all: pd.DataFrame,
    row_df: pd.DataFrame,
    next_pred: dict,
    delta_pred: dict,
    player_id,
    base_year: int,
    pred_year: int,
    id_col="Id",
    year_col="Year",
    age_col="Age",
    team_col="Team",
    pa_col="PA",
    stat_cols=None,
    age_band=1,
    pa_min=223,
    same_team_only=False,
    model_name: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    return_context: bool = False,  # ✅ 기본: 디버그 context 반환 안 함
    llm_timeout: int = 300,         # ✅ 필요하면 여기만 늘리면 됨
) -> tuple[str, dict | None]:
    """
    반환:
      - explanation_text
      - context_used (디버그용)  -> return_context=True일 때만
    """
    if stat_cols is None:
        stat_cols = ["AVG", "OBP", "SLG", "OPS", "WAR", "wRC_plus", "HR", "H", "RBI", "SB", "PA"]

    row_dict = _single_row_dict(row_df)
    player_name = _pick_name(row_dict)
    team = str(row_dict.get(team_col, ""))

    current = {}
    for c in stat_cols:
        if c in row_df.columns:
            current[c] = _safe_float(row_df.iloc[0].get(c, np.nan), default=np.nan)

    career_recent = build_career_context(
        df_all=df_all,
        player_id=player_id,
        base_year=base_year,
        id_col=id_col,
        year_col=year_col,
        age_col=age_col,
        team_col=team_col,
        pa_col=pa_col,
        stat_cols=stat_cols,
        lookback=3,
    )

    age_peers = build_age_peer_context(
        df_all=df_all,
        player_id=player_id,
        base_year=base_year,
        id_col=id_col,
        year_col=year_col,
        age_col=age_col,
        team_col=team_col,
        pa_col=pa_col,
        stat_cols=stat_cols,
        age_band=age_band,
        pa_min=pa_min,
        same_team_only=same_team_only,
        include_peer_yoy=True,
    )

    context = {
        "player": {"id": _to_py(player_id), "name": player_name, "team": team},
        "season": {
            "base_year": int(base_year),
            "pred_year": int(pred_year),
            "age": _to_py(row_dict.get(age_col, None)),
            "pa_min_policy": int(pa_min),
            "peer_age_band": int(age_band),
            "peer_same_team_only": bool(same_team_only),
        },
        "current": current,
        "pred_delta": delta_pred,
        "pred_next": next_pred,
        "career_recent": career_recent,
        "age_peers": age_peers,
        "notes": {
            "explain_policy": "근거 JSON 숫자만 사용, 외부 요인 추측 금지",
            "model_type": "delta_model (next = current + delta)",
        },
    }

    prompt = build_explain_prompt(context)
    text = ollama_chat(
        prompt=prompt,
        model=model_name,
        base_url=base_url,
        temperature=0.4,
        top_p=0.9,
        timeout=llm_timeout,
    )

    return text, (context if return_context else None)

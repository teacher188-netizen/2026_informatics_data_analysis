import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, f1_score,
    confusion_matrix
)

st.set_page_config(page_title="데이터 탐험실", page_icon="🔎", layout="wide")
st.title("🔎 데이터 탐험실")
st.caption("질문 만들기 → 데이터 살펴보기 → 기계학습으로 예측하기")


def read_table(file):
    raw = file.getvalue()
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    raise ValueError("CSV 인코딩을 확인해 주세요. UTF-8 또는 CP949를 권장합니다.")


def prepare_df(df):
    """모든 데이터 열을 수치형으로 정리합니다."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def type_table(df):
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({"열 이름": c, "자료형": "수치형", "결측치": int(s.isna().sum()),
                     "고유값 수": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def make_preprocessor(X):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])


for key, default in {
    "df": None, "source": "", "question": "", "features": [], "target": None,
    "result": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

tab_data, tab_viz, tab_ml = st.tabs(["① 데이터 선정", "② 데이터 시각화", "③ 기계학습"])

with tab_data:
    st.header("① 데이터 입력 및 질문 설정")
    st.write("아래 표에 데이터를 직접 입력하세요. 관측 사례 20개, 변수 3개로 구성되어 있습니다.")

    st.subheader("1. 탐구 질문")
    question = st.text_input(
        "무엇을 알아보고 싶나요?",
        value=st.session_state.question,
        placeholder="예: 공부 시간이 길면 시험 점수가 높을까?",
        key="question_entry"
    )
    st.session_state.question = question

    st.subheader("2. 변수 이름 정하기")
    n1, n2, n3 = st.columns(3)
    with n1:
        name_x1 = st.text_input("입력 변수 1 이름", value="입력 변수 1", key="name_x1")
    with n2:
        name_x2 = st.text_input("입력 변수 2 이름", value="입력 변수 2", key="name_x2")
    with n3:
        name_y = st.text_input("예측 대상 이름", value="예측 대상", key="name_y")

    st.caption("데이터 유형은 모두 수치형으로 통일합니다. 변수 이름과 탐구 질문은 문자로 작성할 수 있지만, 표의 데이터 값에는 숫자만 입력할 수 있습니다.")

    names = [name_x1.strip() or "입력 변수 1", name_x2.strip() or "입력 변수 2", name_y.strip() or "예측 대상"]
    if len(set(names)) < 3:
        st.warning("세 변수의 이름은 서로 다르게 입력해 주세요.")

    st.subheader("3. 데이터 직접 입력")
    st.caption("각 칸에는 숫자만 입력하세요. 소수도 입력할 수 있으며, 사용하지 않는 행이나 값이 없는 칸은 비워 둘 수 있습니다. 최소 5개 이상의 데이터 행이 필요합니다.")

    if "manual_table" not in st.session_state:
        st.session_state.manual_table = pd.DataFrame({
            "입력 변수 1": [""] * 10,
            "입력 변수 2": [""] * 10,
            "예측 대상": [""] * 10,
        })

    # 이름을 바꾼 경우에도 기존에 입력한 값은 열 순서대로 유지
    old_table = st.session_state.manual_table
    if list(old_table.columns) != names:
        old_table = old_table.copy()
        old_table.columns = names
        st.session_state.manual_table = old_table

    edited = st.data_editor(
        st.session_state.manual_table,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        height=620,
        key="manual_data_editor",
        column_config={
            name: st.column_config.NumberColumn(
                name,
                help="숫자만 입력하세요. 소수 입력 가능.",
                format="%.3f",
                step=0.1
            )
            for name in names
        }
    )
    st.session_state.manual_table = edited

    if st.button("입력 데이터 저장하기", type="primary"):
        if not st.session_state.question.strip():
            st.error("먼저 탐구 질문을 입력해 주세요.")
        elif len(set(names)) < 3:
            st.error("세 변수의 이름은 서로 다르게 입력해 주세요.")
        else:
            saved = edited.copy()
            saved.columns = names
            # 빈 행은 제외하고 모든 열을 수치형으로 변환
            saved = saved.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
            saved = prepare_df(saved)
            if len(saved) < 5:
                st.error("데이터가 입력된 행이 5개 이상 필요합니다.")
            elif saved[names[2]].isna().all():
                st.error("예측 대상 열에 숫자를 입력해 주세요.")
            elif saved[names[:2]].notna().sum().sum() == 0:
                st.error("입력 변수 열에 숫자를 입력해 주세요.")
            else:
                st.session_state.df = saved.reset_index(drop=True)
                st.session_state.source = "직접 입력한 데이터"
                st.session_state.features = names[:2]
                st.session_state.target = names[2]
                st.session_state.result = None
                st.success(f"저장 완료: {len(saved)}행 × {len(saved.columns)}열")

    df = st.session_state.df
    if df is not None:
        st.subheader("저장된 데이터 미리보기")
        st.caption(st.session_state.source)
        st.table(df)
        st.info(f"**탐구 질문:** {st.session_state.question or '아직 입력하지 않음'}\n\n**입력 변수:** {', '.join(st.session_state.features)}\n\n**예측 대상:** {st.session_state.target}")

with tab_viz:
    st.header("② 데이터 시각화")
    st.write("두 개의 수치형 변수를 선택해 산점도를 만들고, 두 변수 사이의 피어슨 상관계수를 확인하세요.")

    df = st.session_state.df
    if df is None:
        st.info("먼저 데이터 선정 탭에서 탐구 질문을 입력하고 데이터를 저장하세요.")
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("산점도를 만들려면 숫자로 된 변수가 최소 2개 필요합니다. 데이터 선정 탭에서 수치형 데이터를 입력하고 저장하세요.")
        else:
            controls1, controls2 = st.columns(2)
            with controls1:
                x_col = st.selectbox("X축 변수", numeric_cols, index=0, key="scatter_x")
            y_options = [col for col in numeric_cols if col != x_col]
            with controls2:
                default_y = 0
                y_col = st.selectbox("Y축 변수", y_options, index=default_y, key="scatter_y")

            pair = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            chart_col, stats_col = st.columns([3, 1], gap="large")

            with chart_col:
                if len(pair) < 2:
                    st.warning("선택한 두 변수에 값이 모두 있는 행이 2개 이상 필요합니다.")
                else:
                    fig = px.scatter(
                        pair,
                        x=x_col,
                        y=y_col,
                        title=f"{x_col}와 {y_col}의 산점도",
                        labels={x_col: x_col, y_col: y_col},
                        hover_data={x_col: True, y_col: True},
                    )
                    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"그래프에 표시된 데이터: {len(pair)}개 행 (결측치가 있는 행 제외)")

            with stats_col:
                st.subheader("상관계수")
                if len(pair) < 2 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
                    st.metric("피어슨 r", "계산 불가")
                    st.caption("유효한 데이터가 2개 이상이고 각 변수의 값이 다양해야 계산할 수 있습니다.")
                else:
                    corr = pair[x_col].corr(pair[y_col], method="pearson")
                    st.metric("피어슨 r", f"{corr:.3f}")
                    if corr > 0:
                        direction = "양의 상관관계"
                    elif corr < 0:
                        direction = "음의 상관관계"
                    else:
                        direction = "선형 상관관계가 거의 없음"
                    st.write(f"**방향:** {direction}")
                    st.caption("상관계수는 -1부터 1 사이입니다. 절댓값이 1에 가까울수록 선형 관계가 강하고, 0에 가까울수록 선형 관계가 약합니다.")
                st.divider()
                st.caption("상관관계가 있다고 해서 한 변수가 다른 변수의 원인이라는 뜻은 아닙니다.")

with tab_ml:
    st.header("③ 기계학습")
    df = st.session_state.df
    if df is None:
        st.info("먼저 데이터 선정 탭에서 탐구 질문을 입력하고 표에 데이터를 직접 입력한 뒤 저장하세요.")
    elif len(st.session_state.features) != 2 or st.session_state.target is None:
        st.info("데이터 선정 탭에서 세 열에 데이터를 입력하고 저장하세요.")
    else:
        features, target = st.session_state.features, st.session_state.target
        st.write(f"**입력 변수:** {features[0]}, {features[1]}")
        st.write(f"**예측 대상:** {target}")
        if st.session_state.question:
            st.info(f"탐구 질문: {st.session_state.question}")

        y_series = df[target]
        guess_reg = y_series.nunique(dropna=True) > 10
        task = st.radio("예측 유형", ["회귀", "분류"], horizontal=True,
                        index=0 if guess_reg else 1,
                        help="모든 데이터는 수치형입니다. 회귀는 연속적인 숫자 값을, 분류는 숫자로 표시된 종류/등급을 예측합니다.")
        if task == "회귀":
            model_name = st.selectbox("모델", ["랜덤 포레스트 회귀", "선형 회귀"])
        else:
            model_name = st.selectbox("모델", ["랜덤 포레스트 분류", "로지스틱 회귀"])
        test_size = st.slider("평가용 데이터 비율", 0.1, 0.4, 0.2, 0.05)

        if st.button("모델 학습 및 평가", type="primary"):
            try:
                work = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
                X, y = work[features].copy(), work[target].copy()
                # 입력 데이터는 모두 수치형으로 처리합니다.
                X = X.apply(pd.to_numeric, errors="coerce")

                if task == "회귀":
                    y = pd.to_numeric(y, errors="coerce")
                    valid = y.notna()
                    X, y = X.loc[valid], y.loc[valid]
                    if len(y) < 5:
                        st.error("회귀 분석을 하기에 데이터가 너무 적습니다. 유효한 행을 5개 이상 준비하세요.")
                        st.stop()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    estimator = RandomForestRegressor(n_estimators=200, random_state=42) if model_name == "랜덤 포레스트 회귀" else LinearRegression()
                    model = Pipeline([("prep", make_preprocessor(X)), ("model", estimator)])
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, pred)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    r2 = r2_score(y_test, pred) if len(y_test) > 1 else np.nan
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE (평균 절대 오차)", f"{mae:,.3f}")
                    m2.metric("RMSE (평균 제곱근 오차)", f"{rmse:,.3f}")
                    m3.metric("R² (결정계수)", f"{r2:.3f}" if pd.notna(r2) else "계산 불가")
                    result = X_test.copy()
                    result["실제값"], result["예측값"] = y_test.values, np.round(pred, 3)
                    st.table(result)
                    plot_df = pd.DataFrame({"실제값": y_test.to_numpy(), "예측값": pred})
                    fig = px.scatter(
                        plot_df, x="실제값", y="예측값",
                        title="실제값과 예측값 비교",
                        hover_data={"실제값": ":.3f", "예측값": ":.3f"}
                    )
                    low = min(float(y_test.min()), float(np.min(pred)))
                    high = max(float(y_test.max()), float(np.max(pred)))
                    fig.add_trace(go.Scatter(
                        x=[low, high], y=[low, high], mode="lines",
                        name="완벽한 예측 기준선", line={"dash": "dash"}
                    ))
                    fig.update_layout(xaxis_title="실제값", yaxis_title="예측값")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    y = pd.to_numeric(y, errors="coerce")
                    valid = y.notna()
                    X, y = X.loc[valid], y.loc[valid]
                    if y.nunique() < 2:
                        st.error("분류에는 예측 대상 범주가 최소 2개 필요합니다.")
                        st.stop()
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42,
                        stratify=y if y.value_counts().min() >= 2 else None
                    )
                    estimator = RandomForestClassifier(n_estimators=200, random_state=42) if model_name == "랜덤 포레스트 분류" else LogisticRegression(max_iter=2000)
                    model = Pipeline([("prep", make_preprocessor(X)), ("model", estimator)])
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("정확도", f"{accuracy_score(y_test, pred):.3f}")
                    m2.metric("정밀도 (macro)", f"{precision_score(y_test, pred, average='macro', zero_division=0):.3f}")
                    m3.metric("F1 (macro)", f"{f1_score(y_test, pred, average='macro', zero_division=0):.3f}")
                    labels = sorted(pd.Series(y_test).astype(str).unique())
                    cm = confusion_matrix(y_test, pred, labels=labels)
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                    fig = px.imshow(
                        cm_df,
                        text_auto=True,
                        aspect="auto",
                        labels={"x": "예측값", "y": "실제값", "color": "개수"},
                        title="혼동 행렬",
                        color_continuous_scale="Blues"
                    )
                    fig.update_xaxes(side="bottom")
                    st.plotly_chart(fig, use_container_width=True)
                    result = X_test.copy()
                    result["실제값"], result["예측값"] = y_test.values, pred
                    st.table(result)

                st.session_state.result = {"model": model, "features": features, "target": target, "task": task}
                st.success("모델 학습과 평가가 끝났습니다.")
                st.caption("평가 결과는 이번에 나눈 평가용 데이터에 대한 결과입니다. 데이터의 양과 편향에 따라 실제 성능은 달라질 수 있습니다.")
            except Exception as e:
                st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
                st.info("변수 선택, 자료형, 결측치, 데이터 행 수를 확인해 주세요.")

        result_info = st.session_state.result
        if result_info:
            st.subheader("새로운 값으로 예측해 보기")
            new_values = {}
            input_cols = st.columns(len(result_info["features"]))
            for i, col in enumerate(result_info["features"]):
                s = pd.to_numeric(df[col], errors="coerce")
                with input_cols[i]:
                    default = s.median()
                    new_values[col] = st.number_input(
                        col,
                        value=float(default) if pd.notna(default) else 0.0,
                        key=f"new_{col}"
                    )
            if st.button("새로운 값 예측하기"):
                try:
                    new_df = pd.DataFrame([new_values], columns=result_info["features"])
                    pred = result_info["model"].predict(new_df)[0]
                    st.success(f"예측 결과 — {result_info['target']}: **{pred}**")
                except Exception as e:
                    st.error(f"예측에 실패했습니다: {e}")

st.divider()
st.caption("수업용 탐구 도구입니다. 데이터 값은 모두 수치형으로 입력합니다. 관계가 있다고 해서 인과관계가 증명되는 것은 아닙니다.")

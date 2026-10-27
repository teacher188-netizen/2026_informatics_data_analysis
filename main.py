import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
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
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
            converted = pd.to_numeric(cleaned, errors="coerce")
            n = df[col].notna().sum()
            if n and converted.notna().sum() / n >= 0.85:
                df[col] = converted
    return df


def type_table(df):
    rows = []
    for c in df.columns:
        s = df[c]
        kind = "숫자형" if pd.api.types.is_numeric_dtype(s) else "범주형/문자형"
        rows.append({"열 이름": c, "자료형": kind, "결측치": int(s.isna().sum()),
                     "고유값 수": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def make_preprocessor(X):
    nums = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cats))
    return ColumnTransformer(transformers=transformers)


for key, default in {
    "df": None, "source": "", "question": "", "features": [], "target": None,
    "result": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

tab_data, tab_viz, tab_ml = st.tabs(["① 데이터 선정", "② 데이터 시각화", "③ 기계학습"])

with tab_data:
    st.header("① 데이터 선정")
    st.write("입력 데이터 2개와 예측 대상 데이터 1개를 준비하세요.")
    mode = st.radio("데이터 준비 방식",
                    ["세 변수가 한 파일에 있음", "서로 다른 파일 3개 연결"],
                    horizontal=True)

    loaded_df, source = None, ""

    if mode == "세 변수가 한 파일에 있음":
        f = st.file_uploader("CSV 또는 Excel 파일", type=["csv", "xlsx", "xls"], key="onefile")
        if f:
            try:
                loaded_df = prepare_df(read_table(f))
                source = f"파일: {f.name}"
            except Exception as e:
                st.error(f"파일을 읽을 수 없습니다: {e}")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            f1 = st.file_uploader("파일 1 (입력 변수 후보)", type=["csv", "xlsx", "xls"], key="f1")
        with c2:
            f2 = st.file_uploader("파일 2 (입력 변수 후보)", type=["csv", "xlsx", "xls"], key="f2")
        with c3:
            f3 = st.file_uploader("파일 3 (예측 대상 후보)", type=["csv", "xlsx", "xls"], key="f3")

        if f1 and f2 and f3:
            try:
                d1, d2, d3 = [prepare_df(read_table(f)) for f in (f1, f2, f3)]
                common = sorted(set(d1.columns) & set(d2.columns) & set(d3.columns))
                if not common:
                    st.error("세 파일에 공통 열이 없습니다. 날짜, 지역 코드, 영화 코드 등 연결 기준이 필요합니다.")
                else:
                    key = st.selectbox("세 파일을 연결할 공통 열", common)
                    if st.button("세 파일 연결하기", type="primary"):
                        def rename_cols(d, prefix):
                            return d.rename(columns={c: f"{prefix}__{c}" for c in d.columns if c != key})
                        merged = rename_cols(d1, "데이터1").merge(rename_cols(d2, "데이터2"), on=key, how="inner")
                        merged = merged.merge(rename_cols(d3, "데이터3"), on=key, how="inner")
                        merged = merged.rename(columns={key: "연결키"})
                        st.session_state.df = merged
                        st.session_state.source = f"{f1.name}, {f2.name}, {f3.name} 연결"
                        st.session_state.result = None
                        st.success(f"연결 완료: {len(merged):,}행")
                        st.rerun()
            except Exception as e:
                st.error(f"파일 연결 중 오류: {e}")

    if loaded_df is not None:
        st.session_state.df = loaded_df
        st.session_state.source = source
        st.session_state.result = None

    df = st.session_state.df
    if df is None:
        st.info("데이터 파일을 업로드하세요. 처음에는 세 변수가 한 표에 들어 있는 파일을 권장합니다.")
    else:
        st.success(f"현재 데이터: {len(df):,}행 × {len(df.columns):,}열")
        st.caption(st.session_state.source)
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("탐구 질문 만들기")
        st.session_state.question = st.text_input(
            "‘~이면 ~일까?’ 형태로 작성하세요.",
            value=st.session_state.question,
            placeholder="예: 스크린 수와 상영 기간을 알면 최종 관객 수를 예측할 수 있을까?"
        )
        cols = df.columns.tolist()
        if len(cols) < 3:
            st.warning("입력 변수 2개와 예측 대상 1개를 선택하려면 열이 최소 3개 필요합니다.")
        else:
            a, b, c = st.columns(3)
            with a:
                x1 = st.selectbox("입력 변수 1 (~이면)", cols, key="sel_x1")
            with b:
                x2 = st.selectbox("입력 변수 2 (~이면)", [x for x in cols if x != x1], key="sel_x2")
            with c:
                y = st.selectbox("예측 대상 (~일까?)", [x for x in cols if x not in (x1, x2)], key="sel_y")
            st.session_state.features = [x1, x2]
            st.session_state.target = y
            if st.button("질문과 변수 저장", type="primary"):
                if not st.session_state.question.strip():
                    st.warning("탐구 질문을 먼저 작성하세요.")
                else:
                    st.session_state.result = None
                    st.success("저장했습니다. 데이터 시각화 탭으로 이동하세요.")
            st.info(f"**현재 구조**\n\n입력 변수: {x1}, {x2}\n\n예측 대상: {y}")

with tab_viz:
    st.header("② 데이터 시각화")
    df = st.session_state.df
    if df is None:
        st.info("먼저 데이터 선정 탭에서 파일을 업로드하세요.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("행 수", f"{len(df):,}")
        c2.metric("열 수", f"{len(df.columns):,}")
        c3.metric("결측치 수", f"{int(df.isna().sum().sum()):,}")
        st.subheader("자료형과 데이터 상태")
        st.dataframe(type_table(df), use_container_width=True)

        st.subheader("두 변수의 관계")
        cols = df.columns.tolist()
        a, b = st.columns(2)
        with a:
            va = st.selectbox("변수 A", cols, key="viz_a")
        with b:
            vb = st.selectbox("변수 B", [x for x in cols if x != va], key="viz_b")
        na = pd.api.types.is_numeric_dtype(df[va])
        nb = pd.api.types.is_numeric_dtype(df[vb])
        pair = df[[va, vb]].dropna()

        if na and nb:
            fig, ax = plt.subplots()
            ax.scatter(pair[va], pair[vb], alpha=0.65)
            ax.set_xlabel(va); ax.set_ylabel(vb); ax.set_title(f"{va}와 {vb}")
            fig.tight_layout(); st.pyplot(fig)
            corr = pair[va].corr(pair[vb])
            st.metric("피어슨 상관계수", f"{corr:.3f}" if pd.notna(corr) else "계산 불가")
            st.caption("상관관계가 인과관계를 의미하지는 않습니다.")
        elif na != nb:
            cat, num = (vb, va) if na else (va, vb)
            summary = pair.groupby(cat, observed=False)[num].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
            st.write("범주별 수치 비교")
            st.dataframe(summary, use_container_width=True)
            st.bar_chart(summary["mean"].head(20))
        else:
            counts = pd.crosstab(pair[va].astype(str), pair[vb].astype(str))
            st.write("범주형 변수 교차표 (빈도)")
            st.dataframe(counts, use_container_width=True)
            pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100).round(1)
            st.write("행 기준 비율 (%)")
            st.dataframe(pct, use_container_width=True)

        numeric = df.select_dtypes(include="number")
        st.subheader("숫자형 변수 상관관계")
        if numeric.shape[1] >= 2:
            st.dataframe(numeric.corr().round(2), use_container_width=True)
        else:
            st.info("숫자형 열이 2개 이상일 때 상관계수 표를 표시합니다.")

with tab_ml:
    st.header("③ 기계학습")
    df = st.session_state.df
    if df is None:
        st.info("먼저 데이터 선정 탭에서 파일을 업로드하세요.")
    elif len(st.session_state.features) != 2 or st.session_state.target is None:
        st.info("데이터 선정 탭에서 입력 변수 2개와 예측 대상을 선택하세요.")
    else:
        features, target = st.session_state.features, st.session_state.target
        st.write(f"**입력 변수:** {features[0]}, {features[1]}")
        st.write(f"**예측 대상:** {target}")
        if st.session_state.question:
            st.info(f"탐구 질문: {st.session_state.question}")

        y_series = df[target]
        guess_reg = pd.api.types.is_numeric_dtype(y_series) and y_series.nunique(dropna=True) > 10
        task = st.radio("예측 유형", ["회귀", "분류"], horizontal=True,
                        index=0 if guess_reg else 1,
                        help="회귀는 숫자 값을, 분류는 범주를 예측합니다.")
        if task == "회귀":
            model_name = st.selectbox("모델", ["랜덤 포레스트 회귀", "선형 회귀"])
        else:
            model_name = st.selectbox("모델", ["랜덤 포레스트 분류", "로지스틱 회귀"])
        test_size = st.slider("평가용 데이터 비율", 0.1, 0.4, 0.2, 0.05)

        if st.button("모델 학습 및 평가", type="primary"):
            try:
                work = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
                X, y = work[features].copy(), work[target].copy()
                for col in X.columns:
                    if pd.api.types.is_datetime64_any_dtype(X[col]):
                        X[col] = X[col].astype(str)

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
                    st.dataframe(result, use_container_width=True)
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, pred, alpha=0.7)
                    low, high = min(float(y_test.min()), float(np.min(pred))), max(float(y_test.max()), float(np.max(pred)))
                    ax.plot([low, high], [low, high], linestyle="--")
                    ax.set_xlabel("실제값"); ax.set_ylabel("예측값"); ax.set_title("실제값과 예측값")
                    fig.tight_layout(); st.pyplot(fig)
                else:
                    y = y.astype(str)
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
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=labels), display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
                    fig.tight_layout(); st.pyplot(fig)
                    result = X_test.copy()
                    result["실제값"], result["예측값"] = y_test.values, pred
                    st.dataframe(result, use_container_width=True)

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
                s = df[col]
                with input_cols[i]:
                    if pd.api.types.is_numeric_dtype(s):
                        default = pd.to_numeric(s, errors="coerce").median()
                        new_values[col] = st.number_input(col, value=float(default) if pd.notna(default) else 0.0, key=f"new_{col}")
                    else:
                        options = s.dropna().astype(str).unique().tolist()[:200]
                        new_values[col] = st.selectbox(col, options, key=f"new_{col}") if options else st.text_input(col, key=f"new_{col}")
            if st.button("새로운 값 예측하기"):
                try:
                    new_df = pd.DataFrame([new_values], columns=result_info["features"])
                    pred = result_info["model"].predict(new_df)[0]
                    st.success(f"예측 결과 — {result_info['target']}: **{pred}**")
                except Exception as e:
                    st.error(f"예측에 실패했습니다: {e}")

st.divider()
st.caption("수업용 탐구 도구입니다. 관계가 있다고 해서 인과관계가 증명되는 것은 아닙니다.")

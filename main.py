import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from streamlit_local_storage import LocalStorage

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="데이터 탐험실", page_icon="🔎", layout="wide")
st.title("🔎 데이터 탐험실")
st.caption("탐구 질문 만들기 → 수치 데이터 살펴보기 → 선형 회귀로 예측하기")

# 브라우저 로컬 저장소: 같은 브라우저에서 새로고침해도 입력 내용 복원
local_storage = LocalStorage()

for key, default in {
    "df": None,
    "question": "",
    "features": [],
    "target": None,
    "model": None,
    "result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 최초 실행 시 브라우저에 남아 있는 임시 저장 데이터를 복원
if "local_backup_loaded" not in st.session_state:
    st.session_state.local_backup_loaded = True
    try:
        backup_raw = local_storage.getItem("data_explorer_backup")
        if backup_raw and isinstance(backup_raw, str):
            backup = json.loads(backup_raw)
            st.session_state.question = backup.get("question", "")
            st.session_state.name_x1 = backup.get("name_x1", "입력 변수 1")
            st.session_state.name_x2 = backup.get("name_x2", "입력 변수 2")
            st.session_state.name_y = backup.get("name_y", "예측 대상")
            table_records = backup.get("manual_table", [])
            table_columns = backup.get("manual_columns", ["입력 변수 1", "입력 변수 2", "예측 대상"])
            if table_records:
                st.session_state.manual_table = pd.DataFrame(table_records, columns=table_columns)
    except Exception:
        # 저장 데이터가 없거나 읽을 수 없어도 앱 사용은 계속 가능
        pass


tab_data, tab_viz, tab_ml = st.tabs([
    "① 데이터 입력",
    "② 데이터 시각화",
    "③ 선형 회귀"
])

with tab_data:
    st.header("① 데이터 입력 및 탐구 질문")
    st.write("변수 3개와 데이터 10행을 입력합니다. 모든 데이터 값은 수치형입니다.")

    question = st.text_input(
        "무엇을 알아보고 싶나요?",
        value=st.session_state.question,
        placeholder="예: 주말이고 날씨가 좋으면 서울대공원 입장객이 증가할까?",
        key="question_entry",
    )
    st.session_state.question = question

    c1, c2, c3 = st.columns(3)
    with c1:
        name_x1 = st.text_input("입력 변수 1 이름", value=st.session_state.get("name_x1", "입력 변수 1"), key="name_x1")
    with c2:
        name_x2 = st.text_input("입력 변수 2 이름", value=st.session_state.get("name_x2", "입력 변수 2"), key="name_x2")
    with c3:
        name_y = st.text_input("예측 대상 이름", value=st.session_state.get("name_y", "예측 대상"), key="name_y")

    names = [
        name_x1.strip() or "입력 변수 1",
        name_x2.strip() or "입력 변수 2",
        name_y.strip() or "예측 대상",
    ]

    if len(set(names)) != 3:
        st.warning("세 변수의 이름은 서로 다르게 입력해 주세요.")

    st.caption("데이터 칸에는 숫자만 입력할 수 있습니다. 정수와 소수를 사용할 수 있으며, 입력하지 않는 칸은 비워 두세요.")

    if "manual_table" not in st.session_state:
        st.session_state.manual_table = pd.DataFrame({
            "입력 변수 1": [None] * 10,
            "입력 변수 2": [None] * 10,
            "예측 대상": [None] * 10,
        })

    old_table = st.session_state.manual_table.copy()
    if list(old_table.columns) != names:
        old_table.columns = names
        st.session_state.manual_table = old_table

    edited = st.data_editor(
        st.session_state.manual_table,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        height=410,
        key="manual_data_editor",
        column_config={
            name: st.column_config.NumberColumn(
                name,
                help="숫자만 입력하세요. 소수점 셋째자리까지 입력 가능.",
                format="%.3f",
                step=0.001,
            )
            for name in names
        },
    )
    st.session_state.manual_table = edited

    # 입력할 때마다 브라우저 로컬 저장소에 임시 저장
    try:
        backup_payload = {
            "question": question,
            "name_x1": name_x1,
            "name_x2": name_x2,
            "name_y": name_y,
            "manual_columns": names,
            "manual_table": edited.where(pd.notna(edited), None).to_dict(orient="records"),
        }
        local_storage.setItem("data_explorer_backup", json.dumps(backup_payload, ensure_ascii=False))
    except Exception:
        st.caption("자동 임시 저장을 사용할 수 없습니다. 아래 CSV 다운로드 버튼으로 파일을 저장해 주세요.")

    if st.button("데이터 저장하기", type="primary"):
        if not question.strip():
            st.error("먼저 탐구 질문을 입력해 주세요.")
        elif len(set(names)) != 3:
            st.error("세 변수의 이름은 서로 다르게 입력해 주세요.")
        else:
            saved = edited.copy()
            saved.columns = names
            saved = saved.replace(r"^\s*$", np.nan, regex=True)
            saved = saved.dropna(how="all")
            for col in names:
                saved[col] = pd.to_numeric(saved[col], errors="coerce")
            saved = saved.dropna(subset=names)

            if len(saved) < 5:
                st.error("세 변수에 모두 숫자가 입력된 행이 최소 5개 필요합니다.")
            else:
                st.session_state.df = saved.reset_index(drop=True)
                st.session_state.features = names[:2]
                st.session_state.target = names[2]
                st.session_state.model = None
                st.session_state.result = None
                st.success(f"저장 완료: 수치형 데이터 {len(saved)}행 × 3열")

    if st.session_state.df is not None:
        st.subheader("저장된 데이터")
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)
        st.info(
            f"**탐구 질문:** {st.session_state.question}\n\n"
            f"**입력 변수:** {', '.join(st.session_state.features)}\n\n"
            f"**예측 대상:** {st.session_state.target}"
        )

    st.divider()
    st.subheader("📥 과제 제출용 CSV 저장")
    st.write("입력한 데이터와 탐구 질문을 CSV 파일로 저장한 뒤, Google 클래스룸의 해당 과제에 업로드하세요.")
    export_table = edited.copy()
    export_table.columns = names
    export_table = export_table.where(pd.notna(export_table), "")
    export_csv = export_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV 다운로드",
        data=export_csv,
        file_name="데이터_탐험실_입력데이터.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    st.caption("제출 순서: ① CSV 다운로드 → ② Google 클래스룸에서 해당 과제 열기 → ③ 파일 첨부 후 제출")

with tab_viz:
    st.header("② 데이터 시각화")
    st.write("수치형 변수 두 개의 산점도와 피어슨 상관계수를 확인합니다.")
    df = st.session_state.df

    if df is None:
        st.info("먼저 데이터 입력 탭에서 데이터를 저장해 주세요.")
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        left, right = st.columns(2)
        with left:
            x_col = st.selectbox("X축 변수", numeric_cols, index=0, key="scatter_x")
        with right:
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y축 변수", y_options, index=0, key="scatter_y")

        pair = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        chart_col, stats_col = st.columns([3, 1], gap="large")
        with chart_col:
            if len(pair) >= 2:
                fig = px.scatter(pair, x=x_col, y=y_col, title=f"{x_col}와 {y_col}의 산점도")
                fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"표시된 데이터: {len(pair)}행")
            else:
                st.warning("그래프를 만들려면 두 변수에 모두 값이 있는 행이 2개 이상 필요합니다.")

        with stats_col:
            st.subheader("상관계수")
            if len(pair) < 2 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
                st.metric("피어슨 r", "계산 불가")
            else:
                corr = pair[x_col].corr(pair[y_col])
                st.metric("피어슨 r", f"{corr:.3f}")
                if corr > 0:
                    st.write("**방향:** 양의 상관관계")
                elif corr < 0:
                    st.write("**방향:** 음의 상관관계")
                else:
                    st.write("**방향:** 선형 상관관계가 거의 없음")
                st.caption("상관계수는 -1부터 1 사이입니다. 절댓값이 1에 가까울수록 선형 관계가 강합니다.")
            st.divider()
            st.caption("상관관계가 있다고 해서 인과관계가 증명되는 것은 아닙니다.")

        st.divider()
        st.subheader("세 변수의 관계 해석하기")
        st.write(
            "산점도와 상관계수를 살펴보고 세 변수 사이에 어떤 관계가 있는지 "
            "자신의 말로 정리해 보세요. 관계가 뚜렷하지 않다면 그렇게 판단한 이유도 적어 보세요."
        )
        st.text_area(
            "세 변수의 관계에 대한 나의 해석",
            key="relationship_notes",
            height=180,
            placeholder=(
                f"1. {st.session_state.features[0] if st.session_state.features else '입력 변수 1'}와 "
                f"{st.session_state.target or '예측 대상'}은 어떤 관계인가요? 양의 상관관계/음의 상관관계, 상관계수를 적고 연관성이 깊은지 적은지 작성합니다.\n"
                f"2. {st.session_state.features[1] if len(st.session_state.features) > 1 else '입력 변수 2'}와 "
                f"{st.session_state.target or '예측 대상'}은 어떤 관계인가요? 양의 상관관계/음의 상관관계, 상관계수를 적고 연관성이 깊은지 적은지 작성합니다.\n"
                "3. 두 입력 변수 사이에도 관계가 있나요? 양의 상관관계/음의 상관관계, 상관계수를 적고 연관성이 깊은지 적은지 작성합니다.\n"
                "4. 그래프와 상관계수를 근거로 그렇게 생각한 이유를 적어 보세요."
            ),
        )

with tab_ml:
    st.header("③ 선형 회귀로 예측하기")
    st.write("선형 회귀는 입력 변수와 예측 대상 사이의 관계를 직선 형태의 수식으로 학습해 새로운 값을 예측합니다.")
    st.latex(r"\hat{y} = b_0 + b_1x_1 + b_2x_2")

    df = st.session_state.df
    if df is None:
        st.info("먼저 데이터 입력 탭에서 데이터를 저장해 주세요.")
    else:
        features = st.session_state.features
        target = st.session_state.target
        st.write(f"**입력 변수:** {features[0]}, {features[1]}")
        st.write(f"**예측 대상:** {target}")
        st.info(f"탐구 질문: {st.session_state.question}")

        # 3차원 애니메이션: 경사 하강법으로 예측 평면을 조금씩 조정
        st.subheader("최적의 예측 평면 찾아가기")
        st.write(
            f"점 하나는 실제 데이터 한 행을 뜻합니다. 평면은 두 입력 변수 "
            f"({features[0]}, {features[1]})로 예측 대상({target})을 추정합니다. "
            "애니메이션을 재생하면 예측 오차가 작아지도록 평면이 이동하는 과정을 천천히 관찰할 수 있습니다. 한 프레임은 약 0.45초이며 전체 과정은 약 27초입니다."
        )

        animation_data = df[features + [target]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(animation_data) >= 3 and all(
            animation_data[c].nunique() > 1 for c in features
        ):
            xa = animation_data[features[0]].to_numpy(dtype=float)
            xb = animation_data[features[1]].to_numpy(dtype=float)
            ya = animation_data[target].to_numpy(dtype=float)

            # 경사 하강법이 변수 단위에 지나치게 민감하지 않도록 표준화
            x_means = np.array([xa.mean(), xb.mean()])
            x_stds = np.array([xa.std(), xb.std()])
            x_stds[x_stds == 0] = 1.0
            y_mean = ya.mean()
            y_std = ya.std()
            if y_std == 0:
                y_std = 1.0

            X_scaled = (np.column_stack([xa, xb]) - x_means) / x_stds
            y_scaled = (ya - y_mean) / y_std
            X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
            beta = np.zeros(3, dtype=float)

            x1_grid = np.linspace(float(xa.min()), float(xa.max()), 20)
            x2_grid = np.linspace(float(xb.min()), float(xb.max()), 20)
            gx1, gx2 = np.meshgrid(x1_grid, x2_grid)
            grid_scaled = np.column_stack([
                ((gx1.ravel() - x_means[0]) / x_stds[0]),
                ((gx2.ravel() - x_means[1]) / x_stds[1]),
            ])
            grid_aug = np.column_stack([np.ones(len(grid_scaled)), grid_scaled])

            def plane_values(coefficients):
                z_scaled = grid_aug @ coefficients
                return (z_scaled * y_std + y_mean).reshape(gx1.shape)

            # 첫 화면은 초기 평면, 이후 경사 하강법의 진행 상태를 프레임으로 구성
            initial_surface = go.Surface(
                x=gx1, y=gx2, z=plane_values(beta),
                opacity=0.72, colorscale="Viridis", showscale=False,
                name="예측 평면"
            )
            points = go.Scatter3d(
                x=xa, y=xb, z=ya, mode="markers",
                marker=dict(size=5, color="crimson"),
                name="실제 데이터"
            )

            frames = []
            total_steps = 120
            learning_rate = 0.08
            for step in range(1, total_steps + 1):
                prediction_scaled = X_aug @ beta
                gradient = (2 / len(y_scaled)) * (X_aug.T @ (prediction_scaled - y_scaled))
                beta = beta - learning_rate * gradient

                if step % 2 == 0 or step == 1 or step == total_steps:
                    frames.append(go.Frame(
                        name=str(step),
                        data=[go.Surface(
                            x=gx1, y=gx2, z=plane_values(beta),
                            opacity=0.72, colorscale="Viridis", showscale=False,
                            name="예측 평면"
                        )],
                        traces=[1],
                        layout=go.Layout(title=f"예측 평면 최적화 과정 — {step}/{total_steps}회")
                    ))

            # 마지막 프레임은 최종 학습 결과이며, Play 버튼으로 과정을 재생할 수 있음
            fig3d = go.Figure(data=[points, initial_surface], frames=frames)
            fig3d.update_layout(
                title="예측 평면 최적화 과정 — 초기 상태",
                height=650,
                scene=dict(
                    xaxis_title=features[0],
                    yaxis_title=features[1],
                    zaxis_title=target,
                    aspectmode="cube",
                    camera=dict(eye=dict(x=1.8, y=2.1, z=0.65))
                ),
                margin=dict(l=0, r=0, t=60, b=0),
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    x=0.02, y=1.08,
                    buttons=[
                        dict(label="▶ 재생", method="animate", args=[None, {
                            "frame": {"duration": 450, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate"
                        }]),
                        dict(label="⏸ 멈춤", method="animate", args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]),
                    ]
                )]
            )
            fig3d.update_scenes(camera=dict(eye=dict(x=1.8, y=2.1, z=0.65)))
            st.plotly_chart(fig3d, use_container_width=True)
            st.caption(
                "빨간 점은 실제 데이터, 색이 있는 평면은 모델의 예측값입니다. "
                "평면은 처음부터 정답 위치에 있지 않고, 경사 하강법이 오차를 줄이는 방향으로 계수를 조정합니다. "
                "표준화는 계산을 안정적으로 하기 위한 내부 처리이며, 그래프 축은 원래 단위를 사용합니다."
            )
        else:
            st.info("3차원 평면을 만들려면 유효한 데이터가 3행 이상이고, 두 입력 변수에 각각 서로 다른 값이 있어야 합니다.")

        st.divider()
        test_size = st.slider("평가용 데이터 비율", 0.2, 0.4, 0.2, 0.1,
                              help="일부 데이터로 학습한 뒤, 학습에 사용하지 않은 데이터로 예측 결과를 평가합니다.")

        if st.button("선형 회귀 학습 및 평가", type="primary"):
            try:
                work = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(work) < 5:
                    st.error("선형 회귀를 위해 모든 변수에 값이 있는 행이 최소 5개 필요합니다.")
                elif work[target].nunique() < 2:
                    st.error("예측 대상의 값이 모두 같으면 선형 회귀를 학습할 수 없습니다.")
                else:
                    X = work[features]
                    y = work[target]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, pred)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    r2 = r2_score(y_test, pred) if len(y_test) > 1 else np.nan
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE (평균 절대 오차)", f"{mae:.3f}")
                    m2.metric("RMSE (평균 제곱근 오차)", f"{rmse:.3f}")
                    m3.metric("R² (결정계수)", f"{r2:.3f}" if pd.notna(r2) else "계산 불가")

                    result = X_test.copy()
                    result["실제값"] = y_test.values
                    result["예측값"] = np.round(pred, 3)
                    st.subheader("실제값과 예측값 비교")
                    st.dataframe(result.reset_index(drop=True), use_container_width=True, hide_index=True)

                    plot_df = pd.DataFrame({"실제값": y_test.to_numpy(), "예측값": pred})
                    fig = px.scatter(plot_df, x="실제값", y="예측값", title="실제값과 예측값 비교")
                    low = float(min(y_test.min(), np.min(pred)))
                    high = float(max(y_test.max(), np.max(pred)))
                    if low == high:
                        low -= 1
                        high += 1
                    fig.add_trace(go.Scatter(
                        x=[low, high], y=[low, high], mode="lines",
                        name="완벽한 예측 기준선", line={"dash": "dash"}
                    ))
                    fig.update_layout(xaxis_title="실제값", yaxis_title="예측값")
                    st.plotly_chart(fig, use_container_width=True)

                    coef = pd.DataFrame({"변수": features, "회귀계수": model.coef_})
                    st.subheader("선형 회귀 수식")
                    st.write(f"**절편:** {model.intercept_:.3f}")
                    st.dataframe(coef, use_container_width=True, hide_index=True)
                    st.caption("회귀계수가 양수이면 해당 입력 변수 값이 커질수록 예측값이 커지는 방향, 음수이면 작아지는 방향으로 작용합니다. 다른 입력 변수의 영향을 함께 고려해 해석하세요.")

                    st.session_state.model = model
                    st.session_state.result = {
                        "model": model,
                        "features": features,
                        "target": target,
                    }
                    st.success("선형 회귀 학습과 평가가 끝났습니다.")
                    st.caption("데이터가 10개로 적기 때문에 평가 지표가 데이터 분할 방식에 따라 크게 달라질 수 있습니다.")
            except Exception as e:
                st.error(f"선형 회귀 학습 중 오류가 발생했습니다: {e}")
                st.info("숫자 입력 여부, 결측치, 각 변수의 값이 서로 다른지 확인해 주세요.")

        result_info = st.session_state.result
        if result_info:
            st.subheader("새로운 값으로 예측하기")
            st.write("입력 변수 두 개의 값을 입력하면 선형 회귀 모델이 예측 대상 값을 계산합니다.")
            new_values = {}
            input_cols = st.columns(len(result_info["features"]))
            for i, col in enumerate(result_info["features"]):
                median = pd.to_numeric(df[col], errors="coerce").median()
                with input_cols[i]:
                    new_values[col] = st.number_input(
                        col,
                        value=float(median) if pd.notna(median) else 0.0,
                        key=f"new_{col}",
                    )
            if st.button("새로운 값 예측하기"):
                try:
                    new_df = pd.DataFrame([new_values], columns=result_info["features"])
                    prediction = result_info["model"].predict(new_df)[0]
                    st.success(f"예측 결과 — {result_info['target']}: **{prediction:.3f}**")
                except Exception as e:
                    st.error(f"예측에 실패했습니다: {e}")

st.divider()
st.caption("수업용 탐구 도구입니다. 예측 결과는 입력 데이터의 양과 특성에 따라 달라지며, 관계가 있다고 해서 인과관계가 증명되는 것은 아닙니다.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walmart 프로모션 최적화 대시보드
Streamlit 웹 애플리케이션
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from optimization_engine import PromoOptimizationEngine
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="Walmart 프로모션 최적화",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight: bold;
    color: #1f77b4;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.stButton>button {
    width: 100%;
    background-color: #0066cc;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 엔진 초기화 (캐싱)
@st.cache_resource
def load_engine():
    return PromoOptimizationEngine()

try:
    engine = load_engine()
    risk_df = engine.risk_adjusted
    dept_roi = engine.dept_roi
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# ===== 사이드바 =====
st.sidebar.title("🛒 Walmart 프로모션 최적화")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "메뉴 선택",
    ["📊 경영진 대시보드", "🔍 부서별 분석", "⚡ 예산 최적화", "📈 시나리오 분석", "💰 매출 목표 역산"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 사용 가이드**
- **경영진 대시보드**: 전사 성과 한눈에
- **부서별 분석**: 개별 부서 상세 조회
- **예산 최적화**: AI 기반 최적 배분
- **시나리오 분석**: What-if 시뮬레이션
- **매출 목표 역산**: 필요 예산 계산
""")

# ===== 메인 화면 =====

if mode == "📊 경영진 대시보드":
    st.markdown('<p class="big-font">📊 경영진 대시보드</p>', unsafe_allow_html=True)
    st.markdown("**전사 프로모션 성과 종합**")
    st.markdown("---")
    
    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="총 부서 수",
            value=f"{len(dept_roi)}개",
            delta="분석 완료"
        )
    
    with col2:
        high_roi = dept_roi[dept_roi['marginal_ROI'] > 0.01]
        st.metric(
            label="고ROI 부서",
            value=f"{len(high_roi)}개",
            delta=f"+{high_roi['marginal_ROI'].mean():.3f} 평균"
        )
    
    with col3:
        reverse = dept_roi[dept_roi['marginal_ROI'] < -0.01]
        st.metric(
            label="역효과 부서",
            value=f"{len(reverse)}개",
            delta=f"{reverse['marginal_ROI'].mean():.3f} 평균",
            delta_color="inverse"
        )
    
    with col4:
        potential_savings = reverse['baseline_mean_sales'].sum() * 0.116
        st.metric(
            label="예상 예산 절감",
            value=f"${potential_savings/1000:.0f}K",
            delta="역효과 부서 중단 시"
        )
    
    st.markdown("---")
    
    # 탭
    tab1, tab2, tab3 = st.tabs(["🎯 핵심 인사이트", "📊 부서 분포", "⚡ 즉시 실행 항목"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("리스크-수익 포트폴리오")
            
            # Plotly 산점도
            fig = px.scatter(
                risk_df,
                x='std_ROI',
                y='mean_ROI',
                size='baseline_mean_sales',
                color='sensitivity_group',
                hover_data=['Dept', 'RAROI'],
                title="부서별 리스크-수익 매트릭스",
                labels={'std_ROI': 'ROI 변동성', 'mean_ROI': '평균 ROI'},
                height=500
            )
            
            median_roi = risk_df['mean_ROI'].median()
            median_std = risk_df['std_ROI'].median()
            
            fig.add_hline(y=median_roi, line_dash="dash", line_color="red", opacity=0.5, 
                         annotation_text="중위 ROI")
            fig.add_vline(x=median_std, line_dash="dash", line_color="blue", opacity=0.5,
                         annotation_text="중위 변동성")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("분기별 ROI 변동 (주요 부서)")
            
            # 주요 부서
            key_depts = [85, 56, 30, 18, 45]
            quarter_data = engine.dept_quarter[engine.dept_quarter['Dept'].isin(key_depts)]
            
            fig = go.Figure()
            
            colors = {'85': '#2ecc71', '56': '#3498db', '30': '#9b59b6', 
                     '18': '#e74c3c', '45': '#e67e22'}
            
            for dept in key_depts:
                dept_data = quarter_data[quarter_data['Dept'] == dept]
                if len(dept_data) > 0:
                    rois = []
                    for q in [1, 2, 3, 4]:
                        q_data = dept_data[dept_data['quarter'] == q]
                        if len(q_data) > 0:
                            rois.append(q_data['marginal_ROI'].values[0])
                        else:
                            rois.append(None)
                    
                    fig.add_trace(go.Scatter(
                        x=[1, 2, 3, 4],
                        y=rois,
                        mode='lines+markers',
                        name=f'Dept {dept}',
                        line=dict(width=3, color=colors.get(str(dept), '#95a5a6')),
                        marker=dict(size=10)
                    ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(
                title="주요 부서 분기별 ROI 추이",
                xaxis_title="분기",
                yaxis_title="ROI",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("민감도 그룹 분포")
            
            group_counts = dept_roi['민감도그룹'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=group_counts.index,
                values=group_counts.values,
                hole=.3,
                marker_colors=['#2ecc71', '#3498db', '#95a5a6', '#e74c3c', '#bdc3c7']
            )])
            
            fig.update_layout(title="부서별 할인 민감도 분포", height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("리스크 등급 분포")
            
            risk_counts = risk_df['risk_return_class'].value_counts()
            
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="리스크-수익 등급 분포",
                labels={'x': '등급', 'y': '부서 수'},
                color=risk_counts.values,
                color_continuous_scale='RdYlGn_r',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚡ 이번 주 실행해야 할 액션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("**🚨 즉시 중단 필요 (역효과 부서)**")
            reverse_depts = dept_roi[dept_roi['marginal_ROI'] < -0.01][
                ['Dept', 'marginal_ROI', 'baseline_mean_sales', '민감도그룹']
            ].sort_values('marginal_ROI')
            
            st.dataframe(
                reverse_depts.style.format({
                    'marginal_ROI': '{:.4f}',
                    'baseline_mean_sales': '${:,.0f}'
                }),
                use_container_width=True,
                height=300
            )
        
        with col2:
            st.success("**✅ 예산 증액 추천 (고ROI 부서)**")
            high_roi_depts = dept_roi[dept_roi['marginal_ROI'] > 0.015][
                ['Dept', 'marginal_ROI', 'baseline_mean_sales', '민감도그룹']
            ].sort_values('marginal_ROI', ascending=False)
            
            st.dataframe(
                high_roi_depts.style.format({
                    'marginal_ROI': '{:.4f}',
                    'baseline_mean_sales': '${:,.0f}'
                }),
                use_container_width=True,
                height=300
            )

elif mode == "🔍 부서별 분석":
    st.markdown('<p class="big-font">🔍 부서별 상세 분석</p>', unsafe_allow_html=True)
    st.markdown("부서와 조건을 선택하면 AI가 최적 프로모션 전략을 추천합니다.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dept = st.selectbox("부서 선택", sorted(dept_roi['Dept'].unique()))
    
    with col2:
        quarter = st.selectbox("분기", [1, 2, 3, 4], index=1)
    
    with col3:
        is_holiday = st.checkbox("휴일 주간")
    
    with col4:
        store_type = st.selectbox("매장 타입", ['A', 'B', 'C'])
    
    # 추가 옵션
    col1, col2 = st.columns(2)
    
    with col1:
        budget_input = st.number_input(
            "예산 입력 (선택, $)",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000,
            help="0이면 AI가 자동으로 추천합니다"
        )
        budget = budget_input if budget_input > 0 else None
    
    with col2:
        sales_target = st.number_input(
            "매출 목표 (선택, $)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=10000,
            help="목표 매출을 입력하면 필요한 예산을 계산합니다"
        )
        sales_target = sales_target if sales_target > 0 else None
    
    if st.button("🔮 분석 실행", type="primary"):
        with st.spinner("분석 중..."):
            rec = engine.get_recommendation(
                dept=dept,
                quarter=quarter,
                is_holiday=is_holiday,
                store_type=store_type,
                budget=budget,
                sales_target=sales_target
            )
        
        if 'error' in rec:
            st.error(f"오류: {rec['error']}")
        else:
            st.markdown("---")
            st.subheader("📊 분석 결과")
            
            # 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("예상 ROI", f"{rec['predicted_roi']:.3f}")
            
            with col2:
                roi_sentiment = "🟢" if rec['predicted_roi'] > 0.01 else "🔴" if rec['predicted_roi'] < 0 else "🟡"
                st.metric("액션", roi_sentiment, delta=rec['action'].split(':')[0])
            
            with col3:
                st.metric("권장 예산", f"${rec['recommended_budget']:,.0f}")
            
            with col4:
                st.metric("예상 매출 증가", f"${rec['expected_sales_lift']:,.0f}")
            
            st.markdown("---")
            
            # 상세 정보
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**기준 매출**: ${rec['baseline_sales']:,.0f}/주")
                st.info(f"**리스크 등급**: {rec['risk_class']}")
                st.info(f"**민감도 그룹**: {rec['sensitivity_group']}")
            
            with col2:
                st.success(f"**투자 강도**: {rec['md_intensity']:.1%}")
                st.success(f"**예상 총 매출**: ${rec['expected_total_sales']:,.0f}")
                st.success(f"**최대 투자 한도**: ${rec['max_budget']:,.0f}")
            
            # 매출 목표 역산 결과
            if rec['required_budget_for_target']:
                st.markdown("---")
                st.subheader("💰 매출 목표 달성을 위한 필요 예산")
                st.warning(f"**목표 매출 ${sales_target:,.0f}** 달성을 위해서는 **${rec['required_budget_for_target']:,.0f}** 필요합니다.")
            
            # 액션 플랜
            st.markdown("---")
            st.subheader("📋 실행 계획")
            
            if rec['predicted_roi'] > 0.015:
                st.success(f"""
                **✅ 적극 투자 권장**
                
                1. 이번 주 예산: ${rec['recommended_budget']:,.0f} 배정
                2. 기대 효과: 매출 ${rec['expected_sales_lift']:,.0f} 증가
                3. 모니터링: 주별 ROI 추적
                4. 확장: 성과 좋으면 다음 주 20% 증액
                """)
            elif rec['predicted_roi'] > 0:
                st.warning(f"""
                **⚠️ 제한적 투자 권장**
                
                1. 이번 주 예산: ${rec['recommended_budget']:,.0f} 배정
                2. 조건부 집행: 재고 확보 후 실행
                3. A/B 테스트: 일부 매장만 우선 실행
                4. 평가: 2주 후 효과 재평가
                """)
            else:
                st.error(f"""
                **❌ 프로모션 중단 권장**
                
                1. 현재 진행 중인 프로모션 즉시 중단
                2. 예산 재배분: 고ROI 부서로 이동
                3. 대안 전략: 품질/서비스 개선에 집중
                4. 재평가: 6개월 후 시장 변화 체크
                """)

elif mode == "⚡ 예산 최적화":
    st.markdown('<p class="big-font">⚡ AI 기반 예산 최적화</p>', unsafe_allow_html=True)
    st.markdown("전사 예산을 입력하면 AI가 부서별 최적 배분을 계산합니다.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_budget = st.number_input(
            "총 예산 ($)",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000
        )
    
    with col2:
        quarter = st.selectbox("대상 분기", [1, 2, 3, 4], index=1, key='opt_q')
    
    with col3:
        is_holiday = st.checkbox("휴일 포함", key='opt_holiday')
    
    with col4:
        risk_tolerance = st.selectbox(
            "리스크 성향",
            ['conservative', 'medium', 'aggressive'],
            index=1,
            format_func=lambda x: {'conservative': '보수적', 'medium': '중립', 'aggressive': '공격적'}[x]
        )
    
    if st.button("🚀 최적화 실행", type="primary"):
        with st.spinner("최적 배분 계산 중... (약 30초)"):
            optimal, summary = engine.optimize_portfolio(
                total_budget=total_budget,
                quarter=quarter,
                is_holiday=is_holiday,
                risk_tolerance=risk_tolerance
            )
        
        if optimal is None:
            st.error("최적화 실패: 조건을 만족하는 부서가 없습니다.")
        else:
            st.success("✅ 최적화 완료!")
            
            # 요약 지표
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("실제 배정액", f"${summary['total_allocated']:,.0f}")
            
            with col2:
                st.metric("예상 매출 증가", f"${summary['expected_total_lift']:,.0f}")
            
            with col3:
                st.metric("전사 ROI", f"{summary['overall_ROI']:.2%}")
            
            with col4:
                st.metric("투자 부서 수", f"{summary['n_departments']}개")
            
            st.markdown("---")
            
            # 탭
            tab1, tab2, tab3 = st.tabs(["📊 배분 결과", "📈 시각화", "📥 다운로드"])
            
            with tab1:
                st.subheader("부서별 최적 예산 배분")
                
                # 필터
                col1, col2 = st.columns(2)
                with col1:
                    min_budget_filter = st.slider(
                        "최소 예산 ($)",
                        0,
                        int(optimal['optimal_budget'].max()),
                        0,
                        1000
                    )
                
                filtered = optimal[optimal['optimal_budget'] >= min_budget_filter]
                
                st.dataframe(
                    filtered.style.format({
                        'optimal_budget': '${:,.0f}',
                        'baseline_sales': '${:,.0f}',
                        'expected_sales_lift': '${:,.0f}',
                        'expected_ROI': '{:.4f}',
                        'md_intensity': '{:.1%}',
                        'budget_pct': '{:.1%}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                st.info(f"**총 {len(filtered)}개 부서 표시 중** (필터 조건: 예산 ≥ ${min_budget_filter:,})")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top 20 배분")
                    
                    top20 = optimal.head(20)
                    
                    fig = px.bar(
                        top20,
                        x='Dept',
                        y='optimal_budget',
                        color='expected_ROI',
                        title="상위 20개 부서 예산 배분",
                        labels={'optimal_budget': '배정 예산 ($)', 'Dept': '부서'},
                        color_continuous_scale='RdYlGn',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("누적 매출 증가")
                    
                    sorted_results = optimal.sort_values('expected_sales_lift', ascending=False).reset_index(drop=True)
                    cumsum = sorted_results['expected_sales_lift'].cumsum()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(cumsum))),
                        y=cumsum,
                        fill='tozeroy',
                        name='누적 매출 증가',
                        line=dict(color='#2ecc71', width=3)
                    ))
                    
                    # 80% 지점
                    pct80_value = cumsum.iloc[-1] * 0.8
                    pct80_idx = (cumsum >= pct80_value).idxmax()
                    
                    fig.add_vline(x=pct80_idx, line_dash="dash", line_color="red",
                                 annotation_text=f"80% 달성: {pct80_idx+1}개 부서")
                    
                    fig.update_layout(
                        title="매출 증가 누적 곡선 (파레토)",
                        xaxis_title="부서 수",
                        yaxis_title="누적 매출 증가 ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("📥 결과 다운로드")
                
                csv = optimal.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv,
                    file_name=f"optimal_allocation_Q{quarter}_{'holiday' if is_holiday else 'regular'}.csv",
                    mime="text/csv"
                )
                
                # 요약 리포트
                report = f"""
# 최적 예산 배분 리포트

## 조건
- 총 예산: ${summary['total_budget']:,.0f}
- 분기: Q{summary['quarter']}
- 휴일: {'포함' if summary['is_holiday'] else '미포함'}
- 리스크 성향: {risk_tolerance}

## 결과
- 실제 배정: ${summary['total_allocated']:,.0f}
- 예상 매출 증가: ${summary['expected_total_lift']:,.0f}
- 전사 ROI: {summary['overall_ROI']:.2%}
- 투자 부서: {summary['n_departments']}개

## Top 10 배분
{optimal.head(10)[['Dept', 'optimal_budget', 'expected_ROI', 'expected_sales_lift']].to_string(index=False)}
"""
                
                st.download_button(
                    label="📄 리포트 다운로드 (TXT)",
                    data=report,
                    file_name=f"optimization_report_Q{quarter}.txt",
                    mime="text/plain"
                )

elif mode == "📈 시나리오 분석":
    st.markdown('<p class="big-font">📈 시나리오 분석</p>', unsafe_allow_html=True)
    st.markdown("여러 시나리오를 동시에 비교하여 최적의 전략을 선택하세요.")
    st.markdown("---")
    
    # 공통 설정
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input(
            "총 예산 ($)",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            key='scenario_budget'
        )
    
    with col2:
        quarter = st.selectbox("대상 분기", [1, 2, 3, 4], index=1, key='scenario_q')
    
    # 시나리오 정의
    st.subheader("비교할 시나리오 선택")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario1 = st.checkbox("보수적 전략", value=True)
    with col2:
        scenario2 = st.checkbox("중립 전략", value=True)
    with col3:
        scenario3 = st.checkbox("공격적 전략", value=True)
    
    if st.button("📊 시나리오 비교 실행", type="primary"):
        scenarios = []
        
        if scenario1:
            scenarios.append(('conservative', '보수적'))
        if scenario2:
            scenarios.append(('medium', '중립'))
        if scenario3:
            scenarios.append(('aggressive', '공격적'))
        
        if len(scenarios) == 0:
            st.warning("최소 1개 이상의 시나리오를 선택하세요.")
        else:
            results_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (risk, name) in enumerate(scenarios):
                status_text.text(f"계산 중: {name} 전략...")
                
                optimal, summary = engine.optimize_portfolio(
                    total_budget=total_budget,
                    quarter=quarter,
                    is_holiday=False,
                    risk_tolerance=risk
                )
                
                if optimal is not None:
                    results_list.append({
                        'scenario': name,
                        'risk_tolerance': risk,
                        'expected_roi': summary['overall_ROI'],
                        'expected_lift': summary['expected_total_lift'],
                        'n_departments': summary['n_departments'],
                        'allocation': optimal
                    })
                
                progress_bar.progress((i + 1) / len(scenarios))
            
            status_text.text("완료!")
            progress_bar.empty()
            
            if len(results_list) > 0:
                st.success(f"✅ {len(results_list)}개 시나리오 분석 완료!")
                
                # 비교 표
                st.subheader("📊 시나리오 비교")
                
                comparison_df = pd.DataFrame([
                    {
                        '전략': r['scenario'],
                        '예상 ROI': f"{r['expected_roi']:.2%}",
                        '예상 매출 증가': f"${r['expected_lift']:,.0f}",
                        '투자 부서': f"{r['n_departments']}개"
                    }
                    for r in results_list
                ])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # 시각화
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='예상 ROI',
                        x=[r['scenario'] for r in results_list],
                        y=[r['expected_roi'] * 100 for r in results_list],
                        marker_color='#3498db'
                    ))
                    
                    fig.update_layout(
                        title="시나리오별 예상 ROI 비교",
                        yaxis_title="ROI (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='예상 매출 증가',
                        x=[r['scenario'] for r in results_list],
                        y=[r['expected_lift'] for r in results_list],
                        marker_color='#2ecc71'
                    ))
                    
                    fig.update_layout(
                        title="시나리오별 예상 매출 증가 비교",
                        yaxis_title="매출 증가 ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 권장사항
                st.markdown("---")
                st.subheader("💡 권장사항")
                
                best_roi = max(results_list, key=lambda x: x['expected_roi'])
                best_lift = max(results_list, key=lambda x: x['expected_lift'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **ROI 최대화**
                    
                    {best_roi['scenario']} 전략 추천
                    - 예상 ROI: {best_roi['expected_roi']:.2%}
                    - 매출 증가: ${best_roi['expected_lift']:,.0f}
                    """)
                
                with col2:
                    st.info(f"""
                    **매출 최대화**
                    
                    {best_lift['scenario']} 전략 추천
                    - 예상 ROI: {best_lift['expected_roi']:.2%}
                    - 매출 증가: ${best_lift['expected_lift']:,.0f}
                    """)

elif mode == "💰 매출 목표 역산":
    st.markdown('<p class="big-font">💰 매출 목표 역산</p>', unsafe_allow_html=True)
    st.markdown("목표 매출을 입력하면 필요한 프로모션 예산을 계산합니다.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sales_target = st.number_input(
            "목표 매출 ($)",
            min_value=100000,
            max_value=100000000,
            value=5000000,
            step=100000
        )
    
    with col2:
        quarter = st.selectbox("대상 분기", [1, 2, 3, 4], index=1, key='reverse_q')
    
    with col3:
        is_holiday = st.checkbox("휴일 포함", key='reverse_holiday')
    
    # 현재 baseline 표시
    total_baseline = dept_roi['baseline_mean_sales'].sum()
    required_lift = sales_target - total_baseline
    
    st.info(f"""
    **현재 상황**
    - 현재 기준 매출 (프로모션 없음): ${total_baseline:,.0f}
    - 목표 매출: ${sales_target:,.0f}
    - 필요한 매출 증가: ${required_lift:,.0f} ({required_lift/total_baseline:.1%})
    """)
    
    if required_lift <= 0:
        st.success("✅ 목표 매출이 이미 기준 매출보다 낮습니다. 프로모션이 필요 없습니다!")
    elif st.button("💰 필요 예산 계산", type="primary"):
        with st.spinner("계산 중... (최대 1분 소요)"):
            result = engine.reverse_calculate_budget(
                sales_target=sales_target,
                quarter=quarter,
                is_holiday=is_holiday
            )
        
        if 'error' in result:
            st.error(f"계산 실패: {result.get('error', 'Unknown error')}")
            if 'max_achievable' in result:
                st.warning(f"최대 달성 가능 매출: ${result['max_achievable']:,.0f}")
        else:
            st.success("✅ 계산 완료!")
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("필요 예산", f"${result['required_budget']:,.0f}")
            
            with col2:
                st.metric("예상 ROI", f"{result['expected_roi']:.2%}")
            
            with col3:
                st.metric("예상 매출 증가", f"${result['expected_lift']:,.0f}")
            
            st.markdown("---")
            
            # 배분 결과
            st.subheader("부서별 예산 배분")
            
            allocation = result['allocation']
            
            st.dataframe(
                allocation.head(20).style.format({
                    'optimal_budget': '${:,.0f}',
                    'baseline_sales': '${:,.0f}',
                    'expected_sales_lift': '${:,.0f}',
                    'expected_ROI': '{:.4f}',
                    'md_intensity': '{:.1%}'
                }),
                use_container_width=True
            )
            
            # 시각화
            fig = px.treemap(
                allocation.head(20),
                path=['Dept'],
                values='optimal_budget',
                color='expected_ROI',
                title="Top 20 부서별 예산 배분 (트리맵)",
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Walmart Promo Optimizer v1.0")
st.sidebar.caption("Powered by AI & Data Science")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walmart 프로모션 최적화 엔진
통합 API 클래스
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import os
import warnings
warnings.filterwarnings('ignore')

class PromoOptimizationEngine:
    """
    프로모션 최적화 통합 엔진
    
    기능:
    1. 부서별 ROI 예측
    2. 최적 예산 배분 계산
    3. 매출 목표 역산
    4. 시나리오 비교
    """
    
    def __init__(self, data_path=None):
        """
        초기화
        
        Args:
            data_path: 데이터 파일 경로 (기본값: 현재 디렉토리)
        """
        if data_path:
            os.chdir(data_path)
        
        print("[Optimization Engine] Initializing...")
        self.load_all_data()
        print("[Optimization Engine] Ready!")
    
    def load_all_data(self):
        """모든 분석 결과 로드"""
        try:
            self.dept_roi = pd.read_csv('v3_dept_roi_all.csv')
            self.dept_quarter = pd.read_csv('v3_dept_quarter_cross.csv')
            self.dept_holiday = pd.read_csv('v3_dept_holiday_cross.csv')
            self.dept_storetype = pd.read_csv('v3_dept_storetype_cross.csv')
            self.risk_adjusted = pd.read_csv('risk_adjusted_roi_analysis.csv')
            
            print(f"  [OK] Loaded {len(self.dept_roi)} departments")
            print(f"  [OK] Loaded risk-adjusted analysis for {len(self.risk_adjusted)} departments")
            
        except FileNotFoundError as e:
            print(f"  [ERROR] File not found: {e}")
            raise
    
    def get_department_info(self, dept):
        """
        부서 기본 정보 조회
        
        Args:
            dept: 부서 번호
            
        Returns:
            dict: 부서 정보
        """
        dept_info = self.dept_roi[self.dept_roi['Dept'] == dept]
        
        if len(dept_info) == 0:
            return {'error': f'Department {dept} not found'}
        
        dept_info = dept_info.iloc[0]
        risk_info = self.risk_adjusted[self.risk_adjusted['Dept'] == dept]
        
        result = {
            'Dept': int(dept),
            'baseline_sales': float(dept_info['baseline_mean_sales']),
            'marginal_ROI': float(dept_info['marginal_ROI']),
            'sensitivity_group': dept_info['민감도그룹'],
            'n_observations': int(dept_info['n']),
            'n_markdown': int(dept_info['n_markdown'])
        }
        
        if len(risk_info) > 0:
            risk_info = risk_info.iloc[0]
            result['risk_adjusted_roi'] = float(risk_info['RAROI'])
            result['roi_std'] = float(risk_info['std_ROI'])
            result['roi_range'] = float(risk_info['range_ROI'])
            result['risk_class'] = risk_info['risk_return_class']
        
        return result
    
    def predict_roi(self, dept, quarter=None, is_holiday=None, store_type=None):
        """
        조건별 ROI 예측
        
        Args:
            dept: 부서 번호
            quarter: 분기 (1~4, None이면 전체 평균)
            is_holiday: 휴일 여부 (True/False/None)
            store_type: 매장 타입 ('A'/'B'/'C'/None)
            
        Returns:
            dict: 예측 ROI 및 관련 정보
        """
        rois = []
        
        # 전체 ROI
        dept_info = self.dept_roi[self.dept_roi['Dept'] == dept]
        if len(dept_info) > 0:
            overall_roi = float(dept_info.iloc[0]['marginal_ROI'])
        else:
            return {'error': f'Department {dept} not found'}
        
        # 분기별 ROI
        if quarter is not None:
            q_data = self.dept_quarter[
                (self.dept_quarter['Dept'] == dept) & 
                (self.dept_quarter['quarter'] == quarter)
            ]
            if len(q_data) > 0:
                rois.append(float(q_data.iloc[0]['marginal_ROI']))
        
        # 휴일 ROI
        if is_holiday is not None:
            h_data = self.dept_holiday[
                (self.dept_holiday['Dept'] == dept) & 
                (self.dept_holiday['IsHoliday'] == is_holiday)
            ]
            if len(h_data) > 0:
                rois.append(float(h_data.iloc[0]['marginal_ROI']))
        
        # 매장 타입 ROI
        if store_type is not None:
            col_name = f'Type_{store_type}_ROI'
            s_data = self.dept_storetype[self.dept_storetype['Dept'] == dept]
            if len(s_data) > 0 and col_name in s_data.columns:
                val = s_data.iloc[0][col_name]
                if not pd.isna(val):
                    rois.append(float(val))
        
        # 최종 ROI 계산
        if len(rois) > 0:
            predicted_roi = np.mean([overall_roi] + rois)
        else:
            predicted_roi = overall_roi
        
        return {
            'dept': int(dept),
            'predicted_roi': float(predicted_roi),
            'overall_roi': float(overall_roi),
            'n_conditions': len(rois) + 1,
            'quarter': quarter,
            'is_holiday': is_holiday,
            'store_type': store_type
        }
    
    def get_recommendation(self, dept, quarter, is_holiday, store_type, 
                          budget=None, sales_target=None):
        """
        부서별 프로모션 추천
        
        Args:
            dept: 부서 번호
            quarter: 분기 (1~4)
            is_holiday: 휴일 여부
            store_type: 매장 타입 ('A'/'B'/'C')
            budget: 예산 (선택, $)
            sales_target: 매출 목표 (선택, $)
            
        Returns:
            dict: 추천 및 예측 결과
        """
        # 기본 정보
        dept_info = self.get_department_info(dept)
        if 'error' in dept_info:
            return dept_info
        
        # ROI 예측
        roi_pred = self.predict_roi(dept, quarter, is_holiday, store_type)
        predicted_roi = roi_pred['predicted_roi']
        
        baseline = dept_info['baseline_sales']
        
        # 액션 결정
        if predicted_roi < -0.01:
            action = "STOP: 프로모션 중단 권장"
            if store_type == 'C':
                action += " (단, Type C는 소규모 테스트 가능)"
            recommended_budget = 0
        elif predicted_roi > 0.015:
            action = "GO: 적극 투자 권장"
            recommended_budget = baseline * 0.6  # 60%
        elif predicted_roi > 0:
            action = "CAUTION: 제한적 투자"
            recommended_budget = baseline * 0.3  # 30%
        else:
            action = "HOLD: 관망"
            recommended_budget = 0
        
        # 임계점 체크
        max_budget = baseline * 1.16
        if recommended_budget > max_budget:
            recommended_budget = max_budget
        
        # 예산이 주어진 경우
        if budget:
            md_intensity = budget / baseline if baseline > 0 else 0
            expected_lift = budget * predicted_roi
            expected_total_sales = baseline + expected_lift
        else:
            budget = recommended_budget
            md_intensity = budget / baseline if baseline > 0 else 0
            expected_lift = budget * predicted_roi
            expected_total_sales = baseline + expected_lift
        
        # 매출 목표가 주어진 경우 (역산)
        if sales_target:
            required_lift = sales_target - baseline
            if predicted_roi > 0:
                required_budget = required_lift / predicted_roi
                required_budget = min(required_budget, max_budget)
            else:
                required_budget = 0
        else:
            required_budget = None
        
        return {
            'dept': int(dept),
            'baseline_sales': float(baseline),
            'predicted_roi': float(predicted_roi),
            'action': action,
            'recommended_budget': float(recommended_budget),
            'max_budget': float(max_budget),
            'actual_budget': float(budget) if budget else None,
            'md_intensity': float(md_intensity),
            'expected_sales_lift': float(expected_lift),
            'expected_total_sales': float(expected_total_sales),
            'required_budget_for_target': float(required_budget) if required_budget else None,
            'risk_class': dept_info.get('risk_class', 'Unknown'),
            'sensitivity_group': dept_info['sensitivity_group']
        }
    
    def optimize_portfolio(self, total_budget, quarter, is_holiday=False, 
                          risk_tolerance='medium', sales_target=None):
        """
        전사 포트폴리오 최적화
        
        Args:
            total_budget: 총 예산 ($)
            quarter: 대상 분기 (1~4)
            is_holiday: 휴일 여부
            risk_tolerance: 리스크 성향 ('conservative'/'medium'/'aggressive')
            sales_target: 전사 매출 목표 (선택)
            
        Returns:
            tuple: (배분 결과 DataFrame, 요약 dict)
        """
        print(f"\n[Optimization] Starting portfolio optimization...")
        print(f"  Total Budget: ${total_budget:,.0f}")
        print(f"  Quarter: Q{quarter}")
        print(f"  Holiday: {is_holiday}")
        print(f"  Risk Tolerance: {risk_tolerance}")
        
        # 조건별 ROI 추출
        q_rois = self.dept_quarter[self.dept_quarter['quarter'] == quarter][
            ['Dept', 'marginal_ROI']
        ].rename(columns={'marginal_ROI': 'q_roi'})
        
        h_data = self.dept_holiday[self.dept_holiday['IsHoliday'] == is_holiday]
        h_rois = h_data[['Dept', 'marginal_ROI']].rename(columns={'marginal_ROI': 'h_roi'})
        
        # 통합
        dept_data = self.dept_roi[['Dept', 'baseline_mean_sales', 'marginal_ROI']].merge(
            q_rois, on='Dept', how='left'
        ).merge(
            h_rois, on='Dept', how='left'
        ).merge(
            self.risk_adjusted[['Dept', 'std_ROI', 'RAROI', 'risk_return_class']], 
            on='Dept', how='left'
        )
        
        dept_data['q_roi'] = dept_data['q_roi'].fillna(dept_data['marginal_ROI'])
        dept_data['h_roi'] = dept_data['h_roi'].fillna(dept_data['marginal_ROI'])
        dept_data['final_ROI'] = (dept_data['q_roi'] + dept_data['h_roi']) / 2
        
        # 리스크 성향에 따른 필터링
        if risk_tolerance == 'conservative':
            # A등급만
            eligible = dept_data[dept_data['risk_return_class'] == 'A: 고수익-저위험']
        elif risk_tolerance == 'aggressive':
            # A, B등급
            eligible = dept_data[dept_data['risk_return_class'].isin([
                'A: 고수익-저위험', 'B: 고수익-고위험'
            ])]
        else:
            # ROI > 0인 모든 부서
            eligible = dept_data[dept_data['final_ROI'] > 0]
        
        eligible = eligible[eligible['baseline_mean_sales'] > 0].copy()
        eligible = eligible.dropna(subset=['baseline_mean_sales', 'final_ROI'])
        
        if len(eligible) == 0:
            return None, {'error': 'No eligible departments found'}
        
        print(f"  Eligible Departments: {len(eligible)}")
        
        n_depts = len(eligible)
        dept_list = eligible['Dept'].values
        baseline_list = eligible['baseline_mean_sales'].values
        roi_list = eligible['final_ROI'].values
        std_list = eligible['std_ROI'].fillna(0.01).values
        
        # 목적함수
        def objective(budget_allocation):
            total_sales_lift = np.sum(budget_allocation * roi_list * baseline_list)
            risk_penalty = 0.05 * np.sum(
                (budget_allocation / baseline_list) ** 2 * std_list * baseline_list
            )
            return -(total_sales_lift - risk_penalty)
        
        # 제약조건
        constraint_sum = LinearConstraint(
            A=np.ones(n_depts),
            lb=total_budget * 0.95,
            ub=total_budget
        )
        
        upper_bounds = baseline_list * 1.16
        lower_bounds = np.where(roi_list > 0.01, baseline_list * 0.05, 0)
        lower_bounds = np.minimum(lower_bounds, upper_bounds * 0.9)
        
        bounds = Bounds(lb=lower_bounds, ub=upper_bounds)
        
        # 초기값
        x0 = (roi_list / roi_list.sum()) * total_budget
        x0 = np.clip(x0, lower_bounds, upper_bounds)
        
        # 최적화
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint_sum],
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"  [WARNING] {result.message}")
        
        # 결과 정리
        optimal_budgets = result.x
        
        results = []
        for i, dept in enumerate(dept_list):
            budget = optimal_budgets[i]
            baseline = baseline_list[i]
            roi = roi_list[i]
            
            results.append({
                'Dept': int(dept),
                'optimal_budget': float(budget),
                'baseline_sales': float(baseline),
                'expected_ROI': float(roi),
                'expected_sales_lift': float(budget * roi),
                'expected_total_sales': float(baseline + budget * roi),
                'budget_pct': float(budget / total_budget),
                'md_intensity': float(budget / baseline if baseline > 0 else 0)
            })
        
        results_df = pd.DataFrame(results).sort_values('optimal_budget', ascending=False)
        
        summary = {
            'total_budget': float(total_budget),
            'total_allocated': float(results_df['optimal_budget'].sum()),
            'expected_total_lift': float(results_df['expected_sales_lift'].sum()),
            'overall_ROI': float(results_df['expected_sales_lift'].sum() / total_budget),
            'n_departments': int(len(results_df[results_df['optimal_budget'] > 1000])),
            'risk_tolerance': risk_tolerance,
            'quarter': quarter,
            'is_holiday': is_holiday
        }
        
        print(f"  [OK] Optimization completed")
        print(f"  Expected ROI: {summary['overall_ROI']:.2%}")
        print(f"  Investing in {summary['n_departments']} departments")
        
        return results_df, summary
    
    def reverse_calculate_budget(self, sales_target, quarter, is_holiday=False):
        """
        매출 목표에서 필요 예산 역산
        
        Args:
            sales_target: 목표 매출 ($)
            quarter: 분기 (1~4)
            is_holiday: 휴일 여부
            
        Returns:
            dict: 필요 예산 및 배분
        """
        print(f"\n[Reverse Calculate] Target Sales: ${sales_target:,.0f}")
        
        # 현재 총 baseline
        total_baseline = self.dept_roi['baseline_mean_sales'].sum()
        required_lift = sales_target - total_baseline
        
        print(f"  Current Baseline: ${total_baseline:,.0f}")
        print(f"  Required Lift: ${required_lift:,.0f}")
        
        if required_lift <= 0:
            return {
                'message': 'Target already achievable without promotion',
                'required_budget': 0,
                'target_sales': sales_target,
                'baseline_sales': total_baseline
            }
        
        # 이진 탐색으로 필요 예산 찾기
        min_budget = 0
        max_budget = required_lift * 100  # 매우 큰 값
        
        for _ in range(20):  # 최대 20회 반복
            mid_budget = (min_budget + max_budget) / 2
            
            results, summary = self.optimize_portfolio(
                total_budget=mid_budget,
                quarter=quarter,
                is_holiday=is_holiday
            )
            
            if results is None:
                break
            
            achieved_lift = summary['expected_total_lift']
            
            if abs(achieved_lift - required_lift) < required_lift * 0.01:  # 1% 오차
                print(f"  [OK] Found required budget: ${mid_budget:,.0f}")
                return {
                    'required_budget': float(mid_budget),
                    'expected_lift': float(achieved_lift),
                    'expected_roi': float(summary['overall_ROI']),
                    'target_sales': float(sales_target),
                    'baseline_sales': float(total_baseline),
                    'allocation': results
                }
            elif achieved_lift < required_lift:
                min_budget = mid_budget
            else:
                max_budget = mid_budget
        
        return {
            'error': 'Could not find feasible budget',
            'target_sales': sales_target,
            'max_achievable': total_baseline + achieved_lift
        }

# 사용 예시
if __name__ == "__main__":
    print("="*60)
    print("Optimization Engine Test")
    print("="*60)
    
    # 엔진 초기화
    engine = PromoOptimizationEngine()
    
    # 1. 부서 정보 조회
    print("\n[Test 1] Get Department Info")
    info = engine.get_department_info(85)
    print(f"  Dept 85: {info}")
    
    # 2. ROI 예측
    print("\n[Test 2] Predict ROI")
    roi = engine.predict_roi(85, quarter=2, is_holiday=True, store_type='A')
    print(f"  Predicted ROI: {roi['predicted_roi']:.4f}")
    
    # 3. 추천
    print("\n[Test 3] Get Recommendation")
    rec = engine.get_recommendation(
        dept=85,
        quarter=2,
        is_holiday=True,
        store_type='A',
        sales_target=500000
    )
    print(f"  Action: {rec['action']}")
    print(f"  Recommended Budget: ${rec['recommended_budget']:,.0f}")
    
    # 4. 포트폴리오 최적화
    print("\n[Test 4] Portfolio Optimization")
    results, summary = engine.optimize_portfolio(
        total_budget=1000000,
        quarter=2,
        is_holiday=False,
        risk_tolerance='medium'
    )
    
    print("\n[Test 5] Top 5 Allocation")
    print(results[['Dept', 'optimal_budget', 'expected_ROI', 'expected_sales_lift']].head(5).to_string(index=False))
    
    print("\n="*60)
    print("[Complete] All tests passed!")
    print("="*60)

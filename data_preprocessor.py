#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 전처리 모듈 (개선)

시계열 데이터 전처리와 자원별 사용량 지표 생성을 담당합니다.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    데이터 전처리 클래스
    
    시계열 데이터 전처리와 자원별 사용량 지표 생성을 담당합니다.
    """
    
    def __init__(self, config=None):
        """
        초기화
        
        Args:
            config (dict): 설정 정보
        """
        self.config = config or {}
        self.scaler = None
        self.feature_scaler = None
        self.result_handler = None  # 추가: result_handler 초기화
        
        # 리소스 한계값 설정
        resources_limits = self.config.get('resources_limits', {})
        self.network_max_bandwidth = resources_limits.get('network', {}).get('max_bandwidth', 125000000)  # 기본값 1Gbps
        self.disk_max_io_rate = resources_limits.get('disk', {}).get('max_io_rate', 100000000)  # 기본값 100MB/s
        
        # 이전 데이터 저장용 캐시
        self.prev_data = {}

    def set_result_handler(self, result_handler):
        """
        결과 핸들러 설정
        
        Args:
            result_handler (ResultHandler): 결과 핸들러 객체
        """
        self.result_handler = result_handler
    
    def resample_data(self, df, freq=None):
        """
        데이터 리샘플링 (기본 1시간 단위)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            freq (str): 리샘플링 주기 (None일 경우 config에서 가져옴)
            
        Returns:
            pd.DataFrame: 리샘플링된 데이터프레임
        """
        if df is None or df.empty:
            logger.warning("리샘플링할 데이터가 비어 있습니다.")
            return df
        
        # 설정에서 리샘플링 주기 가져오기 (기본값: 1시간)
        if freq is None:
            freq = self.config.get('advanced', {}).get('resampling', {}).get('freq', '1h')
            logger.info(f"리샘플링 주기: {freq}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("리샘플링을 위해서는 타임스탬프 인덱스가 필요합니다.")
            # 인덱스가 타임스탬프가 아니면서 'timestamp' 또는 '_time' 열이 있는 경우 처리
            if 'timestamp' in df.columns:
                logger.info("'timestamp' 열을 인덱스로 설정합니다.")
                df = df.set_index('timestamp')
            elif '_time' in df.columns:
                logger.info("'_time' 열을 인덱스로 설정합니다.")
                df = df.set_index('_time')
            else:
                logger.error("타임스탬프 열을 찾을 수 없습니다.")
                return df
        
        # 숫자형 열 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        try:
            # 리샘플링 (평균값 사용)
            if len(numeric_cols) > 0:
                resampled = df[numeric_cols].resample(freq).mean()
            else:
                logger.warning("숫자형 열이 없어 리샘플링을 수행할 수 없습니다.")
                return df
            
            # 결측치 처리
            resampled = resampled.interpolate(method='time').bfill().ffill()
            
            # 메타데이터 컬럼(location, origin, device_id 등) 복원
            for col in df.columns:
                if col not in numeric_cols and col not in resampled.columns:
                    try:
                        # 모드(최빈값) 계산
                        mode_series = df[col].mode()
                        if not mode_series.empty:
                            most_common = mode_series.iloc[0]
                        else:
                            # 최빈값이 없으면 첫 번째 비 NaN 값 사용
                            non_nan_values = df[col].dropna()
                            most_common = non_nan_values.iloc[0] if not non_nan_values.empty else None
                        
                        resampled[col] = most_common
                    except Exception as e:
                        logger.warning(f"'{col}' 열의 메타데이터 복원 실패: {e}")
                        resampled[col] = None
            
            logger.info(f"데이터 리샘플링 완료: {len(df)}행 -> {len(resampled)}행 (주기: {freq})")
            return resampled
            
        except Exception as e:
            logger.error(f"리샘플링 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def handle_missing_values(self, df, numeric_cols=None):
        """
        결측치 처리 (개선)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            numeric_cols (list): 숫자형 열 목록
            
        Returns:
            pd.DataFrame: 결측치가 처리된 데이터프레임
        """
        if df is None or df.empty:
            logger.warning("처리할 데이터가 비어 있습니다.")
            return df
            
        # 숫자형 열 목록이 주어지지 않은 경우 자동 탐지
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 결측치 비율 확인
        missing_ratio = df[numeric_cols].isna().mean()
        max_missing_ratio = self.config.get('advanced', {}).get('missing_data', {}).get('max_missing_ratio', 0.8)
        high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
        
        if high_missing_cols:
            logger.warning(f"다음 열은 {max_missing_ratio*100}% 이상의 결측치를 가지고 있습니다: {high_missing_cols}")
        
        try:
            # 타임스탬프 인덱스가 있는 경우 시계열 데이터로 간주
            if isinstance(df.index, pd.DatetimeIndex):
                # 연속적인 인덱스 확인
                if len(df.index) > 1:
                    time_diff = df.index.to_series().diff().median()
                    
                    # 시간 순서대로 정렬
                    df = df.sort_index()
                    
                    # 결측치 처리 방법 가져오기
                    fill_method = self.config.get('advanced', {}).get('missing_data', {}).get('fill_method', 'ffill')
                    
                    # 결측치 처리
                    if fill_method == 'ffill':
                        # 앞의 값으로 채우기
                        df[numeric_cols] = df[numeric_cols].ffill()
                        # 그래도 남아있는 결측치는 뒤의 값으로 채우기
                        df[numeric_cols] = df[numeric_cols].bfill()
                    elif fill_method == 'bfill':
                        # 뒤의 값으로 채우기
                        df[numeric_cols] = df[numeric_cols].bfill()
                        # 그래도 남아있는 결측치는 앞의 값으로 채우기
                        df[numeric_cols] = df[numeric_cols].ffill()
                    else:
                        # 시계열 보간법
                        df[numeric_cols] = df[numeric_cols].interpolate(method='time').ffill().bfill()
                else:
                    # 단일 행인 경우 평균값으로 대체 (해당 열의 값)
                    for col in numeric_cols:
                        if pd.isna(df[col]).any():
                            df[col] = df[col].fillna(0)  # 기본값으로 0 사용
            else:
                # 일반 데이터는 선형 보간법 사용
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
            
            # 여전히 남아있는 결측치는 열별 평균으로 대체
            for col in numeric_cols:
                if df[col].isna().any():
                    mean_val = df[col].mean()
                    if pd.isna(mean_val):  # 열 전체가 NaN인 경우
                        mean_val = 0
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"'{col}' 열의 남은 결측치를 값({mean_val:.2f})으로 대체")
            
            # 비수치 열의 결측치 처리
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            for col in non_numeric_cols:
                if df[col].isna().any():
                    # 가장 많이 등장하는 값으로 대체 (없으면 '알 수 없음')
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else '알 수 없음'
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"'{col}' 열의 결측치를 최빈값('{mode_val}')으로 대체")
            
            return df
            
        except Exception as e:
            logger.error(f"결측치 처리 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def handle_outliers(self, df, numeric_cols=None, method='zscore', threshold=3.0):
        """
        이상치 처리
        
        Args:
            df (pd.DataFrame): 데이터프레임
            numeric_cols (list): 숫자형 열 목록
            method (str): 이상치 탐지 방법 ('zscore', 'iqr')
            threshold (float): 이상치 판단 임계값
            
        Returns:
            pd.DataFrame: 이상치가 처리된 데이터프레임
        """
        if df is None or df.empty:
            logger.warning("처리할 데이터가 비어 있습니다.")
            return df
            
        # 숫자형 열 목록이 주어지지 않은 경우 자동 탐지
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        try:
            for col in numeric_cols:
                # 이상치 탐지 및 처리
                if method == 'zscore':
                    # Z-점수 방식
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:  # 표준편차가 0이 아닌 경우만 처리
                        z_scores = np.abs((df[col] - mean) / std)
                        outliers = z_scores > threshold
                        
                        if outliers.sum() > 0:
                            # 이상치를 중앙값으로 대체
                            median_val = df[col].median()
                            df.loc[outliers, col] = median_val
                            logger.info(f"'{col}' 열에서 {outliers.sum()}개 이상치를 중앙값({median_val:.2f})으로 대체")
                
                elif method == 'iqr':
                    # IQR 방식
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                    if outliers.sum() > 0:
                        # 이상치를 한계값으로 대체
                        df.loc[df[col] < lower_bound, col] = lower_bound
                        df.loc[df[col] > upper_bound, col] = upper_bound
                        logger.info(f"'{col}' 열에서 {outliers.sum()}개 이상치를 IQR 한계값으로 대체")
            
            return df
            
        except Exception as e:
            logger.error(f"이상치 처리 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def calculate_resource_metrics(self, df):
        """
        자원별 사용량 지표 계산 (개선됨)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            
        Returns:
            pd.DataFrame: 자원 지표가 추가된 데이터프레임
        """
        if df is None or df.empty:
            logger.warning("자원 지표를 계산할 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        try:
            # 결과 데이터프레임 초기화 - 원본 인덱스 유지
            metrics_df = pd.DataFrame(index=df.index)
            
            # 기본 메타데이터 복사
            meta_columns = ['device_id', 'companyDomain', 'building', 'location', 'origin']
            for col in meta_columns:
                if col in df.columns:
                    non_na_values = df[col].dropna()
                    if not non_na_values.empty:
                        # 첫 번째 비 NaN 값 사용
                        metrics_df[col] = non_na_values.iloc[0]
            
            # 데이터에 value 컬럼이 있는지 확인
            has_value_column = '_value' in df.columns or 'value' in df.columns
            value_column = 'value' if 'value' in df.columns else '_value' if '_value' in df.columns else None
            
            # 위치(location) 확인
            location = None
            if 'location' in df.columns:
                loc_values = df['location'].dropna().unique()
                location = loc_values[0] if len(loc_values) > 0 else None
            
            # 1. CPU 지표 계산
            if location == 'cpu':
                # CPU 사용률 계산
                if 'usage_idle' in df.columns:
                    metrics_df['cpu_usage'] = 100 - df['usage_idle']
                elif value_column and has_value_column:
                    # value 열을 직접 사용
                    metrics_df['cpu_usage'] = df[value_column]
            
            # 2. 메모리 지표
            elif location == 'mem':
                # 메모리 사용률 계산
                if 'used_percent' in df.columns:
                    metrics_df['memory_used_percent'] = df['used_percent']
                elif value_column and has_value_column:
                    # value 열을 직접 사용
                    metrics_df['memory_used_percent'] = df[value_column]
            
            # 3. 디스크 지표
            elif location == 'disk':
                if 'used_percent' in df.columns:
                    metrics_df['disk_used_percent'] = df['used_percent']
                elif value_column and has_value_column:
                    metrics_df['disk_used_percent'] = df[value_column]
            
            # 4. 디스크 I/O 지표
            elif location == 'diskio':
                # I/O 대기 시간 또는 직접 값 사용
                if 'io_time' in df.columns:
                    metrics_df['disk_io_utilization'] = df['io_time']
                elif value_column and has_value_column:
                    metrics_df['disk_io_utilization'] = df[value_column]
            
            # 5. 네트워크 지표
            elif location == 'net':
                if value_column and has_value_column:
                    metrics_df['net_utilization'] = df[value_column]
                
                # 네트워크 관련 필드가 있으면 사용
                for field in ['bytes_recv', 'bytes_sent', 'drop_in', 'drop_out']:
                    if field in df.columns:
                        metrics_df[f'net_{field}'] = df[field]
            
            # 6. 시스템 부하 지표
            elif location == 'system':
                if 'load1' in df.columns:
                    metrics_df['system_load1'] = df['load1']
                elif value_column and has_value_column:
                    metrics_df['system_load1'] = df[value_column]
            
            # 가용한 데이터가 없는 경우
            if metrics_df.shape[1] <= len(meta_columns):
                logger.warning(f"자원 '{location}'에 대한 지표를 계산할 수 없습니다. 직접 value 열 사용")
                
                # value 열이 있으면 직접 사용
                if value_column and has_value_column:
                    if location == 'cpu':
                        metrics_df['cpu_usage'] = df[value_column]
                    elif location == 'mem':
                        metrics_df['memory_used_percent'] = df[value_column]
                    elif location == 'disk':
                        metrics_df['disk_used_percent'] = df[value_column]
                    elif location == 'diskio':
                        metrics_df['disk_io_utilization'] = df[value_column]
                    elif location == 'net':
                        metrics_df['net_utilization'] = df[value_column]
                    elif location == 'system':
                        metrics_df['system_load1'] = df[value_column]
                    else:
                        metrics_df[f'{location}_value'] = df[value_column]
            
            # NaN 값 처리
            metrics_df = metrics_df.fillna(0)
            
            # 범위 제한 (0-100% 사이로 제한)
            pct_columns = [col for col in metrics_df.columns if 'percent' in col or 'utilization' in col or 'usage' in col]
            for col in pct_columns:
                metrics_df[col] = metrics_df[col].clip(0, 100)
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"자원 지표 계산 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 최소한의 결과 반환 시도
            try:
                # 메타데이터만이라도 포함하는 빈 결과 반환
                min_result = pd.DataFrame(index=df.index)
                for col in ['device_id', 'companyDomain', 'building', 'location', 'origin']:
                    if col in df.columns:
                        min_result[col] = df[col].iloc[0] if len(df) > 0 else None
                
                # value 값이 있으면 추가
                if 'value' in df.columns:
                    min_result['value'] = df['value']
                elif '_value' in df.columns:
                    min_result['value'] = df['_value']
                
                if 'location' in df.columns:
                    loc = df['location'].iloc[0] if len(df) > 0 else 'unknown'
                    min_result[f'{loc}_value'] = df['value'] if 'value' in df.columns else df['_value'] if '_value' in df.columns else 0
                
                return min_result
            except:
                return pd.DataFrame()
    
    def process_resource_data(self, df, location, incremental=True):
        """
        자원별 데이터 처리 (중복 처리 로직 개선)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            location (str): 자원 위치
            incremental (bool): 증분 처리 여부
            
        Returns:
            pd.DataFrame: 처리된 자원 지표 데이터프레임
        """
        if df is None or df.empty:
            logger.warning(f"{location} 위치의 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        try:
            logger.info(f"'{location}' 자원 데이터 처리 시작 ({len(df)}행)")
            
            # 기존 데이터 로드 (증분 처리용)
            existing_data = None
            existing_keys = set()  # (timestamp, device_id, metric_name) 튜플 저장
            
            if incremental and self.result_handler:
                # 최근 처리된 데이터 로드 (마지막 30분)
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=30)
                
                # EAV 모델 사용 여부 확인
                use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
                
                if use_eav_model:
                    # 자원 메트릭 테이블에서 로드
                    existing_data = self.result_handler.load_resource_metrics(
                        resource_type=location,
                        start_time=start_time,
                        end_time=end_time
                    )
                else:
                    # 기존 테이블에서 로드
                    filter_cond = None
                    if location == 'cpu':
                        filter_cond = "cpu_usage IS NOT NULL"
                    elif location == 'mem':
                        filter_cond = "memory_used_percent IS NOT NULL"
                    elif location == 'disk':
                        filter_cond = "disk_used_percent IS NOT NULL"
                    elif location == 'diskio':
                        filter_cond = "disk_io_utilization IS NOT NULL"
                    elif location == 'net':
                        filter_cond = "net_utilization IS NOT NULL"
                    elif location == 'system':
                        filter_cond = "system_load1 IS NOT NULL"
                    
                    existing_data = self.result_handler.load_from_mysql(
                        table_name="processed_resource_data",
                        start_time=start_time,
                        end_time=end_time,
                        filter_cond=filter_cond
                    )
                
                # 기존 데이터에서 고유 키 추출 (timestamp, device_id, metric_name)
                if existing_data is not None and not existing_data.empty:
                    # device_id 확인
                    device_id_col = 'device_id' if 'device_id' in existing_data.columns else None
                    
                    # 각 열에 대해 고유 키 생성
                    for col in existing_data.select_dtypes(include=[np.number]).columns:
                        # 메타데이터 열은 건너뜀
                        if col in ['id', 'device_id', 'companyDomain', 'building'] or 'timestamp' in col.lower():
                            continue
                        
                        # 타임스탬프가 인덱스인 경우
                        if isinstance(existing_data.index, pd.DatetimeIndex):
                            for idx, row in existing_data.iterrows():
                                device_id = row[device_id_col] if device_id_col else "unknown"
                                # (timestamp, device_id, metric_name) 튜플 추가
                                existing_keys.add((idx, device_id, col))
                        else:
                            # 타임스탬프가 열인 경우
                            timestamp_col = next((col for col in existing_data.columns if 'timestamp' in col.lower()), None)
                            if timestamp_col:
                                for idx, row in existing_data.iterrows():
                                    device_id = row[device_id_col] if device_id_col else "unknown"
                                    timestamp = row[timestamp_col]
                                    # (timestamp, device_id, metric_name) 튜플 추가
                                    existing_keys.add((timestamp, device_id, col))
            
            # 중복 제거 (기존 데이터와 겹치는 부분 제외)
            if existing_keys:
                # 새 데이터에서 중복 아닌 행만 필터링
                filtered_rows = []
                
                # device_id 확인
                device_id_col = 'device_id' if 'device_id' in df.columns else None
                device_id_default = self.config.get('data_processing', {}).get('default_values', {}).get('device_id', 'device_001')
                
                # 타임스탬프가 인덱스인 경우
                if isinstance(df.index, pd.DatetimeIndex):
                    for idx, row in df.iterrows():
                        device_id = row[device_id_col] if device_id_col else device_id_default
                        # 이 행의 모든 측정값에 대해 중복 체크
                        duplicate = False
                        
                        for col in df.select_dtypes(include=[np.number]).columns:
                            # 메타데이터 열은 건너뜀
                            if col in ['id', 'device_id', 'companyDomain', 'building'] or 'timestamp' in col.lower():
                                continue
                            
                            # 고유 키 확인
                            if (idx, device_id, col) in existing_keys:
                                duplicate = True
                                break
                        
                        # 중복이 아니면 추가
                        if not duplicate:
                            filtered_rows.append(idx)
                    
                    # 필터링된 행으로 새 데이터프레임 생성
                    if filtered_rows:
                        df = df.loc[filtered_rows]
                    else:
                        logger.info(f"증분 처리: 새로운 데이터가 없습니다. ({location})")
                        return pd.DataFrame()
                else:
                    # 타임스탬프가 열인 경우
                    timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower()), None)
                    if timestamp_col:
                        for idx, row in df.iterrows():
                            device_id = row[device_id_col] if device_id_col else device_id_default
                            timestamp = row[timestamp_col]
                            # 이 행의 모든 측정값에 대해 중복 체크
                            duplicate = False
                            
                            for col in df.select_dtypes(include=[np.number]).columns:
                                # 메타데이터 열은 건너뜀
                                if col in ['id', 'device_id', 'companyDomain', 'building'] or 'timestamp' in col.lower():
                                    continue
                                
                                # 고유 키 확인
                                if (timestamp, device_id, col) in existing_keys:
                                    duplicate = True
                                    break
                            
                            # 중복이 아니면 추가
                            if not duplicate:
                                filtered_rows.append(idx)
                        
                        # 필터링된 행으로 새 데이터프레임 생성
                        if filtered_rows:
                            df = df.loc[filtered_rows]
                        else:
                            logger.info(f"증분 처리: 새로운 데이터가 없습니다. ({location})")
                            return pd.DataFrame()
            
            # 1. 일관된 시간 간격으로 리샘플링
            resample_freq = self.config.get('advanced', {}).get('resampling', {}).get('freq', '5min')
            if isinstance(df.index, pd.DatetimeIndex) and self.config.get('advanced', {}).get('resampling', {}).get('enabled', True):
                df = self.resample_data(df, freq=resample_freq)
            
            # 2. 결측치 처리
            if self.config.get('advanced', {}).get('missing_data', {}).get('handle_missing', True):
                df = self.handle_missing_values(df)
            
            # 3. 이상치 처리
            if self.config.get('advanced', {}).get('outliers', {}).get('handle_outliers', True):
                method = self.config.get('advanced', {}).get('outliers', {}).get('method', 'iqr')
                threshold = self.config.get('advanced', {}).get('outliers', {}).get('threshold', 2.0)
                df = self.handle_outliers(df, method=method, threshold=threshold)
            
            # 4. 자원 지표 계산
            result_df = self.calculate_resource_metrics(df)
            
            # CSV 저장 (필요시)
            if self.config.get('advanced', {}).get('save_csv', False):
                csv_path = f"{location}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                result_df.to_csv(csv_path)
                logger.info(f"자원 지표 CSV 저장: {csv_path}")
            
            logger.info(f"'{location}' 자원 데이터 처리 완료 ({len(result_df)}행)")
            return result_df
            
        except Exception as e:
            logger.error(f"'{location}' 자원 데이터 처리 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def prepare_data_for_prediction(self, df, target_cols=None, scale=True):
        """
        예측을 위한 데이터 준비
        
        Args:
            df (pd.DataFrame): 데이터프레임
            target_cols (list): 타겟 열 목록
            scale (bool): 스케일링 여부
            
        Returns:
            pd.DataFrame: 예측을 위해 준비된 데이터프레임
            object: 스케일러 객체
        """
        if df is None or df.empty:
            logger.warning("준비할 데이터가 비어 있습니다.")
            return pd.DataFrame(), None
        
        try:
            # 타겟 열이 지정되지 않은 경우 모든 숫자형 열 사용
            if target_cols is None:
                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 타겟 열이 데이터프레임에 있는지 확인
            valid_targets = [col for col in target_cols if col in df.columns]
            
            if not valid_targets:
                logger.warning(f"유효한 타겟 열이 없습니다: {target_cols}")
                return pd.DataFrame(), None
            
            # 필요한 열만 선택
            pred_df = df[valid_targets].copy()
            
            # 시간 관련 특성 추가 (인덱스가 DatetimeIndex인 경우)
            if isinstance(df.index, pd.DatetimeIndex):
                # 시간 특성
                pred_df['hour'] = df.index.hour
                pred_df['dayofweek'] = df.index.dayofweek
                pred_df['month'] = df.index.month
                
                # 주기성 표현 (sin/cos 변환)
                pred_df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                pred_df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
                pred_df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                pred_df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
                
                # 주말, 업무 시간 플래그
                pred_df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
                pred_df['is_business_hour'] = ((df.index.hour >= 9) & (df.index.hour < 18) & 
                                              ~df.index.dayofweek.isin([5, 6])).astype(int)
            
            # 스케일링
            if scale:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(pred_df)
                pred_df = pd.DataFrame(scaled_data, index=pred_df.index, columns=pred_df.columns)
                return pred_df, scaler
            
            return pred_df, None
            
        except Exception as e:
            logger.error(f"예측 데이터 준비 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(), None
    
    def create_sequence_dataset(self, df, target_cols, input_window=24, pred_horizon=1):
        """
        시계열 시퀀스 데이터셋 생성 (개선됨)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            target_cols (list): 타겟 열 목록
            input_window (int): 입력 시퀀스 길이
            pred_horizon (int): 예측 지평선
            
        Returns:
            tuple: (X, y, y_timestamps) - 입력 시퀀스, 출력 값, 출력 타임스탬프
        """
        if df is None or df.empty:
            logger.warning("시퀀스 생성을 위한 데이터가 비어 있습니다.")
            return None, None, None
        
        # 데이터가 충분한지 확인 (개선)
        if len(df) < input_window + pred_horizon:
            logger.warning(f"데이터가 충분하지 않습니다. 필요: {input_window + pred_horizon}, 실제: {len(df)}")
            return None, None, None
        
        # 타겟 열이 데이터프레임에 있는지 확인
        valid_targets = [col for col in target_cols if col in df.columns]
        
        if not valid_targets:
            logger.warning(f"유효한 타겟 열이 없습니다: {target_cols}")
            return None, None, None
        
        try:
            # 시퀀스 데이터 생성
            X, y = [], []
            y_timestamps = []
            
            # 각 시점에 대해
            for i in range(len(df) - input_window - pred_horizon + 1):
                # 입력 시퀀스 (모든 특성)
                X.append(df.iloc[i:i + input_window].values)
                
                # 출력 시퀀스 (타겟 열만)
                y_vals = []
                for j in range(pred_horizon):
                    idx = i + input_window + j
                    if idx < len(df):
                        y_row = [df.iloc[idx][col] for col in valid_targets]
                        y_vals.append(y_row)
                
                if len(y_vals) == pred_horizon:
                    y.append(y_vals)
                    
                    # 예측 타임스탬프 저장
                    if isinstance(df.index, pd.DatetimeIndex) and i + input_window < len(df):
                        y_timestamps.append(df.index[i + input_window:i + input_window + pred_horizon])
            
            if not X or not y:
                logger.warning("생성된 시퀀스가 없습니다.")
                return None, None, None
            
            # 데이터 형태 검증 (개선)
            X_array = np.array(X)
            y_array = np.array(y)
            
            logger.info(f"시퀀스 데이터셋 생성 완료: X 형태={X_array.shape}, y 형태={y_array.shape}")
            
            return X_array, y_array, y_timestamps
            
        except Exception as e:
            logger.error(f"시퀀스 데이터셋 생성 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
    
    def create_classification_dataset(self, df, target_col, threshold=None, input_window=24):
        """
        분류를 위한 시퀀스 데이터셋 생성 (개선됨)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            target_col (str): 타겟 열 이름
            threshold (float): 분류 임계값
            input_window (int): 입력 시퀀스 길이
            
        Returns:
            tuple: (X, y, y_timestamps) - 입력 시퀀스, 이진 레이블, 출력 타임스탬프
        """
        if df is None or df.empty:
            logger.warning("분류 데이터셋 생성을 위한 데이터가 비어 있습니다.")
            return None, None, None
        
        # 데이터가 충분한지 확인 (개선)
        if len(df) <= input_window:
            logger.warning(f"데이터가 충분하지 않습니다. 필요: {input_window+1}+, 실제: {len(df)}")
            return None, None, None
        
        if target_col not in df.columns:
            logger.error(f"타겟 열 '{target_col}'이 데이터프레임에 없습니다.")
            return None, None, None
        
        try:
            # 임계값 자동 계산 (전달받은 임계값이 None인 경우)
            if threshold is None:
                # 기본적으로 75 퍼센타일을 임계값으로 사용
                threshold = np.percentile(df[target_col], 75)
                logger.info(f"임계값 자동 설정: {threshold}")
            
            # 분류 데이터셋 생성
            X, y = [], []
            y_timestamps = []
            
            for i in range(len(df) - input_window):
                # 입력 시퀀스
                X.append(df.iloc[i:i + input_window].values)
                
                # 다음 시점의 타겟 값
                next_val = df.iloc[i + input_window][target_col]
                
                # 이진 레이블 (임계값에 따라 분류)
                y.append(1 if next_val > threshold else 0)
                
                # 예측 타임스탬프 저장
                if isinstance(df.index, pd.DatetimeIndex):
                    y_timestamps.append(df.index[i + input_window])
            
            if not X or not y:
                logger.warning("생성된 분류 데이터셋이 없습니다.")
                return None, None, None
            
            # 데이터 배열로 변환 및 차원 검증 (개선)
            X_array = np.array(X)
            y_array = np.array(y)
            
            logger.info(f"분류 데이터셋 생성 완료: X 형태={X_array.shape}, y 형태={y_array.shape}")
            
            # 클래스 불균형 확인
            pos_ratio = np.mean(y)
            logger.info(f"클래스 비율 - 음성(0): {1-pos_ratio:.2f}, 양성(1): {pos_ratio:.2f}")
            
            return X_array, y_array, y_timestamps
            
        except Exception as e:
            logger.error(f"분류 데이터셋 생성 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
    
    def inverse_transform_data(self, df, scaler, columns=None):
        """
        스케일링된 데이터를 원래 스케일로 변환
        
        Args:
            df (pd.DataFrame): 스케일링된 데이터프레임
            scaler (object): 스케일러 객체
            columns (list): 변환할 열 목록
            
        Returns:
            pd.DataFrame: 원래 스케일로 변환된 데이터프레임
        """
        if df is None or df.empty or scaler is None:
            return df
        
        try:
            # 복사본 생성
            result_df = df.copy()
            
            # 변환할 열 결정
            if columns is None:
                columns = df.columns
            
            # 스케일러가 학습된 열만 선택
            valid_columns = [col for col in columns if col in df.columns]
            
            if not valid_columns:
                logger.warning("역변환할 유효한 열이 없습니다.")
                return df
            
            # 역변환 수행
            scaled_data = result_df[valid_columns].values
            original_data = scaler.inverse_transform(scaled_data)
            
            # 결과 적용
            for i, col in enumerate(valid_columns):
                result_df[col] = original_data[:, i]
            
            return result_df
            
        except Exception as e:
            logger.error(f"데이터 역변환 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df
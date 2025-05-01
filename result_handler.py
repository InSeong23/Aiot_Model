#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
결과 처리 모듈 (개선)

전처리된 데이터 저장, 예측 결과를 MySQL에 저장하고 API로 전송하는 기능을 제공합니다.
"""

import logging
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)

class ResultHandler:
    """
    결과 처리 클래스
    
    데이터 저장 및 예측 결과 전송을 담당합니다.
    """
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config (dict): 설정 정보
        """
        self.config = config
        self.mysql_config = config.get("mysql", {})
        self.api_url = config.get("api", {}).get("url", "")
    
    def init_tables(self):
        """
        필요한 테이블 생성
        
        Returns:
            bool: 성공 여부
        """
        if not self.mysql_config or not self.mysql_config.get('host') or not self.mysql_config.get('database'):
            logger.warning("MySQL 설정이 완전하지 않습니다.")
            return False
        
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config.get('host'),
                port=self.mysql_config.get('port'),
                user=self.mysql_config.get('user'),
                password=self.mysql_config.get('password'),
                database=self.mysql_config.get('database')
            )
            
            cursor = conn.cursor()
            
            # 전처리된 데이터 저장 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_resource_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,               -- 측정 시간
                device_id VARCHAR(50),            -- 장치 ID
                companyDomain VARCHAR(100),       -- 회사 도메인
                building VARCHAR(100),            -- 건물
                
                -- CPU 지표
                cpu_usage FLOAT,                  -- CPU 사용률 (100 - usage_idle)
                cpu_user_system_ratio FLOAT,      -- 사용자/시스템 CPU 사용 비율
                
                -- 메모리 지표
                memory_used_percent FLOAT,        -- 메모리 사용률
                
                -- 디스크 지표 
                disk_used_percent FLOAT,          -- 디스크 사용률
                
                -- 디스크 I/O 지표
                disk_read_rate FLOAT,             -- 디스크 읽기 속도 (bytes/sec)
                disk_write_rate FLOAT,            -- 디스크 쓰기 속도 (bytes/sec)
                disk_io_utilization FLOAT,        -- 디스크 I/O 사용률 (%)
                
                -- 네트워크 지표
                net_throughput_sent FLOAT,        -- 네트워크 송신 처리량 (bytes/sec)
                net_throughput_recv FLOAT,        -- 네트워크 수신 처리량 (bytes/sec)
                net_drop_rate_in FLOAT,           -- 수신 패킷 손실률
                net_drop_rate_out FLOAT,          -- 송신 패킷 손실률
                net_error_rate_in FLOAT,          -- 수신 에러율 (errors/sec)
                net_error_rate_out FLOAT,         -- 송신 에러율 (errors/sec)
                net_utilization FLOAT,            -- 네트워크 사용률 (%)
                
                -- 시스템 부하 지표
                system_load1 FLOAT,               -- 1분 평균 시스템 로드
                
                -- 인덱스
                INDEX idx_timestamp (timestamp),
                INDEX idx_device (device_id)
            )
            """)
            
            # 자원 사용량 예측 테이블 (수정)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,               -- 예측 시점
                device_id VARCHAR(50),            -- 장치 ID
                companyDomain VARCHAR(100),       -- 회사 도메인
                building VARCHAR(100),            -- 건물
                
                -- 각 자원별 예측값
                cpu_usage_predicted FLOAT,        -- 예측된 CPU 사용률
                memory_used_percent_predicted FLOAT, -- 예측된 메모리 사용률
                disk_used_percent_predicted FLOAT, -- 예측된 디스크 사용률
                disk_io_utilization_predicted FLOAT, -- 예측된 디스크 I/O 사용률
                net_utilization_predicted FLOAT,  -- 예측된 네트워크 사용률
                system_load1_predicted FLOAT,     -- 예측된 시스템 로드
                
                prediction_time DATETIME,         -- 예측이 수행된 시간
                prediction_horizon VARCHAR(20),   -- 예측 기간 (short_term, mid_term, long_term)
                
                -- 인덱스
                INDEX idx_timestamp (timestamp),
                INDEX idx_device (device_id),
                INDEX idx_prediction (prediction_time, prediction_horizon)
            )
            """)
            
            # 고장 예측 테이블 (수정)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,               -- 예측 시점
                device_id VARCHAR(50),            -- 장치 ID
                companyDomain VARCHAR(100),       -- 회사 도메인
                building VARCHAR(100),            -- 건물
                
                -- 각 자원별 고장 확률
                cpu_failure_probability FLOAT,    -- CPU 고장 확률
                memory_failure_probability FLOAT, -- 메모리 고장 확률
                disk_failure_probability FLOAT,   -- 디스크 고장 확률
                network_failure_probability FLOAT, -- 네트워크 고장 확률
                system_failure_probability FLOAT, -- 시스템 고장 확률
                
                -- 고장 예측 여부 (임계값 초과 시 TRUE)
                is_failure_predicted BOOLEAN,     -- 종합 고장 예측 여부
                failure_resource VARCHAR(50),     -- 고장 예측된 주요 자원
                
                prediction_time DATETIME,         -- 예측이 수행된 시간
                
                -- 인덱스
                INDEX idx_timestamp (timestamp),
                INDEX idx_device (device_id),
                INDEX idx_failure (is_failure_predicted)
            )
            """)
            
            # 용량 계획 테이블 (수정)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS capacity_planning (
                id INT AUTO_INCREMENT PRIMARY KEY,
                resource_type VARCHAR(50),        -- 자원 유형 (cpu, memory, disk, network)
                scenario VARCHAR(50),             -- 시나리오 (기본, 저성장, 고성장 등)
                companyDomain VARCHAR(100),       -- 회사 도메인
                building VARCHAR(100),            -- 건물
                device_id VARCHAR(50),            -- 장치 ID (여러 장치 관리 시)
                expansion_date DATE,              -- 증설 필요 예상 일자
                threshold_value FLOAT,            -- 임계값
                prediction_time DATETIME,         -- 예측이 수행된 시간
                
                -- 인덱스
                INDEX idx_resource (resource_type),
                INDEX idx_expansion (expansion_date)
            )
            """)
            
            # 모델 성능 평가 테이블 (신규)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_type VARCHAR(50),           -- 모델 유형 (resource_prediction, failure_prediction)
                resource_type VARCHAR(50),        -- 자원 유형 (cpu, memory, disk, network, system)
                evaluation_time DATETIME,         -- 평가 시간
                
                -- 성능 지표
                rmse FLOAT,                       -- Root Mean Squared Error
                mae FLOAT,                        -- Mean Absolute Error
                accuracy FLOAT,                   -- 정확도 (분류 모델)
                precision_score FLOAT,            -- 정밀도 (분류 모델)
                recall_score FLOAT,               -- 재현율 (분류 모델)
                f1_score FLOAT,                   -- F1 점수 (분류 모델)
                
                sample_count INT,                 -- 평가에 사용된 샘플 수
                model_version VARCHAR(50),        -- 모델 버전
                
                -- 인덱스
                INDEX idx_evaluation (evaluation_time),
                INDEX idx_model (model_type, resource_type)
            )
            """)
            
            # 예측 실행 정보 테이블 (유지)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                run_start_time DATETIME,          -- 예측 실행 시작 시간
                run_end_time DATETIME,            -- 예측 실행 종료 시간
                prediction_type VARCHAR(50),      -- 예측 유형 (resource, failure)
                status VARCHAR(20),               -- 상태 (success, failed, partial)
                locations_processed VARCHAR(255), -- 처리된 위치 목록 (쉼표로 구분)
                error_message TEXT,               -- 오류 메시지 (있는 경우)
                resource_count INT,               -- 처리된 자원 수
                model_version VARCHAR(50),        -- 사용된 모델 버전
                
                -- 인덱스
                INDEX idx_run_time (run_start_time),
                INDEX idx_status (status)
            )
            """)
            
            conn.commit()
            logger.info("MySQL 테이블 생성 완료")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"MySQL 테이블 생성 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def save_processed_data(self, df):
        """
        전처리된 자원 데이터를 MySQL에 저장
        
        Args:
            df (pd.DataFrame): 전처리된 데이터프레임
            
        Returns:
            bool: 성공 여부
        """
        if df is None or df.empty:
            logger.warning("저장할 전처리 데이터가 비어 있습니다.")
            return False
        
        return self.save_to_mysql(df, "processed_resource_data")
    
    def save_to_mysql(self, df, table_name):
        """
        데이터프레임을 MySQL에 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            table_name (str): 테이블 이름
            
        Returns:
            bool: 성공 여부
        """
        if df is None or df.empty:
            logger.warning("저장할 데이터가 비어 있습니다.")
            return False
        
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config.get('host'),
                port=self.mysql_config.get('port'),
                user=self.mysql_config.get('user'),
                password=self.mysql_config.get('password'),
                database=self.mysql_config.get('database')
            )
            
            cursor = conn.cursor()
            
            # 컬럼 목록 확인
            cursor.execute(f"DESC {table_name}")
            table_columns = [column[0] for column in cursor.fetchall()]
            
            # 데이터프레임에서 테이블 컬럼에 맞는 데이터만 선택
            df_columns = []
            for col in table_columns:
                if col in df.columns:
                    df_columns.append(col)
                elif col == 'id':  # auto_increment 컬럼은 건너뜀
                    continue
                else:
                    logger.debug(f"컬럼 '{col}'이 데이터프레임에 없습니다.")
            
            # 데이터프레임의 열이 테이블에 없는 경우 경고
            for col in df.columns:
                if col not in table_columns and col != 'id':
                    logger.warning(f"데이터프레임 열 '{col}'이 테이블에 없습니다.")
            
            # 필수 컬럼 확인
            # timestamp가 인덱스인 경우 열로 변환
            if isinstance(df.index, pd.DatetimeIndex) and 'timestamp' in table_columns and 'timestamp' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'timestamp'})
            
            # 컬럼 목록이 비어있는 경우 처리
            if not df_columns:
                logger.error(f"테이블 '{table_name}'에 맞는 열이 데이터프레임에 없습니다.")
                return False
            
            # 선택된 컬럼만 사용
            subset_df = df[df_columns]
            
            # 컬럼 목록 생성
            columns_str = ', '.join([f"`{col}`" for col in df_columns])
            placeholders = ', '.join(['%s'] * len(df_columns))
            
            # 삽입 쿼리
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # 데이터 타입 변환 (numpy -> python 네이티브 타입)
            values = []
            for _, row in subset_df.iterrows():
                row_data = []
                for col in df_columns:
                    val = row[col]
                    if isinstance(val, (np.int64, np.int32)):
                        val = int(val)
                    elif isinstance(val, (np.float64, np.float32)):
                        val = float(val)
                    elif isinstance(val, pd.Timestamp):
                        val = val.to_pydatetime()
                    elif isinstance(val, np.bool_):
                        val = bool(val)
                    elif col == 'expansion_date' and isinstance(val, str):
                        # 날짜 문자열을 datetime으로 변환
                        try:
                            val = datetime.strptime(val, '%Y-%m-%d')
                        except:
                            pass
                    row_data.append(val)
                values.append(tuple(row_data))
            
            # 데이터 삽입
            cursor.executemany(query, values)
            conn.commit()
            
            logger.info(f"{len(subset_df)}개 행이 '{table_name}' 테이블에 저장되었습니다.")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"MySQL 저장 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_from_mysql(self, table_name, start_time=None, end_time=None, filter_cond=None, limit=None):
        """
        MySQL에서 데이터 로드
        
        Args:
            table_name (str): 테이블 이름
            start_time (datetime): 시작 시간
            end_time (datetime): 종료 시간
            filter_cond (str): 추가 필터 조건
            limit (int): 최대 행 수
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config.get('host'),
                port=self.mysql_config.get('port'),
                user=self.mysql_config.get('user'),
                password=self.mysql_config.get('password'),
                database=self.mysql_config.get('database')
            )
            
            # 쿼리 구성
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            # 시간 필터
            if start_time:
                conditions.append(f"timestamp >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'")
            
            # 추가 필터 조건
            if filter_cond:
                conditions.append(filter_cond)
            
            # 조건 적용
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # 시간순 정렬
            query += " ORDER BY timestamp"
            
            # 제한 적용
            if limit:
                query += f" LIMIT {limit}"
            
            # 쿼리 실행
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            
            conn.close()
            
            # 'id' 컬럼 제거
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # timestamp를 인덱스로 설정
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            logger.info(f"{table_name}에서 {len(df)}개 행을 로드했습니다.")
            return df
            
        except Exception as e:
            logger.error(f"MySQL 데이터 로드 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def send_to_api(self, data, endpoint_suffix):
        """
        데이터를 API로 전송
        
        Args:
            data (pd.DataFrame or dict): 전송할 데이터
            endpoint_suffix (str): API 엔드포인트 접미사
            
        Returns:
            bool: 성공 여부
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            logger.warning("전송할 데이터가 없습니다.")
            return False
        
        # API URL이 설정되어 있지 않으면 건너뜀
        if not self.api_url:
            logger.warning("API URL이 설정되지 않았습니다.")
            return False
        
        try:
            # 데이터프레임을 레코드 리스트로 변환
            if isinstance(data, pd.DataFrame):
                # datetime 객체를 문자열로 변환
                df_copy = data.copy()
                
                # 인덱스가 DatetimeIndex인 경우 열로 변환
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy = df_copy.reset_index().rename(columns={'index': 'timestamp'})
                
                # 날짜/시간 형식 데이터 변환
                for col in df_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    elif isinstance(df_copy[col].iloc[0], datetime) if len(df_copy) > 0 else False:
                        df_copy[col] = df_copy[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if x else None)
                    elif isinstance(df_copy[col].iloc[0], bool) if len(df_copy) > 0 else False:
                        df_copy[col] = df_copy[col].astype(int)
                
                # 데이터프레임을 레코드 리스트로 변환
                records = df_copy.to_dict(orient='records')
            else:
                # 이미 dictionary 형태인 경우
                records = data
            
            # API URL 설정
            endpoint = f"{self.api_url}/{endpoint_suffix}"
            
            # 메타데이터 추가
            payload = {
                "data": records,
                "meta": {
                    "source": "iot_prediction_system",
                    "version": "2.0",
                    "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                }
            }
            
            # API 호출
            logger.info(f"API 결과 전송 중: {endpoint}")
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10  # 타임아웃 10초
            )
            
            # 응답 확인
            if response.status_code == 200:
                logger.info(f"API 결과 전송 성공: {response.status_code}")
                return True
            else:
                logger.error(f"API 결과 전송 실패: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"API 결과 전송 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_resource_predictions(self, predictions_df, horizon='short_term'):
        """
        자원 사용량 예측 결과 처리
        
        Args:
            predictions_df (pd.DataFrame): 예측 결과 데이터프레임
            horizon (str): 예측 지평선 유형 (short_term, mid_term, long_term)
            
        Returns:
            bool: 성공 여부
        """
        if predictions_df is None or predictions_df.empty:
            logger.warning("처리할 자원 예측 결과가 없습니다.")
            return False
        
        # 예측 시간 추가
        if 'prediction_time' not in predictions_df.columns:
            predictions_df['prediction_time'] = datetime.now()
        
        # 예측 지평선 추가
        predictions_df['prediction_horizon'] = horizon
        
        # 1. MySQL에 저장
        mysql_success = self.save_to_mysql(predictions_df, "resource_predictions")
        
        # 2. API로 전송
        api_success = self.send_to_api(predictions_df, "resource_predictions")
        
        return mysql_success and api_success
    
    def process_failure_predictions(self, predictions_df):
        """
        고장 예측 결과 처리
        
        Args:
            predictions_df (pd.DataFrame): 예측 결과 데이터프레임
            
        Returns:
            bool: 성공 여부
        """
        if predictions_df is None or predictions_df.empty:
            logger.warning("처리할 고장 예측 결과가 없습니다.")
            return False
        
        # 예측 시간 추가
        if 'prediction_time' not in predictions_df.columns:
            predictions_df['prediction_time'] = datetime.now()
        
        # 종합 고장 예측 여부 계산 (어느 하나라도 고장 예측이 있으면 True)
        failure_cols = [col for col in predictions_df.columns if 'failure_probability' in col]
        if failure_cols:
            # 고장 임계값 설정
            failure_threshold = self.config.get("prediction", {}).get("failure", {}).get("threshold", 0.5)
            
            # 가장 높은 고장 확률을 갖는 자원 찾기
            max_prob_resource = None
            max_prob = 0
            
            for col in failure_cols:
                resource_type = col.replace('_failure_probability', '')
                mask = predictions_df[col] > failure_threshold
                
                if mask.any():
                    # 자원별 고장 건수 확인
                    failure_count = mask.sum()
                    logger.info(f"{resource_type} 고장 예측: {failure_count}건")
                    
                    # 최대 확률 자원 업데이트
                    curr_max = predictions_df[col].max()
                    if curr_max > max_prob:
                        max_prob = curr_max
                        max_prob_resource = resource_type
            
            # 종합 고장 예측 및 주요 고장 자원 설정
            predictions_df['is_failure_predicted'] = (predictions_df[failure_cols] > failure_threshold).any(axis=1)
            predictions_df['failure_resource'] = max_prob_resource if max_prob_resource else 'none'
        
        # 1. MySQL에 저장
        mysql_success = self.save_to_mysql(predictions_df, "failure_predictions")
        
        # 2. API로 전송
        api_success = self.send_to_api(predictions_df, "failure_predictions")
        
        return mysql_success and api_success
    
    def process_capacity_planning(self, capacity_df):
        """
        용량 계획 결과 처리
        
        Args:
            capacity_df (pd.DataFrame): 용량 계획 데이터프레임
            
        Returns:
            bool: 성공 여부
        """
        if capacity_df is None or capacity_df.empty:
            logger.warning("처리할 용량 계획 결과가 없습니다.")
            return False
        
        # 1. MySQL에 저장
        mysql_success = self.save_to_mysql(capacity_df, "capacity_planning")
        
        # 2. API로 전송
        api_success = self.send_to_api(capacity_df, "capacity_planning")
        
        return mysql_success and api_success
    
    def record_prediction_run(self, prediction_type, start_time, locations_processed, status="success", error_message=None, resource_count=0):
        """
        예측 실행 정보를 MySQL에 기록
        
        Args:
            prediction_type (str): 예측 유형 (resource 또는 failure)
            start_time (datetime): 실행 시작 시간
            locations_processed (list): 처리된 location 목록
            status (str): 실행 상태 (success, failed, partial)
            error_message (str): 오류 메시지 (있는 경우)
            resource_count (int): 처리된 자원 수
            
        Returns:
            bool: 기록 성공 여부
        """
        if not self.mysql_config or not self.mysql_config.get('host') or not self.mysql_config.get('database'):
            logger.warning("MySQL 설정이 완전하지 않아 예측 실행 기록을 건너뜁니다.")
            return False
        
        try:
            # 예측 실행 정보 데이터프레임 생성
            run_df = pd.DataFrame({
                'run_start_time': [start_time],
                'run_end_time': [datetime.now()],
                'prediction_type': [prediction_type],
                'status': [status],
                'locations_processed': [','.join(locations_processed) if isinstance(locations_processed, list) else str(locations_processed)],
                'error_message': [error_message if error_message else None],
                'resource_count': [resource_count],
                'model_version': ['lstm_v2.0']  # 모델 버전
            })
            
            # MySQL에 저장
            success = self.save_to_mysql(run_df, "prediction_runs")
            
            if success:
                logger.info(f"예측 실행 정보가 기록되었습니다. 유형: {prediction_type}, 상태: {status}")
            else:
                logger.warning("예측 실행 정보 기록 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"예측 실행 정보 기록 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def record_model_performance(self, model_type, resource_type, performance_metrics):
        """
        모델 성능 평가 정보를 MySQL에 기록
        
        Args:
            model_type (str): 모델 유형 (resource_prediction 또는 failure_prediction)
            resource_type (str): 자원 유형 (cpu, memory, disk, network, system)
            performance_metrics (dict): 성능 지표
            
        Returns:
            bool: 기록 성공 여부
        """
        if not self.mysql_config or not self.mysql_config.get('host') or not self.mysql_config.get('database'):
            logger.warning("MySQL 설정이 완전하지 않아 모델 성능 기록을 건너뜁니다.")
            return False
        
        try:
            # 성능 지표 데이터프레임 생성
            perf_df = pd.DataFrame({
                'model_type': [model_type],
                'resource_type': [resource_type],
                'evaluation_time': [datetime.now()],
                'rmse': [performance_metrics.get('rmse', None)],
                'mae': [performance_metrics.get('mae', None)],
                'accuracy': [performance_metrics.get('accuracy', None)],
                'precision_score': [performance_metrics.get('precision', None)],
                'recall_score': [performance_metrics.get('recall', None)],
                'f1_score': [performance_metrics.get('f1', None)],
                'sample_count': [performance_metrics.get('sample_count', 0)],
                'model_version': [performance_metrics.get('model_version', 'lstm_v2.0')]
            })
            
            # MySQL에 저장
            success = self.save_to_mysql(perf_df, "model_performance")
            
            if success:
                logger.info(f"모델 성능 평가 정보가 기록되었습니다. 유형: {model_type}, 자원: {resource_type}")
            else:
                logger.warning("모델 성능 평가 정보 기록 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"모델 성능 평가 정보 기록 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_connections(self):
        """
        연결 테스트
        
        Returns:
            tuple: (mysql_success, api_success)
        """
        # MySQL 연결 테스트
        mysql_success = False
        if self.mysql_config and self.mysql_config.get('host') and self.mysql_config.get('database'):
            try:
                conn = mysql.connector.connect(
                    host=self.mysql_config.get('host'),
                    port=self.mysql_config.get('port'),
                    user=self.mysql_config.get('user'),
                    password=self.mysql_config.get('password'),
                    database=self.mysql_config.get('database')
                )
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchall()
                cursor.close()
                conn.close()
                mysql_success = True
                logger.info("MySQL 연결 테스트 성공")
            except Exception as e:
                logger.error(f"MySQL 연결 테스트 실패: {e}")
        
        # API 연결 테스트
        api_success = False
        if self.api_url:
            try:
                # 간단한 헬스체크 요청
                response = requests.get(
                    f"{self.api_url}/health" if '/health' in self.api_url else self.api_url,
                    timeout=5
                )
                api_success = response.status_code < 400  # 4xx, 5xx 이외의 상태 코드는 성공으로 간주
                if api_success:
                    logger.info(f"API 연결 테스트 성공: {response.status_code}")
                else:
                    logger.warning(f"API 연결 테스트 실패: {response.status_code}")
            except Exception as e:
                logger.error(f"API 연결 테스트 실패: {e}")
        
        return mysql_success, api_success
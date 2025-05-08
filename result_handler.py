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
        필요한 테이블 생성 (모든 테이블 포함)
        
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
            
            # 1. company_metadata 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_metadata (
                id INT AUTO_INCREMENT PRIMARY KEY,
                companyDomain VARCHAR(100) NOT NULL UNIQUE,
                company_name VARCHAR(100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 2. building_metadata 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS building_metadata (
                id INT AUTO_INCREMENT PRIMARY KEY,
                company_id INT NOT NULL,
                building VARCHAR(100) NOT NULL,
                location VARCHAR(255),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY building_unique (company_id, building),
                FOREIGN KEY (company_id) REFERENCES company_metadata(id)
            )
            """)
            
            # 3. device_metadata 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS device_metadata (
                id INT AUTO_INCREMENT PRIMARY KEY,
                building_id INT NOT NULL,
                device_id VARCHAR(50) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY device_unique (building_id, device_id),
                FOREIGN KEY (building_id) REFERENCES building_metadata(id)
            )
            """)
            
            # 4. resource_metrics 테이블 생성 (타임스탬프에 인덱스 추가, 고유 키 제약 추가)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                device_id VARCHAR(50) NOT NULL,
                resource_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(50) NOT NULL,
                metric_value FLOAT,
                unit VARCHAR(10),
                companyDomain VARCHAR(100),
                building VARCHAR(100),
                
                INDEX idx_device_time (device_id, timestamp),
                INDEX idx_metric (resource_type, metric_name),
                INDEX idx_timestamp (timestamp),
                INDEX idx_company_building (companyDomain, building),
                UNIQUE KEY unique_metric (timestamp, device_id, resource_type, metric_name)
            )
            """)
            
            # 5. prediction_results 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                device_id VARCHAR(50) NOT NULL, 
                resource_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(50) NOT NULL,
                predicted_value FLOAT,
                prediction_type VARCHAR(20),
                prediction_horizon VARCHAR(20),
                prediction_time DATETIME,
                companyDomain VARCHAR(100),
                building VARCHAR(100),
                
                INDEX idx_device_time (device_id, timestamp),
                INDEX idx_metric (resource_type, metric_name),
                INDEX idx_prediction (prediction_time, prediction_type)
            )
            """)
            
            # 6. model_performance 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_type VARCHAR(50) NOT NULL,
                resource_type VARCHAR(50) NOT NULL,
                evaluation_time DATETIME NOT NULL,
                rmse FLOAT,
                mae FLOAT,
                accuracy FLOAT,
                precision_score FLOAT,
                recall_score FLOAT,
                f1_score FLOAT,
                sample_count INT,
                model_version VARCHAR(20),
                
                INDEX idx_model (model_type, resource_type),
                INDEX idx_time (evaluation_time)
            )
            """)
            
            # 7. collection_metadata 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_metadata (
                id INT AUTO_INCREMENT PRIMARY KEY,
                collection_type VARCHAR(50) NOT NULL,
                device_id VARCHAR(50) NOT NULL,
                companyDomain VARCHAR(100) NOT NULL,
                building VARCHAR(100) NOT NULL,
                last_collection_time DATETIME NOT NULL,
                
                INDEX idx_device (device_id, companyDomain, building),
                INDEX idx_type (collection_type)
            )
            """)
            
            # 8. prediction_runs 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                run_start_time DATETIME NOT NULL,
                run_end_time DATETIME NOT NULL,
                prediction_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL,
                locations_processed TEXT,
                error_message TEXT,
                resource_count INT,
                model_version VARCHAR(20),
                
                INDEX idx_type (prediction_type),
                INDEX idx_time (run_start_time)
            )
            """)
            
            # 기본 회사 정보 추가 (없는 경우)
            cursor.execute("""
            INSERT IGNORE INTO company_metadata (companyDomain, company_name) 
            VALUES (%s, %s)
            """, ('javame', 'JavaMe Corp'))
            
            # 기본 건물 정보 추가
            cursor.execute("""
            SELECT id FROM company_metadata WHERE companyDomain = 'javame'
            """)
            company_id = cursor.fetchone()[0]
            
            cursor.execute("""
            INSERT IGNORE INTO building_metadata (company_id, building, location) 
            VALUES (%s, %s, %s)
            """, (company_id, 'gyeongnam_campus', 'Gyeongnam, Korea'))
            
            # 기본 디바이스 정보 추가
            cursor.execute("""
            SELECT id FROM building_metadata WHERE company_id = %s AND building = %s
            """, (company_id, 'gyeongnam_campus'))
            building_id = cursor.fetchone()[0]
            
            cursor.execute("""
            INSERT IGNORE INTO device_metadata (building_id, device_id) 
            VALUES (%s, %s)
            """, (building_id, '192.168.71.74'))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("MySQL 테이블 생성 완료")
            return True
            
        except Exception as e:
            logger.error(f"MySQL 테이블 생성 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    def _init_partitions(self):
        """
        초기 파티션 생성 (수정됨 - 고정 값 사용)
        """
        try:
            conn = self.mysql_connect()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # 먼저 resource_metrics 테이블이 존재하는지 확인
            cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = 'resource_metrics'
            """, (self.mysql_config.get('database'),))
            
            if cursor.fetchone()[0] == 0:
                logger.warning("resource_metrics 테이블이 존재하지 않습니다. 파티션 초기화를 건너뜁니다.")
                cursor.close()
                conn.close()
                return
            
            # 파티션 지원이 활성화되어 있는지 확인
            partition_strategy = self.config.get("database", {}).get("partition_strategy", {})
            if not partition_strategy.get("enabled", False):
                cursor.close()
                conn.close()
                return
            
            # 현재 및 다음 달 파티션 생성
            for i in range(3):  # 현재 월 + 2개월
                next_month = datetime.now() + timedelta(days=30 * i)
                next_month_str = next_month.strftime('%Y-%m-01')
                
                # 다음 달 1일의 TO_DAYS 값 계산
                cursor.execute(f"SELECT TO_DAYS('{next_month_str}')")
                to_days_value = cursor.fetchone()[0]
                
                # 다음 달에 대한 파티션 이름
                partition_name = f"p_{next_month.strftime('%Y_%m')}"
                
                try:
                    # 파티션 이미 존재하는지 확인
                    cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.PARTITIONS
                    WHERE TABLE_SCHEMA = %s
                    AND TABLE_NAME = 'resource_metrics'
                    AND PARTITION_NAME = %s
                    """, (self.mysql_config.get('database'), partition_name))
                    
                    if cursor.fetchone()[0] == 0:
                        # 파티션 추가 (고정 값 사용)
                        cursor.execute(f"""
                        ALTER TABLE resource_metrics ADD PARTITION (
                            PARTITION {partition_name} VALUES LESS THAN ({to_days_value})
                        )
                        """)
                        logger.info(f"파티션 추가됨: {partition_name}")
                except Exception as e:
                    logger.warning(f"파티션 추가 실패 ({partition_name}): {e}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"파티션 초기화 중 오류 발생: {e}")

    def manage_partitions(self):
        """
        테이블 파티션 관리 (월별 자동 파티셔닝)
        """
        if not self.config.get("database", {}).get("partition_strategy", {}).get("enabled", False):
            return
        
        try:
            conn = self.mysql_connect()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # 먼저 resource_metrics 테이블이 존재하는지 확인
            cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = 'resource_metrics'
            """, (self.mysql_config.get('database'),))
            
            if cursor.fetchone()[0] == 0:
                logger.warning("resource_metrics 테이블이 존재하지 않습니다. 파티션 관리를 건너뜁니다.")
                cursor.close()
                conn.close()
                return
            
            # 파티션 테이블로 변경
            try:
                cursor.execute("""
                ALTER TABLE resource_metrics PARTITION BY RANGE (TO_DAYS(timestamp)) (
                    PARTITION p_default VALUES LESS THAN MAXVALUE
                )
                """)
                logger.info("resource_metrics 테이블에 파티션 설정 추가됨")
            except Exception as e:
                # 이미 파티션이 설정되어 있으면 무시
                if "already defined with different partition" not in str(e):
                    logger.warning(f"파티션 설정 변경 실패: {e}")
            
            # 현재 날짜 기준 미래 3개월에 대한 파티션 생성
            for i in range(4):  # 현재 월 + 3개월
                next_month = datetime.now() + timedelta(days=30 * i)
                next_month_str = next_month.strftime('%Y-%m-01')
                
                # 다음 달 1일의 TO_DAYS 값 계산
                cursor.execute(f"SELECT TO_DAYS('{next_month_str}')")
                to_days_value = cursor.fetchone()[0]
                
                # 다음 달에 대한 파티션 이름
                partition_name = f"p_{next_month.strftime('%Y_%m')}"
                
                # 파티션 존재 여부 확인
                cursor.execute("""
                SELECT COUNT(*) FROM information_schema.PARTITIONS
                WHERE TABLE_SCHEMA = %s
                AND TABLE_NAME = 'resource_metrics'
                AND PARTITION_NAME = %s
                """, (self.mysql_config.get('database'), partition_name))
                
                if cursor.fetchone()[0] == 0:
                    # 파티션 추가 (고정된 TO_DAYS 값 사용)
                    try:
                        cursor.execute(f"""
                        ALTER TABLE resource_metrics ADD PARTITION (
                            PARTITION {partition_name} VALUES LESS THAN ({to_days_value})
                        )
                        """)
                        logger.info(f"파티션 추가됨: {partition_name}")
                        pass
                    except Exception as e:
                        logger.warning(f"파티션 관리 중 오류 발생 (무시됨): {e}")
            
            cursor.close()
            conn.close()
        
        except Exception as e:
            logger.error(f"파티션 관리 중 오류 발생: {e}")

    def save_resource_metrics_batch(self, df_list, resource_types):
        """
        여러 자원 데이터프레임을 배치로 저장 (성능 최적화)
        
        Args:
            df_list (list): 데이터프레임 목록
            resource_types (list): 자원 유형 목록
                
        Returns:
            bool: 성공 여부
        """
        if not df_list or len(df_list) != len(resource_types):
            logger.warning("저장할 데이터 목록이 유효하지 않습니다.")
            return False
        
        try:
            # MySQL 연결 생성
            conn = self.mysql_connect()
            if not conn:
                return False
            
            # 자동 커밋 비활성화
            conn.autocommit = False
            cursor = conn.cursor()
            
            total_records = 0
            
            try:
                # 트랜잭션 시작
                conn.start_transaction()
                
                # 각 데이터프레임 처리
                for idx, df in enumerate(df_list):
                    resource_type = resource_types[idx]
                    
                    if df is None or df.empty:
                        continue
                    
                    # 배치 크기 설정
                    batch_size = self.config.get("database", {}).get("batch_size", 1000)
                    records = []
                    
                    # 기본 메타데이터 추출
                    device_id = df['device_id'].iloc[0] if 'device_id' in df.columns else self.config.get('data_processing', {}).get('default_values', {}).get('device_id', 'device_001')
                    company_domain = df['companyDomain'].iloc[0] if 'companyDomain' in df.columns else self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
                    building = df['building'].iloc[0] if 'building' in df.columns else self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
                    
                    # 인덱스가 DatetimeIndex인 경우 열로 변환
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                        # 첫 번째 열(원래 인덱스)이 무슨 이름이든 timestamp로 변경
                        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                    
                    # 디버깅 로그
                    logger.info(f"처리 중인 데이터: {resource_type}, 컬럼: {df.columns.tolist()}")
                    logger.info(f"데이터 샘플:\n{df.head()}")
                    
                    # _value 열 직접 처리
                    if '_value' in df.columns:
                        logger.info(f"{resource_type}의 _value 열 직접 처리 중...")
                        logger.info(f"_value 열 샘플: {df['_value'].head()}")
                        logger.info(f"_value 열 타입: {df['_value'].dtype}")
                        logger.info(f"_value 열 통계: 평균={df['_value'].mean()}, 최소={df['_value'].min()}, 최대={df['_value'].max()}")
                        
                        for idx, row in df.iterrows():
                            # 타임스탬프 처리
                            if '_time' in df.columns:
                                # _time 열이 있으면 그대로 사용
                                ts = row['_time']
                                if isinstance(ts, pd.Timestamp):
                                    time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    time_str = str(ts)
                            elif 'timestamp' in df.columns:
                                # timestamp 열 사용
                                ts = row['timestamp']
                                if isinstance(ts, pd.Timestamp):
                                    time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    time_str = str(ts)
                            else:
                                # 모든 방법이 실패하면 기본값 사용
                                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            # _value 열 값 처리 - 핵심 수정 부분
                            try:
                                # 명시적으로 파이썬 기본 타입으로 변환
                                raw_value = row['_value']
                                
                                # 디버깅 출력 추가
                                logger.info(f"원본 _value: {raw_value}, 타입: {type(raw_value)}")
                                
                                # None, NaN 체크
                                if pd.isna(raw_value) or raw_value is None:
                                    logger.warning(f"null 값 감지: {raw_value}")
                                    continue
                                
                                # 확실한 float 변환
                                value = float(raw_value)
                                logger.info(f"변환된 값: {value}, 타입: {type(value)}")
                                
                                # 메트릭 이름 설정
                                if resource_type == 'cpu':
                                    metric_name = 'cpu_usage'
                                elif resource_type == 'mem':
                                    metric_name = 'memory_used_percent'
                                elif resource_type == 'disk':
                                    metric_name = 'disk_used_percent'
                                else:
                                    metric_name = f"{resource_type}_value"
                                
                                # 단위 설정
                                unit = '%' if resource_type in ['cpu', 'mem', 'disk'] or 'percent' in metric_name else ''
                                
                                # 레코드 추가 (값을 명시적으로 float로 변환)
                                records.append((
                                    time_str,
                                    device_id,
                                    resource_type,
                                    metric_name,
                                    value,  # 변환된 Python 기본 float 사용
                                    unit,
                                    company_domain,
                                    building
                                ))
                            except Exception as e:
                                logger.warning(f"값 변환 실패: {e}")
                    
                    else:
                        # _value 열이 없는 경우 자원별 특정 열 사용
                        logger.warning(f"{resource_type}에 _value 열이 없습니다. 자원별 처리 열 사용...")
                        
                        # 자원별 특정 열 매핑
                        resource_columns = {
                            'cpu': 'cpu_usage',
                            'disk': 'disk_used_percent',
                            'diskio': 'disk_io_utilization',
                            'mem': 'memory_used_percent',
                            'net': 'net_utilization',
                            'system': 'system_load1'
                        }
                        
                        # 자원에 맞는 열 선택
                        target_column = resource_columns.get(resource_type)
                        
                        if target_column and target_column in df.columns:
                            logger.info(f"{resource_type} 자원의 {target_column} 열 사용")
                            logger.info(f"{target_column} 열 통계: 평균={df[target_column].mean()}, 최소={df[target_column].min()}, 최대={df[target_column].max()}")
                            
                            for idx, row in df.iterrows():
                                # 타임스탬프 처리 (생략)...
                                if 'timestamp' in df.columns:
                                    ts = row['timestamp']
                                    if isinstance(ts, pd.Timestamp):
                                        time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                                    else:
                                        time_str = str(ts)
                                else:
                                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                
                                # 값 처리
                                try:
                                    raw_value = row[target_column]
                                    # 디버깅
                                    logger.info(f"사용할 값: {raw_value}, 타입: {type(raw_value)}")
                                    
                                    # NaN 체크
                                    if pd.isna(raw_value) or raw_value is None:
                                        continue
                                    
                                    # float 변환
                                    value = float(raw_value)
                                    
                                    # 레코드 추가
                                    records.append((
                                        time_str,
                                        device_id,
                                        resource_type,
                                        target_column,
                                        value,
                                        '%' if 'percent' in target_column or resource_type in ['cpu', 'mem', 'disk'] else '',
                                        company_domain,
                                        building
                                    ))
                                except Exception as e:
                                    logger.error(f"값 변환 실패: {e}")
                        else:
                            # 대체 처리 - 모든 숫자형 열 처리
                            logger.warning(f"{resource_type}에 대한 특정 열을 찾을 수 없음. 모든 숫자형 열 사용")
                            # 각 지표별로 EAV 형식으로 변환 처리 (생략)...
                    
                    # 레코드 삽입
                    if records:
                        logger.info(f"{resource_type}에 대해 {len(records)}개 레코드 삽입 중...")
                        
                        sql = """
                        INSERT INTO resource_metrics 
                        (timestamp, device_id, resource_type, metric_name, metric_value, unit, companyDomain, building)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        metric_value = VALUES(metric_value),
                        unit = VALUES(unit)
                        """
                        
                        # 각 레코드를 개별적으로 삽입하여 중복 오류 처리
                        success_count = 0
                        duplicate_count = 0
                        
                        for record in records:
                            try:
                                # 디버깅 로그 추가
                                logger.debug(f"SQL 실행 직전 metric_value: {record[4]}, 타입: {type(record[4])}")
                                cursor.execute(sql, record)
                                success_count += 1
                            except mysql.connector.errors.IntegrityError as e:
                                if "Duplicate entry" in str(e):
                                    duplicate_count += 1
                                    # 로깅 제한
                                    if duplicate_count <= 5:
                                        logger.warning(f"중복 데이터 건너뜀: {record[0]}, {record[3]}")
                                else:
                                    logger.error(f"삽입 오류: {e}")
                            except Exception as e:
                                logger.error(f"SQL 실행 오류: {e}, 레코드: {record}")
                        
                        total_records += success_count
                        logger.info(f"{resource_type}: {success_count}개 성공, {duplicate_count}개 중복")
                    
                    # 마지막 수집 시간 업데이트
                    self.update_last_collection_time(resource_type, device_id, company_domain, building)
                
                # 명시적 커밋
                conn.commit()
                logger.info(f"총 {total_records}개의 자원 메트릭이 배치로 저장되었습니다.")
                return True
                    
            except Exception as e:
                # 오류 시 롤백
                conn.rollback()
                logger.error(f"배치 저장 중 오류 발생: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
            finally:
                cursor.close()
                conn.close()
                    
        except Exception as e:
            logger.error(f"자원 메트릭 배치 저장 실패: {e}")
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
        
        # EAV 모델 사용 여부 확인
        use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
        
        if use_eav_model:
            # 자원 유형 확인
            resource_type = df['location'].iloc[0] if 'location' in df.columns else None
            if not resource_type:
                logger.warning("자원 유형을 확인할 수 없습니다.")
                return False
            
            # EAV 모델로 변환하여 저장
            return self.save_resource_metrics(df, resource_type)
        else:
            # 기존 방식으로 저장
            return self.save_to_mysql(df, "processed_resource_data")
    
    def save_resource_metrics(self, df, resource_type):
        """
        자원 메트릭을 MySQL에 저장 (개선 버전)
        """
        if df is None or df.empty:
            logger.warning("저장할 자원 메트릭 데이터가 비어 있습니다.")
            return False
        
        try:
            # MySQL 연결
            conn = self.mysql_connect()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 기본 메타데이터
            device_id = df['device_id'].iloc[0] if 'device_id' in df.columns else self.config.get('data_processing', {}).get('default_values', {}).get('device_id', 'device_001')
            company_domain = df['companyDomain'].iloc[0] if 'companyDomain' in df.columns else self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
            building = df['building'].iloc[0] if 'building' in df.columns else self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
            
            # 데이터 변환 및 삽입
            records = []
            
            # 인덱스가 DatetimeIndex면 timestamp 열로 변환
            if isinstance(df.index, pd.DatetimeIndex):
                df_with_ts = df.reset_index()
                df_with_ts.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                df_with_ts = df.copy()
            
            # 데이터 로깅으로 디버깅
            logger.info(f"저장할 데이터 컬럼: {df_with_ts.columns.tolist()}")
            logger.info(f"데이터 샘플: \n{df_with_ts.head()}")
            
            # InfluxDB _value 열 직접 처리
            if '_value' in df_with_ts.columns:
                logger.info("InfluxDB _value 열 처리 중...")
                
                # 디버깅 정보 출력
                logger.info(f"_value 열 샘플: {df_with_ts['_value'].head()}")
                logger.info(f"_value 열 타입: {df_with_ts['_value'].dtype}")
                
                for idx, row in df_with_ts.iterrows():
                    # 타임스탬프 처리
                    if '_time' in df_with_ts.columns:
                        # _time 열이 있으면 그대로 사용
                        ts = row['_time']
                        if isinstance(ts, pd.Timestamp):
                            time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            time_str = str(ts)
                    elif 'timestamp' in df_with_ts.columns:
                        # timestamp 열 사용
                        ts = row['timestamp']
                        if isinstance(ts, pd.Timestamp):
                            time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            time_str = str(ts)
                    else:
                        # 인덱스가 타임스탬프인 경우
                        if isinstance(idx, pd.Timestamp):
                            time_str = idx.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            # 모든 방법이 실패하면 기본값 사용
                            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # _value 열 값 처리 - 핵심 수정 부분
                    try:
                        # 명시적으로 파이썬 기본 타입으로 변환
                        raw_value = row['_value']
                        # 디버깅 출력 추가
                        logger.info(f"원본 _value: {raw_value}, 타입: {type(raw_value)}")
                        
                        # None, NaN 체크
                        if pd.isna(raw_value) or raw_value is None:
                            logger.warning(f"null 값 감지: {raw_value}")
                            continue
                        
                        # 확실한 float 변환
                        value = float(raw_value)
                        logger.info(f"변환된 값: {value}, 타입: {type(value)}")
                        
                        # 메트릭 이름 설정
                        if resource_type == 'cpu':
                            metric_name = 'cpu_usage'
                        elif resource_type == 'mem':
                            metric_name = 'memory_used_percent'
                        elif resource_type == 'disk':
                            metric_name = 'disk_used_percent'
                        else:
                            metric_name = f"{resource_type}_value"
                        
                        # 단위 설정
                        unit = '%' if resource_type in ['cpu', 'mem', 'disk'] or 'percent' in metric_name else ''
                        
                        # 레코드 추가
                        records.append((
                            time_str,
                            device_id,
                            resource_type,
                            metric_name,
                            value,  # 명시적으로 Python 기본 float 사용
                            unit,
                            company_domain,
                            building
                        ))
                        
                    except (ValueError, TypeError) as e:
                        logger.error(f"값 변환 실패: {e}")
            else:
                logger.warning("_value 열이 데이터에 없습니다. 자원별 처리 열 사용...")
                
                # 자원별 특정 열 매핑
                resource_columns = {
                    'cpu': 'cpu_usage',
                    'disk': 'disk_used_percent',
                    'diskio': 'disk_io_utilization',
                    'mem': 'memory_used_percent',
                    'net': 'net_utilization',
                    'system': 'system_load1'
                }
                
                # 자원에 맞는 열 선택
                target_column = resource_columns.get(resource_type)
                
                if target_column and target_column in df_with_ts.columns:
                    logger.info(f"{resource_type} 자원의 {target_column} 열 사용")
                    
                    for idx, row in df_with_ts.iterrows():
                        # 타임스탬프 처리 (기존 코드 유지)
                        if '_time' in df_with_ts.columns:
                            ts = row['_time']
                            if isinstance(ts, pd.Timestamp):
                                time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                time_str = str(ts)
                        elif 'timestamp' in df_with_ts.columns:
                            ts = row['timestamp']
                            if isinstance(ts, pd.Timestamp):
                                time_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                time_str = str(ts)
                        else:
                            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # 값 처리
                        try:
                            raw_value = row[target_column]
                            # 디버깅
                            logger.info(f"사용할 값: {raw_value}, 타입: {type(raw_value)}")
                            
                            # NaN 체크
                            if pd.isna(raw_value) or raw_value is None:
                                continue
                            
                            # float 변환
                            value = float(raw_value)
                            
                            # 레코드 추가
                            records.append((
                                time_str,
                                device_id,
                                resource_type,
                                target_column,
                                value,
                                '%' if 'percent' in target_column or resource_type in ['cpu', 'mem', 'disk'] else '',
                                company_domain,
                                building
                            ))
                        except Exception as e:
                            logger.error(f"값 변환 실패: {e}")
                else:
                    # 대체 처리 - 모든 숫자형 열 처리
                    logger.warning(f"{resource_type}에 대한 특정 열을 찾을 수 없음. 모든 숫자형 열 사용")
                    for idx, row in df_with_ts.iterrows():
                        # 타임스탬프 처리 (생략)...
                        
                        # 숫자형 열만 처리
                        for col in df_with_ts.select_dtypes(include=[np.number]).columns:
                            # 메타데이터 열은 건너뜀
                            if col in ['id', 'device_id', 'companyDomain', 'building', 'table'] or 'timestamp' in col.lower():
                                continue
                            
                            # 단위 결정
                            unit = '%' if 'percent' in col or 'usage' in col else ''
                            if 'bytes' in col:
                                unit = 'bytes/s'
                            elif 'rate' in col:
                                unit = '/s'
                            
                            # 값 검증 - NaN, inf 제외
                            value = row[col]
                            if pd.isna(value) or np.isinf(value):
                                continue
                            
                            try:
                                # 명시적 float 변환
                                value = float(value)
                                logger.info(f"열 {col}의 값: {value}")
                                
                                # 레코드 추가
                                records.append((
                                    time_str,
                                    device_id,
                                    resource_type,
                                    col,
                                    value,
                                    unit,
                                    company_domain,
                                    building
                                ))
                            except Exception as e:
                                logger.warning(f"값 변환 실패: {value}, 오류: {e}")
            
            # 저장할 레코드가 있는지 확인
            if not records:
                logger.warning(f"{resource_type} 자원에 저장할 유효한 데이터가 없습니다.")
                cursor.close()
                conn.close()
                return False
            
            # 레코드 삽입
            sql = """
            INSERT INTO resource_metrics 
            (timestamp, device_id, resource_type, metric_name, metric_value, unit, companyDomain, building)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            metric_value = VALUES(metric_value),
            unit = VALUES(unit)
            """
            
            # 각 레코드를 개별적으로 삽입하여 중복 오류 처리
            success_count = 0
            duplicate_count = 0
            
            for record in records:
                try:
                    # 디버깅 로그 추가
                    logger.debug(f"SQL 실행 직전 metric_value: {record[4]}, 타입: {type(record[4])}")
                    cursor.execute(sql, record)
                    success_count += 1
                except mysql.connector.errors.IntegrityError as e:
                    if "Duplicate entry" in str(e):
                        duplicate_count += 1
                        # 오류 로깅 제한
                        if duplicate_count <= 5:
                            logger.warning(f"중복 데이터 건너뜀: {record[0]}, {record[3]}")
                    else:
                        logger.error(f"삽입 오류: {e}")
                except Exception as e:
                    logger.error(f"SQL 실행 오류: {e}, 레코드: {record}")
            
            # 명시적 커밋
            conn.commit()
            
            logger.info(f"{resource_type} 자원 메트릭 저장 완료: {success_count}개 성공, {duplicate_count}개 중복")
            
            # 마지막 수집 시간 업데이트
            self.update_last_collection_time(resource_type, device_id, company_domain, building)
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"자원 메트릭 저장 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
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
    
    def load_resource_metrics(self, resource_type, device_id=None, start_time=None, end_time=None, metrics=None):
        """
        EAV 모델에서 자원 메트릭 로드
        
        Args:
            resource_type (str): 자원 유형 (cpu, mem, disk, diskio, net, system)
            device_id (str): 장치 ID
            start_time (datetime): 시작 시간
            end_time (datetime): 종료 시간
            metrics (list): 지표 이름 목록
            
        Returns:
            pd.DataFrame: 로드된 자원 메트릭 데이터프레임
        """
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config.get('host'),
                port=self.mysql_config.get('port'),
                user=self.mysql_config.get('user'),
                password=self.mysql_config.get('password'),
                database=self.mysql_config.get('database')
            )
            
            # 기본 쿼리
            query = "SELECT * FROM resource_metrics"
            conditions = []
            
            # 필터 조건 추가
            if resource_type:
                conditions.append(f"resource_type = '{resource_type}'")
            
            if device_id:
                conditions.append(f"device_id = '{device_id}'")
            
            if metrics and len(metrics) > 0:
                metrics_cond = " OR ".join([f"metric_name = '{m}'" for m in metrics])
                conditions.append(f"({metrics_cond})")
            
            # 시간 필터
            if start_time:
                conditions.append(f"timestamp >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'")
            
            # 조건 적용
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # 시간순 정렬
            query += " ORDER BY timestamp"
            
            # 쿼리 실행
            raw_df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            
            conn.close()
            
            if raw_df.empty:
                return pd.DataFrame()
            
            # EAV 형식을 피벗하여 일반 시계열 형식으로 변환
            df_pivot = raw_df.pivot_table(
                index=['timestamp', 'device_id', 'companyDomain', 'building'],
                columns='metric_name',
                values='metric_value',
                aggfunc='first'
            ).reset_index()
            
            # timestamp를 인덱스로 설정
            df_pivot.set_index('timestamp', inplace=True)
            
            # 메타데이터 열 추가
            if 'device_id' in df_pivot.columns:
                df_pivot['device_id'] = raw_df['device_id'].iloc[0]
            if 'companyDomain' in df_pivot.columns:
                df_pivot['companyDomain'] = raw_df['companyDomain'].iloc[0]
            if 'building' in df_pivot.columns:
                df_pivot['building'] = raw_df['building'].iloc[0]
            
            # 자원 유형 열 추가
            df_pivot['resource_type'] = resource_type
            
            logger.info(f"{len(df_pivot)}개의 자원 메트릭을 로드했습니다. (자원: {resource_type})")
            return df_pivot
            
        except Exception as e:
            logger.error(f"자원 메트릭 로드 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def send_to_api(self, data, endpoint_suffix):
        """
        데이터를 API로 전송 (재시도 로직 추가)
        
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
        
        # 재시도 설정 가져오기
        api_retry = self.config.get("api", {}).get("retry", {})
        retry_enabled = api_retry.get("enabled", False)
        max_attempts = api_retry.get("max_attempts", 3)
        backoff_factor = api_retry.get("backoff_factor", 2)
        timeout = api_retry.get("timeout", 30)
        
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
                    # numpy 데이터 타입 변환
                    elif isinstance(df_copy[col].iloc[0], (np.int64, np.int32, np.float64, np.float32)) if len(df_copy) > 0 else False:
                        df_copy[col] = df_copy[col].astype(float).astype(object)
                
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
            
            # 재시도 로직
            attempt = 0
            success = False
            last_error = None
            
            while attempt < max_attempts and not success:
                try:
                    attempt += 1
                    logger.info(f"API 결과 전송 중: {endpoint} (시도 {attempt}/{max_attempts})")
                    
                    # API 호출
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=timeout
                    )
                    
                    # 응답 확인
                    if response.status_code == 200:
                        logger.info(f"API 결과 전송 성공: {response.status_code}")
                        success = True
                        break
                    else:
                        logger.warning(f"API 결과 전송 실패: {response.status_code} - {response.text}")
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        
                        # 5xx 서버 오류만 재시도, 4xx 클라이언트 오류는 재시도하지 않음
                        if not (500 <= response.status_code < 600) and retry_enabled:
                            logger.warning(f"클라이언트 오류로 재시도하지 않음 ({response.status_code})")
                            break
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"API 요청 예외 발생: {e}")
                    last_error = str(e)
                
                # 재시도 대기 (지수 백오프)
                if attempt < max_attempts and retry_enabled:
                    wait_time = backoff_factor ** attempt
                    logger.info(f"{wait_time:.1f}초 후 재시도...")
                    time.sleep(wait_time)
            
            if not success and retry_enabled:
                logger.error(f"최대 시도 횟수({max_attempts})를 초과했습니다. 마지막 오류: {last_error}")
            
            return success
        
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
        
        # EAV 모델 사용 여부 확인
        use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
        
        if use_eav_model:
            # 자원 유형 확인
            resource_type = None
            
            # 컬럼명에서 자원 유형 추출
            for col in predictions_df.columns:
                if 'cpu' in col.lower() and '_pred' in col.lower():
                    resource_type = 'cpu'
                    break
                elif 'mem' in col.lower() and '_pred' in col.lower():
                    resource_type = 'mem'
                    break
                elif 'disk' in col.lower() and '_pred' in col.lower():
                    resource_type = 'disk'
                    break
                elif 'net' in col.lower() and '_pred' in col.lower():
                    resource_type = 'net'
                    break
                elif 'system' in col.lower() and '_pred' in col.lower():
                    resource_type = 'system'
                    break
            
            if not resource_type:
                logger.warning("자원 유형을 확인할 수 없습니다.")
                resource_type = "unknown"
            
            # EAV 모델로 변환하여 저장
            return self.save_prediction_results(predictions_df, "resource", resource_type, horizon)
        else:
            # 기존 방식으로 저장
            mysql_success = self.save_to_mysql(predictions_df, "resource_predictions")
        
        # API로 전송
        api_success = self.send_to_api(predictions_df, "resource_predictions")
        
        return mysql_success and api_success
    
    def save_prediction_results(self, df, prediction_type, resource_type, prediction_horizon='short_term'):
        """
        예측 결과를 MySQL에 저장 (타임스탬프 변환 오류 해결)
        
        Args:
            df (pd.DataFrame): 예측 결과 데이터프레임
            prediction_type (str): 예측 유형 (resource, failure)
            resource_type (str): 자원 유형
            prediction_horizon (str): 예측 지평선
            
        Returns:
            bool: 성공 여부
        """
        if df is None or df.empty:
            logger.warning("저장할 예측 결과가 비어 있습니다.")
            return False
        
        try:
            # MySQL 연결 생성
            conn = self.mysql_connect()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 데이터 변환 및 삽입
            records = []
            
            # 인덱스가 DatetimeIndex인 경우 열로 변환
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # 첫 번째 열(원래 인덱스)이 무슨 이름이든 timestamp로 변경
                df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
            
            # 각 예측 결과를 SQL 레코드로 변환
            for idx, row in df.iterrows():
                try:
                    # 타임스탬프 처리 - 문자열로 변환하여 MySQL 타입 변환 오류 해결
                    if 'timestamp' in df.columns:
                        timestamp = row['timestamp']
                        # 타임스탬프 객체를 문자열로 변환
                        if isinstance(timestamp, pd.Timestamp) or isinstance(timestamp, datetime):
                            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 예측값을 포함하는 열만 찾기 - 실제 수치 데이터
                    pred_cols = [col for col in df.columns if ('predict' in col or 'probability' in col) 
                                and isinstance(row[col], (int, float, np.number))]
                    
                    for col in pred_cols:
                        # 메트릭 이름 추출
                        metric_name = col.replace('_predicted', '').replace('_probability', '')
                        
                        # 값 확인 (NaN 또는 비수치 값 처리)
                        val = row[col]
                        if pd.isna(val):
                            val = 0.0
                        else:
                            try:
                                val = float(val)
                            except (ValueError, TypeError):
                                logger.warning(f"수치로 변환할 수 없는 값: {val}, 0.0으로 대체")
                                val = 0.0
                        
                        # prediction_time도 문자열로 변환
                        prediction_time = row['prediction_time'] if 'prediction_time' in df.columns else datetime.now()
                        if isinstance(prediction_time, pd.Timestamp) or isinstance(prediction_time, datetime):
                            prediction_time = prediction_time.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # 메타데이터 추출
                        device_id = row['device_id'] if 'device_id' in df.columns else 'device_001'
                        company_domain = row['companyDomain'] if 'companyDomain' in df.columns else 'javame'
                        building = row['building'] if 'building' in df.columns else 'gyeongnam_campus'
                        
                        # 레코드 추가 - 모든 타임스탬프를 문자열로 변환
                        records.append((
                            timestamp,           # 문자열로 변환된 타임스탬프
                            device_id,           # 문자열
                            resource_type,       # 문자열
                            metric_name,         # 문자열
                            val,                 # 숫자
                            prediction_type,     # 문자열
                            prediction_horizon,  # 문자열
                            prediction_time,     # 문자열로 변환된 타임스탬프
                            company_domain,      # 문자열
                            building             # 문자열
                        ))
                except Exception as e:
                    logger.error(f"행 {idx} 처리 중 오류 발생: {e}")
                    continue
            
            # 일괄 삽입
            if records:
                query = """
                INSERT INTO prediction_results 
                (timestamp, device_id, resource_type, metric_name, predicted_value, 
                prediction_type, prediction_horizon, prediction_time, companyDomain, building)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.executemany(query, records)
                conn.commit()
                
                logger.info(f"{len(records)}개의 예측 결과가 저장되었습니다. (유형: {prediction_type}, 자원: {resource_type})")
                
                cursor.close()
                conn.close()
                return True
            else:
                logger.warning("저장할 레코드가 없습니다.")
                cursor.close()
                conn.close()
                return False
            
        except Exception as e:
            logger.error(f"예측 결과 저장 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def send_to_api(self, data, endpoint_suffix):
        """
        데이터를 API로 전송 (개선됨)
        
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
        
        # API 비활성화 확인
        api_enabled = self.config.get("api", {}).get("enabled", True)
        if not api_enabled:
            logger.info("API 호출이 비활성화되어 있습니다.")
            return True  # 성공으로 처리
        
        # 재시도 설정 가져오기
        api_retry = self.config.get("api", {}).get("retry", {})
        retry_enabled = api_retry.get("enabled", False)
        max_attempts = api_retry.get("max_attempts", 3)
        backoff_factor = api_retry.get("backoff_factor", 2)
        timeout = api_retry.get("timeout", 30)
        
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
                    # numpy 데이터 타입 변환
                    elif isinstance(df_copy[col].iloc[0], (np.int64, np.int32, np.float64, np.float32)) if len(df_copy) > 0 else False:
                        df_copy[col] = df_copy[col].astype(float).astype(object)
                
                # NaN 값 처리
                df_copy = df_copy.fillna('null')
                
                # 데이터프레임을 레코드 리스트로 변환
                records = df_copy.to_dict(orient='records')
            else:
                # 이미 dictionary 형태인 경우
                records = data
            
            # API URL 설정
            endpoint = f"{self.api_url}/{endpoint_suffix}" if endpoint_suffix else self.api_url
            
            # 메타데이터 추가
            payload = {
                "data": records,
                "meta": {
                    "source": "iot_prediction_system",
                    "version": "2.0",
                    "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                }
            }
            
            # 재시도 로직
            attempt = 0
            success = False
            last_error = None
            
            while attempt < max_attempts and not success:
                try:
                    attempt += 1
                    logger.info(f"API 결과 전송 중: {endpoint} (시도 {attempt}/{max_attempts})")
                    
                    # API 호출
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=timeout
                    )
                    
                    # 응답 확인
                    if response.status_code == 200:
                        logger.info(f"API 결과 전송 성공: {response.status_code}")
                        success = True
                        break
                    else:
                        logger.warning(f"API 결과 전송 실패: {response.status_code} - {response.text}")
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        
                        # 5xx 서버 오류만 재시도, 4xx 클라이언트 오류는 재시도하지 않음
                        if not (500 <= response.status_code < 600) and retry_enabled:
                            logger.warning(f"클라이언트 오류로 재시도하지 않음 ({response.status_code})")
                            break
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"API 요청 예외 발생: {e}")
                    last_error = str(e)
                
                # 재시도 대기 (지수 백오프)
                if attempt < max_attempts and retry_enabled:
                    wait_time = backoff_factor ** attempt
                    logger.info(f"{wait_time:.1f}초 후 재시도...")
                    import time
                    time.sleep(wait_time)
            
            if not success and retry_enabled:
                logger.error(f"최대 시도 횟수({max_attempts})를 초과했습니다. 마지막 오류: {last_error}")
            
            # 테스트 모드인 경우 성공으로 처리
            test_mode = self.config.get("api", {}).get("test_mode", False)
            if test_mode and not success:
                logger.info("테스트 모드: API 오류를 무시하고 성공으로 처리합니다.")
                return True
            
            return success
        
        except Exception as e:
            logger.error(f"API 결과 전송 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 테스트 모드인 경우 성공으로 처리
            test_mode = self.config.get("api", {}).get("test_mode", False)
            if test_mode:
                logger.info("테스트 모드: 오류를 무시하고 성공으로 처리합니다.")
                return True
                
            return False
    
    def mysql_connect(self):
        """
        MySQL 연결 생성 (오류 처리 포함)
        
        Returns:
            connection: MySQL 연결 객체 또는 None (실패 시)
        """
        try:
            # 설정 유효성 검사
            host = self.mysql_config.get('host')
            port = self.mysql_config.get('port')
            user = self.mysql_config.get('user')
            password = self.mysql_config.get('password')
            database = self.mysql_config.get('database')
            
            # 필수 설정 확인
            if not all([host, user, database]):
                logger.error("MySQL 연결 정보가 불완전합니다.")
                return None
            
            # 포트 설정이 없거나 None이면 기본값 사용
            if port is None:
                port = 3306
                logger.warning(f"MySQL 포트가 지정되지 않아 기본값({port})을 사용합니다.")
            
            # 포트 문자열인 경우 정수로 변환
            if isinstance(port, str):
                try:
                    port = int(port)
                except ValueError:
                    logger.error(f"MySQL 포트 번호가 유효하지 않습니다: {port}")
                    port = 3306
                    logger.warning(f"기본 포트({port})를 사용합니다.")
            
            # 연결 시도
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connection_timeout=10
            )
            
            return conn
        
        except Exception as e:
            logger.error(f"MySQL 연결 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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
        
        # EAV 모델 사용 여부 확인
        use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
        
        if use_eav_model:
            # 각 자원별로 처리
            success = True
            for col in failure_cols:
                resource_type = col.replace('_failure_probability', '')
                
                # 해당 자원의 고장 확률만 추출
                resource_df = predictions_df[['timestamp', 'device_id', 'companyDomain', 'building', 
                                             'prediction_time', 'is_failure_predicted', 'failure_resource', col]]
                
                # 컬럼명 변경
                resource_df = resource_df.rename(columns={col: 'failure_probability'})
                
                # EAV 모델로 변환하여 저장
                resource_success = self.save_prediction_results(resource_df, "failure", resource_type)
                success = success and resource_success
            
            return success
        else:
            # 기존 방식으로 저장
            mysql_success = self.save_to_mysql(predictions_df, "failure_predictions")
        
        # API로 전송
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
                
                # 재학습 필요 여부 확인
                self.check_model_performance(model_type, resource_type, performance_metrics)
            else:
                logger.warning("모델 성능 평가 정보 기록 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"모델 성능 평가 정보 기록 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def check_model_performance(self, model_type, resource_type, performance_metrics):
        """
        모델 성능 확인 및 재학습 트리거
        
        Args:
            model_type (str): 모델 유형 (resource_prediction 또는 failure_prediction)
            resource_type (str): 자원 유형 (cpu, memory, disk, network, system)
            performance_metrics (dict): 성능 지표
            
        Returns:
            bool: 재학습 필요 여부
        """
        # 자동 재학습 설정 확인
        auto_retraining = self.config.get("prediction", {}).get("performance_monitoring", {}).get("auto_retraining", False)
        if not auto_retraining:
            return False
        
        # 성능 임계값 설정
        thresholds = self.config.get("prediction", {}).get("performance_monitoring", {}).get("thresholds", {})
        model_thresholds = thresholds.get(model_type, {})
        
        # 성능 확인
        retraining_needed = False
        
        if model_type == "resource_prediction":
            rmse_threshold = model_thresholds.get("rmse", 10.0)
            mae_threshold = model_thresholds.get("mae", 5.0)
            
            if (performance_metrics.get("rmse", 0) > rmse_threshold or 
                performance_metrics.get("mae", 0) > mae_threshold):
                retraining_needed = True
                
        elif model_type == "failure_prediction":
            accuracy_threshold = model_thresholds.get("accuracy", 0.7)
            precision_threshold = model_thresholds.get("precision", 0.6)
            recall_threshold = model_thresholds.get("recall", 0.6)
            
            if (performance_metrics.get("accuracy", 1.0) < accuracy_threshold or 
                performance_metrics.get("precision", 1.0) < precision_threshold or
                performance_metrics.get("recall", 1.0) < recall_threshold):
                retraining_needed = True
        
        if retraining_needed:
            logger.warning(f"{model_type}/{resource_type} 모델 성능이 임계값 미달: 재학습 필요")
            
            # 재학습 트리거를 위한 테이블 생성
            try:
                conn = mysql.connector.connect(
                    host=self.mysql_config.get('host'),
                    port=self.mysql_config.get('port'),
                    user=self.mysql_config.get('user'),
                    password=self.mysql_config.get('password'),
                    database=self.mysql_config.get('database')
                )
                
                cursor = conn.cursor()
                
                # 재학습 트리거 테이블 생성
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_retraining_triggers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_type VARCHAR(50),
                    resource_type VARCHAR(50),
                    trigger_time DATETIME,
                    reason VARCHAR(255),
                    status VARCHAR(20),
                    
                    INDEX idx_model (model_type, resource_type),
                    INDEX idx_status (status)
                )
                """)
                
                # 재학습 트리거 추가
                cursor.execute("""
                INSERT INTO model_retraining_triggers
                (model_type, resource_type, trigger_time, reason, status)
                VALUES (%s, %s, %s, %s, %s)
                """, (
                    model_type, 
                    resource_type, 
                    datetime.now(),
                    f"성능 저하: {json.dumps(performance_metrics)}",
                    "pending"
                ))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                logger.info(f"{model_type}/{resource_type} 모델 재학습 트리거가 추가되었습니다.")
                return True
                
            except Exception as e:
                logger.error(f"모델 재학습 트리거 추가 실패: {e}")
                return False
        
        return False
    
    def update_last_collection_time(self, collection_type, device_id, company_domain, building):
        """
        마지막 데이터 수집 시간 업데이트
        
        Args:
            collection_type (str): 수집 유형 (resource_type 또는 'all')
            device_id (str): 장치 ID
            company_domain (str): 회사 도메인
            building (str): 건물
            
        Returns:
            bool: 성공 여부
        """
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config.get('host'),
                port=self.mysql_config.get('port'),
                user=self.mysql_config.get('user'),
                password=self.mysql_config.get('password'),
                database=self.mysql_config.get('database')
            )
            
            cursor = conn.cursor()
            
            # 현재 시간
            current_time = datetime.now()
            
            # 데이터 타입 확인 및 변환 (수정: float64 처리)
            if isinstance(device_id, (np.number, float)):
                device_id = str(device_id)
            if isinstance(company_domain, (np.number, float)):
                company_domain = str(company_domain)
            if isinstance(building, (np.number, float)):
                building = str(building)
            
            # 이미 존재하는지 확인
            cursor.execute("""
            SELECT id FROM collection_metadata 
            WHERE collection_type = %s 
            AND device_id = %s 
            AND companyDomain = %s 
            AND building = %s
            """, (collection_type, device_id, company_domain, building))
            
            result = cursor.fetchone()
            
            if result:
                # 기존 레코드 업데이트
                cursor.execute("""
                UPDATE collection_metadata 
                SET last_collection_time = %s 
                WHERE collection_type = %s 
                AND device_id = %s 
                AND companyDomain = %s 
                AND building = %s
                """, (current_time, collection_type, device_id, company_domain, building))
            else:
                # 새 레코드 삽입
                cursor.execute("""
                INSERT INTO collection_metadata 
                (collection_type, device_id, companyDomain, building, last_collection_time)
                VALUES (%s, %s, %s, %s, %s)
                """, (collection_type, device_id, company_domain, building, current_time))
            
            # 'all' 유형도 함께 업데이트
            if collection_type != 'all':
                # 이미 존재하는지 확인
                cursor.execute("""
                SELECT id FROM collection_metadata 
                WHERE collection_type = 'all' 
                AND device_id = %s 
                AND companyDomain = %s 
                AND building = %s
                """, (device_id, company_domain, building))
                
                result = cursor.fetchone()
                
                if result:
                    # 기존 레코드 업데이트
                    cursor.execute("""
                    UPDATE collection_metadata 
                    SET last_collection_time = %s 
                    WHERE collection_type = 'all' 
                    AND device_id = %s 
                    AND companyDomain = %s 
                    AND building = %s
                    """, (current_time, device_id, company_domain, building))
                else:
                    # 새 레코드 삽입
                    cursor.execute("""
                    INSERT INTO collection_metadata 
                    (collection_type, device_id, companyDomain, building, last_collection_time)
                    VALUES (%s, %s, %s, %s, %s)
                    """, ('all', device_id, company_domain, building, current_time))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"마지막 수집 시간 업데이트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_last_collection_time(self, collection_type, device_id, company_domain, building):
        """
        마지막 데이터 수집 시간 조회
        
        Args:
            collection_type (str): 수집 유형 (resource_type 또는 'all')
            device_id (str): 장치 ID
            company_domain (str): 회사 도메인
            building (str): 건물
            
        Returns:
            datetime: 마지막 수집 시간 (없으면 None)
        """
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config.get('host'),
                port=self.mysql_config.get('port'),
                user=self.mysql_config.get('user'),
                password=self.mysql_config.get('password'),
                database=self.mysql_config.get('database')
            )
            
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT last_collection_time FROM collection_metadata 
            WHERE collection_type = %s 
            AND device_id = %s 
            AND companyDomain = %s 
            AND building = %s
            """, (collection_type, device_id, company_domain, building))
            
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result:
                return result[0]
            else:
                # 기본값: 7일 전
                max_days = self.config.get("data_collection", {}).get("max_days_initial", 7)
                return datetime.now() - timedelta(days=max_days)
            
        except Exception as e:
            logger.error(f"마지막 수집 시간 조회 실패: {e}")
            # 기본값: 7일 전
            max_days = self.config.get("data_collection", {}).get("max_days_initial", 7)
            return datetime.now() - timedelta(days=max_days)
    
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
    def mysql_connect(self):
        """
        MySQL 연결 생성 (오류 처리 포함)
        
        Returns:
            connection: MySQL 연결 객체 또는 None (실패 시)
        """
        try:
            # 설정 유효성 검사
            host = self.mysql_config.get('host')
            port = self.mysql_config.get('port')
            user = self.mysql_config.get('user')
            password = self.mysql_config.get('password')
            database = self.mysql_config.get('database')
            
            # 필수 설정 확인
            if not all([host, user, database]):
                logger.error("MySQL 연결 정보가 불완전합니다.")
                return None
            
            # 포트 설정이 없거나 None이면 기본값 사용
            if port is None:
                port = 3306
                logger.warning(f"MySQL 포트가 지정되지 않아 기본값({port})을 사용합니다.")
            
            # 포트 문자열인 경우 정수로 변환
            if isinstance(port, str):
                try:
                    port = int(port)
                except ValueError:
                    logger.error(f"MySQL 포트 번호가 유효하지 않습니다: {port}")
                    port = 3306
                    logger.warning(f"기본 포트({port})를 사용합니다.")
            
            # 연결 시도
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connection_timeout=10
            )
            
            return conn
        
        except Exception as e:
            logger.error(f"MySQL 연결 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
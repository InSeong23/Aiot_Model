#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 수집 모듈 (개선)

InfluxDB에서 데이터를 쿼리하고 전처리하는 기능을 제공합니다.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient

logger = logging.getLogger(__name__)

class DataCollector:
    """
    데이터 수집 클래스
    
    InfluxDB에서 데이터를 수집하고 MySQL에서 전처리된 데이터를 불러옵니다.
    """
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config (dict): 설정 정보
        """
        self.config = config
        self.influx_config = config.get("influxdb", {})
        self.client = None
        self.result_handler = None
    
    def set_result_handler(self, result_handler):
        """
        결과 핸들러 설정
        
        Args:
            result_handler (ResultHandler): 결과 핸들러 객체
        """
        self.result_handler = result_handler
    
    def connect(self):
        """
        InfluxDB 연결
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            # URL에 http 또는 https가 없는 경우 http:// 추가
            url = self.influx_config.get("url", "http://influxdb.javame.live")
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"http://{url}"
                
            token = self.influx_config.get("token", "")
            org = self.influx_config.get("org", "javame")
            
            # 디버그 로그 추가
            logger.debug(f"InfluxDB 연결 시도: URL={url}, ORG={org}, TOKEN 길이={len(token)}")
            
            # 클라이언트 생성 시 추가 옵션 설정
            self.client = InfluxDBClient(
                url=url,
                token=token,
                org=org,
                timeout=30000,  # 타임아웃 증가
                verify_ssl=False  # SSL 인증서 검증 비활성화
            )
            
            # 연결 테스트 - health 확인
            health = self.client.health()
            if health and health.status == "pass":
                logger.info(f"InfluxDB 연결 성공: {url}")
                return True
            else:
                logger.error(f"InfluxDB 상태 확인 실패: {health.message if health else 'No health info'}")
                return False
        except Exception as e:
            logger.error(f"InfluxDB 연결 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def disconnect(self):
        """InfluxDB 연결 해제"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("InfluxDB 연결 해제")
    
    def query_data(self, start_time, end_time=None, location=None, origins=None, tags=None, fields=None):
        """
        InfluxDB에서 데이터 쿼리 (타입 안전성 강화)
        
        Args:
            start_time (datetime or int): 시작 시간 (datetime 객체 또는 일 수)
            end_time (datetime): 종료 시간 (기본값: 현재 시간)
            location (str): location 태그 값
            origins (list): origin 태그 값 목록
            tags (dict): 추가 태그 필터
            fields (list): 필드 목록
            
        Returns:
            pd.DataFrame: 쿼리 결과
        """
        if not self.client:
            if not self.connect():
                return pd.DataFrame()
        
        # 기본값 설정
        if end_time is None:
            end_time = datetime.now()
        
        # start_time이 정수인 경우 일 수로 처리
        if isinstance(start_time, int):
            start_time = end_time - timedelta(days=start_time)
        
        bucket = self.influx_config.get("bucket", "data")
        
        try:
            query_api = self.client.query_api()
            
            # 시간 범위
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # 최소한의 필터만 사용 (location만 적용)
            filters = []
            
            if location:
                filters.append(f'r.location == "{location}"')
            
            # 필터를 AND로 결합
            filter_str = " and ".join(filters) if filters else "true"
            
            # 간소화된 쿼리
            query = f'''
            from(bucket: "{bucket}")
            |> range(start: {start_time_str}, stop: {end_time_str})
            |> filter(fn: (r) => {filter_str})
            |> limit(n: 2000)
            '''
            
            # 쿼리 실행
            result = query_api.query_data_frame(query)
            
            # 결과 처리
            if isinstance(result, list):
                result = pd.concat(result) if result else pd.DataFrame()
            
            if result.empty:
                logger.warning(f"쿼리 결과가 비어있습니다. location={location}")
                return pd.DataFrame()
            
            # _time을 인덱스로 설정 (UTC 시간대 유지)
            if '_time' in result.columns:
                result = result.set_index('_time')
            
            # 안전한 데이터 타입 변환 (타임스탬프 건너뛰기)
            date_columns = ['_start', '_stop']
            string_columns = ['result', 'table', '_field', '_measurement', 'location', 'origin', 'device_id', 'companyDomain', 'building', 'place']
            
            for col in result.columns:
                # 날짜 컬럼 무시, 문자열 컬럼 무시
                if col in date_columns or col in string_columns:
                    continue
                    
                # _value 또는 value 컬럼은 숫자로 변환 시도
                if col == '_value' or col == 'value':
                    try:
                        # 안전하게 숫자 변환
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"'{col}' 열 숫자 변환 실패: {e}")
            
            logger.info(f"쿼리 결과: {len(result)}행, {result.shape[1]}열")
            return result
            
        except Exception as e:
            logger.error(f"InfluxDB 쿼리 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def collect_all_resources(self, start_time=None, end_time=None, save_to_mysql=True):
        """
        모든 자원 데이터 수집 및 전처리
        
        Args:
            start_time (datetime): 시작 시간 (None이면 마지막 수집 시간 사용)
            end_time (datetime): 종료 시간 (None이면 현재 시간)
            save_to_mysql (bool): MySQL에 저장 여부
            
        Returns:
            dict: 자원별 전처리된 데이터프레임
        """
        if end_time is None:
            end_time = datetime.now()
        
        # 회사/건물/디바이스 목록 가져오기
        companies = self.config.get("companies", [])
        if not companies:
            # 기본 설정 사용
            default_company = self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
            default_building = self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
            default_device = self.config.get('data_processing', {}).get('default_values', {}).get('device_id', 'device_001')
            
            companies = [{
                "companyDomain": default_company,
                "buildings": [{
                    "name": default_building,
                    "devices": [default_device]
                }]
            }]
        
        # 결과 저장용 딕셔너리
        all_processed_data = {}
        
        # 파티션 관리
        if self.result_handler and self.config.get("database", {}).get("partition_strategy", {}).get("enabled", True):
            self.result_handler.manage_partitions()
        
        # 각 회사/건물/디바이스별로 처리
        for company in companies:
            company_domain = company.get("companyDomain")
            for building in company.get("buildings", []):
                building_name = building.get("name")
                for device_id in building.get("devices", []):
                    logger.info(f"수집 중: {company_domain}/{building_name}/{device_id}")
                    
                    # 각 자원별로 데이터 수집
                    device_data = {}
                    df_list = []
                    resource_types = []
                    
                    # 자원 위치 목록
                    locations = self.influx_config.get("locations", ["cpu", "disk", "diskio", "mem", "net", "system"])
                    
                    for location in locations:
                        logger.info(f"'{location}' 자원 데이터 수집 시작...")
                        
                        # 태그 필터 추가
                        tags = {
                            "companyDomain": company_domain,
                            "building": building_name,
                            "device_id": device_id
                        }
                        
                        # 이 자원에 대한 마지막 수집 시간 가져오기 (없으면 디바이스 전체 수집 시간 사용)
                        if start_time is None and self.result_handler:
                            resource_start_time = self.result_handler.get_last_collection_time(
                                location, device_id, company_domain, building_name
                            )
                            # 두 시간 중 더 최근 시간을 사용하여 중복을 최소화
                            if resource_start_time:
                                loc_start_time = resource_start_time
                            else:
                                # 디바이스 전체 수집 시간 사용
                                device_start_time = self.result_handler.get_last_collection_time(
                                    'all', device_id, company_domain, building_name
                                )
                                if device_start_time:
                                    loc_start_time = device_start_time
                                else:
                                    # 기본: 7일
                                    loc_start_time = end_time - timedelta(days=7)
                        else:
                            loc_start_time = start_time
                        
                        # 해당 자원 데이터 수집
                        df = self.get_resource_data(location, loc_start_time, end_time, tags)
                        
                        if df is None or df.empty:
                            logger.warning(f"'{location}' 자원 데이터 수집 실패")
                            continue
                        
                        # 기본 메타데이터 추가
                        if 'device_id' not in df.columns:
                            df['device_id'] = device_id
                        if 'companyDomain' not in df.columns:
                            df['companyDomain'] = company_domain
                        if 'building' not in df.columns:
                            df['building'] = building_name
                        
                        # 데이터 전처리
                        from data_preprocessor import DataPreprocessor
                        preprocessor = DataPreprocessor(self.config)
                        processed_df = preprocessor.process_resource_data(df, location)
                        
                        if processed_df is None or processed_df.empty:
                            logger.warning(f"'{location}' 자원 데이터 전처리 실패")
                            continue
                        
                        # 결과 저장
                        device_data[location] = processed_df
                        
                        # 배치 처리를 위한 리스트 추가
                        df_list.append(processed_df)
                        resource_types.append(location)
                    
                    # MySQL에 일괄 저장 (배치 처리)
                    if save_to_mysql and self.result_handler and df_list:
                        # EAV 모델 사용 여부 확인
                        use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
                        
                        if use_eav_model:
                            # 배치 처리 사용
                            save_success = self.result_handler.save_resource_metrics_batch(df_list, resource_types)
                            
                            if save_success:
                                logger.info(f"{len(df_list)}개 자원의 전처리 데이터를 MySQL에 일괄 저장했습니다.")
                            else:
                                logger.warning("자원 전처리 데이터 일괄 저장 실패")
                        else:
                            # 기존 개별 저장 방식
                            for i, resource_type in enumerate(resource_types):
                                save_success = self.result_handler.save_processed_data(df_list[i])
                                
                                if save_success:
                                    logger.info(f"'{resource_type}' 자원 전처리 데이터를 MySQL에 저장했습니다.")
                                else:
                                    logger.warning(f"'{resource_type}' 자원 전처리 데이터를 MySQL에 저장하지 못했습니다.")
                    
                    # 디바이스별 결과 저장
                    if device_data:
                        key = f"{company_domain}_{building_name}_{device_id}"
                        all_processed_data[key] = device_data
        
        return all_processed_data
    
    def get_resource_data(self, resource_type, start_time=None, end_time=None, tags=None):
        """
        리소스 유형별 데이터 수집 (개선)
        
        Args:
            resource_type (str): 자원 유형
            start_time (datetime): 시작 시간 (None이면 마지막 수집 시간 또는 7일 전부터)
            end_time (datetime): 종료 시간 (None이면 현재 시간)
            tags (dict): 태그 필터
            
        Returns:
            pd.DataFrame: 자원 데이터
        """
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # 초기 데이터 수집 시 7일 기본값 사용
            max_days = self.config.get("data_collection", {}).get("max_days_initial", 7)
            start_time = end_time - timedelta(days=max_days)
            logger.info(f"{resource_type} 자원의 초기 데이터 수집: {max_days}일 기간")
        
        # 간소화된 쿼리: 단일 필터 (location만)
        logger.info(f"{resource_type} 데이터 수집 중... (기간: {start_time} ~ {end_time})")
        df = self.query_data(
            start_time=start_time,
            end_time=end_time,
            location=resource_type
        )
        
        if df is None or df.empty:
            logger.warning(f"{resource_type} 데이터 수집 실패")
            return pd.DataFrame()
        
        # 기본 메타데이터 추가
        if 'device_id' not in df.columns or df['device_id'].isna().all():
            df['device_id'] = self.config.get('data_processing', {}).get('default_values', {}).get('device_id', '192.168.71.74')
        
        if 'companyDomain' not in df.columns or df['companyDomain'].isna().all():
            df['companyDomain'] = self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
        
        if 'building' not in df.columns or df['building'].isna().all():
            df['building'] = self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
        
        logger.info(f"{resource_type} 데이터 수집 완료: {len(df)}행")
        return df
    
    def load_processed_data(self, days=7, resource_types=None):
        """
        MySQL에서 전처리된 데이터 로드
        
        Args:
            days (int): 로드할 일 수
            resource_types (list): 리소스 유형 목록
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        if not self.result_handler:
            logger.error("결과 핸들러가 설정되지 않았습니다.")
            return pd.DataFrame()
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # EAV 모델 사용 여부 확인
        use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
        
        if use_eav_model:
            # 각 자원별로 데이터 로드 후 병합
            result_dfs = []
            
            for resource_type in resource_types:
                resource_df = self.result_handler.load_resource_metrics(
                    resource_type=resource_type,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not resource_df.empty:
                    result_dfs.append(resource_df)
            
            # 결과 병합
            if result_dfs:
                return pd.concat(result_dfs, axis=0)
            else:
                return pd.DataFrame()
        else:
            # 기존 방식으로 데이터 로드
            
            # 필터 조건 생성
            filter_cond = None
            if resource_types:
                # 자원 유형 필드 확인
                filters = []
                for resource in resource_types:
                    if resource == 'cpu':
                        filters.append("cpu_usage IS NOT NULL")
                    elif resource == 'mem':
                        filters.append("memory_used_percent IS NOT NULL")
                    elif resource == 'disk':
                        filters.append("disk_used_percent IS NOT NULL")
                    elif resource == 'diskio':
                        filters.append("disk_io_utilization IS NOT NULL")
                    elif resource == 'net':
                        filters.append("net_utilization IS NOT NULL")
                    elif resource == 'system':
                        filters.append("system_load1 IS NOT NULL")
                
                if filters:
                    filter_cond = " OR ".join(filters)
            
            # 데이터 로드
            return self.result_handler.load_from_mysql(
                table_name="processed_resource_data",
                start_time=start_time,
                end_time=end_time,
                filter_cond=filter_cond
            )
    
    def get_prediction_data(self, resource_type, days=7, use_mysql=True):
        """
        예측을 위한 자원 데이터 가져오기
        
        Args:
            resource_type (str): 자원 유형 (cpu, mem, disk, diskio, net, system)
            days (int): 가져올 일 수
            use_mysql (bool): MySQL에서 데이터 로드 여부
            
        Returns:
            pd.DataFrame: 예측을 위한 데이터
        """
        if use_mysql and self.result_handler:
            # EAV 모델 사용 여부 확인
            use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
            
            if use_eav_model:
                # 자원 유형에 맞는 메트릭 로드
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
                
                df = self.result_handler.load_resource_metrics(
                    resource_type=resource_type,
                    start_time=start_time,
                    end_time=end_time
                )
            else:
                # MySQL에서 전처리된 데이터 로드
                df = self.load_processed_data(days, [resource_type])
            
            if not df.empty:
                logger.info(f"{resource_type} 예측 데이터를 MySQL에서 로드했습니다: {len(df)}행")
                return df
        
        # MySQL에서 데이터를 로드하지 못한 경우 InfluxDB에서 직접 가져오기
        logger.info(f"{resource_type} 예측 데이터를 InfluxDB에서 수집합니다...")
        
        # 원시 데이터 수집
        raw_df = self.get_resource_data(resource_type, days)
        
        if raw_df.empty:
            logger.warning(f"{resource_type} 데이터 수집 실패")
            return pd.DataFrame()
        
        # 회사/건물/디바이스 정보 설정
        default_company = self.config.get('default_company', 'javame')
        default_building = self.config.get('default_building', 'gyeongnam_campus')
        default_device = self.config.get('default_device', 'device_001')
        
        # 기본 메타데이터 추가
        if 'device_id' not in raw_df.columns:
            raw_df['device_id'] = default_device
        if 'companyDomain' not in raw_df.columns:
            raw_df['companyDomain'] = default_company
        if 'building' not in raw_df.columns:
            raw_df['building'] = default_building
        
        from data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(self.config)
        preprocessor.set_result_handler(self.result_handler)  # result_handler 설정
        processed_df = preprocessor.process_resource_data(raw_df, resource_type)
        
        if processed_df.empty:
            logger.warning(f"{resource_type} 데이터 전처리 실패")
            return pd.DataFrame()
        
        # MySQL에 저장
        if use_mysql and self.result_handler:
            # EAV 모델 사용 여부 확인
            use_eav_model = self.config.get("database", {}).get("use_eav_model", False)
            
            if use_eav_model:
                save_success = self.result_handler.save_resource_metrics(processed_df, resource_type)
            else:
                save_success = self.result_handler.save_processed_data(processed_df)
                
            if save_success:
                logger.info(f"{resource_type} 전처리 데이터를 MySQL에 저장했습니다.")
        
        return processed_df
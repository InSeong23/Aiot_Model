#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfluxDB 데이터 구조 탐색 스크립트
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.json'):
    """설정 파일 로드"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def explore_influxdb():
    """InfluxDB 데이터 구조 탐색"""
    # 설정 로드
    config = load_config()
    influx_config = config.get("influxdb", {})
    
    # 환경 변수에서 토큰 가져오기
    token = os.environ.get('INFLUXDB_TOKEN') or influx_config.get("token", "")
    url = influx_config.get("url", "http://influxdb.javame.live")
    org = influx_config.get("org", "javame")
    bucket = influx_config.get("bucket", "data")
    
    logger.info(f"InfluxDB 연결 시도: {url}, 조직: {org}, 버킷: {bucket}")
    
    try:
        # InfluxDB 클라이언트 생성
        client = InfluxDBClient(url=url, token=token, org=org)
        query_api = client.query_api()
        
        # 기본 탐색 쿼리 실행 - 최근 데이터 일부만 가져오기
        logger.info("기본 데이터 구조 탐색 중...")
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -1d)
          |> limit(n: 10)
        '''
        result = query_api.query_data_frame(query)
        
        if isinstance(result, list):
            if not result:
                logger.warning("쿼리 결과가 비어있습니다.")
                return
            result = pd.concat(result)
        
        logger.info("\n=== 기본 데이터 구조 ===")
        logger.info(f"결과 형태: {type(result)}")
        logger.info(f"결과 크기: {result.shape}")
        logger.info(f"컬럼 목록: {result.columns.tolist()}")
        
        # 데이터 샘플 출력
        logger.info("\n=== 데이터 샘플 ===")
        pd.set_option('display.max_columns', None)
        logger.info(f"\n{result.head()}")
        
        # 고유한 측정값(measurement) 확인
        if '_measurement' in result.columns:
            measurements = result['_measurement'].unique()
            logger.info(f"\n=== 고유 측정값(measurement) ===\n{measurements}")
        
        # 고유한 필드(field) 확인
        if '_field' in result.columns:
            fields = result['_field'].unique()
            logger.info(f"\n=== 고유 필드(field) ===\n{fields}")
        
        # 고유한 위치(location) 확인
        if 'location' in result.columns:
            locations = result['location'].unique()
            logger.info(f"\n=== 고유 위치(location) ===\n{locations}")
        
        # 각 위치별 데이터 확인
        logger.info("\n=== 위치별 데이터 탐색 ===")
        locations = influx_config.get("locations", ["cpu", "mem", "disk", "diskio", "net", "system"])
        
        for location in locations:
            query = f'''
            from(bucket: "{bucket}")
              |> range(start: -1d)
              |> filter(fn: (r) => r["location"] == "{location}")
              |> limit(n: 5)
            '''
            try:
                loc_result = query_api.query_data_frame(query)
                
                if isinstance(loc_result, list):
                    if not loc_result:
                        logger.info(f"위치 '{location}'에 대한 데이터가 없습니다.")
                        continue
                    loc_result = pd.concat(loc_result)
                
                logger.info(f"\n--- 위치: {location} ---")
                logger.info(f"결과 크기: {loc_result.shape}")
                logger.info(f"컬럼 목록: {loc_result.columns.tolist()}")
                
                if '_field' in loc_result.columns:
                    fields = loc_result['_field'].unique()
                    logger.info(f"필드 목록: {fields}")
                
                if '_measurement' in loc_result.columns:
                    measurements = loc_result['_measurement'].unique()
                    logger.info(f"측정값 목록: {measurements}")
                
                # 샘플 데이터 출력
                logger.info(f"샘플 데이터:\n{loc_result.head(2)}")
                
            except Exception as e:
                logger.error(f"위치 '{location}' 쿼리 실패: {e}")
        
        # 원본 데이터의 필드 구조 확인을 위한 피벗 전 데이터 확인
        logger.info("\n=== CPU 데이터 상세 확인 (피벗 없음) ===")
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["location"] == "cpu")
          |> limit(n: 10)
        '''
        cpu_result = query_api.query_data_frame(query)
        
        if isinstance(cpu_result, list) and cpu_result:
            cpu_result = pd.concat(cpu_result)
            logger.info(f"CPU 데이터 샘플 (피벗 없음):\n{cpu_result.head()}")
            
            # 필드별 고유 값 확인
            for field in ['_field', '_value', 'location', 'origin']:
                if field in cpu_result.columns:
                    unique_values = cpu_result[field].unique()
                    logger.info(f"필드 '{field}'의 고유 값: {unique_values}")
        
        # CPU 데이터에 대한 피벗 쿼리 실행
        logger.info("\n=== CPU 데이터 피벗 확인 ===")
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["location"] == "cpu")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> limit(n: 10)
        '''
        cpu_pivot = query_api.query_data_frame(query)
        
        if isinstance(cpu_pivot, list) and cpu_pivot:
            cpu_pivot = pd.concat(cpu_pivot)
            logger.info(f"CPU 데이터 샘플 (피벗 후):\n{cpu_pivot.head()}")
            logger.info(f"피벗 후 컬럼 목록: {cpu_pivot.columns.tolist()}")
        
        # 연결 해제
        client.close()
        logger.info("InfluxDB 탐색 완료")
        
    except Exception as e:
        logger.error(f"InfluxDB 탐색 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    explore_influxdb()
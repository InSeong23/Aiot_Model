#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InfluxDB 연결 테스트 스크립트

InfluxDB에 간단한 쿼리를 실행하여 연결을 테스트합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """설정 파일 로드"""
    config_path = 'config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"설정 파일 로드: {config_path}")
                return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}
    else:
        logger.error(f"설정 파일이 존재하지 않습니다: {config_path}")
        return {}

def test_influxdb_connection():
    """InfluxDB 연결 테스트"""
    # 설정 로드
    config = load_config()
    if not config:
        return False
    
    # InfluxDB 설정 추출
    influx_config = config.get('influxdb', {})
    token = influx_config.get('token', '')
    url = influx_config.get('url', '')
    org = influx_config.get('org', '')
    bucket = influx_config.get('bucket', '')
    
    if not token or not url or not org or not bucket:
        logger.error("InfluxDB 설정이 불완전합니다.")
        return False
    
    try:
        # InfluxDB 클라이언트 생성
        client = InfluxDBClient(
            url=url,
            token=token,
            org=org,
            timeout=30000,  # 타임아웃 30초
            verify_ssl=False  # SSL 인증서 검증 비활성화
        )
        
        # 연결 상태 확인
        health = client.health()
        logger.info(f"InfluxDB 상태: {health.status}")
        logger.info(f"InfluxDB 메시지: {health.message}")
        
        # 간단한 쿼리 실행
        query_api = client.query_api()
        
        # 최근 1시간 데이터 쿼리
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # CPU 데이터 쿼리
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: {start_time_str}, stop: {end_time_str})
          |> filter(fn: (r) => r["location"] == "cpu")
          |> limit(n: 5)
        '''
        
        logger.info("CPU 데이터 쿼리 실행...")
        result = query_api.query(query)
        
        if result:
            logger.info("쿼리 결과 확인:")
            for table in result:
                for record in table.records:
                    logger.info(f" - {record}")
        else:
            logger.warning("쿼리 결과가 비어있습니다.")
        
        # 연결 해제
        client.close()
        logger.info("InfluxDB 연결 테스트 완료")
        return True
    
    except Exception as e:
        logger.error(f"InfluxDB 연결 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_influxdb_connection()
    sys.exit(0 if success else 1)
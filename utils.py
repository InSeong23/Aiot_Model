#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 함수 모듈

데이터 수집 및 전처리를 위한 유틸리티 함수들을 제공합니다.
"""

import os
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_last_collection_time_from_file(resource_type='all'):
    """
    파일에서 마지막 데이터 수집 시간 가져오기
    
    Args:
        resource_type (str): 자원 유형 (all, cpu, mem, disk, diskio, net, system)
        
    Returns:
        datetime: 마지막 수집 시간
    """
    try:
        # 파일 이름 설정
        filename = f'last_collection_time_{resource_type}.txt'
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                time_str = f.read().strip()
                return datetime.fromisoformat(time_str)
        
        return None
    except Exception as e:
        logger.error(f"마지막 수집 시간 파일 읽기 실패: {e}")
        return None

def update_last_collection_time_to_file(time, resource_type='all'):
    """
    파일에 마지막 데이터 수집 시간 저장
    
    Args:
        time (datetime): 저장할 시간
        resource_type (str): 자원 유형 (all, cpu, mem, disk, diskio, net, system)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 파일 이름 설정
        filename = f'last_collection_time_{resource_type}.txt'
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # 시간 저장
        with open(filename, 'w') as f:
            f.write(time.isoformat())
        
        return True
    except Exception as e:
        logger.error(f"마지막 수집 시간 파일 저장 실패: {e}")
        return False

def get_default_start_time(config, days=7):
    """
    기본 시작 시간 가져오기
    
    Args:
        config (dict): 설정 정보
        days (int): 기본 일 수
        
    Returns:
        datetime: 시작 시간
    """
    max_days = config.get("data_collection", {}).get("max_days_initial", days)
    return datetime.now() - timedelta(days=max_days)

def parse_datetime(date_str):
    """
    날짜/시간 문자열 파싱
    
    Args:
        date_str (str): 날짜/시간 문자열
        
    Returns:
        datetime: 파싱된 날짜/시간
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d',
        '%Y%m%d',
        '%Y%m%d%H%M%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"날짜/시간 형식을 인식할 수 없습니다: {date_str}")

def is_table_created(conn, table_name):
    """
    테이블 생성 여부 확인
    
    Args:
        conn: 데이터베이스 연결 객체
        table_name (str): 테이블 이름
        
    Returns:
        bool: 테이블 존재 여부
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        result = cursor.fetchone()
        cursor.close()
        
        return result is not None
    except Exception as e:
        logger.error(f"테이블 확인 실패: {e}")
        return False

def generate_device_id():
    """
    고유한 장치 ID 생성
    
    Returns:
        str: 장치 ID
    """
    import uuid
    return f"device_{uuid.uuid4().hex[:8]}"

def get_column_type(value):
    """
    값에 따른 SQL 컬럼 타입 가져오기
    
    Args:
        value: 값
        
    Returns:
        str: SQL 컬럼 타입
    """
    if isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, int):
        return "INT"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, datetime):
        return "DATETIME"
    else:
        # 기본값: VARCHAR(255)
        return "VARCHAR(255)"
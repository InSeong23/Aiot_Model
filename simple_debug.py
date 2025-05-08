#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
간소화된 디버깅 스크립트
"""

import json
import pandas as pd
import numpy as np
import logging
import mysql.connector
from datetime import datetime
from influxdb_client import InfluxDBClient

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정 로드
with open('config.json', 'r') as f:
    config = json.load(f)

# InfluxDB 연결
influx_config = config.get("influxdb", {})
client = InfluxDBClient(
    url=influx_config.get("url"),
    token=influx_config.get("token"),
    org=influx_config.get("org")
)

# MySQL 연결
mysql_config = config.get("mysql", {})
mysql_conn = mysql.connector.connect(
    host=mysql_config.get('host'),
    port=mysql_config.get('port'),
    user=mysql_config.get('user'),
    password=mysql_config.get('password'),
    database=mysql_config.get('database')
)
mysql_cursor = mysql_conn.cursor()

# InfluxDB에서 데이터 가져와서 값만 출력
def check_influx_values():
    query_api = client.query_api()
    bucket = influx_config.get("bucket")
    
    # 5월 7일 데이터만 가져오기
    query = '''
    from(bucket: "data")
    |> range(start: 2025-05-07T00:00:00Z, stop: 2025-05-08T00:00:00Z)
    |> filter(fn: (r) => r.location == "cpu")
    |> limit(n: 10)
    '''
    
    result = query_api.query_data_frame(query)
    
    if isinstance(result, list):
        result = pd.concat(result) if result else pd.DataFrame()
    
    print("InfluxDB 데이터 샘플:")
    print(result.head())
    
    # _value 열 확인
    if '_value' in result.columns:
        print("\n_value 열 값:")
        for i, val in enumerate(result['_value'].head(10)):
            print(f"{i}: {val} (타입: {type(val)})")
    
    return result

# 테스트 삽입
def test_simple_insert():
    """단순 삽입 테스트"""
    try:
        # 테스트 데이터 삭제
        mysql_cursor.execute("DELETE FROM resource_metrics WHERE resource_type = 'cpu'")
        mysql_conn.commit()
        print("기존 데이터 삭제 완료")
        
        # 테스트 데이터
        test_data = [
            ('2025-05-07 01:00:00', '192.168.71.74', 'cpu', 'cpu_usage', 24.99, '%', 'javame', 'gyeongnam_campus'),
            ('2025-05-07 02:00:00', '192.168.71.74', 'cpu', 'cpu_usage', 25.01, '%', 'javame', 'gyeongnam_campus'),
            ('2025-05-07 03:00:00', '192.168.71.74', 'cpu', 'cpu_usage', 24.97, '%', 'javame', 'gyeongnam_campus')
        ]
        
        # 테스트 데이터 삽입
        sql = """
        INSERT INTO resource_metrics 
        (timestamp, device_id, resource_type, metric_name, metric_value, unit, companyDomain, building)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for data in test_data:
            mysql_cursor.execute(sql, data)
            mysql_conn.commit()
        
        print("테스트 데이터 삽입 성공")
        
        # 확인
        mysql_cursor.execute("SELECT * FROM resource_metrics WHERE resource_type = 'cpu'")
        rows = mysql_cursor.fetchall()
        print("\nMySQL 테스트 데이터:")
        for row in rows:
            print(row)
    
    except Exception as e:
        print(f"오류 발생: {e}")

# 실제 데이터로 테스트
def test_with_real_data(df):
    if df.empty:
        print("실제 데이터가 비어있습니다")
        return
    
    try:
        # 테스트 데이터 삭제
        mysql_cursor.execute("DELETE FROM resource_metrics WHERE resource_type = 'cpu'")
        mysql_conn.commit()
        print("기존 데이터 삭제 완료")
        
        # 기본 설정
        device_id = "192.168.71.74"
        resource_type = "cpu"
        company_domain = "javame"
        building = "gyeongnam_campus"
        
        records = []
        
        # _value 열이 있는지 확인
        if '_value' in df.columns:
            print("_value 열 사용")
            
            for idx, row in df.iterrows():
                # 타임스탬프 처리
                if isinstance(idx, pd.Timestamp):
                    time_str = idx.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = "2025-05-07 12:00:00"  # 기본값
                
                # 값 확인
                try:
                    value = float(row['_value'])
                    records.append((
                        time_str,
                        device_id,
                        resource_type,
                        'cpu_usage',
                        value,
                        '%',
                        company_domain,
                        building
                    ))
                    print(f"추가된 레코드: 시간={time_str}, 값={value}")
                except (ValueError, TypeError) as e:
                    print(f"값 변환 실패: {row['_value']}, 오류: {e}")
        
        # 테스트 데이터 삽입
        if records:
            sql = """
            INSERT INTO resource_metrics 
            (timestamp, device_id, resource_type, metric_name, metric_value, unit, companyDomain, building)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for record in records:
                try:
                    mysql_cursor.execute(sql, record)
                    mysql_conn.commit()
                    print(f"레코드 삽입 성공: {record}")
                except Exception as e:
                    print(f"레코드 삽입 실패: {record}, 오류: {e}")
            
            # 확인
            mysql_cursor.execute("SELECT * FROM resource_metrics WHERE resource_type = 'cpu'")
            rows = mysql_cursor.fetchall()
            print("\nMySQL에 저장된 실제 데이터:")
            for row in rows:
                print(row)
        else:
            print("저장할 레코드가 없습니다")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        print(traceback.format_exc())

# 데이터베이스 테이블 구조 확인
def check_table_structure():
    try:
        # 테이블 구조 확인
        mysql_cursor.execute("DESCRIBE resource_metrics")
        columns = mysql_cursor.fetchall()
        
        print("\nMySQL 테이블 구조:")
        for column in columns:
            print(column)
        
        # 제약 조건 확인
        mysql_cursor.execute("""
        SELECT COLUMN_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME = 'resource_metrics' AND TABLE_SCHEMA = %s
        """, (mysql_config.get('database'),))
        
        constraints = mysql_cursor.fetchall()
        print("\n제약 조건:")
        for constraint in constraints:
            print(constraint)
            
    except Exception as e:
        print(f"테이블 구조 확인 중 오류 발생: {e}")

if __name__ == "__main__":
    try:
        print("===== InfluxDB 데이터 확인 =====")
        df = check_influx_values()
        
        print("\n===== 단순 삽입 테스트 =====")
        test_simple_insert()
        
        print("\n===== 테이블 구조 확인 =====")
        check_table_structure()
        
        print("\n===== 실제 데이터로 테스트 =====")
        test_with_real_data(df)
    finally:
        # 연결 종료
        mysql_cursor.close()
        mysql_conn.close()
        client.close()
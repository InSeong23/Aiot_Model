#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfluxDB에서 MySQL로 데이터 직접 저장 테스트
"""

import json
import pandas as pd
import numpy as np
import logging
import mysql.connector
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정 로드
with open('config.json', 'r') as f:
    config = json.load(f)

# InfluxDB 연결
influx_config = config.get("influxdb", {})
influx_client = InfluxDBClient(
    url=influx_config.get("url"),
    token=influx_config.get("token"),
    org=influx_config.get("org"),
    timeout=30000
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

# 1. InfluxDB에서 데이터 가져오기
def get_influx_data():
    # 날짜 범위 설정
    end_time = datetime.now()
    start_time = datetime(2025, 5, 7)  # 5월 7일 데이터만 가져오기
    
    query_api = influx_client.query_api()
    bucket = influx_config.get("bucket")
    
    query = f'''
    from(bucket: "{bucket}")
    |> range(start: {start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}, stop: {end_time.strftime('%Y-%m-%dT%H:%M:%SZ')})
    |> filter(fn: (r) => r.location == "cpu")
    '''
    
    logger.info("InfluxDB 쿼리 실행 중...")
    result = query_api.query_data_frame(query)
    
    if isinstance(result, list):
        result = pd.concat(result) if result else pd.DataFrame()
    
    if '_time' in result.columns:
        result = result.set_index('_time')
    
    logger.info(f"가져온 데이터: {len(result)}행, 열: {result.columns.tolist()}")
    logger.info(f"데이터 샘플:\n{result.head()}")
    
    # 데이터 타입 확인
    logger.info(f"데이터 타입:\n{result.dtypes}")
    
    # _value 열 값 확인
    if '_value' in result.columns:
        logger.info(f"_value 열 통계: 평균={result['_value'].mean()}, 최소={result['_value'].min()}, 최대={result['_value'].max()}")
        logger.info(f"_value 열 샘플: {result['_value'].head(10).tolist()}")
    
    return result

# 2. MySQL에 직접 저장
def save_to_mysql(df, resource_type='cpu'):
    if df.empty:
        logger.error("저장할 데이터가 없습니다")
        return
    
    # 기본 설정
    device_id = "192.168.71.74"
    company_domain = "javame"
    building = "gyeongnam_campus"
    
    # 먼저 기존 데이터 삭제
    mysql_cursor.execute("DELETE FROM resource_metrics WHERE resource_type = 'cpu' AND DATE(timestamp) = '2025-05-07'")
    mysql_conn.commit()
    logger.info("기존 데이터 삭제 완료")
    
    # 데이터 저장용 레코드 생성
    records = []
    
    # _value 열이 있는지 확인
    if '_value' in df.columns:
        logger.info("_value 열이 있습니다. 이 값을 사용해 저장합니다.")
        
        for idx, row in df.iterrows():
            # 타임스탬프 처리
            if isinstance(idx, pd.Timestamp):
                time_str = idx.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 값 확인 및 로깅
            value = row['_value']
            logger.info(f"저장 중: 시간={time_str}, 값={value}, 타입={type(value)}")
            
            # 유효한 값인지 확인
            if pd.isna(value) or np.isinf(value):
                logger.warning(f"유효하지 않은 값: {value}")
                continue
            
            # 값이 문자열이면 변환 시도
            if isinstance(value, str):
                try:
                    value = float(value)
                    logger.info(f"문자열을 숫자로 변환: {value}")
                except ValueError:
                    logger.warning(f"숫자로 변환할 수 없는 문자열: {value}")
                    continue
            
            # 레코드 추가
            records.append((
                time_str,
                device_id,
                resource_type,
                'cpu_usage',
                float(value),  # 명시적으로 float 변환
                '%',
                company_domain,
                building
            ))
    
    # 중요: executemany 대신 각 레코드를 개별적으로 삽입하고 중복 오류 처리
    insert_query = """
    INSERT INTO resource_metrics 
    (timestamp, device_id, resource_type, metric_name, metric_value, unit, companyDomain, building)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
    metric_value = VALUES(metric_value),
    unit = VALUES(unit)
    """
    
    success_count = 0
    duplicate_count = 0
    error_count = 0
    
    for record in records:
        try:
            mysql_cursor.execute(insert_query, record)
            mysql_conn.commit()
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"진행 상황: {success_count}/{len(records)} 레코드 삽입 완료")
        except mysql.connector.errors.IntegrityError as e:
            # 중복 키 오류 처리
            if "Duplicate entry" in str(e):
                duplicate_count += 1
                # 중복 데이터 업데이트 쿼리 (선택 사항)
                try:
                    update_query = """
                    UPDATE resource_metrics 
                    SET metric_value = %s, unit = %s
                    WHERE timestamp = %s AND device_id = %s AND resource_type = %s AND metric_name = %s
                    """
                    mysql_cursor.execute(update_query, (
                        record[4], record[5], record[0], record[1], record[2], record[3]
                    ))
                    mysql_conn.commit()
                except Exception as update_error:
                    logger.warning(f"중복 데이터 업데이트 실패: {update_error}")
            else:
                error_count += 1
                logger.error(f"삽입 오류: {e}")
        except Exception as e:
            error_count += 1
            logger.error(f"기타 오류: {e}")
    
    logger.info(f"총 {len(records)}개 레코드 중 {success_count}개 성공, {duplicate_count}개 중복, {error_count}개 오류")

# 3. 저장된 데이터 확인
def check_mysql_data():
    mysql_cursor.execute("""
    SELECT DATE(timestamp) as date, HOUR(timestamp) as hour, COUNT(*) as count, 
           AVG(metric_value) as avg_value, MIN(metric_value) as min_value, MAX(metric_value) as max_value
    FROM resource_metrics 
    WHERE resource_type = 'cpu' AND DATE(timestamp) = '2025-05-07'
    GROUP BY DATE(timestamp), HOUR(timestamp)
    ORDER BY date, hour
    """)
    
    rows = mysql_cursor.fetchall()
    
    logger.info("MySQL에 저장된 데이터:")
    for row in rows:
        date, hour, count, avg, min_val, max_val = row
        logger.info(f"- {date} {hour:02d}시: {count}행, 평균={avg:.2f}, 최소={min_val:.2f}, 최대={max_val:.2f}")

# 메인 실행
try:
    # 1. InfluxDB에서 데이터 가져오기
    df = get_influx_data()
    
    # 2. MySQL에 직접 저장
    if not df.empty:
        save_to_mysql(df)
        
        # 3. 저장된 데이터 확인
        check_mysql_data()
    else:
        logger.error("InfluxDB에서 가져온 데이터가 없습니다")
    
except Exception as e:
    logger.error(f"오류 발생: {e}")
    import traceback
    logger.error(traceback.format_exc())
finally:
    # 연결 종료
    mysql_cursor.close()
    mysql_conn.close()
    influx_client.close()
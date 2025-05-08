# test_mysql.py
import mysql.connector
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 설정 로드
with open('config.json', 'r') as f:
    config = json.load(f)

mysql_config = config.get("mysql", {})

try:
    # 연결
    conn = mysql.connector.connect(
        host=mysql_config.get('host'),
        port=mysql_config.get('port'),
        user=mysql_config.get('user'),
        password=mysql_config.get('password'),
        database=mysql_config.get('database')
    )
    cursor = conn.cursor()
    
    # 테이블 구조 확인
    cursor.execute("DESCRIBE resource_metrics")
    for col in cursor.fetchall():
        logger.info(f"컬럼: {col}")
    
    # 테스트 데이터 삽입
    test_data = (
        '2025-05-08 12:00:00',
        '192.168.71.74',
        'cpu',
        'cpu_usage',
        98.25,  # 명시적 float 값
        '%',
        'javame',
        'gyeongnam_campus'
    )
    
    cursor.execute("""
    INSERT INTO resource_metrics 
    (timestamp, device_id, resource_type, metric_name, metric_value, unit, companyDomain, building)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
    metric_value = VALUES(metric_value),
    unit = VALUES(unit)
    """, test_data)
    
    # 영향 받은 행 수 확인
    logger.info(f"영향 받은 행 수: {cursor.rowcount}")
    
    # 명시적 커밋
    conn.commit()
    
    # 방금 삽입한 데이터 확인
    cursor.execute("""
    SELECT * FROM resource_metrics 
    WHERE timestamp = %s AND device_id = %s AND resource_type = %s AND metric_name = %s
    """, (test_data[0], test_data[1], test_data[2], test_data[3]))
    
    result = cursor.fetchone()
    logger.info(f"삽입된 데이터: {result}")
    
except Exception as e:
    logger.error(f"오류 발생: {e}")
finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        logger.info("MySQL 연결 종료")
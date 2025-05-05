#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
테스트 스크립트: 주요 기능 개별 테스트
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def test_influxdb_connection():
    """InfluxDB 연결 테스트"""
    logger.info("===== InfluxDB 연결 테스트 =====")
    
    config = load_config()
    
    from data_collector import DataCollector
    collector = DataCollector(config)
    
    connected = collector.connect()
    if connected:
        logger.info("InfluxDB 연결 성공")
        
        # 기본 쿼리 테스트
        query_result = collector.query_data(start_time=1)  # 1일 전부터 데이터 조회
        
        if query_result is not None and not query_result.empty:
            logger.info(f"쿼리 성공: {len(query_result)}행 반환")
            
            # 데이터 기본 정보 출력
            logger.info(f"컬럼: {query_result.columns.tolist()}")
            
            # 고유 위치 확인
            if 'location' in query_result.columns:
                locations = query_result['location'].unique()
                logger.info(f"위치 목록: {locations}")
        else:
            logger.warning("쿼리 결과가 비어있습니다.")
        
        collector.disconnect()
    else:
        logger.error("InfluxDB 연결 실패")
    
    logger.info("===== InfluxDB 연결 테스트 완료 =====")
    return connected

def test_mysql_connection():
    """MySQL 연결 테스트"""
    logger.info("===== MySQL 연결 테스트 =====")
    
    config = load_config()
    
    from result_handler import ResultHandler
    result_handler = ResultHandler(config)
    
    # 연결 테스트
    connection = result_handler.mysql_connect()
    if connection:
        logger.info("MySQL 연결 성공")
        connection.close()
        return True
    else:
        logger.error("MySQL 연결 실패")
        return False

def test_data_collection():
    """데이터 수집 테스트"""
    logger.info("===== 데이터 수집 테스트 =====")
    
    config = load_config()
    
    from result_handler import ResultHandler
    result_handler = ResultHandler(config)
    
    from data_collector import DataCollector
    collector = DataCollector(config)
    collector.set_result_handler(result_handler)
    
    # 데이터 수집 테스트
    connected = collector.connect()
    if not connected:
        logger.error("InfluxDB 연결 실패")
        return False
    
    try:
        # CPU 데이터 수집 테스트
        logger.info("CPU 데이터 수집 테스트")
        cpu_df = collector.get_resource_data("cpu", start_time=datetime.now() - timedelta(days=1))
        
        if cpu_df is not None and not cpu_df.empty:
            logger.info(f"CPU 데이터 수집 성공: {len(cpu_df)}행")
            
            # 데이터 전처리
            from data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(config)
            preprocessor.set_result_handler(result_handler)
            
            logger.info("CPU 데이터 전처리 시작")
            processed_df = preprocessor.process_resource_data(cpu_df, "cpu")
            
            if processed_df is not None and not processed_df.empty:
                logger.info(f"CPU 데이터 전처리 성공: {len(processed_df)}행")
                
                # 데이터 저장 (EAV 모델)
                save_success = result_handler.save_resource_metrics(processed_df, "cpu")
                logger.info(f"CPU 데이터 저장: {'성공' if save_success else '실패'}")
            else:
                logger.warning("CPU 데이터 전처리 실패")
        else:
            logger.warning("CPU 데이터 수집 실패")
        
        collector.disconnect()
        return True
    except Exception as e:
        logger.error(f"데이터 수집 테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        collector.disconnect()
        return False

def test_resource_prediction():
    """자원 사용량 예측 테스트"""
    logger.info("===== 자원 사용량 예측 테스트 =====")
    
    config = load_config()
    
    from result_handler import ResultHandler
    result_handler = ResultHandler(config)
    
    from data_collector import DataCollector
    collector = DataCollector(config)
    collector.set_result_handler(result_handler)
    
    # 자원 사용량 예측 테스트
    try:
        # 예측 데이터 로드
        logger.info("CPU 예측 데이터 로드")
        df = collector.get_prediction_data("cpu", days=7, use_mysql=True)
        
        if df is not None and not df.empty:
            logger.info(f"CPU 예측 데이터 로드 성공: {len(df)}행")
            
            # 예측 실행
            from predictor import Predictor
            predictor = Predictor(config)
            predictor.set_result_handler(result_handler)
            
            logger.info("CPU 자원 사용량 예측 시작")
            pred_horizon = config.get("prediction", {}).get("resource", {}).get("pred_horizon", 24)
            predictions_df = predictor.predict_resource_usage(df, "cpu", horizon=pred_horizon)
            
            if predictions_df is not None and not predictions_df.empty:
                logger.info(f"CPU 자원 사용량 예측 성공: {len(predictions_df)}행")
                
                # 예측 결과 처리
                success = result_handler.process_resource_predictions(predictions_df, horizon='short_term')
                logger.info(f"CPU 자원 사용량 예측 결과 처리: {'성공' if success else '실패'}")
            else:
                logger.warning("CPU 자원 사용량 예측 실패")
        else:
            logger.warning("CPU 예측 데이터 로드 실패")
        
        return True
    except Exception as e:
        logger.error(f"자원 사용량 예측 테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_failure_prediction():
    """고장 예측 테스트"""
    logger.info("===== 고장 예측 테스트 =====")
    
    config = load_config()
    
    from result_handler import ResultHandler
    result_handler = ResultHandler(config)
    
    from data_collector import DataCollector
    collector = DataCollector(config)
    collector.set_result_handler(result_handler)
    
    # 고장 예측 테스트
    try:
        # 예측 데이터 로드
        logger.info("System 예측 데이터 로드")
        df = collector.get_prediction_data("system", days=7, use_mysql=True)
        
        if df is not None and not df.empty:
            logger.info(f"System 예측 데이터 로드 성공: {len(df)}행")
            
            # 예측 실행
            from predictor import Predictor
            predictor = Predictor(config)
            predictor.set_result_handler(result_handler)
            
            logger.info("System 고장 예측 시작")
            predictions_df = predictor.predict_failures(df, "system")
            
            if predictions_df is not None and not predictions_df.empty:
                logger.info(f"System 고장 예측 성공: {len(predictions_df)}행")
                
                # 예측 결과 처리
                success = result_handler.process_failure_predictions(predictions_df)
                logger.info(f"System 고장 예측 결과 처리: {'성공' if success else '실패'}")
            else:
                logger.warning("System 고장 예측 실패")
        else:
            logger.warning("System 예측 데이터 로드 실패")
        
        return True
    except Exception as e:
        logger.error(f"고장 예측 테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """테스트 메인 함수"""
    logger.info("===== IoT 예측 시스템 테스트 시작 =====")
    
    # InfluxDB 연결 테스트
    influx_ok = test_influxdb_connection()
    
    # MySQL 연결 테스트
    mysql_ok = test_mysql_connection()
    
    if not influx_ok or not mysql_ok:
        logger.error("기본 연결 테스트 실패. 테스트를 중단합니다.")
        return 1
    
    # 데이터 수집 테스트
    collection_ok = test_data_collection()
    
    if not collection_ok:
        logger.warning("데이터 수집 테스트 실패.")
    
    # 자원 사용량 예측 테스트
    resource_ok = test_resource_prediction()
    
    # 고장 예측 테스트
    failure_ok = test_failure_prediction()
    
    logger.info("===== IoT 예측 시스템 테스트 완료 =====")
    logger.info(f"- InfluxDB 연결: {'성공' if influx_ok else '실패'}")
    logger.info(f"- MySQL 연결: {'성공' if mysql_ok else '실패'}")
    logger.info(f"- 데이터 수집: {'성공' if collection_ok else '실패'}")
    logger.info(f"- 자원 사용량 예측: {'성공' if resource_ok else '실패'}")
    logger.info(f"- 고장 예측: {'성공' if failure_ok else '실패'}")
    
    if all([influx_ok, mysql_ok, collection_ok, resource_ok, failure_ok]):
        logger.info("모든 테스트가 성공적으로 완료되었습니다.")
        return 0
    else:
        logger.warning("일부 테스트가 실패했습니다.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.critical(f"테스트 중 예상치 못한 오류 발생: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
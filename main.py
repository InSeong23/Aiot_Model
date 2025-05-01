#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IoT 예측 시스템 메인 스크립트 (개선)

InfluxDB에서 데이터를 수집하고 고장 및 자원 사용량을 예측하여 
결과를 MySQL에 저장하고 API로 전송합니다.
"""

import os
import sys

# GPU 사용 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import signal
import logging
import logging.handlers
import traceback
from datetime import datetime, timedelta
import argparse
import schedule

# 내부 모듈
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from predictor import Predictor
from result_handler import ResultHandler

# 기본 설정
CONFIG_PATH = 'config.json'
LOG_FILE = 'logs/prediction.log'

# 전역 변수
logger = None
config = None
collector = None
preprocessor = None
predictor = None
result_handler = None

def setup_logging(log_level=logging.INFO):
    """
    로깅 설정
    
    Args:
        log_level (int): 로그 레벨
        
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 핸들러 설정
    handlers = []
    
    # 파일 핸들러
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    handlers.append(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    handlers.append(console_handler)
    
    # 로거 설정
    logging.basicConfig(
        level=log_level,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def load_config(config_path=CONFIG_PATH):
    """
    설정 파일 로드
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        dict: 설정 정보
    """
    # 기본 설정
    default_config = {
        "influxdb": {
            "url": os.environ.get('INFLUXDB_URL', 'http://influxdb.javame.live'),
            "token": os.environ.get('INFLUXDB_TOKEN', ''),
            "org": os.environ.get('INFLUXDB_ORG', 'javame'),
            "bucket": os.environ.get('INFLUXDB_BUCKET', 'data'),
            "origins": ["server_data", "sensor_data"],
            "locations": ["cpu", "disk", "diskio", "mem", "net", "sensors", "system"]
        },
        "mysql": {
            "host": os.environ.get('MYSQL_HOST', 's4.java21.net'),
            "port": int(os.environ.get('MYSQL_PORT', 13306)),
            "user": os.environ.get('MYSQL_USER', ''),
            "password": os.environ.get('MYSQL_PASSWORD', ''),
            "database": os.environ.get('MYSQL_DATABASE', '')
        },
        "prediction": {
            "failure": {
                "enabled": True,
                "interval_hours": int(os.environ.get('FAILURE_INTERVAL', 6)),
                "threshold": 10.0,
                "input_window": 24,
                "target_column": "usage_idle"
            },
            "resource": {
                "enabled": True,
                "interval_hours": int(os.environ.get('RESOURCE_INTERVAL', 12)),
                "capacity_threshold": 85,
                "input_window": 48,
                "pred_horizon": 24,
                "target_column": "used_percent"
            },
            "retraining_days": 7,  # 일주일에 한 번 모델 재학습
            "visualization": False  # 시각화 비활성화 (서버용)
        },
        "resources": {
            "cpu": {
                "target_columns": ["usage_user", "usage_system", "usage_idle"],
                "prediction_type": ["failure", "resource"]
            },
            "mem": {
                "target_columns": ["used_percent", "available_percent"],
                "prediction_type": ["failure", "resource"]
            },
            "disk": {
                "target_columns": ["used_percent"],
                "prediction_type": ["failure", "resource"]
            },
            "diskio": {
                "target_columns": ["io_time", "read_bytes", "write_bytes"],
                "prediction_type": ["resource"]
            },
            "net": {
                "target_columns": ["bytes_recv", "bytes_sent", "drop_in", "drop_out", "err_in", "err_out"],
                "prediction_type": ["resource"]
            },
            "system": {
                "target_columns": ["load1", "load5", "load15"],
                "prediction_type": ["failure"]
            }
        },
        "resources_limits": {
            "network": {
                "max_bandwidth": 125000000  # 1Gbps = 125_000_000 bytes/sec
            },
            "disk": {
                "max_io_rate": 100000000  # 100 MB/sec
            }
        },
        "api": {
            "url": os.environ.get('API_URL', 'http://localhost:10272/ai/data')
        },
        "data_processing": {
            "default_values": {
                "device_id": "device_001",
                "companyDomain": "javame",
                "building": "gyeongnam_campus",
                "resource_type": "value"
            }
        },
        "advanced": {
            "resampling": {
                "enabled": True,
                "freq": "5min"
            },
            "missing_data": {
                "handle_missing": True,
                "max_missing_ratio": 0.5
            },
            "outliers": {
                "handle_outliers": True,
                "method": "zscore",
                "threshold": 3.0
            },
            "save_csv": False,
            "log_level": os.environ.get('LOG_LEVEL', 'INFO')
        }
    }
    
    # 설정 파일 로드
    loaded_config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                print(f"설정 파일 로드: {config_path}")
        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
    
    # 설정 병합 (로드된 설정이 우선)
    final_config = deep_update(default_config.copy(), loaded_config)
    
    # 환경 변수 적용 (환경 변수가 우선)
    if os.environ.get('INFLUXDB_TOKEN'):
        final_config["influxdb"]["token"] = os.environ.get('INFLUXDB_TOKEN')
    if os.environ.get('MYSQL_HOST'):
        final_config["mysql"]["host"] = os.environ.get('MYSQL_HOST')
    if os.environ.get('MYSQL_PORT'):
        final_config["mysql"]["port"] = int(os.environ.get('MYSQL_PORT'))
    if os.environ.get('MYSQL_USER'):
        final_config["mysql"]["user"] = os.environ.get('MYSQL_USER')
    if os.environ.get('MYSQL_PASSWORD'):
        final_config["mysql"]["password"] = os.environ.get('MYSQL_PASSWORD')
    if os.environ.get('MYSQL_DATABASE'):
        final_config["mysql"]["database"] = os.environ.get('MYSQL_DATABASE')
    if os.environ.get('API_URL'):
        final_config["api"]["url"] = os.environ.get('API_URL')
    
    # 설정 저장 (환경 변수 적용 후)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"설정 파일 저장 실패: {e}")
    
    return final_config

def deep_update(d, u):
    """
    딕셔너리 깊은 업데이트
    
    Args:
        d (dict): 대상 딕셔너리
        u (dict): 업데이트할 딕셔너리
    
    Returns:
        dict: 업데이트된 딕셔너리
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def init_components():
    """
    구성 요소 초기화
    
    Returns:
        tuple: (collector, preprocessor, predictor, result_handler)
    """
    global config, collector, preprocessor, predictor, result_handler
    
    # 결과 핸들러 초기화
    result_handler = ResultHandler(config)
    
    # MySQL 테이블 초기화
    result_handler.init_tables()
    
    # 데이터 수집기 초기화
    collector = DataCollector(config)
    collector.set_result_handler(result_handler)
    
    # 데이터 전처리기 초기화
    preprocessor = DataPreprocessor(config)
    
    # 예측기 초기화
    predictor = Predictor(config)
    predictor.set_result_handler(result_handler)
    
    return collector, preprocessor, predictor, result_handler

def run_data_collection():
    """
    데이터 수집 및 전처리 작업 실행
    
    Returns:
        bool: 성공 여부
    """
    logger.info("===== 데이터 수집 및 전처리 작업 시작 =====")
    
    # 실행 시작 시간
    start_time = datetime.now()
    
    try:
        # 최근 7일간의 데이터 수집 및 전처리
        processed_data = collector.collect_all_resources(days=7)
        
        if processed_data:
            resources_count = len(processed_data)
            logger.info(f"데이터 수집 및 전처리 완료: {resources_count}개 자원")
            
            # 각 자원별 데이터 행 수 로깅
            for resource_type, df in processed_data.items():
                logger.info(f"- {resource_type}: {len(df)}행")
            
            logger.info("===== 데이터 수집 및 전처리 작업 완료 =====")
            return True
        else:
            logger.error("데이터 수집 및 전처리 실패")
            logger.info("===== 데이터 수집 및 전처리 작업 실패 =====")
            return False
    
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        logger.info("===== 데이터 수집 및 전처리 작업 실패 =====")
        return False

def run_resource_prediction():
    """
    자원 사용량 예측 작업 실행
    
    Returns:
        bool: 성공 여부
    """
    logger.info("===== 자원 사용량 예측 작업 시작 =====")
    
    # 실행 시작 시간
    start_time = datetime.now()
    locations_processed = []
    resource_count = 0
    overall_status = "success"
    error_messages = []
    
    try:
        # 예측할 자원 유형 목록
        resources_config = config.get("resources", {})
        resources_to_predict = []
        
        for resource_type, resource_conf in resources_config.items():
            prediction_types = resource_conf.get("prediction_type", [])
            if isinstance(prediction_types, str):
                prediction_types = [prediction_types]
                
            if "resource" in prediction_types:
                resources_to_predict.append(resource_type)
        
        if not resources_to_predict:
            logger.warning("예측할 자원이 없습니다.")
            return False
        
        # 각 자원에 대해 예측 수행
        for resource_type in resources_to_predict:
            logger.info(f"===== {resource_type} 자원 사용량 예측 시작 =====")
            locations_processed.append(resource_type)
            
            try:
                # 예측 데이터 가져오기
                df = collector.get_prediction_data(resource_type, days=30, use_mysql=True)
                
                if df is None or df.empty:
                    logger.error(f"{resource_type} 예측 데이터 가져오기 실패")
                    error_messages.append(f"{resource_type} 데이터 가져오기 실패")
                    overall_status = "partial" if overall_status != "failed" else "failed"
                    continue
                
                # 예측 기간 설정
                pred_horizon = config.get("prediction", {}).get("resource", {}).get("pred_horizon", 24)
                
                # 자원 사용량 예측
                predictions_df = predictor.predict_resource_usage(df, resource_type, horizon=pred_horizon)
                
                if predictions_df is None or predictions_df.empty:
                    logger.error(f"{resource_type} 자원 사용량 예측 실패")
                    error_messages.append(f"{resource_type} 예측 실패")
                    overall_status = "partial" if overall_status != "failed" else "failed"
                    continue
                
                # 예측 결과 처리 (MySQL 저장 및 API 전송)
                success = result_handler.process_resource_predictions(predictions_df, horizon='short_term')
                
                if success:
                    logger.info(f"{resource_type} 자원 사용량 예측 결과 처리 완료")
                    resource_count += 1
                else:
                    logger.warning(f"{resource_type} 자원 사용량 예측 결과 처리 실패")
                    overall_status = "partial" if overall_status != "failed" else "failed"
                
            except Exception as e:
                logger.error(f"{resource_type} 자원 사용량 예측 중 오류 발생: {e}")
                logger.error(traceback.format_exc())
                error_messages.append(f"{resource_type}: {str(e)}")
                overall_status = "partial" if overall_status != "failed" else "failed"
        
        # 예측 실행 정보 기록
        error_message = "; ".join(error_messages) if error_messages else None
        result_handler.record_prediction_run(
            prediction_type="resource",
            start_time=start_time,
            locations_processed=locations_processed,
            status=overall_status,
            error_message=error_message,
            resource_count=resource_count
        )
        
        logger.info(f"===== 자원 사용량 예측 작업 완료 ({overall_status}) =====")
        return overall_status == "success"
    
    except Exception as e:
        logger.error(f"자원 사용량 예측 작업 중 예상치 못한 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return False

def run_failure_prediction():
    """
    고장 예측 작업 실행
    
    Returns:
        bool: 성공 여부
    """
    logger.info("===== 고장 예측 작업 시작 =====")
    
    # 실행 시작 시간
    start_time = datetime.now()
    locations_processed = []
    resource_count = 0
    overall_status = "success"
    error_messages = []
    
    try:
        # 예측할 자원 유형 목록
        resources_config = config.get("resources", {})
        resources_to_predict = []
        
        for resource_type, resource_conf in resources_config.items():
            prediction_types = resource_conf.get("prediction_type", [])
            if isinstance(prediction_types, str):
                prediction_types = [prediction_types]
                
            if "failure" in prediction_types:
                resources_to_predict.append(resource_type)
        
        if not resources_to_predict:
            logger.warning("고장 예측할 자원이 없습니다.")
            return False
        
        # 각 자원에 대해 예측 수행
        for resource_type in resources_to_predict:
            logger.info(f"===== {resource_type} 고장 예측 시작 =====")
            locations_processed.append(resource_type)
            
            try:
                # 예측 데이터 가져오기
                df = collector.get_prediction_data(resource_type, days=30, use_mysql=True)
                
                if df is None or df.empty:
                    logger.error(f"{resource_type} 고장 예측 데이터 가져오기 실패")
                    error_messages.append(f"{resource_type} 데이터 가져오기 실패")
                    overall_status = "partial" if overall_status != "failed" else "failed"
                    continue
                
                # 고장 예측
                predictions_df = predictor.predict_failures(df, resource_type)
                
                if predictions_df is None or predictions_df.empty:
                    logger.error(f"{resource_type} 고장 예측 실패")
                    error_messages.append(f"{resource_type} 고장 예측 실패")
                    overall_status = "partial" if overall_status != "failed" else "failed"
                    continue
                
                # 예측 결과 처리 (MySQL 저장 및 API 전송)
                success = result_handler.process_failure_predictions(predictions_df)
                
                if success:
                    logger.info(f"{resource_type} 고장 예측 결과 처리 완료")
                    resource_count += 1
                else:
                    logger.warning(f"{resource_type} 고장 예측 결과 처리 실패")
                    overall_status = "partial" if overall_status != "failed" else "failed"
                
            except Exception as e:
                logger.error(f"{resource_type} 고장 예측 중 오류 발생: {e}")
                logger.error(traceback.format_exc())
                error_messages.append(f"{resource_type}: {str(e)}")
                overall_status = "partial" if overall_status != "failed" else "failed"
        
        # 예측 실행 정보 기록
        error_message = "; ".join(error_messages) if error_messages else None
        result_handler.record_prediction_run(
            prediction_type="failure",
            start_time=start_time,
            locations_processed=locations_processed,
            status=overall_status,
            error_message=error_message,
            resource_count=resource_count
        )
        
        logger.info(f"===== 고장 예측 작업 완료 ({overall_status}) =====")
        return overall_status == "success"
    
    except Exception as e:
        logger.error(f"고장 예측 작업 중 예상치 못한 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return False

def test_connection():
    """
    연결 테스트
    
    Returns:
        bool: 성공 여부
    """
    logger.info("===== 연결 테스트 시작 =====")
    
    try:
        # InfluxDB 연결 테스트
        logger.info("InfluxDB 연결 테스트 중...")
        influx_success = collector.connect()
        if influx_success:
            logger.info("InfluxDB 연결 테스트 성공")
            collector.disconnect()
        else:
            logger.error("InfluxDB 연결 테스트 실패")
        
        # MySQL 및 API 연결 테스트
        logger.info("MySQL 및 API 연결 테스트 중...")
        mysql_success, api_success = result_handler.test_connections()
        
        if mysql_success:
            logger.info("MySQL 연결 테스트 성공")
        else:
            logger.error("MySQL 연결 테스트 실패")
        
        if api_success:
            logger.info("API 연결 테스트 성공")
        else:
            logger.warning("API 연결 테스트 실패")
        
        logger.info("===== 연결 테스트 완료 =====")
        return influx_success and mysql_success
        
    except Exception as e:
        logger.error(f"연결 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def show_next_scheduled_runs():
    """다음 예약된 작업 표시"""
    logger.info("다음 예약된 작업:")
    for job in schedule.jobs:
        next_run = job.next_run
        if next_run:
            time_diff = next_run - datetime.now()
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"- {job.job_func.__name__}: {int(hours)}시간 {int(minutes)}분 후 ({next_run.strftime('%Y-%m-%d %H:%M:%S')})")

def handle_exit(signum, frame):
    """프로세스 종료 처리"""
    logger.info("종료 신호 수신. 정리 중...")
    
    # 연결 해제
    if collector:
        collector.disconnect()
    
    sys.exit(0)

def main():
    """메인 함수"""
    global logger, config
    
    try:
        # 명령행 인자 파싱
        parser = argparse.ArgumentParser(description='IoT 예측 시스템')
        
        parser.add_argument('--mode', type=str, default='schedule', 
                            choices=['all', 'collect', 'resource', 'failure', 'schedule'],
                            help='실행 모드 (all, collect, resource, failure, schedule)')
        
        parser.add_argument('--config', type=str, default=CONFIG_PATH,
                            help='설정 파일 경로')
        
        parser.add_argument('--check-db', action='store_true',
                            help='데이터베이스 연결 테스트만 실행')
        
        args = parser.parse_args()
        
        # 초기 로깅 설정 (간단한 버전)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        
        logger.info("=== IoT 예측 시스템 시작 ===")
        logger.debug(f"명령행 인자: {args}")
        
        # 종료 신호 핸들러 등록
        signal.signal(signal.SIGTERM, handle_exit)
        signal.signal(signal.SIGINT, handle_exit)
        
        # 설정 로드
        logger.info("설정 파일 로드 중...")
        config = load_config(args.config)
        logger.debug(f"로드된 설정: {config}")
        
        # 로그 레벨 재설정
        log_level_str = config.get('advanced', {}).get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str)
        logger.setLevel(log_level)
        logger.info(f"로그 레벨 설정: {log_level_str}")
        
        # 필요한 디렉토리 생성
        logger.info("필요한 디렉토리 생성 중...")
        os.makedirs("logs", exist_ok=True)
        os.makedirs("model_weights", exist_ok=True)
        
        # 실제 로깅 설정 (파일 포함)
        logger = setup_logging(log_level)
        logger.info("IoT 예측 시스템 시작")
        
        # 컴포넌트 초기화
        logger.info("시스템 컴포넌트 초기화 중...")
        init_components()
        
        # 데이터베이스 연결 테스트
        if args.check_db:
            logger.info("데이터베이스 연결 테스트 모드")
            db_ok = test_connection()
            if db_ok:
                logger.info("데이터베이스 연결 테스트 성공")
                return 0
            else:
                logger.error("데이터베이스 연결 테스트 실패")
                return 1
        else:
            # 연결 테스트는 항상 실행
            logger.info("데이터베이스 연결 확인...")
            test_connection()
        
        # schedule 모드면 스케줄링 실행
        if args.mode == 'schedule':
            logger.info("스케줄 모드로 실행합니다.")
            
            # 데이터 수집 스케줄링 (4시간마다)
            schedule.every(4).hours.do(run_data_collection)
            logger.info("데이터 수집: 4시간마다 실행 예약됨")
            
            # 고장 예측 스케줄링
            if config["prediction"]["failure"]["enabled"]:
                interval = config["prediction"]["failure"]["interval_hours"]
                schedule.every(interval).hours.do(run_failure_prediction)
                logger.info(f"고장 예측: {interval}시간마다 실행 예약됨")
            
            # 자원 사용량 분석 스케줄링
            if config["prediction"]["resource"]["enabled"]:
                interval = config["prediction"]["resource"]["interval_hours"]
                schedule.every(interval).hours.do(run_resource_prediction)
                logger.info(f"자원 사용량 분석: {interval}시간마다 실행 예약됨")
            
            # 초기 실행
            logger.info("초기 데이터 수집 및 예측 실행 중...")
            
            run_data_collection()
            
            if config["prediction"]["failure"]["enabled"]:
                logger.info("고장 예측 실행 시작...")
                run_failure_prediction()
                
            if config["prediction"]["resource"]["enabled"]:
                logger.info("자원 사용량 예측 실행 시작...")
                run_resource_prediction()
            
            # 다음 작업 시간 표시
            show_next_scheduled_runs()
            
            # 스케줄러 실행 루프
            try:
                logger.info("스케줄러 대기 모드...")
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("사용자에 의해 중지되었습니다.")
        else:
            # 단일 실행 모드
            logger.info(f"{args.mode} 모드로 실행합니다.")
            
            if args.mode == 'all' or args.mode == 'collect':
                run_data_collection()
                
            if args.mode == 'all' or args.mode == 'failure':
                run_failure_prediction()
            
            if args.mode == 'all' or args.mode == 'resource':
                run_resource_prediction()
        
        logger.info("=== IoT 예측 시스템 종료 ===")
        return 0
    
    except Exception as e:
        if logger:
            logger.error(f"심각한 오류 발생: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"심각한 오류 발생: {e}")
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    result = main()
    sys.exit(result)
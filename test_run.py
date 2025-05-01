#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
테스트 스크립트: 데이터 수집 테스트
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import logging
from datetime import datetime, timedelta

print("시작: 데이터 수집 테스트")

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def run_collection_test():
    # 설정 로드
    config = load_config()
    print("설정 로드 완료")
    
    # 결과 핸들러 초기화
    from result_handler import ResultHandler
    result_handler = ResultHandler(config)
    print("결과 핸들러 초기화 완료")
    
    # 데이터 수집기 초기화
    from data_collector import DataCollector
    collector = DataCollector(config)
    collector.set_result_handler(result_handler)
    print("데이터 수집기 초기화 완료")
    
    # InfluxDB 연결
    influx_ok = collector.connect()
    if not influx_ok:
        print("InfluxDB 연결 실패")
        return 1
    
    try:
        # 리소스 데이터 수집 (CPU만)
        print("CPU 데이터 수집 중...")
        cpu_df = collector.get_resource_data("cpu", days=1)
        
        if cpu_df is None or cpu_df.empty:
            print("CPU 데이터 수집 실패")
        else:
            print(f"CPU 데이터 수집 성공: {len(cpu_df)}행")
            print(f"데이터 컬럼: {cpu_df.columns.tolist()}")
            print(f"샘플 데이터:\n{cpu_df.head(2)}")
            
            # 데이터 전처리
            from data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(config)
            print("데이터 전처리기 초기화 완료")
            
            print("CPU 데이터 전처리 중...")
            processed_df = preprocessor.process_resource_data(cpu_df, "cpu")
            
            if processed_df is None or processed_df.empty:
                print("CPU 데이터 전처리 실패")
            else:
                print(f"CPU 데이터 전처리 성공: {len(processed_df)}행")
                print(f"전처리 후 컬럼: {processed_df.columns.tolist()}")
                print(f"샘플 데이터:\n{processed_df.head(2)}")
                
                # MySQL에 저장
                save_ok = result_handler.save_processed_data(processed_df)
                print(f"MySQL 저장: {'성공' if save_ok else '실패'}")
    
    finally:
        # 연결 해제
        collector.disconnect()
    
    print("데이터 수집 테스트 완료")
    return 0

if __name__ == "__main__":
    try:
        result = run_collection_test()
        print(f"테스트 종료. 결과 코드: {result}")
        sys.exit(result)
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
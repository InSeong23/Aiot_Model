#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InfluxDB 예측 시스템 실행 스크립트

환경 변수를 설정하고 메인 스크립트를 실행합니다.
"""

import os
import sys
import subprocess
import json
from dotenv import load_dotenv

def main():
    """메인 함수"""
    print("===== InfluxDB 예측 시스템 실행 =====")
    
    # .env 파일 로드 (환경 변수 설정)
    load_dotenv()
    
    # config.json 파일 로드
    config_path = 'config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"설정 파일 로드: {config_path}")
        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
            return 1
    else:
        print(f"설정 파일이 존재하지 않습니다: {config_path}")
        print("기본 설정을 사용합니다.")
    
    # 환경 변수 설정
    env = os.environ.copy()
    
    # InfluxDB 설정
    if 'INFLUXDB_TOKEN' in env and env['INFLUXDB_TOKEN']:
        print("환경 변수에서 InfluxDB 토큰을 찾았습니다.")
    else:
        print("경고: InfluxDB 토큰이 설정되지 않았습니다.")
        print("config.json 파일에 토큰을 설정하거나 INFLUXDB_TOKEN 환경 변수를 설정해야 합니다.")
    
    # MySQL 설정이 있는지 확인
    mysql_config_ok = True
    for var in ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE']:
        if var not in env or not env[var]:
            mysql_config_ok = False
            break
    
    if not mysql_config_ok:
        print("경고: MySQL 설정이 완전하지 않습니다. MySQL 저장 기능이 비활성화될 수 있습니다.")
    
    # 명령행 인수를 그대로 전달
    args = ["python", "main.py"] + sys.argv[1:]
    
    print(f"실행 명령: {' '.join(args)}")
    
    # 서비스 실행
    try:
        process = subprocess.run(args, env=env)
        return process.returncode
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("1. 기본 라이브러리 임포트")
import pandas as pd
import numpy as np
print("성공: 기본 라이브러리")

print("2. 데이터베이스 클라이언트 임포트")
import influxdb_client
import mysql.connector
print("성공: 데이터베이스 클라이언트")

print("3. TensorFlow 임포트")
try:
    import tensorflow as tf
    print(f"성공: TensorFlow {tf.__version__}")
    print(f"사용 가능한 장치: {tf.config.list_physical_devices()}")
except Exception as e:
    print(f"실패: TensorFlow - {e}")

print("모든 테스트 완료")
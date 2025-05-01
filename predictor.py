#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
예측 기능 관리 모듈 (개선)

자원 사용량 및 고장 예측 기능을 제공합니다.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta

# 로깅 설정
logger = logging.getLogger(__name__)

class Predictor:
    """
    예측 클래스
    
    데이터 전처리, 모델 생성, 학습, 예측 등의 과정을 관리합니다.
    """
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config (dict): 설정 정보
        """
        self.config = config
        self.model_dir = os.path.join(os.getcwd(), 'model_weights')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 자원별 모델 저장
        self.models = {}
        self.result_handler = None
    
    def set_result_handler(self, result_handler):
        """
        결과 핸들러 설정
        
        Args:
            result_handler (ResultHandler): 결과 핸들러 객체
        """
        self.result_handler = result_handler
    
    def predict_resource_usage(self, df, resource_type, horizon=24, steps=None):
        """
        자원 사용량 예측
        
        Args:
            df (pd.DataFrame): 입력 데이터
            resource_type (str): 자원 유형 (cpu, mem, disk, diskio, net, system)
            horizon (int): 예측 기간 (시간)
            steps (int): 예측 단계 (None일 경우 1시간 간격)
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        if df is None or df.empty:
            logger.error("입력 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        # 자원별 타겟 지표 결정
        target_cols = self._get_target_columns(resource_type)
        
        if not target_cols:
            logger.error(f"자원 유형 '{resource_type}'에 대한 타겟 열을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        try:
            # 데이터 전처리
            from data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(self.config)
            
            # 예측 단계 설정 (1시간 간격)
            if steps is None:
                # 인덱스의 시간 간격을 분석하여 1시간에 해당하는 스텝 수 계산
                if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                    time_diff = (df.index[-1] - df.index[0]) / (len(df) - 1)
                    steps = int(pd.Timedelta('1 hour') / time_diff)
                    steps = max(1, steps)  # 최소 1 스텝
                else:
                    steps = 1
            
            # 데이터 스케일링
            pred_data, scaler = preprocessor.prepare_data_for_prediction(df, target_cols, scale=True)
            
            if pred_data.empty:
                logger.error("데이터 준비 실패")
                return pd.DataFrame()
            
            # 모델 생성 또는 로드
            model_key = f"{resource_type}_usage"
            
            if model_key not in self.models:
                model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
                
                if os.path.exists(model_path):
                    logger.info(f"기존 모델 로드: {model_path}")
                    try:
                        model = load_model(model_path)
                        self.models[model_key] = model
                    except Exception as e:
                        logger.warning(f"모델 로드 실패: {e}, 새 모델을 생성합니다.")
                        model = self._create_lstm_model(pred_data.shape[1], horizon)
                        self.models[model_key] = model
                else:
                    logger.info(f"새 모델 생성: {resource_type}")
                    model = self._create_lstm_model(pred_data.shape[1], horizon)
                    self.models[model_key] = model
            else:
                model = self.models[model_key]
            
            # 시퀀스 데이터 생성
            input_window = self.config.get("prediction", {}).get("resource", {}).get("input_window", 48)
            input_window = min(input_window, len(pred_data) // 2)  # 데이터 길이에 맞게 조정
            
            X, y, y_dates = preprocessor.create_sequence_dataset(
                pred_data, target_cols, input_window=input_window, pred_horizon=horizon
            )
            
            if X is None or X.shape[0] == 0:
                logger.error("시퀀스 데이터 생성 실패")
                return pd.DataFrame()
            
            # 모델 학습
            training_needed = self._check_training_needed(model_key)
            
            if training_needed:
                logger.info(f"{resource_type} 모델 학습 시작...")
                
                # 훈련/검증 세트 분할
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # 모델 저장 경로
                model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
                
                # 콜백 설정
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(filepath=model_path, save_best_only=True)
                ]
                
                # 모델 학습
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # 모델 저장
                model.save(model_path)
                logger.info(f"{resource_type} 모델 학습 완료 및 저장: {model_path}")
                
                # 모델 성능 평가
                y_pred = model.predict(X_val)
                
                # 성능 지표 계산
                if len(y_val) > 0 and len(y_pred) > 0:
                    # RMSE 및 MAE 계산 (첫 번째 타겟 열에 대해)
                    rmse = np.sqrt(mean_squared_error(y_val[:, 0, 0], y_pred[:, 0]))
                    mae = mean_absolute_error(y_val[:, 0, 0], y_pred[:, 0])
                    
                    logger.info(f"{resource_type} 모델 성능: RMSE={rmse:.2f}, MAE={mae:.2f}")
                    
                    # 성능 지표 기록
                    if self.result_handler:
                        self.result_handler.record_model_performance(
                            model_type="resource_prediction",
                            resource_type=resource_type,
                            performance_metrics={
                                'rmse': rmse,
                                'mae': mae,
                                'sample_count': len(X_val),
                                'model_version': '2.0'
                            }
                        )
            
            # 예측 수행
            logger.info(f"{resource_type} 자원 사용량 예측 시작...")
            
            # 마지막 시퀀스 준비
            last_sequence = X[-1:] if len(X) > 0 else None
            
            if last_sequence is None:
                logger.error("예측을 위한 시퀀스가 없습니다.")
                return pd.DataFrame()
            
            # 예측 수행
            predictions = model.predict(last_sequence)
            
            # 예측 시간 생성
            last_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
            
            if steps > 1:
                # 1시간 간격으로 타임스탬프 생성
                pred_times = [last_time + timedelta(hours=i+1) for i in range(horizon)]
            else:
                # 원본 데이터의 시간 간격과 동일하게 타임스탬프 생성
                if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                    time_diff = (df.index[-1] - df.index[-2])
                    pred_times = [last_time + time_diff * (i+1) for i in range(horizon)]
                else:
                    # 기본값: 1시간 간격
                    pred_times = [last_time + timedelta(hours=i+1) for i in range(horizon)]
            
            # 결과 데이터프레임 생성
            result_df = pd.DataFrame(index=pred_times)
            
            # 각 타겟 열에 대한 예측 결과 추가
            for i, col in enumerate(target_cols):
                pred_col = f"{col}_predicted"
                result_df[pred_col] = predictions[0, :, i]
            
            # 스케일링 역변환
            if scaler:
                # 예측값 열 추출
                pred_columns = result_df.columns
                
                # 역변환할 값들 준비
                values_to_inverse = result_df[pred_columns].values
                
                # 역변환
                try:
                    inversed_values = scaler.inverse_transform(values_to_inverse)
                    
                    # 역변환 결과 적용
                    for i, col in enumerate(pred_columns):
                        result_df[col] = inversed_values[:, i]
                except Exception as e:
                    logger.warning(f"스케일링 역변환 실패: {e}")
            
            # 메타데이터 추가
            device_id = self._get_device_id(df)
            company_domain = self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
            building = self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
            
            result_df['device_id'] = device_id
            result_df['companyDomain'] = company_domain
            result_df['building'] = building
            result_df['prediction_time'] = datetime.now()
            
            # 자원별 예측 결과 열 이름 매핑
            resource_col_map = {
                'cpu': 'cpu_usage_predicted',
                'mem': 'memory_used_percent_predicted',
                'disk': 'disk_used_percent_predicted',
                'diskio': 'disk_io_utilization_predicted',
                'net': 'net_utilization_predicted',
                'system': 'system_load1_predicted'
            }
            
            # 예측 열 이름이 리소스 테이블과 일치하도록 매핑
            for old_col in list(result_df.columns):
                if old_col.endswith('_predicted') and old_col not in resource_col_map.values():
                    for resource_key, resource_col in resource_col_map.items():
                        if resource_type == resource_key:
                            result_df[resource_col] = result_df[old_col]
                            if old_col != resource_col:
                                result_df.drop(old_col, axis=1, inplace=True)
                            break
            
            logger.info(f"{resource_type} 자원 사용량 예측 완료: {len(result_df)}행")
            return result_df
            
        except Exception as e:
            logger.error(f"{resource_type} 자원 사용량 예측 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def predict_failures(self, df, resource_type):
        """
        자원 고장 예측
        
        Args:
            df (pd.DataFrame): 입력 데이터
            resource_type (str): 자원 유형 (cpu, mem, disk, diskio, net, system)
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        if df is None or df.empty:
            logger.error("입력 데이터가 비어 있습니다.")
            return pd.DataFrame()
        
        # 자원별 타겟 지표 결정
        target_cols = self._get_target_columns(resource_type)
        
        if not target_cols:
            logger.error(f"자원 유형 '{resource_type}'에 대한 타겟 열을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        try:
            # 데이터 전처리
            from data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(self.config)
            
            # 타겟 열 선택
            target_col = target_cols[0]  # 첫 번째 타겟 열만 사용
            
            # 임계값 설정
            failure_threshold = self.config.get("prediction", {}).get("failure", {}).get("threshold", 10.0)
            
            # 데이터 스케일링
            pred_data, scaler = preprocessor.prepare_data_for_prediction(df, [target_col], scale=True)
            
            if pred_data.empty:
                logger.error("데이터 준비 실패")
                return pd.DataFrame()
            
            # 모델 생성 또는 로드
            model_key = f"{resource_type}_failure"
            
            if model_key not in self.models:
                model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
                
                if os.path.exists(model_path):
                    logger.info(f"기존 모델 로드: {model_path}")
                    try:
                        model = load_model(model_path)
                        self.models[model_key] = model
                    except Exception as e:
                        logger.warning(f"모델 로드 실패: {e}, 새 모델을 생성합니다.")
                        model = self._create_classification_model(pred_data.shape[1])
                        self.models[model_key] = model
                else:
                    logger.info(f"새 모델 생성: {resource_type}")
                    model = self._create_classification_model(pred_data.shape[1])
                    self.models[model_key] = model
            else:
                model = self.models[model_key]
            
            # 분류 데이터 생성
            input_window = self.config.get("prediction", {}).get("failure", {}).get("input_window", 24)
            input_window = min(input_window, len(pred_data) // 2)  # 데이터 길이에 맞게 조정
            
            X, y, y_dates = preprocessor.create_classification_dataset(
                pred_data, target_col, threshold=failure_threshold, input_window=input_window
            )
            
            if X is None or X.shape[0] == 0:
                logger.error("분류 데이터 생성 실패")
                return pd.DataFrame()
            
            # 모델 학습
            training_needed = self._check_training_needed(model_key)
            
            if training_needed:
                logger.info(f"{resource_type} 고장 예측 모델 학습 시작...")
                
                # 훈련/검증 세트 분할
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # 클래스 가중치 계산 (불균형 처리)
                class_counts = np.bincount(y_train)
                if len(class_counts) > 1 and class_counts[1] > 0:
                    class_weight = {
                        0: 1.0,
                        1: class_counts[0] / class_counts[1]  # 클래스 1의 가중치 증가
                    }
                else:
                    class_weight = None
                
                # 모델 저장 경로
                model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
                
                # 콜백 설정
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(filepath=model_path, save_best_only=True)
                ]
                
                # 모델 학습
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1
                )
                
                # 모델 저장
                model.save(model_path)
                logger.info(f"{resource_type} 고장 예측 모델 학습 완료 및 저장: {model_path}")
                
                # 모델 성능 평가
                if len(X_val) > 0 and len(y_val) > 0:
                    y_pred_prob = model.predict(X_val)
                    y_pred = (y_pred_prob > 0.5).astype(int)
                    
                    # 성능 지표 계산
                    accuracy = accuracy_score(y_val, y_pred)
                    
                    # 양성 샘플이 있는 경우에만 계산
                    if np.sum(y_val) > 0 and np.sum(y_pred) > 0:
                        precision = precision_score(y_val, y_pred)
                        recall = recall_score(y_val, y_pred)
                        f1 = f1_score(y_val, y_pred)
                    else:
                        precision = recall = f1 = 0
                    
                    logger.info(f"{resource_type} 고장 예측 모델 성능: 정확도={accuracy:.2f}, 정밀도={precision:.2f}, 재현율={recall:.2f}, F1={f1:.2f}")
                    
                    # 성능 지표 기록
                    if self.result_handler:
                        self.result_handler.record_model_performance(
                            model_type="failure_prediction",
                            resource_type=resource_type,
                            performance_metrics={
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'sample_count': len(X_val),
                                'model_version': '2.0'
                            }
                        )
            
            # 예측 수행
            logger.info(f"{resource_type} 고장 예측 시작...")
            
            # 전체 데이터셋에 대해 예측
            failure_probs = model.predict(X)
            
            # 임계값에 따른 고장 여부 결정
            failures = (failure_probs > 0.5).astype(int)
            
            # 결과 데이터프레임 생성
            result_df = pd.DataFrame()
            
            # 예측 시간 설정
            if y_dates:
                result_df['timestamp'] = y_dates
            elif isinstance(df.index, pd.DatetimeIndex):
                # 입력 시퀀스 길이를 고려하여 시간 설정
                result_df['timestamp'] = df.index[input_window:input_window + len(failure_probs)]
            else:
                # 현재 시간 기준
                now = datetime.now()
                result_df['timestamp'] = [now + timedelta(hours=i) for i in range(len(failure_probs))]
            
            # 예측 결과 추가
            resource_failure_col = f"{resource_type}_failure_probability"
            result_df[resource_failure_col] = failure_probs
            
            # 메타데이터 추가
            device_id = self._get_device_id(df)
            company_domain = self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
            building = self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
            
            result_df['device_id'] = device_id
            result_df['companyDomain'] = company_domain
            result_df['building'] = building
            result_df['prediction_time'] = datetime.now()
            
            # 자원별 고장 확률 열 이름 매핑
            resource_col_map = {
                'cpu': 'cpu_failure_probability',
                'mem': 'memory_failure_probability',
                'disk': 'disk_failure_probability',
                'net': 'network_failure_probability',
                'system': 'system_failure_probability'
            }
            
            # 예측 열 이름이 고장 테이블과 일치하도록 매핑
            if resource_type in resource_col_map:
                result_df[resource_col_map[resource_type]] = result_df[resource_failure_col]
                if resource_failure_col != resource_col_map[resource_type]:
                    result_df.drop(resource_failure_col, axis=1, inplace=True)
            
            # 통합 고장 예측 여부
            result_df['is_failure_predicted'] = (result_df[[col for col in result_df.columns if 'failure_probability' in col]] > 0.5).any(axis=1)
            
            # 주요 고장 자원
            result_df['failure_resource'] = resource_type
            
            logger.info(f"{resource_type} 고장 예측 완료: {len(result_df)}행")
            return result_df
            
        except Exception as e:
            logger.error(f"{resource_type} 고장 예측 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _create_lstm_model(self, input_dim, output_steps=24):
        """
        LSTM 기반 시계열 예측 모델 생성
        
        Args:
            input_dim (int): 입력 차원 수
            output_steps (int): 출력 스텝 수
            
        Returns:
            tf.keras.Model: LSTM 모델
        """
        model = Sequential()
        
        # 양방향 LSTM 층
        model.add(Bidirectional(
            LSTM(64, activation='tanh', return_sequences=True),
            input_shape=(None, input_dim)
        ))
        model.add(Dropout(0.2))
        
        # 두 번째 LSTM 층
        model.add(LSTM(32, activation='tanh'))
        model.add(Dropout(0.2))
        
        # 출력층
        model.add(Dense(output_steps))
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_classification_model(self, input_dim):
        """
        고장 예측을 위한 분류 모델 생성
        
        Args:
            input_dim (int): 입력 차원 수
            
        Returns:
            tf.keras.Model: 분류 모델
        """
        model = Sequential()
        
        # LSTM 층
        model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(None, input_dim)))
        model.add(Dropout(0.2))
        
        # 두 번째 LSTM 층
        model.add(LSTM(32, activation='tanh'))
        model.add(Dropout(0.2))
        
        # 출력층
        model.add(Dense(1, activation='sigmoid'))
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _check_training_needed(self, model_key):
        """
        모델 학습 필요 여부 확인
        
        Args:
            model_key (str): 모델 키
            
        Returns:
            bool: 학습 필요 여부
        """
        # 모델이 없으면 학습 필요
        if model_key not in self.models:
            return True
        
        # 모델 파일이 없으면 학습 필요
        model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
        if not os.path.exists(model_path):
            return True
        
        # 모델 재학습 주기 확인 (일주일에 한 번)
        model_stats = os.stat(model_path)
        last_modified = datetime.fromtimestamp(model_stats.st_mtime)
        days_since_modified = (datetime.now() - last_modified).days
        
        # 설정에서 재학습 주기 가져오기
        retraining_days = self.config.get("prediction", {}).get("retraining_days", 7)
        
        return days_since_modified >= retraining_days
    
    def _get_target_columns(self, resource_type):
        """
        자원 유형별 타겟 열 가져오기
        
        Args:
            resource_type (str): 자원 유형
            
        Returns:
            list: 타겟 열 목록
        """
        # 자원별 대표 지표 매핑
        resource_metrics = {
            'cpu': ['cpu_usage'],
            'mem': ['memory_used_percent'],
            'disk': ['disk_used_percent'],
            'diskio': ['disk_io_utilization', 'disk_read_rate', 'disk_write_rate'],
            'net': ['net_utilization', 'net_throughput_sent', 'net_throughput_recv'],
            'system': ['system_load1']
        }
        
        return resource_metrics.get(resource_type, [])
    
    def _get_device_id(self, df):
        """
        데이터프레임에서 장치 ID 가져오기
        
        Args:
            df (pd.DataFrame): 데이터프레임
            
        Returns:
            str: 장치 ID
        """
        if 'device_id' in df.columns:
            return df['device_id'].iloc[0]
        
        # 기본값 반환
        return self.config.get('data_processing', {}).get('default_values', {}).get('device_id', 'device_001')
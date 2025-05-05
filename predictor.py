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
        
        # 초기 모델 확인 및 필요시 학습
        self._check_models_exist()
    def _check_models_exist(self):
        """
        모델 파일 존재 여부 확인 및 필요시 초기 학습 예약
        """
        model_exists = False
        resources_config = self.config.get("resources", {})
        
        for resource_type, resource_conf in resources_config.items():
            prediction_types = resource_conf.get("prediction_type", [])
            if isinstance(prediction_types, str):
                prediction_types = [prediction_types]
            
            if "resource" in prediction_types:
                model_key = f"{resource_type}_usage"
                model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
                if os.path.exists(model_path):
                    model_exists = True
            
            if "failure" in prediction_types:
                model_key = f"{resource_type}_failure"
                model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
                if os.path.exists(model_path):
                    model_exists = True
        
        # 모델이 없으면 초기 데이터 수집 및 학습 필요성 기록
        if not model_exists:
            logger.info("모델 파일이 존재하지 않습니다. 초기 학습이 필요합니다.")
            # 초기 학습은 main.py에서 시스템 시작 시 진행
    def set_result_handler(self, result_handler):
        """
        결과 핸들러 설정
        
        Args:
            result_handler (ResultHandler): 결과 핸들러 객체
        """
        self.result_handler = result_handler
    
    def predict_resource_usage(self, df, resource_type, horizon=None, steps=None):
        """
        자원 사용량 예측 (개선됨)
        
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
        
        # 예측 주기 계산
        if horizon is None:
            # 자원 예측 주기 가져오기 (기본값: 12시간)
            prediction_interval = self.config.get("prediction", {}).get("resource", {}).get("interval_hours", 12)
            # 예측 주기 + 여유시간(1시간) 설정
            horizon = prediction_interval + 1
            logger.info(f"예측 지평선 자동 설정: {horizon}시간 (다음 예측 주기 + 1시간)")
            
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
            
            # 시퀀스 데이터 생성 실패 시 간단한 대체 예측 (개선)
            if X is None or X.shape[0] == 0:
                logger.warning("시퀀스 데이터 생성 실패, 간단한 예측 방법 사용")
                return self._fallback_resource_prediction(df, resource_type, horizon, target_cols)
            
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
                return self._fallback_resource_prediction(df, resource_type, horizon, target_cols)
            
            # 예측 수행
            try:
                predictions = model.predict(last_sequence)
            except Exception as e:
                logger.error(f"모델 예측 중 오류 발생: {e}")
                return self._fallback_resource_prediction(df, resource_type, horizon, target_cols)
            
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
                if predictions.shape[1] > i:
                    result_df[pred_col] = predictions[0, :, i] if predictions.shape[2] > i else np.nan
            
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
            return self._fallback_resource_prediction(df, resource_type, horizon, target_cols)
    
    def _fallback_resource_prediction(self, df, resource_type, horizon, target_cols):
        """
        간단한 대체 예측 방법 (개선)
        
        Args:
            df (pd.DataFrame): 원본 데이터
            resource_type (str): 자원 유형
            horizon (int): 예측 기간
            target_cols (list): 타겟 열 목록
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        logger.info(f"{resource_type} 대체 예측 방법 사용")
        
        # 예측 시간 생성
        last_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0 else datetime.now()
        pred_times = [last_time + timedelta(hours=i+1) for i in range(horizon)]
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame(index=pred_times)
        
        # 타겟 열별 마지막 값 또는 평균값 기준 예측
        for col in target_cols:
            if col in df.columns:
                # 마지막 3개 값의 평균 사용
                last_values = df[col].iloc[-3:] if len(df) >= 3 else df[col]
                pred_val = last_values.mean()
                
                # 결과에 추가
                pred_col = f"{col}_predicted"
                result_df[pred_col] = pred_val
        
        # 자원별 예측 결과 열 이름 매핑
        resource_col_map = {
            'cpu': 'cpu_usage_predicted',
            'mem': 'memory_used_percent_predicted',
            'disk': 'disk_used_percent_predicted',
            'diskio': 'disk_io_utilization_predicted',
            'net': 'net_utilization_predicted',
            'system': 'system_load1_predicted'
        }
        
        # 예측 열 이름 매핑
        if resource_type in resource_col_map:
            result_df[resource_col_map[resource_type]] = result_df[f"{target_cols[0]}_predicted"] if target_cols else 0
        
        # 메타데이터 추가
        device_id = self._get_device_id(df)
        company_domain = self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')
        building = self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')
        
        result_df['device_id'] = device_id
        result_df['companyDomain'] = company_domain
        result_df['building'] = building
        result_df['prediction_time'] = datetime.now()
        
        logger.info(f"{resource_type} 대체 예측 완료: {len(result_df)}행")
        return result_df
    def predict_integrated_failures(self, resource_data_dict):
        """
        여러 자원 데이터를 통합하여 디바이스 단위 고장 예측
        
        Args:
            resource_data_dict (dict): 자원별 데이터프레임 딕셔너리 {resource_type: df}
            
        Returns:
            pd.DataFrame: 통합 고장 예측 결과
        """
        if not resource_data_dict:
            logger.warning("고장 예측을 위한 자원 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 자원별 예측 결과 저장
        resource_predictions = {}
        device_ids = set()
        meta_info = {}
        
        # 각 자원별로 고장 예측 수행
        for resource_type, df in resource_data_dict.items():
            if df is None or df.empty:
                continue
            
            # 자원 유형의 고장 예측 지원 여부 확인
            resource_config = self.config.get('resources', {}).get(resource_type, {})
            prediction_types = resource_config.get('prediction_type', [])
            
            if isinstance(prediction_types, str):
                prediction_types = [prediction_types]
                
            if 'failure' not in prediction_types:
                logger.debug(f"{resource_type} 자원은 고장 예측을 지원하지 않습니다.")
                continue
            
            # 고장 예측 수행
            failure_df = self.predict_failures(df, resource_type)
            
            if failure_df is not None and not failure_df.empty:
                resource_predictions[resource_type] = failure_df
                
                # 디바이스 ID 및 메타 정보 수집
                if 'device_id' in failure_df.columns:
                    device_id = failure_df['device_id'].iloc[0]
                    device_ids.add(device_id)
                    
                    # 메타 정보 저장
                    if device_id not in meta_info:
                        meta_info[device_id] = {
                            'companyDomain': failure_df['companyDomain'].iloc[0] if 'companyDomain' in failure_df.columns else None,
                            'building': failure_df['building'].iloc[0] if 'building' in failure_df.columns else None
                        }
        
        if not resource_predictions:
            logger.warning("고장 예측 결과가 없습니다.")
            return pd.DataFrame()
        
        # 디바이스별 통합 고장 예측
        integrated_results = []
        
        for device_id in device_ids:
            device_result = {
                'device_id': device_id,
                'prediction_time': datetime.now(),
                'is_failure_predicted': False,
                'integrated_failure_probability': 0.0,
                'failure_resource': None,
                'resource_probabilities': {}
            }
            
            # 메타 정보 추가
            if device_id in meta_info:
                device_result.update(meta_info[device_id])
            
            max_prob = 0.0
            max_prob_resource = None
            
            # 각 자원별 고장 확률 통합
            for resource_type, pred_df in resource_predictions.items():
                # 해당 디바이스의 데이터만 필터링
                device_preds = pred_df[pred_df['device_id'] == device_id] if 'device_id' in pred_df.columns else pred_df
                
                if device_preds.empty:
                    continue
                
                # 고장 확률 열 찾기
                prob_col = f"{resource_type}_failure_probability"
                alt_prob_col = None
                
                for col in device_preds.columns:
                    if 'failure_probability' in col:
                        if resource_type in col:
                            prob_col = col
                        else:
                            alt_prob_col = col
                            
                # 확률 열이 없으면 대체 열 사용
                if prob_col not in device_preds.columns and alt_prob_col:
                    prob_col = alt_prob_col
                
                # 고장 확률 얻기
                if prob_col in device_preds.columns:
                    # 최대 고장 확률 추출
                    failure_prob = device_preds[prob_col].max()
                    device_result['resource_probabilities'][resource_type] = float(failure_prob)
                    
                    # 최대 확률 갱신
                    if failure_prob > max_prob:
                        max_prob = failure_prob
                        max_prob_resource = resource_type
            
            # 최종 고장 확률 및 자원 설정
            device_result['integrated_failure_probability'] = float(max_prob)
            device_result['failure_resource'] = max_prob_resource
            
            # 임계값 기반 고장 예측
            failure_threshold = self.config.get("prediction", {}).get("failure", {}).get("threshold", 0.5)
            device_result['is_failure_predicted'] = max_prob > failure_threshold
            
            # 타임스탬프 설정
            now = datetime.now()
            future_time = now + timedelta(hours=1)  # 1시간 후 예측
            device_result['timestamp'] = future_time
            
            # 자원별 실제 확률 열 추가
            for resource_type, prob in device_result['resource_probabilities'].items():
                col_name = f"{resource_type}_failure_probability"
                device_result[col_name] = prob
            
            # 리소스 확률 딕셔너리 제거 (DB 저장시 불필요)
            del device_result['resource_probabilities']
            
            integrated_results.append(device_result)
        
        # 결과를 데이터프레임으로 변환
        if integrated_results:
            result_df = pd.DataFrame(integrated_results)
            
            # 자원별 고장 확률 열 표준화
            for resource_type in self.config.get("resources", {}).keys():
                prob_col = f"{resource_type}_failure_probability"
                if prob_col not in result_df.columns:
                    result_df[prob_col] = 0.0
            
            return result_df
        else:
            return pd.DataFrame()    
    def predict_failures(self, df, resource_type):
        """
        자원 고장 예측 (개선됨)
        
        Args:
            df (pd.DataFrame): 입력 데이터
            resource_type (str): 자원 유형 (cpu, mem, disk, diskio, net, system)
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        if df is None or df.empty:
            logger.error("입력 데이터가 비어 있습니다.")
            return self._fallback_failure_prediction(df, resource_type)
        
        # 최소 필요 데이터 크기 확인
        min_data_size = 10  # 최소 10개 이상의 데이터 포인트 필요
        if len(df) < min_data_size:
            logger.warning(f"입력 데이터가 너무 적습니다. 필요: {min_data_size}, 실제: {len(df)}. 대체 예측 사용.")
            return self._fallback_failure_prediction(df, resource_type)
        
        
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
                return self._fallback_failure_prediction(df, resource_type)
            
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
            
            # 시퀀스 데이터 생성 실패 시 간단한 대체 예측 (개선)
            if X is None or X.shape[0] == 0:
                logger.warning("분류 데이터 생성 실패, 간단한 예측 방법 사용")
                return self._fallback_failure_prediction(df, resource_type)
            
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
            
            # 예측 수행
            try:
                failure_probs = model.predict(X)
            except Exception as e:
                logger.error(f"모델 예측 중 오류 발생: {e}")
                return self._fallback_failure_prediction(df, resource_type)
            
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
            return self._fallback_failure_prediction(df, resource_type)
    
    def _fallback_failure_prediction(self, df, resource_type):
        """
        간단한 대체 고장 예측 방법 (개선)
        
        Args:
            df (pd.DataFrame): 원본 데이터
            resource_type (str): 자원 유형
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        logger.info(f"{resource_type} 대체 고장 예측 방법 사용")
        
        # 임계값 설정
        failure_threshold = self.config.get("prediction", {}).get("failure", {}).get("threshold", 10.0)
        
        # 예측 시간 생성
        now = datetime.now()
        
        # 단일 행 결과 데이터프레임 생성
        result_df = pd.DataFrame({
            'timestamp': [now + timedelta(hours=1)],  # 1시간 후 예측
            'device_id': [self._get_device_id(df)],
            'companyDomain': [self.config.get('data_processing', {}).get('default_values', {}).get('companyDomain', 'javame')],
            'building': [self.config.get('data_processing', {}).get('default_values', {}).get('building', 'gyeongnam_campus')],
            'prediction_time': [now],
            'failure_resource': [resource_type],
            'is_failure_predicted': [False]  # 기본값: 고장 없음
        })
        
        # 자원별 고장 확률 열 이름 매핑
        resource_col_map = {
            'cpu': 'cpu_failure_probability',
            'mem': 'memory_failure_probability',
            'disk': 'disk_failure_probability',
            'net': 'network_failure_probability',
            'system': 'system_failure_probability'
        }
        
        # 고장 확률 추가
        if resource_type in resource_col_map:
            result_df[resource_col_map[resource_type]] = 0.1  # 기본값: 낮은 고장 확률
        
        return result_df
    
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
        try:
            # 모델 파일이 없으면 학습 필요
            model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
            if not os.path.exists(model_path):
                return True
            
            # 모델 로드 시도
            try:
                # 커스텀 객체 없이 모델 로드
                model = load_model(model_path, compile=False)  # compile=False로 로드
                # 나중에 다시 컴파일
                if model_key.endswith('_usage'):
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                else:
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                self.models[model_key] = model
                return False
            except Exception as e:
                logger.warning(f"모델 로드 실패: {e}, 재학습 필요")
                return True
        except Exception as e:
            logger.error(f"모델 체크 중 오류: {e}")
            return True
    
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
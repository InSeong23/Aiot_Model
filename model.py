#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IoT 시계열 예측 모델

고장 예측과 자원 사용량 예측을 위한 통합 모델 클래스 제공
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# Prophet 패키지 임포트 시도 (설치되지 않았을 수 있음)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class TimeSeriesModel:
    """
    시계열 예측 모델 기본 클래스
    
    LSTM과 Prophet 모델을 사용하여 시계열 예측 수행
    """
    
    def __init__(self, config=None):
        """
        초기화
        
        Args:
            config (dict): 설정 정보
        """
        self.config = config or {}
        self.model = None
        self.prophet_model = None
        self.history = None
        self.last_train_time = None
        self.input_dim = None
        self.input_window = None
        self.pred_horizon = None
        self.model_path = None
        self.mode = None
        self.scaler = None
        
        # 시나리오 정의 (자원 예측에만 사용)
        self.scenarios = {
            '기본': 1.0,       # 기본 시나리오 (현재 추세)
            '저성장': 0.7,     # 저성장 시나리오
            '고성장': 1.5      # 고성장 시나리오
        }
    
    def build_model(self):
        """
        모델 아키텍처 구축 (추상 메서드)
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=16, callbacks=None, **kwargs):
        """
        모델 학습
        
        Args:
            X (np.array): 입력 데이터
            y (np.array): 타겟 데이터
            validation_data (tuple): 검증 데이터 (X_val, y_val)
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            callbacks (list): 콜백 함수 목록
            
        Returns:
            history: 학습 이력
        """
        if self.model is None:
            logger.error("모델이 구축되지 않았습니다.")
            return None
        
        self.last_train_time = datetime.now()
        
        # 기본 콜백 설정
        if callbacks is None:
            callbacks = []
            
            # 조기 종료
            early_stopping = EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # 모델 저장
            if self.model_path:
                checkpoint = ModelCheckpoint(
                    filepath=self.model_path,
                    save_best_only=True,
                    monitor='val_loss' if validation_data else 'loss',
                    mode='min'
                )
                callbacks.append(checkpoint)
        
        # 모델 학습
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """
        예측 수행
        
        Args:
            X (np.array): 입력 데이터
            
        Returns:
            np.array: 예측 결과
        """
        if self.model is None:
            logger.error("모델이 구축되지 않았습니다.")
            return None
        
        return self.model.predict(X)
    
    def save(self, path=None):
        """
        모델 저장
        
        Args:
            path (str): 저장 경로
            
        Returns:
            bool: 성공 여부
        """
        if self.model is None:
            logger.error("저장할 모델이 없습니다.")
            return False
        
        save_path = path or self.model_path
        
        if not save_path:
            logger.error("저장 경로가 지정되지 않았습니다.")
            return False
        
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 모델 저장
            self.model.save(save_path)
            logger.info(f"모델 저장 완료: {save_path}")
            return True
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False
    
    def load(self, path=None):
        """
        모델 로드
        
        Args:
            path (str): 로드 경로
            
        Returns:
            bool: 성공 여부
        """
        load_path = path or self.model_path
        
        if not load_path:
            logger.error("로드 경로가 지정되지 않았습니다.")
            return False
        
        if not os.path.exists(load_path):
            logger.error(f"모델 파일을 찾을 수 없습니다: {load_path}")
            return False
        
        try:
            self.model = load_model(load_path)
            logger.info(f"모델 로드 완료: {load_path}")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False
    
    def get_model_info(self):
        """
        모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        if self.model is None:
            return {"status": "모델이 구축되지 않았습니다."}
        
        info = {
            "type": self.__class__.__name__,
            "mode": self.mode,
            "input_dim": self.input_dim,
            "input_window": self.input_window,
            "pred_horizon": self.pred_horizon,
            "model_path": self.model_path,
            "last_train_time": self.last_train_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_train_time else None,
            "trained": self.history is not None
        }
        
        return info


class FailurePredictionModel(TimeSeriesModel):
    """
    고장 예측 모델
    
    LSTM을 사용한 이진 분류 모델
    """
    
    def __init__(self, input_dim, input_window=24, lstm_units=64, dropout_rate=0.2, model_path=None):
        """
        초기화
        
        Args:
            input_dim (int): 입력 특성 수
            input_window (int): 입력 시퀀스 길이
            lstm_units (int): LSTM 유닛 수
            dropout_rate (float): 드롭아웃 비율
            model_path (str): 모델 저장 경로
        """
        super().__init__()
        self.input_dim = input_dim
        self.input_window = input_window
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.mode = 'failure'
        self.pred_horizon = 1
        self.build_model()
    
    def build_model(self):
        """
        고장 예측 모델 구축 (이진 분류)
        """
        model = Sequential()
        
        # 첫 번째 LSTM 층
        model.add(LSTM(self.lstm_units, activation='tanh', return_sequences=True, 
                      input_shape=(self.input_window, self.input_dim)))
        model.add(Dropout(self.dropout_rate))
        
        # 두 번째 LSTM 층
        model.add(LSTM(self.lstm_units//2, activation='tanh'))
        model.add(Dropout(self.dropout_rate))
        
        # 출력층
        model.add(Dense(1, activation='sigmoid'))  # 이진 분류
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("고장 예측 모델 구축 완료")
        
        return model
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=16, callbacks=None, class_weight=None):
        """
        모델 학습 (클래스 가중치 지원)
        
        Args:
            X (np.array): 입력 데이터
            y (np.array): 타겟 데이터
            validation_data (tuple): 검증 데이터 (X_val, y_val)
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            callbacks (list): 콜백 함수 목록
            class_weight (dict): 클래스 가중치
            
        Returns:
            history: 학습 이력
        """
        # 클래스 불균형 처리
        if class_weight is None:
            # 클래스 비율 계산
            pos_ratio = np.mean(y)
            
            if 0 < pos_ratio < 1:
                class_weight = {0: pos_ratio, 1: 1 - pos_ratio}
                logger.info(f"클래스 가중치 자동 설정: {class_weight}")
        
        # 부모 클래스의 fit 메서드 호출
        return super().fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight
        )
    
    def predict_failures(self, X, threshold=0.5):
        """
        고장 확률 및 예측 결과 반환
        
        Args:
            X (np.array): 입력 데이터
            threshold (float): 고장 판단 임계값
            
        Returns:
            tuple: (고장 확률, 고장 예측 결과)
        """
        if self.model is None:
            logger.error("모델이 구축되지 않았습니다.")
            return None, None
        
        # 고장 확률 예측
        probs = self.model.predict(X)
        
        # 임계값 기준 이진 예측
        predictions = (probs >= threshold).astype(int)
        
        return probs, predictions


class ResourcePredictionModel(TimeSeriesModel):
    """
    자원 사용량 예측 모델
    
    LSTM을 사용한 회귀 모델
    """
    
    def __init__(self, input_dim, input_window=24, pred_horizon=24, lstm_units=64, dropout_rate=0.2, model_path=None):
        """
        초기화
        
        Args:
            input_dim (int): 입력 특성 수
            input_window (int): 입력 시퀀스 길이
            pred_horizon (int): 예측 지평선
            lstm_units (int): LSTM 유닛 수
            dropout_rate (float): 드롭아웃 비율
            model_path (str): 모델 저장 경로
        """
        super().__init__()
        self.input_dim = input_dim
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.mode = 'resource'
        self.build_model()
    
    def build_model(self):
        """
        자원 사용량 예측 모델 구축 (회귀)
        """
        model = Sequential()
        
        # 양방향 LSTM 층
        model.add(Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=True),
                           input_shape=(self.input_window, self.input_dim)))
        model.add(Dropout(self.dropout_rate))
        
        # 두 번째 LSTM 층
        model.add(LSTM(self.lstm_units, activation='tanh'))
        model.add(Dropout(self.dropout_rate))
        
        # 출력층
        model.add(Dense(self.pred_horizon))  # 여러 시점 예측
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info("자원 사용량 예측 모델 구축 완료")
        
        return model
    
    def train_prophet_model(self, df, target_col):
        """
        Prophet 모델 학습 (장기 예측용)
        
        Args:
            df (pd.DataFrame): 데이터프레임
            target_col (str): 타겟 열 이름
            
        Returns:
            bool: 성공 여부
        """
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet 모듈이 설치되지 않았습니다. 장기 예측 기능을 사용할 수 없습니다.")
            return False
        
        try:
            # Prophet 데이터 준비
            prophet_df = df.reset_index().rename(columns={df.index.name or 'index': 'ds', target_col: 'y'})
            prophet_df = prophet_df[['ds', 'y']].dropna()
            
            # 타임존 정보 제거 - Prophet 요구사항
            if hasattr(prophet_df['ds'], 'dt'):
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            
            # Prophet 모델 생성 및 학습
            self.prophet_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            
            # 모델 학습
            self.prophet_model.fit(prophet_df)
            logger.info("Prophet 모델 학습 완료")
            return True
            
        except Exception as e:
            logger.error(f"Prophet 모델 학습 실패: {e}")
            self.prophet_model = None
            return False
    
    def predict_long_term(self, periods=90, capacity_threshold=None):
        """
        장기 자원 사용량 예측 (Prophet 사용)
        
        Args:
            periods (int): 예측 기간 (일)
            capacity_threshold (float): 용량 임계값
            
        Returns:
            dict: 시나리오별 예측 결과
            dict: 시나리오별 용량 증설 필요 시점
        """
        if self.prophet_model is None:
            logger.error("Prophet 모델이 학습되지 않았습니다.")
            return {}, {}
        
        try:
            # 미래 데이터프레임 생성
            future = self.prophet_model.make_future_dataframe(periods=periods, freq='D')
            
            # 기본 예측 수행
            forecast = self.prophet_model.predict(future)
            
            scenario_results = {}
            expansion_dates = {}
            
            # 각 시나리오별 시뮬레이션
            for scenario_name, growth_factor in self.scenarios.items():
                # 시나리오별 예측 조정
                scenario_forecast = forecast.copy()
                
                # 예측 시작점부터 성장률 조정
                forecast_start_date = scenario_forecast['ds'].iloc[-periods]
                future_mask = scenario_forecast['ds'] >= forecast_start_date
                
                # 트렌드 성분에 성장 계수 적용
                scenario_forecast.loc[future_mask, 'trend'] = scenario_forecast.loc[future_mask, 'trend'] * growth_factor
                
                # 최종 예측값 재계산
                scenario_forecast.loc[future_mask, 'yhat'] = (
                    scenario_forecast.loc[future_mask, 'trend'] +
                    scenario_forecast.loc[future_mask, 'seasonal'] +
                    scenario_forecast.loc[future_mask, 'weekly'] +
                    scenario_forecast.loc[future_mask, 'yearly']
                )
                
                # 예측 구간 조정
                scenario_forecast.loc[future_mask, 'yhat_lower'] = scenario_forecast.loc[future_mask, 'yhat'] - \
                                                              (forecast.loc[future_mask, 'yhat_upper'] - 
                                                               forecast.loc[future_mask, 'yhat'])
                
                scenario_forecast.loc[future_mask, 'yhat_upper'] = scenario_forecast.loc[future_mask, 'yhat'] + \
                                                              (forecast.loc[future_mask, 'yhat_upper'] - 
                                                               forecast.loc[future_mask, 'yhat'])
                
                # 용량 초과 시점 찾기 (임계값이 제공된 경우)
                if capacity_threshold is not None:
                    over_capacity = scenario_forecast[scenario_forecast['yhat'] > capacity_threshold]
                    
                    if not over_capacity.empty:
                        expansion_date = over_capacity['ds'].iloc[0]
                        expansion_dates[scenario_name] = expansion_date.strftime('%Y-%m-%d')
                    else:
                        expansion_dates[scenario_name] = None
                
                scenario_results[scenario_name] = scenario_forecast
            
            return scenario_results, expansion_dates
            
        except Exception as e:
            logger.error(f"장기 예측 중 오류 발생: {e}")
            return {}, {}
    
    def generate_forecasts(self, last_sequence, df_index, pred_steps, scaler=None):
        """
        미래 시점에 대한 예측 생성
        
        Args:
            last_sequence (np.array): 마지막 입력 시퀀스
            df_index (pd.DatetimeIndex): 원본 데이터프레임 인덱스
            pred_steps (int): 예측할 스텝 수
            scaler (object): 스케일러 (역변환용)
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        if self.model is None:
            logger.error("모델이 구축되지 않았습니다.")
            return None
        
        # 마지막 시간 확인
        last_time = df_index[-1] if len(df_index) > 0 else datetime.now()
        
        # 입력 시퀀스 준비
        current_sequence = last_sequence.copy()
        
        # 예측 결과 저장
        predictions = []
        forecast_times = []
        
        # 시간 간격 추정
        if len(df_index) > 1:
            time_diff = pd.Series(df_index).diff().median()
        else:
            # 기본 간격: 1시간
            time_diff = pd.Timedelta(hours=1)
        
        # 순차적 예측 수행
        for i in range(pred_steps):
            # 다음 시점 예측
            pred = self.model.predict(np.array([current_sequence]))[0]
            
            # 첫 번째 예측값만 사용
            next_val = pred[0]
            predictions.append(next_val)
            
            # 다음 시간
            next_time = last_time + time_diff * (i + 1)
            forecast_times.append(next_time)
            
            # 입력 시퀀스 업데이트 (맨 앞의 데이터 제거하고 예측값 추가)
            # 다차원 특성이 있는 경우를 처리하기 위해 복잡한 로직 필요
            if len(current_sequence.shape) > 1 and current_sequence.shape[1] > 1:
                # 현재 시퀀스에서 첫 번째 행을 제거
                current_sequence = np.roll(current_sequence, -1, axis=0)
                
                # 마지막 행은 이전 값 유지하면서 첫 번째 열만 예측값으로 변경
                current_sequence[-1, 0] = next_val
            else:
                # 단변량 시계열의 경우 간단히 처리
                current_sequence = np.append(current_sequence[1:], next_val)
        
        # 예측 결과를 데이터프레임으로 변환
        forecast_df = pd.DataFrame({
            'predicted_value': predictions
        }, index=forecast_times)
        
        # 스케일러가 제공된 경우 역변환
        if scaler is not None:
            try:
                inv_values = scaler.inverse_transform(
                    forecast_df[['predicted_value']].values.reshape(-1, 1)
                ).flatten()
                forecast_df['predicted_value'] = inv_values
            except Exception as e:
                logger.warning(f"예측값 역변환 실패: {e}")
        
        return forecast_df
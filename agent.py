import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import config

class KOSPIEnvironment:
    """ KOSPI 대표 종목 및 벤치마크(코스피 지수) 데이터를 관리하는 환경 """
    def __init__(self):
        # KOSPI 대표 20종목 (yfinance: .KS = 코스피)
        self.tickers = [
            "005930.KS", "000660.KS", "051910.KS", "006400.KS", "035420.KS",  # 삼성전자, SK하이닉스, LG화학, 삼성SDI, 네이버
            "035720.KS", "005380.KS", "000270.KS", "003550.KS", "068270.KS",  # 카카오, 현대차, 기아, LG, 셀트리온
            "207940.KS", "105560.KS", "055550.KS", "032830.KS", "051900.KS",  # 삼성바이오, KB금융, 신한지주, 삼성생명, LG생활건강
            "009150.KS", "017670.KS", "000810.KS", "096770.KS", "066570.KS",  # 삼성전기, SK텔레콤, 삼성화재, SK이노베이션, LG에너지솔루션
        ]
        self.benchmark = "^KS11"  # 코스피 지수
        self.all_symbols = self.tickers + [self.benchmark]
        
        self.data, self.tickers = self._download_data()
        self.vocab_size = len(self.tickers)

    @st.cache_data(ttl=3600)
    def _download_data(_self):
        # 벤치마크를 포함하여 데이터 다운로드 (5년, 약 1260 거래일)
        data = yf.download(_self.all_symbols, period="5y", interval="1d")['Close']
        data = data.ffill().bfill().dropna(axis=1)
        # yfinance 다중 종목 시 컬럼이 MultiIndex ('Close', '티커') → 단일 레벨(티커명)로 평탄화
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)
        tickers = [t for t in data.columns if t != _self.benchmark]
        return data, list(tickers)

class StaticConstraintEngine:
    def __init__(self, env, current_step):
        self.env = env
        self.vocab_size = env.vocab_size
        self.valid_mask = np.ones(self.vocab_size, dtype=bool)
        
        if current_step >= 20:
            history = self.env.data[self.env.tickers].iloc[current_step-20 : current_step]
            sma_20 = history.mean()
            current_prices = self.env.data[self.env.tickers].iloc[current_step]
            
            for i, ticker in enumerate(self.env.tickers):
                if current_prices[ticker] < sma_20[ticker]:
                    self.valid_mask[i] = False
                    
            if not np.any(self.valid_mask):
                self.valid_mask = np.ones(self.vocab_size, dtype=bool)

    def apply_mask(self, logits):
        return np.where(self.valid_mask, logits, -np.inf)

class RecommendationAgent:
    # lr, gamma, eps 인자를 추가하고 기본값을 설정합니다.
    def __init__(self, env, use_constraints=True, lr=0.01, gamma=0.98, eps=0.1):
        self.env = env
        self.use_constraints = use_constraints
        
        # 주입받은 하이퍼파라미터를 에이전트 속성으로 저장
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        
    def select_action(self, current_step):
        logits = np.random.randn(self.env.vocab_size)
        engine = StaticConstraintEngine(self.env, current_step)
        
        if self.use_constraints:
            logits = engine.apply_mask(logits)
            
        chosen_action = int(np.argmax(logits))
        
        if current_step + 1 < len(self.env.data):
            current_price = float(self.env.data[self.env.tickers[chosen_action]].iloc[current_step])
            next_price = float(self.env.data[self.env.tickers[chosen_action]].iloc[current_step + 1])
            reward = ((next_price - current_price) / current_price) * 100 if current_price > 0 else 0.0
        else:
            reward = 0.0
            
        is_valid = engine.valid_mask[chosen_action]
        chosen_ticker = self.env.tickers[chosen_action]
        
        return chosen_ticker, is_valid, reward
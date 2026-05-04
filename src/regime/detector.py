"""Regime Detector
- KOSPI 60일 (log-return, 20일 rolling vol) 시계열 → GaussianMixture 3-state
- 사후 라벨링: (mean_return, mean_volatility) 통계로 cluster→label 매핑
- LLM 시장 뉴스 sentiment 와 결합 (override 임계 |score|>θ)

NOTE: 원래 hmmlearn.GaussianHMM 으로 설계했으나 Python 3.14 환경에 prebuilt
      wheel 부재 + MSVC 미설치로 GaussianMixture 로 대체. 시간적 평탄화는
      "전일 라벨과 동일한 cluster posterior 가 ≥0.4 이면 유지" 휴리스틱으로 보완.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.mixture import GaussianMixture

from src.config import get_config
from src.db.sappo_models import (
    SentimentScore, get_sappo_session, get_latest_regime,
)


LABEL_RISK_ON = "risk_on_trend"
LABEL_RISK_OFF = "high_vol_risk_off"
LABEL_REVERT = "mean_revert"
ALL_LABELS = (LABEL_RISK_ON, LABEL_RISK_OFF, LABEL_REVERT)

MARKET_CODE = "_MARKET_"


@dataclass
class RegimeResult:
    label: str
    hmm_state: int                          # GMM cluster idx
    hmm_probs: dict[str, float]             # {label: posterior probability}
    kospi_return_60d: float                 # 60일 누적 log-return
    kospi_vol_60d: float                    # 60일 일수익률 std (annualized)
    llm_score: float | None
    overridden_by_llm: bool
    notes: str = ""
    macro_features: dict[str, float] | None = None  # A1+A2: USD/KRW + VIX 최신값

    @property
    def hmm_probs_tuple(self) -> tuple[float, float, float]:
        return (
            self.hmm_probs.get(LABEL_RISK_ON, 0.0),
            self.hmm_probs.get(LABEL_RISK_OFF, 0.0),
            self.hmm_probs.get(LABEL_REVERT, 0.0),
        )


class RegimeDetector:
    """매일 호출되는 KOSPI regime 분류기."""

    def __init__(
        self,
        lookback_days: int | None = None,
        n_states: int | None = None,
        llm_override_threshold: float | None = None,
        random_state: int = 42,
    ) -> None:
        cfg = get_config().regime
        self.lookback_days = lookback_days or cfg.hmm_lookback_days
        self.n_states = n_states or cfg.hmm_n_states
        self.llm_override_threshold = (
            llm_override_threshold if llm_override_threshold is not None
            else cfg.llm_override_threshold
        )
        self.random_state = random_state

    # ──────────────────────────────────────────────
    # 데이터 수집
    # ──────────────────────────────────────────────
    def _fetch_kospi(self, end: str | None = None) -> pd.DataFrame:
        """KOSPI 지수 (FDR 'KS11') 의 lookback_days+30일 일봉을 반환."""
        import FinanceDataReader as fdr
        end_dt = datetime.now() if end is None else datetime.strptime(end, "%Y%m%d")
        # 휴장일 여유로 lookback × 1.6 + 30 일 앞 당겨서 받음
        start_dt = end_dt - timedelta(days=int(self.lookback_days * 1.6) + 30)
        df = fdr.DataReader(
            "KS11",
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            raise RuntimeError("KOSPI(KS11) 데이터 로드 실패")
        df = df.rename(columns=str.lower).reset_index()
        if "date" not in df.columns:
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        return df[["date", "open", "high", "low", "close", "volume"]]

    def _build_features(self, kospi_df: pd.DataFrame, end_date: str | None = None) -> pd.DataFrame:
        """일별 (log-return, 20일 rolling vol) feature 매트릭스.

        매크로 feature 사용 가능 시 (USD/KRW + VIX) 6D 로 확장.
        DB 에 데이터 없으면 KOSPI 2D 만으로 fallback (graceful — A1+A2 dark-launch).
        """
        df = kospi_df.copy()
        df["log_ret"] = np.log(df["close"]).diff()
        df["roll_vol"] = df["log_ret"].rolling(20).std()
        df = df.dropna(subset=["log_ret", "roll_vol"]).reset_index(drop=True)
        # 마지막 lookback_days 만 사용
        feats = df.tail(self.lookback_days).reset_index(drop=True)

        # 매크로 overlay (A1+A2): sappo_macro_features 가 비어있거나 결측이면 KOSPI-only
        try:
            from src.db.sappo_models import get_macro_features_window
            macro_rows = get_macro_features_window(
                end_date or datetime.now().strftime("%Y%m%d"),
                days=self.lookback_days * 2 + 30,  # 휴장일 여유
            )
            if not macro_rows:
                return feats
            macro_df = pd.DataFrame([
                {
                    "macro_date": r.date,
                    "usdkrw_log_ret": float(r.usdkrw_log_ret or 0.0),
                    "usdkrw_vol_20d": float(r.usdkrw_vol_20d or 0.0),
                    "vix_log_ret": float(r.vix_log_ret or 0.0),
                    "vix_close": float(r.vix_close or 0.0),
                }
                for r in macro_rows
            ])
            # KOSPI 의 date 컬럼과 inner join 위해 macro_date 를 datetime 으로
            macro_df["macro_date"] = pd.to_datetime(macro_df["macro_date"], format="%Y%m%d")
            feats = feats.copy()
            feats["join_date"] = pd.to_datetime(feats["date"]).dt.normalize()
            macro_df = macro_df.set_index("macro_date").sort_index()
            # 가장 가까운 과거 macro 매칭 (asof) — 거래일정 mismatch 흡수
            feats["usdkrw_log_ret"] = feats["join_date"].map(
                lambda d: macro_df["usdkrw_log_ret"].asof(d) if not pd.isna(d) else 0.0
            )
            feats["usdkrw_vol_20d"] = feats["join_date"].map(
                lambda d: macro_df["usdkrw_vol_20d"].asof(d) if not pd.isna(d) else 0.0
            )
            feats["vix_log_ret"] = feats["join_date"].map(
                lambda d: macro_df["vix_log_ret"].asof(d) if not pd.isna(d) else 0.0
            )
            # VIX z-score (60일 표본 안에서 z-score 로 정규화 — 분포 평탄화)
            vix_close_series = feats["join_date"].map(
                lambda d: macro_df["vix_close"].asof(d) if not pd.isna(d) else np.nan
            ).astype(float)
            vix_mean = vix_close_series.mean()
            vix_std = vix_close_series.std() if vix_close_series.std() > 0 else 1.0
            feats["vix_z"] = (vix_close_series - vix_mean) / vix_std
            feats = feats.drop(columns=["join_date"])
            feats = feats.fillna(0.0)
        except Exception as e:
            logger.warning(f"매크로 feature join 실패 — KOSPI-only fallback: {e}")
        return feats

    def _fetch_market_sentiment(self, date: str) -> float | None:
        """sappo_sentiment_scores 에서 stock_code='_MARKET_' 의 sentiment 점수 (없으면 None)."""
        session = get_sappo_session()
        try:
            row = (
                session.query(SentimentScore)
                .filter(
                    SentimentScore.stock_code == MARKET_CODE,
                    SentimentScore.date <= date,
                )
                .order_by(SentimentScore.date.desc())
                .first()
            )
            return float(row.score) if row else None
        finally:
            session.close()

    # ──────────────────────────────────────────────
    # GMM fit + 사후 라벨링
    # ──────────────────────────────────────────────
    def _fit_and_assign(
        self,
        feats: pd.DataFrame,
    ) -> tuple[int, np.ndarray, dict[int, str], float, float]:
        """GMM 적합 → 마지막 시점의 cluster 와 사후확률, cluster→label 매핑 반환.

        Returns:
            (last_cluster_idx, last_posterior(K,), cluster_to_label, ret_60d, vol_60d_annualized)
        """
        # 매크로 feature 가 있으면 6D, 없으면 KOSPI 2D (A1+A2 dark-launch fallback)
        feature_cols = ["log_ret", "roll_vol"]
        for opt_col in ("usdkrw_log_ret", "usdkrw_vol_20d", "vix_log_ret", "vix_z"):
            if opt_col in feats.columns:
                feature_cols.append(opt_col)
        X = feats[feature_cols].values
        # 차원 ↑ 시 reg_covar 도 ↑ (60×6=360 datapoint 안정성)
        reg_cov = 1e-4 if X.shape[1] > 2 else 1e-5
        gmm = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
            n_init=3,
            reg_covar=reg_cov,
        )
        gmm.fit(X)
        post = gmm.predict_proba(X)              # (T, K)
        last_post = post[-1]
        last_cluster = int(np.argmax(last_post))

        # cluster 별 통계 — 라벨 할당은 KOSPI 컬럼(idx 0=ret, 1=vol) 기준 유지
        clusters = gmm.predict(X)
        cluster_stats: list[tuple[int, float, float]] = []
        for k in range(self.n_states):
            mask = clusters == k
            if mask.sum() == 0:
                cluster_stats.append((k, 0.0, 0.0))
                continue
            mean_ret = X[mask, 0].mean()
            mean_vol = X[mask, 1].mean()
            cluster_stats.append((k, mean_ret, mean_vol))

        # 라벨 할당 규칙 (3-state 가정):
        #   1) 가장 높은 mean_vol → high_vol_risk_off
        #   2) 남은 두 cluster 중 mean_ret 높은 쪽 → risk_on_trend
        #   3) 나머지 → mean_revert
        cluster_to_label: dict[int, str] = {}
        if self.n_states == 3:
            stats_by_vol = sorted(cluster_stats, key=lambda t: t[2], reverse=True)
            high_vol_cluster = stats_by_vol[0][0]
            cluster_to_label[high_vol_cluster] = LABEL_RISK_OFF
            remaining = [s for s in cluster_stats if s[0] != high_vol_cluster]
            remaining_by_ret = sorted(remaining, key=lambda t: t[1], reverse=True)
            cluster_to_label[remaining_by_ret[0][0]] = LABEL_RISK_ON
            cluster_to_label[remaining_by_ret[1][0]] = LABEL_REVERT
        else:
            # n_states != 3 의 경우 단순 규칙: vol 최고=risk_off, ret 최고=risk_on, 나머지=revert
            stats_by_vol = sorted(cluster_stats, key=lambda t: t[2], reverse=True)
            cluster_to_label[stats_by_vol[0][0]] = LABEL_RISK_OFF
            stats_by_ret = sorted(
                [s for s in cluster_stats if s[0] != stats_by_vol[0][0]],
                key=lambda t: t[1], reverse=True,
            )
            cluster_to_label[stats_by_ret[0][0]] = LABEL_RISK_ON
            for s in stats_by_ret[1:]:
                cluster_to_label[s[0]] = LABEL_REVERT

        # 60일 통계 (보고용)
        ret_60d = float(X[:, 0].sum())                    # 누적 log-return
        vol_60d_ann = float(X[:, 0].std() * np.sqrt(252))  # annualized vol from daily log-ret

        return last_cluster, last_post, cluster_to_label, ret_60d, vol_60d_ann

    # ──────────────────────────────────────────────
    # LLM 결합 (override)
    # ──────────────────────────────────────────────
    def _combine_with_llm(
        self,
        hmm_label: str,
        llm_score: float | None,
    ) -> tuple[str, bool, str]:
        """LLM sentiment 가 강한 음/양수면 HMM 결과를 강등/승격.

        규칙:
          - llm_score < -threshold AND hmm_label == risk_on_trend  → high_vol_risk_off
          - llm_score >  threshold AND hmm_label == high_vol_risk_off → mean_revert (한 단계 완화)
          - 그 외 → HMM 라벨 그대로
        """
        if llm_score is None:
            return hmm_label, False, "no_llm_score"
        thr = self.llm_override_threshold
        if llm_score < -thr and hmm_label == LABEL_RISK_ON:
            return LABEL_RISK_OFF, True, f"llm={llm_score:+.2f} demoted risk_on→risk_off"
        if llm_score > thr and hmm_label == LABEL_RISK_OFF:
            return LABEL_REVERT, True, f"llm={llm_score:+.2f} promoted risk_off→revert"
        return hmm_label, False, f"llm={llm_score:+.2f} no_override"

    # ──────────────────────────────────────────────
    # 시간적 평탄화 (휴리스틱)
    # ──────────────────────────────────────────────
    def _smooth_with_yesterday(
        self,
        new_label: str,
        new_post_by_label: dict[str, float],
    ) -> tuple[str, str]:
        """전일 라벨 posterior 가 0.4 이상이면 라벨 유지 (단발성 노이즈 흡수)."""
        prev = get_latest_regime()
        if prev is None or prev.label == new_label:
            return new_label, ""
        prev_post = new_post_by_label.get(prev.label, 0.0)
        if prev_post >= 0.4:
            return prev.label, f"smoothed_to_prev({prev.label}, prev_post={prev_post:.2f})"
        return new_label, ""

    # ──────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────
    def detect_today(self, date: str | None = None) -> RegimeResult:
        date_str = date or datetime.now().strftime("%Y%m%d")
        kospi = self._fetch_kospi(end=date_str)
        feats = self._build_features(kospi, end_date=date_str)
        if len(feats) < max(20, self.n_states * 5):
            raise RuntimeError(
                f"KOSPI feature 부족 ({len(feats)}일) — lookback {self.lookback_days} 점검"
            )
        cluster, post, cluster_to_label, ret_60d, vol_60d = self._fit_and_assign(feats)
        hmm_label = cluster_to_label[cluster]
        post_by_label: dict[str, float] = {lbl: 0.0 for lbl in ALL_LABELS}
        for k, lbl in cluster_to_label.items():
            post_by_label[lbl] = float(post[k])

        # LLM override
        llm_score = self._fetch_market_sentiment(date_str)
        final_label, overridden, llm_note = self._combine_with_llm(hmm_label, llm_score)

        # 평탄화
        smoothed_label, smooth_note = self._smooth_with_yesterday(final_label, post_by_label)
        if smoothed_label != final_label:
            final_label = smoothed_label

        notes_parts = [f"hmm={hmm_label}", llm_note]
        if smooth_note:
            notes_parts.append(smooth_note)
        notes = "; ".join(notes_parts)

        # A1+A2: 매크로 feature 가 join 됐다면 마지막 행 값 첨부 (보고/저장 용)
        macro_payload: dict[str, float] | None = None
        if "usdkrw_log_ret" in feats.columns:
            last_row = feats.iloc[-1]
            macro_payload = {
                "usdkrw_log_ret": float(last_row.get("usdkrw_log_ret", 0.0)),
                "usdkrw_vol_20d": float(last_row.get("usdkrw_vol_20d", 0.0)),
                "vix_log_ret": float(last_row.get("vix_log_ret", 0.0)),
                "vix_z": float(last_row.get("vix_z", 0.0)),
            }

        result = RegimeResult(
            label=final_label,
            hmm_state=cluster,
            hmm_probs=post_by_label,
            kospi_return_60d=ret_60d,
            kospi_vol_60d=vol_60d,
            llm_score=llm_score,
            overridden_by_llm=overridden,
            notes=notes,
            macro_features=macro_payload,
        )
        logger.info(
            f"[REGIME] {date_str} label={final_label} cluster={cluster} "
            f"posts={ {k: round(v,2) for k,v in post_by_label.items()} } "
            f"ret60d={ret_60d:+.3f} vol60d={vol_60d:.3f} llm={llm_score} "
            f"override={overridden}"
        )
        return result

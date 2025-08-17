from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from advanced_optimizer import AdvancedBayesianOptimizer
import numpy as np
import math
from typing import List, Optional, Tuple, Iterable, Dict


UTC = ZoneInfo("UTC")
LON = ZoneInfo("Europe/London")
BER = ZoneInfo("Europe/Berlin")

def parse_processing_date(s: str, year: int = 2025) -> date:
    """Convert a label like '22 July' to a date in a given year."""
    return datetime.strptime(f"{s} {year}", "%d %B %Y").date()

@dataclass
class BusinessCalendar:
    tz: ZoneInfo = LON
    start: time = time(10, 0)  # 10:00
    end: time = time(16, 0)    # 16:00
    # UK Bank Holidays (YYYY-MM-DD). Can be empty.
    holidays: Tuple[str, ...] = ("2025-08-25",)


    def _is_business_day(self, d: date) -> bool:
        if d.weekday() >= 5:  # Sat, Sun
            return False
        if self.holidays:
            if d.isoformat() in self.holidays:
                return False
        return True

    def business_minutes_between(self, start_dt: datetime, end_dt: datetime) -> int:
        """Business minutes between two tz-aware datetimes in self.tz (mon–fri, start-end)."""
        if end_dt <= start_dt:
            return 0
        s = start_dt.astimezone(self.tz)
        e = end_dt.astimezone(self.tz)

        def day_start(dt):
            return dt.replace(hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0)
        def day_end(dt):
            return dt.replace(hour=self.end.hour, minute=self.end.minute, second=0, microsecond=0)

        total = 0
        # same day quick path
        if s.date() == e.date():
            if self._is_business_day(s.date()):
                a = max(s, day_start(s))
                b = min(e, day_end(e))
                if b > a:
                    total += int((b - a).total_seconds() // 60)
            return total

        # first partial
        if self._is_business_day(s.date()):
            a = max(s, day_start(s))
            b = day_end(s)
            if b > a:
                total += int((b - a).total_seconds() // 60)

        # full days
        d = s.date() + timedelta(days=1)
        while d < e.date():
            if self._is_business_day(d):
                total += (self.end.hour - self.start.hour)*60 + (self.end.minute - self.start.minute)
            d += timedelta(days=1)

        # last partial
        if self._is_business_day(e.date()):
            a = day_start(e)
            b = min(e, day_end(e))
            if b > a:
                total += int((b - a).total_seconds() // 60)

        return total

    def add_business_minutes(self, start_dt: datetime, minutes: int) -> datetime:
        """Add business minutes to start_dt in self.tz, skipping weekends/holidays."""
        if minutes <= 0:
            return start_dt
        cur = start_dt.astimezone(self.tz)

        # move to next business start if outside hours
        def normalize_start(dt: datetime) -> datetime:
            # Achtung: dt.time() ist tz-naiv und daher mit self.start/self.end vergleichbar
            while dt.weekday() >= 5 or (self.holidays and dt.date().isoformat() in self.holidays) or dt.time() >= self.end:
                dt = (dt + timedelta(days=1)).replace(hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0)
            if dt.time() < self.start:
                dt = dt.replace(hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0)
            return dt

        cur = normalize_start(cur)
        remaining = minutes

        while remaining > 0:
            end_of_day = cur.replace(hour=self.end.hour, minute=self.end.minute, second=0, microsecond=0)
            span = int((end_of_day - cur).total_seconds() // 60)
            if span >= remaining:
                return cur + timedelta(minutes=remaining)
            remaining -= span
            # next day start
            cur = (cur + timedelta(days=1)).replace(hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0)
            cur = normalize_start(cur)

        return cur

@dataclass
class RobustLinear:
    """Theil–Sen robust line y = a + b x with slope quantiles for uncertainty."""
    a: float
    b: float
    slope_q20: float
    slope_q80: float

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray) -> "RobustLinear":
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        slopes = []
        n = len(x)
        if n < 2:
            # Fallback: keine robuste Schätzung möglich
            a = float(y[0]) if n == 1 else 0.0
            b = 0.0
            return RobustLinear(a=a, b=b, slope_q20=b, slope_q80=b)
        for i in range(n):
            for j in range(i+1, n):
                dx = x[j] - x[i]
                if dx != 0:
                    slopes.append((y[j] - y[i]) / dx)
        if not slopes:
            # alle x identisch
            a = float(np.median(y))
            b = 0.0
            return RobustLinear(a=a, b=b, slope_q20=b, slope_q80=b)
        b = float(np.median(slopes))
        a = float(np.median(y - b*x))
        q20, q80 = float(np.quantile(slopes, 0.2)), float(np.quantile(slopes, 0.8))
        return RobustLinear(a=a, b=b, slope_q20=q20, slope_q80=q80)

    def predict(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.a + self.b * np.asarray(x)

@dataclass
class LOESS:
    """Local linear regression with tricube weights."""
    frac: float = 0.6

    def predict_one(self, x_train: np.ndarray, y_train: np.ndarray, xi: float) -> float:
        x = np.asarray(x_train, float)
        y = np.asarray(y_train, float)
        n = len(x)
        k = max(2, int(math.ceil(self.frac*n)))
        d = np.abs(x - xi)
        h = np.sort(d)[k-1]
        if h == 0:
            y_same = y[d == 0]
            return float(y_same.mean())
        w = (1 - (d/h)**3)**3
        w[d>h] = 0.0
        X0 = np.ones_like(x)
        X1 = x
        W = w
        s00 = float((W*X0*X0).sum())
        s01 = float((W*X0*X1).sum())
        s11 = float((W*X1*X1).sum())
        t0  = float((W*X0*y).sum())
        t1  = float((W*X1*y).sum())
        det = s00*s11 - s01*s01
        if det == 0:
            if W.sum() == 0:
                return float(y.mean())
            return float((W*y).sum()/W.sum())
        a = (t0*s11 - t1*s01) / det
        b = (t1*s00 - t0*s01) / det
        return float(a + b*xi)

    def predict(self, x_train: np.ndarray, y_train: np.ndarray, xs: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x_train, y_train, float(xi)) for xi in xs])

@dataclass
class IntegratedRegressor:
    """Integrated model for LSE processing forecasts.
    
    - x: cumulative UK business hours since first observation
    - y: processing date as days since base (2025-01-01)
    - Model: distance-aware blend of Theil–Sen (global) and LOESS (local)
             y_hat(x) = (1-w)*TS(x) + w*LOESS(x), w = exp(-dist/tau)
    """
    cal: BusinessCalendar
    base_date: date = date(2025,1,1)
    loess_frac: float = 0.6
    tau_hours: float = 12.0  # scale for blending; ~2 Geschäftstage à 6h
    # fitted:
    x_: np.ndarray = field(default=None, init=False)
    y_: np.ndarray = field(default=None, init=False)
    ts_: RobustLinear = field(default=None, init=False)
    loess_: LOESS = field(default=None, init=False)
    rmse_: float = field(default=None, init=False)
    t0_: datetime = field(default=None, init=False)

    def _prepare_xy(self, rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, datetime]:
        # rows: dicts with keys: timestamp(str, UTC naive e.g. 'YYYY-MM-DDTHH:MM:SS'), date(str like "22 July")
        # Convert to arrays
        data = []
        for r in rows:
            ts_utc = datetime.fromisoformat(r["timestamp"]).replace(tzinfo=UTC)
            ts_lon = ts_utc.astimezone(self.cal.tz)
            y_date = parse_processing_date(r["date"])
            data.append((ts_lon, y_date))
        data.sort(key=lambda t: t[0])  # by time
        t0 = data[0][0]
        xs = []
        ys = []
        for ts_lon, y_date in data:
            mins = self.cal.business_minutes_between(t0, ts_lon)
            xs.append(mins/60.0)
            ys.append((y_date - self.base_date).days)
        return np.array(xs, float), np.array(ys, float), t0

    def fit(self, rows: List[Dict]) -> "IntegratedRegressor":
        x, y, t0 = self._prepare_xy(rows)
        self.x_, self.y_, self.t0_ = x, y, t0
        self.ts_ = RobustLinear.fit(x, y)
        self.loess_ = LOESS(self.loess_frac)
        # residual RMSE using blended fit at observed xs
        y_hat = self._blend_predict(x)
        self.rmse_ = float(np.sqrt(np.mean((y - y_hat)**2)))
        return self

    def _weight(self, x: float) -> float:
        """Blend weight for LOESS near the data edge (more LOESS when close)."""
        # distance to last observed x
        if self.x_ is None:
            return 0.0
        d = max(0.0, x - float(self.x_[-1]))
        return float(math.exp(-d / self.tau_hours))

    def _blend_predict_scalar(self, x: float) -> float:
        ts_val = float(self.ts_.predict(x))
        lo_val = float(self.loess_.predict_one(self.x_, self.y_, x))
        w = self._weight(x)
        return (1.0 - w)*ts_val + w*lo_val

    def _blend_predict(self, xs: np.ndarray) -> np.ndarray:
        return np.array([self._blend_predict_scalar(float(v)) for v in xs])

    # --- Inversion: find x for target y (monotone bisection with guard) ---
    def _invert_y(self, y_target: float, x_lo: float, x_hi: float, tol: float=1e-3, maxit: int=200) -> float:
        """Find x such that blend(x) ~= y_target using bisection; assumes overall monotone increasing."""
        def f(x): return self._blend_predict_scalar(x) - y_target
        flo, fhi = f(x_lo), f(x_hi)
        # expand hi if needed
        it = 0
        while fhi < 0 and it < 50:
            x_hi += 6.0  # add one business day window
            fhi = f(x_hi)
            it += 1
        # if still not bracketed, fall back to Theil–Sen closed form (guard b==0)
        if (flo > 0 and fhi > 0) or (flo < 0 and fhi < 0):
            if self.ts_.b != 0:
                return (y_target - self.ts_.a) / self.ts_.b
            # no slope information -> return best-effort lower bracket
            return x_lo
        for _ in range(maxit):
            xm = 0.5*(x_lo + x_hi)
            fm = f(xm)
            if abs(fm) < tol or (x_hi - x_lo) < tol:
                return xm
            if fm > 0:
                x_hi = xm
            else:
                x_lo = xm
        return 0.5*(x_lo + x_hi)

    def predict_datetime(self, target_date: str | date, tz_out: ZoneInfo = BER) -> Dict[str, object]:
        """Predict when the processing date will reach 'target_date'.
        Returns prediction and simple uncertainty band as datetimes in tz_out.
        """
        if isinstance(target_date, str):
            d = parse_processing_date(target_date)
        else:
            d = target_date
        y_t = (d - self.base_date).days
        # bracket search starting from last x
        x0 = float(self.x_[-1]) if self.x_ is not None else 0.0
        x_pred = self._invert_y(y_t, x0, x0 + 24.0)  # try within next 4 business days first
        # uncertainty from Theil–Sen slope quantiles and RMSE
        # Convert RMSE-y to hours-x via Theil–Sen slope
        eps_hours = 0.0
        if self.ts_.b > 0:
            eps_hours = self.rmse_ / self.ts_.b
        # slope-quantile bounds (guard division by zero)
        def pred_x_with_slope(s: float) -> float:
            s_eff = s if abs(s) >= 1e-12 else (self.ts_.b if abs(self.ts_.b) >= 1e-12 else 1e-6)
            a_med = float(np.median(self.y_ - s_eff*self.x_))
            return (y_t - a_med) / s_eff

        x_q1 = pred_x_with_slope(self.ts_.slope_q20)
        x_q2 = pred_x_with_slope(self.ts_.slope_q80)
        x_lo = min(x_q1, x_q2) - eps_hours
        x_hi = max(x_q1, x_q2) + eps_hours

        # map business hours -> datetimes
        def to_dt(x_hours: float) -> datetime:
            start = self.t0_.astimezone(self.cal.tz)
            dt = self.cal.add_business_minutes(start, int(round(x_hours*60)))
            return dt.astimezone(tz_out)

        return {
            "target_date": d,
            "x_hours_point": float(x_pred),
            "x_hours_low": float(x_lo),
            "x_hours_high": float(x_hi),
            "when_point": to_dt(x_pred),
            "when_low": to_dt(x_lo),
            "when_high": to_dt(x_hi),
        }


@dataclass
class ProfessionalAutoCalibratedRegressor(IntegratedRegressor):
    """
    Production-grade auto-calibrated regressor with advanced optimization.
    """
    
    # Optimization settings
    optimization_strategy: str = 'hybrid'  # 'skopt', 'optuna', 'hybrid', 'ensemble'
    optimization_budget: int = 30  # Number of evaluations
    use_parallel: bool = False  # Parallel evaluation
    
    # Calibration control
    min_data_for_calibration: int = 5
    recalibration_threshold: int = 10  # Recalibrate every N new points
    
    # History tracking
    calibration_history: List[Dict] = field(default_factory=list, init=False)
    last_calibration_size: int = field(default=0, init=False)
    performance_metrics: Dict = field(default=None, init=False)
    
    def fit(self, rows: List[Dict]) -> "ProfessionalAutoCalibratedRegressor":
        """Fit with state-of-the-art automatic calibration."""
        
        # Check if calibration needed
        should_calibrate = (
            len(rows) >= self.min_data_for_calibration and
            (self.last_calibration_size == 0 or 
             len(rows) - self.last_calibration_size >= self.recalibration_threshold)
        )
        
        if should_calibrate:
            print(f"\n🎯 Professional Auto-Calibration Starting...")
            print(f"   Data points: {len(rows)}")
            print(f"   Strategy: {self.optimization_strategy}")
            
            # Run advanced optimization
            optimal_params, metrics = self._advanced_calibration(rows)
            
            # Apply optimal parameters
            self.loess_frac = optimal_params['loess_frac']
            self.tau_hours = optimal_params['tau_hours']
            
            # Advanced: Also optimize these if in param space
            if 'base_tau' in optimal_params:
                self.base_tau = optimal_params['base_tau']
            if 'recency_weight' in optimal_params:
                self.recency_weight = optimal_params['recency_weight']
            
            # Store calibration info
            self.calibration_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'n_points': len(rows),
                'optimal_params': optimal_params,
                'metrics': metrics,
                'strategy': self.optimization_strategy
            })
            
            self.last_calibration_size = len(rows)
            self.performance_metrics = metrics
            
            print(f"✅ Calibration Complete!")
            print(f"   Optimal: loess_frac={self.loess_frac:.3f}, tau={self.tau_hours:.1f}h")
            print(f"   Performance: {metrics.get('cv_score', 0):.4f} hours mean error")
            if 'feature_importance' in metrics:
                print(f"   Parameter importance: {metrics['feature_importance']}")
        
        # Continue with standard fit
        return super().fit(rows)
    
    def _advanced_calibration(self, rows: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Run advanced Bayesian optimization for parameter tuning.
        """
        # Define sophisticated parameter space
        param_space = {
            'loess_frac': {
                'type': 'float',
                'low': 0.35,
                'high': 0.85,
                'prior': 'uniform'  # Could use 'log-uniform' for some params
            },
            'tau_hours': {
                'type': 'float', 
                'low': 6.0,
                'high': 30.0,
                'prior': 'uniform'
            }
        }
        
        # Advanced: Add more parameters to optimize
        if self.optimization_strategy in ['hybrid', 'ensemble']:
            param_space.update({
                'base_tau': {
                    'type': 'float',
                    'low': 8.0,
                    'high': 24.0,
                    'prior': 'uniform'
                },
                'recency_weight': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 0.9,
                    'prior': 'uniform'
                }
            })
        
        # Create advanced optimizer
        optimizer = AdvancedBayesianOptimizer(
            param_space=param_space,
            strategy=self.optimization_strategy,
            acquisition_func='gp_hedge',  # Hedge between multiple acquisition functions
            n_initial_points=min(10, max(5, len(rows) // 3)),
            n_calls=self.optimization_budget,
            n_jobs=-1 if self.use_parallel else 1,
            random_state=42,
            early_stop_patience=5
        )
        
        # Sophisticated cross-validation objective
        def objective(params: Dict) -> float:
            """
            Advanced objective with multiple validation strategies.
            """
            try:
                errors = []
                
                # Strategy 1: Time series cross-validation
                n_splits = min(5, len(rows) - self.min_data_for_calibration + 1)
                for split_size in np.linspace(0.6, 0.9, n_splits):
                    n_train = int(len(rows) * split_size)
                    if n_train < self.min_data_for_calibration:
                        continue
                    
                    train_rows = rows[:n_train]
                    test_rows = rows[n_train:min(n_train + 3, len(rows))]
                    
                    if not test_rows:
                        continue
                    
                    # Fit model with proposed parameters
                    temp_model = IntegratedRegressor(
                        cal=self.cal,
                        base_date=self.base_date,
                        loess_frac=params['loess_frac'],
                        tau_hours=params['tau_hours']
                    )
                    
                    try:
                        temp_model.fit(train_rows)
                        
                        # Evaluate predictions
                        for test_row in test_rows:
                            pred = temp_model.predict_datetime(test_row['date'])
                            if pred and pred.get('when_point'):
                                actual = datetime.fromisoformat(test_row['timestamp']).replace(tzinfo=UTC)
                                pred_time = pred['when_point']
                                
                                # Error in business hours (more meaningful)
                                error_hours = abs(self.cal.business_minutes_between(actual, pred_time) / 60.0)
                                
                                # Weighted by recency (recent predictions more important)
                                recency_weight = np.exp(-0.1 * (len(rows) - n_train))
                                errors.append(error_hours * recency_weight)
                    except:
                        continue
                
                # Strategy 2: Leave-one-out for recent points
                if len(rows) >= 7:
                    for i in range(max(0, len(rows) - 3), len(rows)):
                        loo_train = rows[:i] + rows[i+1:]
                        loo_test = rows[i]
                        
                        if len(loo_train) < self.min_data_for_calibration:
                            continue
                        
                        temp_model = IntegratedRegressor(
                            cal=self.cal,
                            base_date=self.base_date,
                            loess_frac=params['loess_frac'],
                            tau_hours=params['tau_hours']
                        )
                        
                        try:
                            temp_model.fit(loo_train)
                            pred = temp_model.predict_datetime(loo_test['date'])
                            
                            if pred and pred.get('when_point'):
                                actual = datetime.fromisoformat(loo_test['timestamp']).replace(tzinfo=UTC)
                                error_hours = abs(self.cal.business_minutes_between(actual, pred['when_point']) / 60.0)
                                errors.append(error_hours * 1.5)  # Weight recent LOO higher
                        except:
                            continue
                
                if not errors:
                    return 1000.0
                
                # Sophisticated error aggregation
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                max_error = np.max(errors)
                
                # Composite score (balance mean and worst-case)
                score = mean_error + 0.5 * std_error + 0.1 * max_error
                
                # Regularization (prefer moderate parameters)
                regularization = (
                    0.01 * abs(params['loess_frac'] - 0.6) +
                    0.001 * abs(params['tau_hours'] - 12.0)
                )
                
                return score + regularization
                
            except Exception as e:
                return 1000.0
        
        # Run optimization
        result = optimizer.optimize(
            objective_func=objective,
            n_calls=self.optimization_budget,
            verbose=True
        )
        
        # Extract advanced metrics
        metrics = {
            'cv_score': result['best_score'],
            'n_evaluations': result['n_evaluations'],
            'strategy_used': self.optimization_strategy
        }
        
        # Add advanced analytics if available
        if 'model' in result and result['model']:
            importance = optimizer.get_feature_importance()
            if importance:
                metrics['feature_importance'] = importance
        
        if 'convergence' in result:
            metrics['convergence_history'] = result['convergence']
        
        # Get next best candidates for future exploration
        next_candidates = optimizer.predict_next_best(3)
        if next_candidates:
            metrics['next_candidates'] = next_candidates
        
        return result['best_params'], metrics

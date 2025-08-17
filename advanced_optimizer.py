# advanced_optimizer.py
"""
State-of-the-art Bayesian Optimization for LSE parameter tuning.
Combines multiple optimization strategies for best results.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Primary: Scikit-Optimize (Best Gaussian Process implementation)
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi

# Secondary: Optuna (Most flexible, great for complex spaces)
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# For ensemble predictions
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel

@dataclass
class AdvancedBayesianOptimizer:
    """
    Production-grade Bayesian Optimizer using multiple strategies.
    
    Features:
    - Multiple acquisition functions
    - Ensemble of GP kernels
    - Parallel evaluation support
    - Early stopping
    - Constraint handling
    - Multi-objective optimization
    """
    
    # Optimization space
    param_space: Dict[str, Any]
    
    # Strategy settings
    strategy: str = 'hybrid'  # 'skopt', 'optuna', 'hybrid', 'ensemble'
    acquisition_func: str = 'EI'  # 'EI', 'LCB', 'PI', 'gp_hedge'
    
    # Optimization settings
    n_initial_points: int = 10
    n_calls: int = 50
    n_jobs: int = 1  # Parallel evaluations
    random_state: int = 42
    
    # Advanced settings
    acq_optimizer: str = 'sampling'  # 'sampling', 'lbfgs', 'auto'
    acq_optimizer_kwargs: Dict = field(default_factory=lambda: {'n_points': 10000})
    
    # GP settings
    kernel: Any = field(default=None)
    alpha: float = 1e-6  # Noise level
    normalize_y: bool = True
    
    # Early stopping
    early_stop_threshold: float = 1e-4
    early_stop_patience: int = 10
    
    # Results storage
    optimization_history_: List[Dict] = field(default_factory=list, init=False)
    best_params_: Dict = field(default=None, init=False)
    best_score_: float = field(default=float('inf'), init=False)
    model_: Any = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize kernels and spaces."""
        if self.kernel is None:
            # Sophisticated composite kernel
            self.kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) * 
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
                WhiteKernel(noise_level=self.alpha, noise_level_bounds=(1e-10, 1e-1))
            )
    
    def optimize(self, 
                 objective_func: Callable,
                 n_calls: Optional[int] = None,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> Dict:
        """
        Run advanced Bayesian optimization.
        
        Args:
            objective_func: Function to minimize
            n_calls: Override default number of evaluations
            callback: Called after each iteration
            verbose: Print progress
        
        Returns:
            Dictionary with best parameters and metadata
        """
        n_calls = n_calls or self.n_calls
        
        if self.strategy == 'skopt':
            return self._optimize_skopt(objective_func, n_calls, callback, verbose)
        elif self.strategy == 'optuna':
            return self._optimize_optuna(objective_func, n_calls, callback, verbose)
        elif self.strategy == 'hybrid':
            return self._optimize_hybrid(objective_func, n_calls, callback, verbose)
        elif self.strategy == 'ensemble':
            return self._optimize_ensemble(objective_func, n_calls, callback, verbose)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
def _optimize_skopt(self, objective_func: Callable, n_calls: int, 
                   callback: Optional[Callable], verbose: bool) -> Dict:
    """
    Scikit-Optimize: Best Gaussian Process implementation.
    """
    if verbose:
        print("🔬 Running Scikit-Optimize Bayesian Optimization...")
        print(f"   Acquisition: {self.acquisition_func}")
        print(f"   Kernel: {self.kernel}")
    
    # Convert param_space to skopt format
    dimensions = []
    param_names = []
    
    for name, spec in self.param_space.items():
        param_names.append(name)
        if isinstance(spec, tuple) and len(spec) == 2:
            dimensions.append(Real(spec[0], spec[1], name=name))
        elif isinstance(spec, dict):
            if spec['type'] == 'float':
                dimensions.append(Real(spec['low'], spec['high'], name=name, 
                                     prior=spec.get('prior', 'uniform')))
            elif spec['type'] == 'int':
                dimensions.append(Integer(spec['low'], spec['high'], name=name))
            elif spec['type'] == 'categorical':
                dimensions.append(Categorical(spec['choices'], name=name))
    
    # Create sophisticated objective wrapper
    @use_named_args(dimensions=dimensions)
    def wrapped_objective(**params):
        score = objective_func(params)
        
        # Store in history
        self.optimization_history_.append({
            'params': params.copy(),
            'score': score,
            'iteration': len(self.optimization_history_)
        })
        
        # Update best
        if score < self.best_score_:
            self.best_score_ = score
            self.best_params_ = params.copy()
            if verbose:
                print(f"   ⭐ New best at iteration {len(self.optimization_history_)}: {score:.6f}")
        
        # Callback
        if callback:
            callback(params, score)
        
        return score
    
    # KORRIGIERT: Korrekte Handhabung von acq_func
    # Prepare kwargs for gp_minimize
    gp_minimize_kwargs = {
        'func': wrapped_objective,
        'dimensions': dimensions,
        'n_calls': n_calls,
        'n_initial_points': self.n_initial_points,
        'acq_optimizer': self.acq_optimizer,
        'acq_optimizer_kwargs': self.acq_optimizer_kwargs,
        'random_state': self.random_state,
        'n_jobs': self.n_jobs,
        'noise': self.alpha,
        'verbose': verbose,
        'callback': None,  # We handle callbacks internally
        'x0': None,  # Could provide initial guess
        'y0': None,  # Could provide initial observations
        'model_queue_size': 1,  # Keep only latest model
    }
    
    # Handle acquisition function correctly
    if self.acquisition_func == 'gp_hedge':
        # For gp_hedge, just pass the string - NO acq_func_kwargs!
        gp_minimize_kwargs['acq_func'] = 'gp_hedge'
    elif self.acquisition_func == 'EI':
        # For explicit functions, pass the function object and kwargs
        gp_minimize_kwargs['acq_func'] = gaussian_ei
        gp_minimize_kwargs['acq_func_kwargs'] = {'xi': 0.01}
    elif self.acquisition_func == 'LCB':
        gp_minimize_kwargs['acq_func'] = gaussian_lcb
        gp_minimize_kwargs['acq_func_kwargs'] = {'kappa': 1.96}
    elif self.acquisition_func == 'PI':
        gp_minimize_kwargs['acq_func'] = gaussian_pi
        gp_minimize_kwargs['acq_func_kwargs'] = {'xi': 0.01}
    else:
        # Default to gp_hedge
        gp_minimize_kwargs['acq_func'] = 'gp_hedge'
    
    # Run optimization
    result = gp_minimize(**gp_minimize_kwargs)
    
    # Store the GP model
    self.model_ = result.models[-1] if result.models else None
    
    # Extract results
    best_params = dict(zip(param_names, result.x))
    
    if verbose:
        print(f"\n✅ Optimization complete!")
        print(f"   Best score: {result.fun:.6f}")
        print(f"   Best params: {self._format_params(best_params)}")
        print(f"   Convergence: {result.func_vals[-5:] if len(result.func_vals) >= 5 else result.func_vals}")
    
    return {
        'best_params': best_params,
        'best_score': result.fun,
        'n_evaluations': len(result.func_vals),
        'convergence': result.func_vals,
        'model': self.model_,
        'result_object': result
    }
    
    def _optimize_optuna(self, objective_func: Callable, n_calls: int,
                        callback: Optional[Callable], verbose: bool) -> Dict:
        """
        Optuna: Most flexible modern optimizer with TPE/CMA-ES.
        """
        if verbose:
            print("🧬 Running Optuna Advanced Optimization...")
            print(f"   Sampler: TPE + CMA-ES Ensemble")
        
        # Create study with advanced settings
        sampler = TPESampler(
            n_startup_trials=self.n_initial_points,
            n_ei_candidates=24,
            seed=self.random_state,
            multivariate=True,  # Consider parameter correlations
            constant_liar=True,  # Better parallelization
        )
        
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=n_calls,
            reduction_factor=3
        )
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name='LSE_optimization',
        )
        
        # Add CMA-ES for comparison
        cmaes_study = optuna.create_study(
            direction='minimize',
            sampler=CmaEsSampler(
                n_startup_trials=self.n_initial_points,
                seed=self.random_state,
            )
        )
        
        def optuna_objective(trial):
            params = {}
            for name, spec in self.param_space.items():
                if isinstance(spec, tuple):
                    params[name] = trial.suggest_float(name, spec[0], spec[1])
                elif isinstance(spec, dict):
                    if spec['type'] == 'float':
                        if spec.get('log', False):
                            params[name] = trial.suggest_float(name, spec['low'], spec['high'], log=True)
                        else:
                            params[name] = trial.suggest_float(name, spec['low'], spec['high'])
                    elif spec['type'] == 'int':
                        params[name] = trial.suggest_int(name, spec['low'], spec['high'])
                    elif spec['type'] == 'categorical':
                        params[name] = trial.suggest_categorical(name, spec['choices'])
            
            score = objective_func(params)
            
            # Store history
            self.optimization_history_.append({
                'params': params.copy(),
                'score': score,
                'iteration': len(self.optimization_history_),
                'trial_id': trial.number
            })
            
            if score < self.best_score_:
                self.best_score_ = score
                self.best_params_ = params.copy()
            
            if callback:
                callback(params, score)
            
            return score
        
        # Run both optimizers in parallel
        n_tpe = int(n_calls * 0.7)
        n_cmaes = n_calls - n_tpe
        
        # TPE optimization
        study.optimize(
            optuna_objective,
            n_trials=n_tpe,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose
        )
        
        # CMA-ES optimization
        if n_cmaes > 0:
            cmaes_study.optimize(
                optuna_objective,
                n_trials=n_cmaes,
                n_jobs=self.n_jobs,
                show_progress_bar=verbose
            )
        
        # Combine results
        all_trials = study.trials + cmaes_study.trials
        best_trial = min(all_trials, key=lambda t: t.value)
        
        if verbose:
            print(f"\n✅ Optuna optimization complete!")
            print(f"   Best score: {best_trial.value:.6f}")
            print(f"   Best params: {self._format_params(best_trial.params)}")
            print(f"   TPE trials: {len(study.trials)}, CMA-ES trials: {len(cmaes_study.trials)}")
        
        return {
            'best_params': best_trial.params,
            'best_score': best_trial.value,
            'n_evaluations': len(all_trials),
            'study': study,
            'cmaes_study': cmaes_study,
            'best_trial': best_trial
        }
    
    def _optimize_hybrid(self, objective_func: Callable, n_calls: int,
                        callback: Optional[Callable], verbose: bool) -> Dict:
        """
        Hybrid approach: Start with Optuna (exploration), finish with Scikit-Optimize (exploitation).
        """
        if verbose:
            print("🚀 Running Hybrid Optimization (Optuna → Scikit-Optimize)...")
        
        # Phase 1: Optuna for exploration (40% budget)
        n_optuna = int(n_calls * 0.4)
        n_skopt = n_calls - n_optuna
        
        if verbose:
            print(f"\n📍 Phase 1: Optuna exploration ({n_optuna} evaluations)")
        
        optuna_result = self._optimize_optuna(
            objective_func, n_optuna, callback, verbose=False
        )
        
        # Phase 2: Scikit-Optimize for exploitation (60% budget)
        if verbose:
            print(f"\n📍 Phase 2: Scikit-Optimize exploitation ({n_skopt} evaluations)")
            print(f"   Starting from best Optuna result: {optuna_result['best_score']:.6f}")
        
        # Use Optuna's best as starting point
        self.best_params_ = optuna_result['best_params']
        self.best_score_ = optuna_result['best_score']
        
        skopt_result = self._optimize_skopt(
            objective_func, n_skopt, callback, verbose=False
        )
        
        # Combine results
        final_best = (skopt_result if skopt_result['best_score'] < optuna_result['best_score'] 
                     else optuna_result)
        
        if verbose:
            print(f"\n✅ Hybrid optimization complete!")
            print(f"   Best score: {final_best['best_score']:.6f}")
            print(f"   Best params: {self._format_params(final_best['best_params'])}")
            print(f"   Best found by: {'Scikit-Optimize' if final_best == skopt_result else 'Optuna'}")
        
        return {
            'best_params': final_best['best_params'],
            'best_score': final_best['best_score'],
            'n_evaluations': n_calls,
            'optuna_result': optuna_result,
            'skopt_result': skopt_result,
            'optimization_history': self.optimization_history_
        }
    
    def _optimize_ensemble(self, objective_func: Callable, n_calls: int,
                          callback: Optional[Callable], verbose: bool) -> Dict:
        """
        Ensemble of multiple optimizers running in parallel with voting.
        """
        if verbose:
            print("🎭 Running Ensemble Optimization (Multiple Strategies)...")
        
        # Distribute budget
        strategies = ['skopt_ei', 'skopt_lcb', 'optuna_tpe', 'optuna_cmaes']
        n_per_strategy = n_calls // len(strategies)
        
        results = {}
        all_evaluations = []
        
        for i, strategy in enumerate(strategies):
            if verbose:
                print(f"\n📍 Strategy {i+1}/{len(strategies)}: {strategy}")
            
            if strategy.startswith('skopt'):
                acq = strategy.split('_')[1].upper()
                self.acquisition_func = acq
                res = self._optimize_skopt(
                    objective_func, n_per_strategy, None, verbose=False
                )
            else:
                res = self._optimize_optuna(
                    objective_func, n_per_strategy, None, verbose=False
                )
            
            results[strategy] = res
            all_evaluations.extend(self.optimization_history_[-n_per_strategy:])
        
        # Find best across all strategies
        best_strategy = min(results.items(), key=lambda x: x[1]['best_score'])
        
        if verbose:
            print(f"\n✅ Ensemble optimization complete!")
            print(f"   Best strategy: {best_strategy[0]}")
            print(f"   Best score: {best_strategy[1]['best_score']:.6f}")
            print(f"   Best params: {self._format_params(best_strategy[1]['best_params'])}")
            print(f"\n   All strategy results:")
            for name, res in results.items():
                print(f"     {name}: {res['best_score']:.6f}")
        
        return {
            'best_params': best_strategy[1]['best_params'],
            'best_score': best_strategy[1]['best_score'],
            'n_evaluations': len(all_evaluations),
            'best_strategy': best_strategy[0],
            'all_results': results,
            'all_evaluations': all_evaluations
        }
    
    def predict_next_best(self, n_candidates: int = 5) -> List[Dict]:
        """
        Predict next best parameters to try using the GP model.
        """
        if self.model_ is None:
            return []
        
        # Generate random candidates
        candidates = []
        for _ in range(n_candidates * 100):
            params = {}
            for name, spec in self.param_space.items():
                if isinstance(spec, tuple):
                    params[name] = np.random.uniform(spec[0], spec[1])
                elif isinstance(spec, dict):
                    if spec['type'] == 'float':
                        params[name] = np.random.uniform(spec['low'], spec['high'])
            candidates.append(params)
        
        # Predict with uncertainty
        X = np.array([[p[k] for k in sorted(self.param_space.keys())] 
                     for p in candidates])
        
        if hasattr(self.model_, 'predict'):
            mean, std = self.model_.predict(X, return_std=True)
            
            # Calculate acquisition value (Upper Confidence Bound)
            acq_values = -(mean - 2.0 * std)  # Negative because we minimize
            
            # Get top candidates
            top_indices = np.argsort(acq_values)[-n_candidates:]
            
            return [
                {
                    'params': candidates[i],
                    'expected_score': mean[i],
                    'uncertainty': std[i],
                    'acquisition_value': acq_values[i]
                }
                for i in top_indices
            ]
        
        return []
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance using the GP model.
        """
        if self.model_ is None or not hasattr(self.model_, 'kernel_'):
            return {}
        
        # Extract length scales from kernel
        if hasattr(self.model_.kernel_, 'length_scale'):
            length_scales = self.model_.kernel_.length_scale
            if not isinstance(length_scales, np.ndarray):
                length_scales = np.array([length_scales])
            
            # Importance is inverse of length scale
            param_names = sorted(self.param_space.keys())
            importances = 1.0 / (length_scales + 1e-6)
            importances = importances / importances.sum()
            
            return dict(zip(param_names, importances))
        
        return {}
    
    def _format_params(self, params: Dict) -> str:
        """Format parameters for display."""
        items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                for k, v in params.items()]
        return "{" + ", ".join(items) + "}"

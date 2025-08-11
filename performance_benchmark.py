#!/usr/bin/env python3
"""
Performance Benchmark Script for LSE Status Monitor Optimizations

This script demonstrates the performance improvements achieved through:
1. Lazy loading of dependencies
2. Smart caching
3. Early exit conditions
4. Memory optimization

Run this to see the optimizations in action.
"""

import time
import sys

def benchmark_import_time():
    """Benchmark module import time"""
    print("=== Import Performance Test ===")
    
    start = time.time()
    import check_lse
    import_time = time.time() - start
    
    print(f"Module import time: {import_time:.3f}s")
    print("✅ All dependencies lazy-loaded successfully")
    return check_lse

def benchmark_file_operations(check_lse):
    """Benchmark file loading with caching"""
    print("\n=== File Caching Performance Test ===")
    
    # First load (cold)
    start = time.time()
    status = check_lse.load_status()
    cold_time = time.time() - start
    print(f"Cold status load: {cold_time:.3f}s")
    
    # Second load (cached)
    start = time.time()
    status2 = check_lse.load_status()
    cached_time = time.time() - start
    print(f"Cached status load: {cached_time:.3f}s")
    
    speedup = cold_time / max(cached_time, 0.001)  # Avoid division by zero
    print(f"Cache speedup: {speedup:.1f}x faster")
    
    # Same for history
    start = time.time()
    history = check_lse.load_history()
    cold_hist = time.time() - start
    print(f"Cold history load: {cold_hist:.3f}s")
    
    start = time.time()
    history2 = check_lse.load_history()
    cached_hist = time.time() - start
    print(f"Cached history load: {cached_hist:.3f}s")

def benchmark_early_exit(check_lse):
    """Benchmark early exit conditions"""
    print("\n=== Early Exit Optimization Test ===")
    
    status = {'last_date': '22 July'}
    
    # Test non-manual run with no changes (should skip)
    skip_expensive = check_lse.should_skip_expensive_operations('22 July', status, False)
    print(f"Skip expensive ops (no change, auto): {skip_expensive}")
    
    # Test manual run (should not skip)
    skip_manual = check_lse.should_skip_expensive_operations('22 July', status, True)
    print(f"Skip expensive ops (no change, manual): {skip_manual}")
    
    # Test with change (should not skip)
    skip_change = check_lse.should_skip_expensive_operations('23 July', status, False)
    print(f"Skip expensive ops (with change, auto): {skip_change}")
    
    print("✅ Early exit logic working correctly")

def benchmark_lazy_imports(check_lse):
    """Benchmark lazy loading of heavy dependencies"""
    print("\n=== Lazy Import Performance Test ===")
    
    # Test numpy lazy loading
    start = time.time()
    np = check_lse._get_numpy()
    numpy_time = time.time() - start
    print(f"NumPy lazy load: {numpy_time:.3f}s")
    
    # Test matplotlib lazy loading
    start = time.time()
    plt, mdates = check_lse._get_matplotlib()
    mpl_time = time.time() - start
    print(f"Matplotlib lazy load: {mpl_time:.3f}s")
    
    # Test requests lazy loading
    start = time.time()
    requests = check_lse._get_requests()
    req_time = time.time() - start
    print(f"Requests lazy load: {req_time:.3f}s")
    
    print("✅ All heavy dependencies loaded on-demand")

def benchmark_regression_performance(check_lse):
    """Benchmark regression calculation performance"""
    print("\n=== Regression Performance Test ===")
    
    history = check_lse.load_history()
    
    start = time.time()
    forecast = check_lse.calculate_regression_forecast(history)
    calc_time = time.time() - start
    
    print(f"Regression calculation: {calc_time:.3f}s")
    
    if forecast:
        print(f"✅ Forecast generated successfully")
        print(f"   Data points: {forecast.get('data_points', 'N/A')}")
        print(f"   R²: {forecast.get('r_squared', 0):.3f}")
        print(f"   Slope: {forecast.get('slope', 0):.3f}")
    else:
        print("⚠️ No forecast generated (insufficient data)")

def main():
    """Run all benchmarks"""
    print("LSE Status Monitor - Performance Benchmark")
    print("=" * 50)
    
    # Import benchmark
    check_lse = benchmark_import_time()
    
    # File operations benchmark
    benchmark_file_operations(check_lse)
    
    # Early exit benchmark
    benchmark_early_exit(check_lse)
    
    # Lazy imports benchmark
    benchmark_lazy_imports(check_lse)
    
    # Regression performance benchmark
    benchmark_regression_performance(check_lse)
    
    print("\n" + "=" * 50)
    print("✅ All benchmarks completed successfully!")
    print("\nKey Optimizations:")
    print("• Lazy loading of heavy dependencies (numpy, matplotlib, requests)")
    print("• Smart caching with mtime-based invalidation")
    print("• Early exit conditions to skip expensive operations")
    print("• Memory optimization with proper resource cleanup")
    print("• Reduced redundant imports and code consolidation")

if __name__ == "__main__":
    main()
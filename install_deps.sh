#!/bin/bash
# Smart dependency installer for LSE Status Monitor
# Installs core dependencies always, advanced only when needed

set -e

echo "🚀 Smart dependency installation starting..."
START_TIME=$(date +%s)

# Always install core dependencies (lightweight, <3MB total)
echo "📦 Installing core dependencies..."
pip install --upgrade pip wheel setuptools --disable-pip-version-check --timeout 30 --retries 2
pip install -r requirements-core.txt --prefer-binary --disable-pip-version-check --timeout 30 --retries 2

CORE_TIME=$(date +%s)
CORE_DURATION=$((CORE_TIME - START_TIME))
echo "✅ Core dependencies installed in ${CORE_DURATION}s"

# Install advanced dependencies conditionally
INSTALL_ADVANCED=${INSTALL_ADVANCED:-"false"}
MANUAL_RUN=${MANUAL_RUN:-"false"}

# Always install numpy for regression functionality
echo "📊 Installing numpy (required for regression)..."
pip install "numpy>=1.24.0" --prefer-binary --disable-pip-version-check --timeout 30 --retries 2

NUMPY_TIME=$(date +%s)
NUMPY_DURATION=$((NUMPY_TIME - CORE_TIME))
echo "✅ Numpy installed in ${NUMPY_DURATION}s"

# Install other advanced dependencies only when needed
if [ "$INSTALL_ADVANCED" = "true" ] || [ "$MANUAL_RUN" != "true" ]; then
    echo "📈 Installing additional advanced dependencies for enhanced analytics..."
    pip install matplotlib>=3.7.0 scipy>=1.10.0 scikit-learn>=1.3.0 --prefer-binary --disable-pip-version-check --timeout 60 --retries 2
    
    ADV_TIME=$(date +%s)
    ADV_DURATION=$((ADV_TIME - NUMPY_TIME))
    echo "✅ Additional advanced dependencies installed in ${ADV_DURATION}s"
else
    echo "⚡ Skipping heavy advanced dependencies for fast manual run (numpy still available)"
fi

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
echo "🎯 Total installation time: ${TOTAL_DURATION}s"

# Verify core functionality
echo "🔍 Verifying installation..."
python -c "
import requests
import bs4
print('✅ Core dependencies working')

# Test numpy (always required)
try:
    import numpy
    print('✅ Numpy available for regression calculations')
except ImportError:
    print('❌ ERROR: Numpy missing - regression will fail!')
    exit(1)

# Test optional advanced features
try:
    import scipy, sklearn, matplotlib
    print('✅ Advanced dependencies available for enhanced analytics')
except ImportError:
    print('⚡ Advanced dependencies skipped (fast mode) - numpy still available')
"
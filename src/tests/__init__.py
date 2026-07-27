import warnings

# Suppress the SWIG module attribute deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type .* has no __module__ attribute.*")

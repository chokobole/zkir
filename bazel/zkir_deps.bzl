"""
This module configures dependencies for the ZKIR project.
"""

load("//third_party/omp:omp_configure.bzl", "omp_configure")

def zkir_deps():
    omp_configure(name = "local_config_omp")

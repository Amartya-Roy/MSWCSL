"""Regenerate presentation figures without overwriting updated main tables.

The current benchmark_main_*.tex tables supersede real_benchmark_compact.tex.
The latter remains a historical source and must not regenerate the main tables.
"""
from build_final_figures import main

if __name__ == '__main__':
    main()

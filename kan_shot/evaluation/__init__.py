from .statistical_tests import (
    pairwise_comparison, format_report, bootstrap_ci,
    cohens_d, wilcoxon_test, bonferroni_correction,
    ExperimentReport, ComparisonResult
)

__all__ = [
    'pairwise_comparison', 'format_report', 'bootstrap_ci',
    'cohens_d', 'wilcoxon_test', 'bonferroni_correction',
    'ExperimentReport', 'ComparisonResult',
]

try:
    from .sfa import SplineFeatureAttribution, RandomAttribution, SFAResult
    __all__.extend(['SplineFeatureAttribution', 'RandomAttribution', 'SFAResult'])
except ImportError:
    pass

try:
    from .robustness import (
        apply_corruption, CORRUPTION_REGISTRY,
        fgsm_attack, pgd_attack,
        evaluate_corruption_robustness, RobustnessResult
    )
    __all__.extend([
        'apply_corruption', 'CORRUPTION_REGISTRY',
        'fgsm_attack', 'pgd_attack',
        'evaluate_corruption_robustness', 'RobustnessResult',
    ])
except ImportError:
    pass

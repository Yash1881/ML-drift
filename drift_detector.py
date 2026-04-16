import numpy as np
from scipy import stats

def compute_psi(baseline, new_data, bins=10):
    """
    Compute Population Stability Index (PSI) from scratch.
    """
    # Define common bins based on both distributions to cover the full range
    min_val = min(np.min(baseline), np.min(new_data))
    max_val = max(np.max(baseline), np.max(new_data))
    
    # Create equal-width bins
    bins_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Compute proportions
    baseline_hist, _ = np.histogram(baseline, bins=bins_edges)
    new_hist, _ = np.histogram(new_data, bins=bins_edges)
    
    # Convert to proportions
    baseline_prop = baseline_hist / len(baseline)
    new_prop = new_hist / len(new_data)
    
    # Add small epsilon to avoid division by zero and log(0)
    epsilon = 1e-4
    baseline_prop = np.where(baseline_prop == 0, epsilon, baseline_prop)
    new_prop = np.where(new_prop == 0, epsilon, new_prop)
    
    # Normalize again after adding epsilon
    baseline_prop = baseline_prop / np.sum(baseline_prop)
    new_prop = new_prop / np.sum(new_prop)
    
    # Compute PSI
    psi = np.sum((new_prop - baseline_prop) * np.log(new_prop / baseline_prop))
    return float(psi)

def compute_kl_divergence(baseline, new_data, bins=10):
    """
    Compute Kullback-Leibler (KL) Divergence from scratch.
    """
    min_val = min(np.min(baseline), np.min(new_data))
    max_val = max(np.max(baseline), np.max(new_data))
    
    bins_edges = np.linspace(min_val, max_val, bins + 1)
    
    # P(x) is new data distribution, Q(x) is baseline distribution (commonly D_KL(P || Q))
    # We will use P=new, Q=baseline
    new_hist, _ = np.histogram(new_data, bins=bins_edges)
    baseline_hist, _ = np.histogram(baseline, bins=bins_edges)
    
    p_prop = new_hist / len(new_data)
    q_prop = baseline_hist / len(baseline)
    
    epsilon = 1e-4
    p_prop = np.where(p_prop == 0, epsilon, p_prop)
    q_prop = np.where(q_prop == 0, epsilon, q_prop)
    
    p_prop = p_prop / np.sum(p_prop)
    q_prop = q_prop / np.sum(q_prop)
    
    kl = np.sum(p_prop * np.log(p_prop / q_prop))
    return float(kl)

def compute_ks_test(baseline, new_data):
    """
    Compute Kolmogorov-Smirnov Test using scipy.
    """
    statistic, p_value = stats.ks_2samp(baseline, new_data)
    return float(statistic), float(p_value)

def analyze_feature_drift(baseline, new_data, feature_name):
    """
    Analyze drift for a single numeric feature.
    """
    psi = compute_psi(baseline, new_data)
    kl = compute_kl_divergence(baseline, new_data)
    ks_stat, ks_pvalue = compute_ks_test(baseline, new_data)
    
    if psi > 0.2:
        status = "HIGH RISK"
    elif psi > 0.1 or ks_pvalue < 0.05:
        status = "MONITOR"
    else:
        status = "STABLE"
        
    return {
        "feature": feature_name,
        "psi": psi,
        "kl_divergence": kl,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "status": status
    }

def get_overall_health(features_status):
    """
    Determine overall model health based on all features.
    features_status is a list of statuses (e.g., ["STABLE", "MONITOR", ...])
    """
    if "HIGH RISK" in features_status:
        return "HIGH RISK"
    if "MONITOR" in features_status:
        return "MONITOR"
    return "STABLE"

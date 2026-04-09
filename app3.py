import streamlit as st
import numpy as np
from scipy.stats import t

# Function
def ttest_1samp_two_sided(X, popmean, alpha):
    n = len(X)
    x_bar = np.mean(X)
    sigma = np.std(X, ddof=1)
    mu = popmean
    
    standard_error = sigma / np.sqrt(n)
    t_calculated = (x_bar - mu) / standard_error
    
    t_table_negative = t.ppf(alpha/2, n-1)
    t_table_positive = t.ppf(1 - alpha/2, n-1)
    
    p_value = 2 * (1 - t.cdf(abs(t_calculated), n-1))
    
    return t_calculated, t_table_negative, t_table_positive, p_value

# UI
st.title("📊 One-Sample Two-Sided T-Test")

# Input data
data_input = st.text_area("Enter sample values (comma separated):")
popmean = st.number_input("Population Mean (μ):", value=0.0)
alpha = st.number_input("Significance Level (α):", value=0.05)

# Button
if st.button("Run T-Test"):
    try:
        # Convert input to list
        X = np.array([float(i) for i in data_input.split(",")])
        
        t_calc, t_neg, t_pos, p_val = ttest_1samp_two_sided(X, popmean, alpha)
        
        st.subheader("Results")
        st.write(f"t-calculated: {t_calc:.4f}")
        st.write(f"t-critical (negative): {t_neg:.4f}")
        st.write(f"t-critical (positive): {t_pos:.4f}")
        st.write(f"p-value: {p_val:.6f}")
        
        # Decision
        if p_val < alpha:
            st.success("Reject H₀ (Significant difference)")
        else:
            st.warning("Fail to reject H₀ (Not significant)")
    
    except:
        st.error("Please enter valid numeric data.")
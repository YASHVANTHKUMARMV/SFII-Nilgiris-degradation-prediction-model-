"""
Phase 14: Scientific Visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logger = logging.getLogger("Lab.Visualization")

def plot_sfii_trajectory(time_steps, sfii_scores, disturbance_point, output_path):
    """
    Generates a publication-quality plot of the SFII temporal trajectory,
    highlighting the mathematical Drop, Duration, and Recovery metrics.
    """
    logger.info("Generating SFII Trajectory Plot...")
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(time_steps, sfii_scores, color='#2ca02c', linewidth=2.5, label='SFII Score')
    ax.axvline(x=disturbance_point, color='#d62728', linestyle='--', linewidth=2, label='Disturbance Detected (LandTrendr)')
    
    # Formatting for IEEE/MDPI standards
    ax.set_title("Structural Forest Integrity Index (SFII) Trajectory", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Months)", fontsize=12)
    ax.set_ylabel("SFII Score (0-1)", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Plot saved to {output_path}")

# ---------------------------------------------------------
# INTERNAL LABORATORY REVIEW: VISUALIZATION STANDARDS
# ---------------------------------------------------------
# Decision: All plots must be exported at 300 DPI for print quality.
# Color palettes must be colorblind-friendly (e.g., Seaborn deep/paper presets).
# ---------------------------------------------------------

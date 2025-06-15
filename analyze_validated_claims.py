import json
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, f_oneway
from sklearn.metrics import mean_squared_error

def analyze_validated_claims(json_file_path):
    """
    Analyze validated claims JSON to calculate average cosine scores by confidence level
    and perform linear regression analysis.
    
    Args:
        json_file_path (str): Path to the validated_claims.json file
    
    Returns:
        dict: Dictionary containing analysis results
    """
    
    # Load JSON data with UTF-8 encoding
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract all entries into a flat list
    all_entries = []
    for author, entries in data.items():
        for entry in entries:
            entry['author'] = author
            all_entries.append(entry)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_entries)
    
    # Calculate average cosine scores by confidence level
    confidence_levels = ['LOW', 'MEDIUM', 'HIGH']
    cosine_averages = {}
    
    for confidence in confidence_levels:
        filtered_df = df[df['confidence'] == confidence]
        if len(filtered_df) > 0:
            avg_cosine = filtered_df['cosine'].mean()
            cosine_averages[confidence] = {
                'average': avg_cosine,
                'count': len(filtered_df),
                'std': filtered_df['cosine'].std()
            }
        else:
            cosine_averages[confidence] = {
                'average': None,
                'count': 0,
                'std': None
            }
    
    # Print results
    print("Average Cosine Scores by Confidence Level:")
    print("-" * 50)
    for confidence, stats in cosine_averages.items():
        if stats['average'] is not None:
            print(f"{confidence}: {stats['average']:.4f} (n={stats['count']}, std={stats['std']:.4f})")
        else:
            print(f"{confidence}: No data available")
    
    # Prepare data for linear regression
    # Map confidence levels to numeric values
    confidence_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    df['confidence_numeric'] = df['confidence'].map(confidence_mapping)
    
    # Remove rows with missing values
    regression_df = df[['confidence_numeric', 'cosine']].dropna()
    
    # Perform linear regression: cosine ~ confidence_numeric
    if len(regression_df) > 1:
        X = regression_df[['confidence_numeric']].values
        y = regression_df['cosine'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        # Calculate statistical significance
        n = len(X)
        k = X.shape[1]  # number of predictors
        
        # Calculate predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate standard error of the coefficient
        mse = mean_squared_error(y, y_pred)
        var_residuals = np.sum(residuals**2) / (n - k - 1)
        var_X = np.sum((X - np.mean(X))**2)
        se_coef = np.sqrt(var_residuals / var_X)
        
        # Calculate t-statistic and p-value
        t_stat = model.coef_[0] / se_coef
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), n - k - 1))
        
        # Calculate confidence intervals (95%)
        t_critical = t.ppf(0.975, n - k - 1)
        ci_lower = model.coef_[0] - t_critical * se_coef
        ci_upper = model.coef_[0] + t_critical * se_coef
        
        print("\n\nLinear Regression Analysis (Cosine ~ Confidence):")
        print("-" * 50)
        print(f"Sample size: {n}")
        print(f"Coefficient: {model.coef_[0]:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Standard error: {se_coef:.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"95% CI for coefficient: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Interpret statistical significance
        if p_value < 0.001:
            sig_level = "*** (p < 0.001)"
        elif p_value < 0.01:
            sig_level = "** (p < 0.01)"
        elif p_value < 0.05:
            sig_level = "* (p < 0.05)"
        else:
            sig_level = "Not significant (p >= 0.05)"
        
        print(f"Statistical significance: {sig_level}")
        print(f"\nEquation: cosine = {model.intercept_:.4f} + {model.coef_[0]:.4f} * confidence_level")
        
        # ANOVA for additional perspective
        print("\n\nANOVA Analysis (Cosine by Confidence Groups):")
        print("-" * 50)
        
        # Group data by confidence level
        groups = []
        group_labels = []
        for conf in ['LOW', 'MEDIUM', 'HIGH']:
            group_data = df[df['confidence'] == conf]['cosine'].values
            if len(group_data) > 0:
                groups.append(group_data)
                group_labels.append(conf)
        
        if len(groups) >= 2:
            f_stat, p_value_anova = f_oneway(*groups)
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_value_anova:.4f}")
            
            # Effect size (eta-squared)
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((all_data - grand_mean)**2)
            eta_squared = ss_between / ss_total
            print(f"Effect size (eta-squared): {eta_squared:.4f}")
            
            # Post-hoc tests if significant
            if p_value_anova < 0.05:
                print("\nPost-hoc Tukey HSD Test:")
                print("-" * 30)
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                # Prepare data for Tukey test
                tukey_data = []
                tukey_groups = []
                for i, (group, label) in enumerate(zip(groups, group_labels)):
                    tukey_data.extend(group)
                    tukey_groups.extend([label] * len(group))
                
                tukey_result = pairwise_tukeyhsd(tukey_data, tukey_groups, alpha=0.05)
                print(tukey_result)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot Cosine vs Confidence
        for conf in ['LOW', 'MEDIUM', 'HIGH']:
            conf_data = regression_df[regression_df['confidence_numeric'] == confidence_mapping.get(conf, 0)]
            if len(conf_data) > 0:
                ax.scatter(conf_data['confidence_numeric'], conf_data['cosine'], 
                          label=conf, alpha=0.6, s=50)
        
        # Add regression line
        x_range = np.linspace(0.5, 3.5, 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r--', label='Regression line', linewidth=2)
        
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Cosine Score')
        ax.set_title('Cosine Score vs Confidence Level')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['LOW', 'MEDIUM', 'HIGH'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cosine_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("\nInsufficient data for regression analysis")
    
    # Return results
    results = {
        'cosine_averages': cosine_averages,
        'total_entries': len(all_entries),
        'entries_by_confidence': df['confidence'].value_counts().to_dict(),
        'overall_cosine_mean': df['cosine'].mean(),
        'overall_cosine_std': df['cosine'].std()
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    json_file_path = "validated_claims.json"
    
    try:
        results = analyze_validated_claims(json_file_path)
        
        print("\n\nSummary Statistics:")
        print("-" * 50)
        print(f"Total entries: {results['total_entries']}")
        print(f"Overall cosine mean: {results['overall_cosine_mean']:.4f}")
        print(f"Overall cosine std: {results['overall_cosine_std']:.4f}")
        print("\nEntries by confidence level:")
        for conf, count in results['entries_by_confidence'].items():
            print(f"  {conf}: {count}")
            
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        print("Please ensure the file path is correct.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
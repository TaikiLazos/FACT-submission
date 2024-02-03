import re
import ast
import os
import matplotlib.pyplot as plt

def extract_results(file_path):
    results = []

    with open(file_path, 'r') as file:
        content = file.read()

    matches = re.finditer(r'Params: alpha = (.*?), beta = (.*?), gamma = (.*?), lamda = (.*?)\nResults: (.*?)\n', content, re.DOTALL)

    for match in matches:
        alpha, beta, gamma, lamda, results_str = match.groups()

        results_dict = ast.literal_eval(results_str)

        for epochs, metrics in results_dict.items():
            avg_accuracy = metrics['accuracy'][0]
            avg_dp = metrics['dp'][0]
            avg_eo = metrics['eo'][0]

            results.append({
                'alpha': float(alpha),
                'beta': float(beta),
                'gamma': float(gamma),
                'lamda': float(lamda),
                'epochs': epochs,
                'avg_accuracy': avg_accuracy,
                'avg_dp': avg_dp * 100,
                'avg_eo': avg_eo * 100
            })

    return results

def plot_acc_dp_tradeoff(file_path):
    results = extract_results(file_path)
    accuracies = [r['avg_accuracy'] for r in results]
    dps = [r['avg_dp'] for r in results]

    plt.scatter(dps, accuracies)
    plt.xlabel('Demographic Parity (DP)')
    plt.ylabel('Average Accuracy')
    plt.title('DP-Accuracy Tradeoff')
    plt.grid(True)
    plt.show()
    
def plot_acc_dp_tradeoff(file_path):
    results = extract_results(file_path)
    
    # Sort the results by DP and then by accuracy in descending order
    sorted_results = sorted(results, key=lambda x: (x['avg_dp'], -x['avg_accuracy']))

    best_case_results = []
    current_best_accuracy = 0

    for result in sorted_results:
        if result['avg_accuracy'] > current_best_accuracy:
            best_case_results.append(result)
            current_best_accuracy = result['avg_accuracy']
        # Limit to 5 points for the "best case" scenario to keep the plot clean
        if len(best_case_results) == 5:
            break
    
    # Extracting accuracies and DP values for the best case scenarios
    accuracies = [r['avg_accuracy'] for r in best_case_results]
    dps = [r['avg_dp'] for r in best_case_results]

    # Plot a line with markers
    plt.plot(dps, accuracies, '-o', label='Acc-DP Tradeoff')

    plt.xlabel('Demographic Parity (DP)')
    plt.ylabel('Average Accuracy')
    plt.title('Best Case DP-Accuracy Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def process_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith("results_log_pockec_z_epoch=500") and filename.endswith("16.txt"):
            file_path = os.path.join(directory, filename)
            plot_acc_dp_tradeoff(file_path)

# Example usage
file_path = 'results/'
process_files(file_path)

import pandas as pd
from cfi.core import CFI

# Placeholder eval
def run_benchmark(dataset='harmbench'):
    # Load dataset (assume CSV with 'prompt', 'history', 'expected_band')
    df = pd.read_csv(f'{dataset}.csv')
    cfi = CFI()
    results = []
    for _, row in df.iterrows():
        u_t = row['prompt']
        H_t = row['history'].split('|')
        S_prev = cfi.get_initial_state()
        band, _, _ = cfi(u_t, H_t, S_prev)
        results.append(band == row['expected_band'])

    print(f"Accuracy on {dataset}: {np.mean(results):.2f}")

if __name__ == "__main__":
    run_benchmark()

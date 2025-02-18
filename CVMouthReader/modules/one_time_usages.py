# import pandas as pd
# import multiprocessing
# import time

# def csv_cleaner(input_csv, columns_to_drop):
#     df = pd.read_csv(input_csv)
#     df = df.drop(columns=columns_to_drop, errors='ignore')
#     df.to_csv(input_csv, index=False)


# def worker(n):
#     time.sleep(0.1)  # Simulate work
#     return n * n

# def benchmark():
#     cpu_count = multiprocessing.cpu_count()
#     best_time = float("inf")
#     best_workers = 1

#     for num_workers in range(1, cpu_count + 2):  # Test different worker counts
#         pool = multiprocessing.Pool(num_workers)
#         start_time = time.time()
#         pool.map(worker, range(100))  # Adjust workload as needed
#         pool.close()
#         pool.join()
#         elapsed_time = time.time() - start_time

#         print(f"Workers: {num_workers}, Time: {elapsed_time:.4f}s")

#         if elapsed_time < best_time:
#             best_time = elapsed_time
#             best_workers = num_workers

#     return best_workers



# def main():
#     # 5 seems best for workers/time tradeoff, 9 is best for time
#     optimal_workers = benchmark()
#     print(f"Optimal number of workers: {optimal_workers}")
#     # cols = ["word_type","error_type","song","annotations"]
#     # csv = "CVMouthReader/data/input/script.csv"
#     # csv_cleaner(csv, cols)

# if __name__ == "__main__":
#     main()
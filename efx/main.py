import concurrent.futures
import random


def thread_function(a, b):
    print("{}, {}".format(a, b))
    return 1, 1, 1


def test_threading():
    number_of_threads = 10
    shared_resource = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        future_results = {executor.submit(thread_function, 5, 10): thread_index
                          for thread_index in range(number_of_threads)}
        for future_result in concurrent.futures.as_completed(future_results):
            result = future_results[future_result]
            try:
                data = future_result.result()
            except Exception as exc:
                print("{} generated an exception {}".format(result, exc))
            else:
                shared_resource += data[0] + data[1] + data[2]
                print("Thread#{} returns the result: {}".format(result, data))
    print(shared_resource)


if __name__ == "__main__":
    test_threading()
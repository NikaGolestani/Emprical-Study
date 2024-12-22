import time
import random
import pandas as pd

class SortAlgorithms:
    def __init__(self, array=None, size=None, value_range=None):
        if array is not None:
            self.original_array = array
            self.array = array.copy()
        elif size is not None:
            if value_range is None:
                value_range = (1, 1000)  # Default range if not specified
            self.original_array = [random.randint(value_range[0], value_range[1]) for _ in range(size)]
            self.array = self.original_array.copy()
            self.input_count = size
            self.min_value = value_range[0]
            self.max_value = value_range[1]
        else:
            self.original_array = []
            self.array = []

        self.df = pd.DataFrame(columns=["Algorithm Name", "Time Taken (ms)", "Input Count", "Min", "Max"])

    def _log_to_dataframe(self, method, execution_time):
        """Log sorting method results to a DataFrame."""
        self.df = pd.concat([self.df, pd.DataFrame({
            "Algorithm Name": [method],
            "Time Taken (ms)": [execution_time * 1000],  # Convert seconds to milliseconds
            "Input Count": [self.input_count],
            "Min": [self.min_value],
            "Max": [self.max_value]
        })], ignore_index=True)
        self.resetArray()

    def mergeSort(self, array=None):
        if array is None:
            array = self.array

        if len(array) > 1:
            mid = len(array) // 2
            left_half = array[:mid]
            right_half = array[mid:]

            self.mergeSort(left_half)
            self.mergeSort(right_half)

            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    array[k] = left_half[i]
                    i += 1
                else:
                    array[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                array[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                array[k] = right_half[j]
                j += 1
                k += 1

    def selectionSort(self):
        n = len(self.array)
        for i in range(n):
            min_index = i
            for j in range(i + 1, n):
                if self.array[j] < self.array[min_index]:
                    min_index = j
            self.array[i], self.array[min_index] = self.array[min_index], self.array[i]

    def bubbleSort(self):
        n = len(self.array)
        for i in range(n):
            for j in range(0, n-i-1):
                if self.array[j] > self.array[j + 1]:
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]

    def improvedBubbleSort(self):
        n = len(self.array)
        for i in range(n):
            swapped = False
            for j in range(n - i - 1):
                if self.array[j] > self.array[j + 1]:
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]
                    swapped = True
            if not swapped:
                break

    def quickSort(self, low=None, high=None):
        if low is None and high is None:
            low = 0
            high = len(self.array) - 1

        if low < high:
            pi = self.partition(low, high)
            self.quickSort(low, pi - 1)
            self.quickSort(pi + 1, high)

    def partition(self, low, high):
        pivot = self.array[high]
        i = low - 1
        for j in range(low, high):
            if self.array[j] < pivot:
                i += 1
                self.array[i], self.array[j] = self.array[j], self.array[i]
        self.array[i + 1], self.array[high] = self.array[high], self.array[i + 1]
        return i + 1

    def improvedQuickSort(self, low=None, high=None):
        if low is None and high is None:
            low = 0
            high = len(self.array) - 1

        while low < high:
            pi = self.partition(low, high)

            # Process the smaller subarray first
            if pi - low < high - pi:
                self.improvedQuickSort(low, pi - 1)
                low = pi + 1  # Tail recursion optimization
            else:
                self.improvedQuickSort(pi + 1, high)
                high = pi - 1  # Tail recursion optimization

    def radixSort(self):
        """Standard Radix Sort implementation."""
        max_num = max(self.array)  # Find the maximum number to determine the number of digits

        exp = 1  # Start with the least significant digit
        while max_num // exp > 0:
            self.countSort(exp)  # Perform counting sort for the current digit
            exp *= 10  # Move to the next significant digit


    def countSort(self, exp, track_sorted=False):
        """
        Counting Sort used by Radix Sort.
        :param exp: The current digit's place value (1 for units, 10 for tens, etc.).
        :param track_sorted: If True, returns whether the array is already sorted.
        """
        n = len(self.array)
        output = [0] * n  # Output array to store sorted numbers
        count = [0] * 10  # Count array for digits (0-9)

        # Count occurrences of each digit in the current place value
        for i in range(n):
            index = self.array[i] // exp
            count[index % 10] += 1

        # Update count[i] so it contains the position of the next element in output[]
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Build the output array
        i = n - 1
        while i >= 0:
            index = self.array[i] // exp
            output[count[index % 10] - 1] = self.array[i]
            count[index % 10] -= 1
            i -= 1

        # Check if the array is already sorted (optional, for improvedRadixSort)
        already_sorted = True
        for i in range(1, n):
            if output[i] < output[i - 1]:
                already_sorted = False
                break

        # Copy the sorted numbers back to the original array
        for i in range(n):
            self.array[i] = output[i]

        if track_sorted:
            return already_sorted


    def resetArray(self):
        """Reset array to the original state."""
        self.array = self.original_array.copy()



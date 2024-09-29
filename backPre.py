# Apply some thresholds to post-process the model results.

import numpy as np
import torch

# Calculate the distribution of 0 and 1 in the prediction results.
def getConsecutiveNum(array):
    count_zeros = []
    count_ones = []
    start_zeros = []
    start_ones = []

    # Initialize the current count, current value, and starting position.
    current_count = 1
    current_value = array[0]
    start_index = 0

    # Traverse the array, starting from the second element.
    for i in range(1, len(array)):
        if array[i] == current_value:
            current_count += 1
        else:
            # Add the count and starting position to the respective lists based on the current value.
            if current_value == 0:
                count_zeros.append(current_count)
                start_zeros.append(start_index)
            else:
                count_ones.append(current_count)
                start_ones.append(start_index)

            # Reset the current count, current value, and starting position.
            current_value = array[i]
            current_count = 1
            start_index = i

    # Add the count and starting position of the last sequence to the respective lists
    if current_value == 0:
        count_zeros.append(current_count)
        start_zeros.append(start_index)
    else:
        count_ones.append(current_count)
        start_ones.append(start_index)

    return count_zeros, count_ones, start_zeros, start_ones

def back_pre(forecast):
    # Copy forecast tensor
    if isinstance(forecast, np.ndarray):
        forecast_copy = forecast.copy()
    elif isinstance(forecast, torch.Tensor):
        forecast_copy = forecast.clone()

    for i in range(forecast_copy.shape[0]):
        # 1. Handle the occurrence of short sequences of 0 or 1.
        trod_continuity = 10  # A threshold of how many consecutive short points is considered an error.
        count_zeros, count_ones, start_zeros, start_ones = getConsecutiveNum(forecast_copy[i])
        for j in range(len(count_ones)):
            if count_ones[j] < trod_continuity:
                start = start_ones[j]
                end = start_ones[j] + count_ones[j]
                forecast_copy[0, start:end] = 0
        count_zeros, count_ones, start_zeros, start_ones = getConsecutiveNum(forecast_copy[i])  # Update the 0 and 1 distribution in the forecast_copy array.
        for k in range(len(count_zeros)):
            if count_zeros[k] < trod_continuity:
                start = start_zeros[k]
                end = start_zeros[k] + count_zeros[k]
                forecast_copy[0, start:end] = 1

        # 2. Check if there are two consecutive 1s that are too close to each other.
        count_zeros, count_ones, start_zeros, start_ones = getConsecutiveNum(forecast_copy[i])
        if len(count_ones) > 1:  # Only when the segment of 1s consists of at least two elements is it necessary to check if two 1s are too close to each other.
            trod_near120_big = 120
            trod_near120_small = 20
            trod_close = 100  # Based on the heart rate, the interval from the previous QRS offset to the next QRS onset should not be less than the specified threshold.
            for k in range(len(count_ones)-1):
                # If the combined length of the 0 segment between two adjacent 1 segments is close to 120 ms, merge these two 1 segments.
                start = start_ones[k]
                end = start_ones[k+1] + count_ones[k+1]
                if end-start >= trod_near120_small and end-start <= trod_near120_big:  # After merging the two segments, the length of the new QRS region must fall within a specified range.
                    forecast_copy[0, start:end] = 1


                start = start_ones[k] + count_ones[k]
                end = start_ones[k+1]
                if end-start <= trod_close:  # Determine which of the two segments is closer to a real QRS.
                    if abs(count_ones[k]-60) <= abs(count_ones[k+1]-60):
                        forecast_copy[0, start_ones[k+1]:start_ones[k+1] + count_ones[k+1]] = 0
                    else:
                        forecast_copy[0, start_ones[k]:start_ones[k] + count_ones[k]] = 0

        # 3. Remove potential QRS complexes that do not meet the minimum width requirement.
        count_zeros, count_ones, start_zeros, start_ones = getConsecutiveNum(forecast_copy[i])
        trod_longEnough = 30  # The width of the QRS complex should not be below a specified threshold.
        for k in range(len(count_ones)):
            if start_ones[k] != 0 and start_ones[k]+count_ones[k] != forecast_copy.shape[1]-1 and count_ones[k] <= trod_longEnough:
                start = start_ones[k]
                end = start_ones[k] + count_ones[k]
                forecast_copy[0, start:end] = 0
            pass

    return forecast_copy


import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np

##Set Max recursion
import sys
sys.setrecursionlimit(10000)

# Utitilty functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(run_arr, mean, plot_title):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.bar(x,run_arr)
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.text(mean, 10, f'Mean: {mean:.2f}', color='red', fontsize=12)
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms order of 1e-6")
    plt.title(plot_title)
    plt.savefig(plot_title)
    ##plt.show()

# function to generate random list 
# @args : length = number of items 
#       : max_value maximum value
def create_random_list(length, max_value, item=None, item_index=None):
    random_list = [random.randint(0,max_value) for i in range(length)]
    if item!= None:
        random_list.insert(item_index,item)

    return random_list

# function to generate reversed list of a given size and with a given maximum value
def create_reverse_list(length, max_value, item=None, item_index=None):
    reversed_list = create_random_list(length,max_value)
    reversed_list.sort()
    reversed_list.reverse()

    return reversed_list

# function to generate near sorted list of a given size and with a given maximum value
def create_near_sorted_list(length, max_value, item=None, item_index=None):
    near_sorted_list = [i for i in range(length)]

    for _ in range(length // 8):
        swap1 = random.randint(0,length - 1)
        swap2 = random.randint(0,length - 1)
        near_sorted_list[swap1],near_sorted_list[swap2] = near_sorted_list[swap2],near_sorted_list[swap1]
    #include your code here

    return near_sorted_list

# function to generate near sorted list of a given size and with a given maximum value
def reduced_unique_list(length, max_value, item=None, item_index=None):
    reduced_list = []
    random_list = create_random_list(length,max_value)
    seen = set()

    for num in random_list:
        if num not in seen:
            seen.add(num)
            reduced_list.append(num)
    
    #include your code here

    return reduced_list

# Implementation of sorting algorithms
class BubbleSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

    ### your implementation for bubble sort goes here 
    def bubble_sort(self,):
        self.sorted_items = self.items.copy()
        L = self.sorted_items
        n = len(L)
        ##Loop through list
        for i in range (n):
            ##From i to sorted part of the list
            for j in range(0,n-i-1):
                ##Push the max element to the end of the non sorted part of the list
                if L[j] > L[j+1]:
                    L[j], L[j+1] = L[j+1], L[j]
        

    def get_sorted(self,):
        self.bubble_sort()
        return self.sorted_items
    
class InsertionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

       ### your implementation for insertion sort goes here
    def insertion_sort(self,):
        self.sorted_items = self.items.copy()
        L = self.sorted_items
        n = len(L)
        ##Loop through list
        for i in range (n):
            j = i
            ## comment stuff
            while(j > 0):
                if L[j] < L[j-1]:
                    L[j], L[j-1] = L[j-1], L[j]
                j-= 1 

    def get_sorted(self,):
        self.insertion_sort()
        return self.sorted_items
    
class SelectionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

    def selection_sort(self,):
        self.sorted_items = self.items.copy()
        L = self.sorted_items
        n = len(L)
        ##Loop through the list
        for i in range (n):
            minIndex = i
            ##Find the minimum from index i to end of list
            for j in range (i+1, n):
                if( L[j] < L[minIndex]):
                    minIndex = j
            ##Swap the ith element with the index of the mimimum
            L[i], L[minIndex] = L[minIndex], L[i]

    def get_sorted(self,):
        self.selection_sort()
        return self.sorted_items
    
class MergeSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

    ### your implementation for selection sort goes here 
    aux = []
    def merge(self, low, mid, high):
        L = self.sorted_items
        MergeSort.aux = L.copy()
        left = low
        right = mid + 1
        for i in range (low, high + 1 ):
            MergeSort.aux[i] = L[i]
            
        
        for i in range(low,high + 1):
            if(left > mid):
                L[i] = MergeSort.aux[right]
                right += 1
            elif(right > high):
                L[i] = MergeSort.aux[left]
                left += 1
            elif(MergeSort.aux[left] < MergeSort.aux[right]):
                L[i] = MergeSort.aux[left]
                left += 1
            else:
                L[i] = MergeSort.aux[right]
                right += 1


    def merge_sort(self,):
        self.sorted_items = self.items.copy()
        self.sort(0, len(self.sorted_items) - 1)

    def sort(self, low, high):
        if high <= low:
            return
        mid = low + (high - low)//2
        self.sort(low, mid)
        self.sort(mid+1,high)
        self.merge(low,mid,high)

    def get_sorted(self,):
        self.merge_sort()
        return self.sorted_items

class QuickSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

    ### your implementation for selection sort goes here

    def quick_sort(self,):
        self.sorted_items = self.items.copy()
        self.quick(0, len(self.items) - 1)
        

    def quick(self,low,high):
        if high <= low:
            return
        j = self.partition(low,high)
        self.quick(low, j-1)
        self.quick(j+1, high)

    def partition(self,low,high):
        L = self.sorted_items
        pivotValue = L[low]
        i = low + 1
        j = high 

        while(True):
            while (L[i] < pivotValue):
                i += 1
                if i >= high:
                    break

            while (L[j] > pivotValue):
                j -= 1
                if j <= low:
                    break
            
            if (i >= j):
                break
            L[i],L[j] = L[j], L[i]
            i += 1
            j -= 1
        L[low],L[j] = L[j], L[low]
        return j

    def get_sorted(self,):
        self.quick_sort()
        return self.sorted_items

# test all algorithm implementations
test_case = [42, 7, 19, 88, 5, 33, 61, 14, 28, 90, 12, 55, 73, 61, 3, 25, 90, 33, 94, 29, 41]
print("Test case array input: ",test_case)

##Bubble Sort test
bubble_sort = BubbleSort(test_case)
bubble_sort.bubble_sort()
print("Bubble Sort:", bubble_sort.get_sorted())

##Inesrtion Sort test
insertion_sort = InsertionSort(test_case)
insertion_sort.insertion_sort()
print("Insertion Sort:", insertion_sort.get_sorted())

##Selection Sort test
selection_sort = SelectionSort(test_case)
selection_sort.selection_sort()
print("Selection Sort:", selection_sort.get_sorted())

##Merge Sort test
merge_sort = MergeSort(test_case)
merge_sort.merge_sort()
print("Merge Sort:", merge_sort.get_sorted())

#example run for QuickSort
quick_sort = QuickSort(test_case)
quick_sort.quick_sort()
print("Quick Sort: ",quick_sort.get_sorted())


def runExperiment(sortObject, N, title):
    run_times = []
    for i in range (N):
        start = timeit.default_timer()
        sortObject.get_sorted()
        stop = timeit.default_timer()
        run_times.append((stop - start) * (10**6))
    bubbleSortAverage = np.sum(run_times)/len(run_times)
    draw_plot(run_times,bubbleSortAverage, title)

# run all algorithms
def experiment_A():
    
    # Insert your code for experiment A design here

    ##Generate random list and set number of times to run experiment
    toSort = create_random_list(10000,10000) 
    N = 80
    bubble_sort_A = BubbleSort(toSort)
    insertion_sort_A = InsertionSort(toSort)
    selection_sort_A = SelectionSort(toSort)
    merge_sort_A = MergeSort(toSort)
    quick_sort_A = QuickSort(toSort)

    runExperiment(bubble_sort_A,N, "Bubble Sort Experiment A")
    runExperiment(insertion_sort_A,N, "Insert Sort Experiment A")
    runExperiment(selection_sort_A,N, "Selection Sort Experiment A")
    runExperiment(merge_sort_A,N, "Merge Sort Experiment A")
    runExperiment(quick_sort_A,N, "Quick Sort Experiment A")

    return 0

def experiment_B():
    
    # Insert your code for experiment B design here 

    toSort = create_near_sorted_list(5000,5000) 
    N = 100
    bubble_sort_B = BubbleSort(toSort)
    insertion_sort_B = InsertionSort(toSort)
    selection_sort_B = SelectionSort(toSort)
    merge_sort_B = MergeSort(toSort)
    quick_sort_B = QuickSort(toSort)

    runExperiment(bubble_sort_B,N, "Bubble Sort Experiment B")
    runExperiment(insertion_sort_B,N, "Insert Sort Experiment B")
    runExperiment(selection_sort_B,N, "Selection Sort Experiment B")
    runExperiment(merge_sort_B,N, "Merge Sort Experiment B")
    runExperiment(quick_sort_B,N, "Quick Sort Experiment B")
    return 0

def experiment_C():
    
    # Insert your code for experiment C design here 
    toSort = create_reverse_list(10000,10000) 
    N = 100
    bubble_sort_C = BubbleSort(toSort)
    insertion_sort_C = InsertionSort(toSort)
    selection_sort_C = SelectionSort(toSort)
    merge_sort_C = MergeSort(toSort)
    quick_sort_C = QuickSort(toSort)

    runExperiment(bubble_sort_C,N, "Bubble Sort Experiment C")
    runExperiment(insertion_sort_C,N, "Insert Sort Experiment C")
    runExperiment(selection_sort_C,N, "Selection Sort Experiment C")
    runExperiment(merge_sort_C,N, "Merge Sort Experiment C")
    runExperiment(quick_sort_C,N, "Quick Sort Experiment C")
    return 0

def experiment_D():
    
    # Insert your code for experiment D design here 

    return 0

def experiment_E():
    
    # Insert your code for experiment E design here 
    toSort = create_reverse_list(5000,5000) 
    N = 100
    bubble_sort_E = BubbleSort(toSort)
    insertion_sort_E = InsertionSort(toSort)
    selection_sort_E = SelectionSort(toSort)
    merge_sort_E = MergeSort(toSort)
    quick_sort_E = QuickSort(toSort)

    runExperiment(bubble_sort_E,N, "Bubble Sort Experiment E")
    runExperiment(insertion_sort_E,N, "Insert Sort Experiment E")
    runExperiment(selection_sort_E,N, "Selection Sort Experiment E")
    runExperiment(merge_sort_E,N, "Merge Sort Experiment E")
    runExperiment(quick_sort_E,N, "Quick Sort Experiment E")
    return 0

# call each experiment
#experiment_A()
#experiment_B()
experiment_C()
experiment_D()
experiment_E()


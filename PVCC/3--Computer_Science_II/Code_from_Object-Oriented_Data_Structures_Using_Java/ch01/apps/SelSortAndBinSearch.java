//----------------------------------------------------------------------------
// SelSortAndBinSearch.java        by Dale/Joyce/Weems               Chapter 1
//
// Demonstrates binary search.
// Demonstrates Selection Sort.
//----------------------------------------------------------------------------
package ch01.apps;

import java.util.Random;
import java.text.DecimalFormat;

public class SelSortAndBinSearch
{
  static final int SIZE = 100;   // size of array to be searched
  static int[] values = new int[SIZE];   // values to be sorted

  static void initValues()
  // Initializes the values array with random integers from 0 to SIZE.
  {
    Random rand = new Random();
    for (int index = 0; index < SIZE; index++)
      values[index] = Math.abs(rand.nextInt()) % SIZE;
  }

  static public boolean isSorted()
  // Returns true if the array values are sorted and false otherwise.
  {
    boolean sorted = true;
    for (int index = 0; index < (SIZE - 1); index++)
      if (values[index] > values[index + 1])
        sorted = false;
    return sorted;
  }

  static public void swap(int index1, int index2)
  // Precondition: index1 and index2 are >= 0 and < SIZE.
  //
  // Swaps the integers at locations index1 and index2 of the values array. 
  {
    int temp = values[index1];
    values[index1] = values[index2];
    values[index2] = temp;
  }

  static public void printValues()
  // Prints all the values integers.
  {
    int value;
    DecimalFormat fmt = new DecimalFormat("000");
    System.out.println("The values array is:");
    for (int index = 0; index < SIZE; index++)
    {
      value = values[index];
      if (((index + 1) % 10) == 0)
        System.out.println(fmt.format(value));
      else
        System.out.print(fmt.format(value) + " ");
    }
    System.out.println();
  }

  /////////////////////////////////////////////////////////////////
  //
  //  Selection Sort

  static int minIndex(int startIndex, int endIndex)
  // Returns the index of the smallest value in
  // values[startIndex]..values[endIndex].
  {
    int indexOfMin = startIndex;
    for (int index = startIndex + 1; index <= endIndex; index++)
      if (values[index] < values[indexOfMin])
        indexOfMin = index;
    return indexOfMin;
  }

  static void selectionSort()
  // Sorts the values array using the selection sort algorithm.
  {
    int endIndex = SIZE - 1;
    for (int current = 0; current < endIndex; current++)
      swap(current, minIndex(current, endIndex));
  }

  //
  //  Selection Sort End
  /////////////////////////////////////////////////////////////////


  static boolean binarySearch(int target, int first, int last)
  // Preconditions: first and last are legal indices of values
  //                first + last <= Integer.MAX_Value
  //
  // If target is contained in values[first,last] return true
  // otherwise return false
  {
    int midpoint;
    while (first <= last)
    {
      midpoint = (first + last) / 2;
      if (target == values[midpoint])
        return true;
      else
      if (target > values[midpoint])  // target too high
        first = midpoint + 1;
      else                            // target too low
        last = midpoint - 1;
    }
    return false;
  }

 
  /////////////////////////////////////////////////////////////////
  //
  //  Main

  public static void main(String[] args)
  {
    initValues();
    printValues();
    System.out.println("values is sorted: " + isSorted());
    System.out.println();    
    selectionSort();
    printValues();
    System.out.println("values is sorted: " + isSorted());
    System.out.println();
    
    System.out.println("Binary Search:");
    System.out.println("0 is in array: " + binarySearch(0, 0, SIZE - 1));
    System.out.println("1 is in array: " + binarySearch(1, 0, SIZE - 1));
    for (int i = (SIZE/2 - 2); i <= (SIZE/2 + 2); i++)
      System.out.println(i + " is in array: " + binarySearch(i, 0, SIZE - 1));
    System.out.println(SIZE - 1 + " is in array: " + binarySearch(SIZE - 1, 0, SIZE - 1));
    System.out.println(SIZE + " is in array: " + binarySearch(SIZE, 0, SIZE - 1));
    System.out.println();
  }
}

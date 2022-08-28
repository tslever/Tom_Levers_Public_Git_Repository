//----------------------------------------------------------------------------
// BinarySearch.java           by Dale/Joyce/Weems                   Chapter 3
//
// Demonstrates both iterative and resursive binary search.
//----------------------------------------------------------------------------
package ch03.apps;

import java.util.Random;
import java.text.DecimalFormat;

public class BinarySearch
{
  static final int SIZE = 100;   // size of array to be searched
  static int[] values = new int[SIZE];   // array to be searched

  static void initValues()
  // Initializes the values array with non-decreasing 'random' integers 
  {
    Random rand = new Random();
    values[0] = 0;
    for (int index = 1; index < SIZE; index++)
      values[index] = values[index - 1] + rand.nextInt(4);
  }

  static void printValues()
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

  static boolean binarySearchIterative(int target, int first, int last)
  // Precondition: first and last are legal indices of values
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

  static boolean binarySearchRecursive(int target, int first, int last)
  // Precondition: first and last are legal indices of values
  //
  // If target is contained in values[first,last] return true
  // otherwise return false
  {
    int midpoint = (first + last) / 2;
    if (first > last)
      return false;
    else
      if (target == values[midpoint])
        return true;
      else
      if (target > values[midpoint])  // target too high
        return binarySearchRecursive(target, midpoint + 1, last);
      else                            // target too low
        return binarySearchRecursive(target, first, midpoint - 1);
  }

 
  /////////////////////////////////////////////////////////////////
  //
  //  Main

  public static void main(String[] args)
  {
    initValues();
    printValues();
    System.out.println();
    
    System.out.println("Iterative Binary Search:");
    System.out.println("0 is in array: " + binarySearchIterative(0, 0, SIZE - 1));
    System.out.println("1 is in array: " + binarySearchIterative(1, 0, SIZE - 1));
    System.out.println("10 is in array: " + binarySearchIterative(10, 0, SIZE - 1));
    System.out.println("20 is in array: " + binarySearchIterative(20, 0, SIZE - 1));
    System.out.println("100 is in array: " + binarySearchIterative(100, 0, SIZE - 1));
    System.out.println(values[SIZE/2] + " is in array: " 
                       + binarySearchIterative(values[SIZE/2], 0, SIZE - 1));
    System.out.println(values[SIZE - 1] + " is in array: " 
                       + binarySearchIterative(values[SIZE - 1], 0, SIZE - 1));
    System.out.println("5000 is in array: " + binarySearchIterative(5000, 0, SIZE - 1));
    System.out.println();

    System.out.println("Recursive Binary Search:");
    System.out.println("0 is in array: " + binarySearchRecursive(0, 0, SIZE - 1));
    System.out.println("1 is in array: " + binarySearchRecursive(1, 0, SIZE - 1));
    System.out.println("10 is in array: " + binarySearchRecursive(10, 0, SIZE - 1));
    System.out.println("20 is in array: " + binarySearchRecursive(20, 0, SIZE - 1));
    System.out.println("100 is in array: " + binarySearchRecursive(100, 0, SIZE - 1));
    System.out.println(values[SIZE/2] + " is in array: " 
                       + binarySearchRecursive(values[SIZE/2], 0, SIZE - 1));
    System.out.println(values[SIZE - 1] + " is in array: " 
                       + binarySearchRecursive(values[SIZE - 1], 0, SIZE - 1));
    System.out.println("5000 is in array: " + binarySearchRecursive(5000, 0, SIZE - 1));
    System.out.println();
  }
}

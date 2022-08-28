//----------------------------------------------------------------------
// TestFactorial.java         by Dale/Joyce/Weems              Chapter 3
//
// Repeatedly prompts user for a non-negative integer n.
// Outputs n! calculated recursively and iteratively.
// Uses a command line interface.
//----------------------------------------------------------------------
package ch03.apps;

import java.util.Scanner;

public class TestFactorial
{
  private static int factorial(int n)
  // Precondition: n is nonnegative
  //
  // Returns the value of "n!"
  {
    if (n == 0)
      return 1;
    else
      return (n * factorial (n - 1));
  }
   
  public static int factorial2(int n)
  // Precondition: n is nonnegative
  //
  // Returns the value of retValue: n!
  {
    int retValue = 1;     
    while (n != 0)
    {
      retValue = retValue * n;
      n = n - 1;
    }
  return(retValue);
  }
 
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);
    System.out.println("Factorial calculation test: "); 
    int n = 0;
    while (n >= 0)
    {
      System.out.print("\nEnter n (-1 to stop) > ");
      n = scan.nextInt();
      if (n >= 0)
      {
         System.out.println(n + "! is");
         System.out.println("\t recursive " + factorial(n));
         System.out.println("\t iterative " + factorial2(n));
      }
    }
  } 
}

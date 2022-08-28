//----------------------------------------------------------------------
// BalancedCLI.java         by Dale/Joyce/Weems                Chapter 2
//
// Checks for balanced grouping symbols.
// Input consists of a sequence of expressions, one per line.
// Special symbol types are (), [], and {}.
// Uses a command line interface.
//----------------------------------------------------------------------

package ch02.apps;

import java.util.Scanner;
import ch02.balanced.*;

public class BalancedCLI
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);

    // Instantiate new Balanced class with grouping symbols.
    Balanced bal = new Balanced("([{", ")]}");
    
    int result; // 0 = balanced, 1 = unbalanced, 2 = premature end

    String expression = null;    // expression to be evaluated
    final String STOP = "X";     // indicates end of input

    while (!STOP.equals(expression))
    {
      // Get next expression to be processed.
      System.out.print("Expression (" + STOP + " to stop): ");
      expression = scan.nextLine();
      if (!STOP.equals(expression))
      {      
         // Obtain and output result of balanced testing.
         result = bal.test(expression);
         if (result == 1)
           System.out.println("Unbalanced \n");
         else
         if (result == 2)
           System.out.println("Premature end of expression \n");
         else
           System.out.println("Balanced \n");
      }
    }
  }
}
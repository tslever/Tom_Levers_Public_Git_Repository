//---------------------------------------------------------------------
// PalindromeCLI.java       by Dale/Joyce/Weems               Chapter 4
//
// Checks for strings that are palidromes.
// Input consists of a sequence of strings.
// Output consists of whether the input string is a palindrome.
// Interacts with the user through the command line interface.
//---------------------------------------------------------------------
package ch04.apps;

import java.util.Scanner;
import ch04.palindromes.Palindrome;

public class PalindromeCLI 
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);

    String candidate = null;     // string to be evaluated
    final String STOP = "X";     // indicates end of input

    while (!STOP.equals(candidate))
    {
      // Get next candidate string to be processed.
      System.out.print("String (" + STOP + " to stop): ");
      candidate = scan.nextLine();
      if (!STOP.equals(candidate))
      {            
        // Obtain and output result of palindrome testing.
        if (Palindrome.test(candidate))
          System.out.println("is a palindrome.\n");
        else
          System.out.println("is NOT a palindrome.\n");
      }
    }
  }
}
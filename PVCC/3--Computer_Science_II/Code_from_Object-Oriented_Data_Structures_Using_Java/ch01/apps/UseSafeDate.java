//----------------------------------------------------------------------
// UseSafeDate.java          by Dale/Joyce/Weems               Chapter 1
//
// Example of catching exceptions thrown by SafeDate class 
//----------------------------------------------------------------------

package ch01.apps;

import java.util.Scanner;
import ch01.dates.*;

public class UseSafeDate
{
  public static void main(String[] args)
  {
    int month, day, year;
    SafeDate theDate;
    boolean DateOK = false;
    Scanner scan = new Scanner(System.in);
 
    while (!DateOK)
    {
      System.out.println("Enter a date (month day and year):");
      month = scan.nextInt(); day = scan.nextInt(); year = scan.nextInt();
      try
      {
        theDate = new SafeDate(month, day, year);
        DateOK = true;
        System.out.println(theDate + " is a safe date.");
      }
      catch(DateOutOfBoundsException DateOBExcept)
      {
        System.out.println(DateOBExcept.getMessage() + "\n");
      }
    }
  }
}

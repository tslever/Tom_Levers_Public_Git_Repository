//----------------------------------------------------------------------
// DaysBetween.java          by Dale/Joyce/Weems               Chapter 1
//
// Asks the user to enter two "modern" dates and then reports 
// the number of days between the two dates.
//----------------------------------------------------------------------

package ch01.apps;

import java.util.Scanner;
import ch01.dates.*;

public class DaysBetween
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);
    int day, month, year;
      
    System.out.println("Enter two 'modern' dates: month day year");
    System.out.println("For example, January 21, 1939, would be: 1 21 1939");
    System.out.println();
    System.out.println("Modern dates are not before " + Date.MINYEAR + ".");
    System.out.println();
      
    System.out.println("Enter the first date:");
    month = scan.nextInt(); day = scan.nextInt(); year = scan.nextInt();
    Date d1 = new Date(month, day, year);
      
    System.out.println("Enter the second date:");
    month = scan.nextInt(); day = scan.nextInt(); year = scan.nextInt();
    Date d2 = new Date(month, day, year);
      
    if ((d1.getYear() <= Date.MINYEAR) || (d2.getYear() <= Date.MINYEAR))
      System.out.println("You entered a 'pre-modern' date.");
    else
    {
      System.out.println("The number of days between");
      System.out.print(d1 + " and " + d2 + " is ");
      System.out.println(Math.abs(d1.lilian() - d2.lilian()));
    }
  }
}
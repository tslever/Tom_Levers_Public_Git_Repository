//---------------------------------------------------------------------------
// CSPeople.java            by Dale/Joyce/Weems                     Chapter 6
//
// Reads information about famous computer scientists from the file 
// input/FamousCS.txt. Allows user to indicate if they wish to see the list 
// sorted by name or by year of birth.
//---------------------------------------------------------------------------
package ch06.apps;

import java.io.*;
import java.util.*;
import ch06.lists.*;
import support.*;

public class CSPeople
{
  public static void main(String[] args) throws IOException 
  {
    // Get user's display preference
    Scanner scan = new Scanner(System.in);
    int choice;
    System.out.println("  1: Sorted by name? \n  2: Sorted by year of birth?");
    System.out.print("\nHow would you like to see the information > ");
    choice = scan.nextInt();
    
    // Instantiate sorted list
    SortedABList<FamousPerson> people;
    if (choice == 1)
      people = new SortedABList<FamousPerson>(); // defaults to natural order
    else
      people = new SortedABList<FamousPerson>(FamousPerson.yearOfBirthComparator());
    
    // Set up file reading
    FileReader fin = new FileReader("input/FamousCS.txt");
    Scanner info = new Scanner(fin);
    info.useDelimiter("[,\\n]");  // delimiters are commas, line feeds
    FamousPerson person;
    String fname, lname, fact;
    int year;

    // Read the info from the file and add it to the list
    while (info.hasNext())      
    {
      fname = info.next();   lname = info.next();
      year = info.nextInt(); fact = info.next();
      person = new FamousPerson(fname, lname, year, fact);
      people.add(person);
    }
    
    // Display the list, using the advanced for loop
    System.out.println();
    for (FamousPerson fp: people)
       System.out.println(fp);
  } 
} 
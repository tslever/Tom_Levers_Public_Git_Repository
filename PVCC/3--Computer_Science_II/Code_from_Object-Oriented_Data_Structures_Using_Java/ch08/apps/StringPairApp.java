//---------------------------------------------------------------------------
// StringPairApp.java          by Dale/Joyce/Weems                  Chapter 8
//
// Reads # separated pairs of strings from specified input file.
// First pair of strings provides descriptive info.
// Remaining pairs of strings stored in a map as key - value pairs.
// Prompts user for a key and if it exists, displays the associated value.
//---------------------------------------------------------------------------
package ch08.apps;

import java.io.*;
import java.util.*;
import ch08.maps.*;

public class StringPairApp
{
  public static void main(String[] args) throws IOException 
  {
    // Create map
    MapInterface<String, String> pairs = new ArrayListMap<String, String>();

    // Set up file reading
    String fname = args[0];       // input file of text
    FileReader fin = new FileReader(fname);
    Scanner info = new Scanner(fin);
    info.useDelimiter("[#\\n\\r]");  // delimiters are # signs, line feeds,
                                     // carriage returns

    // get information about the key and value
    String keyInfo = info.next();
    String valueInfo = info.next();
    info.nextLine();
    
    // Reads the key/value pairs from the file and puts them into the map
    String key, value;
    while (info.hasNext())      
    {
      key = info.next();   value = info.next();
      info.nextLine();
      pairs.put(key, value);
    }
    
    // Interact with user, getting keys and displaying value
    Scanner scan = new Scanner(System.in);
    final String STOP = "XX";     
    key = null;
    while (!STOP.equals(key))
    {
      System.out.println("\nEnter " + keyInfo + " (" + STOP + " to exit):");
      key = scan.next(); 
      if (!STOP.equals(key))
        if (pairs.contains(key))
          System.out.println(valueInfo + "\t" + pairs.get(key));
        else
          System.out.println("\tNo information available.");
    }
  } 
}
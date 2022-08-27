//---------------------------------------------------------------------------
// StringPairMap.java          by Dale/Joyce/Weems                  Chapter 8
//
// Reads comma separated pairs of strings from specified input file.
// First pair of strings provide descriptive info.
// Remaining pairs of strings stored in a map as key - value pairs.
// Prompts user for a key and if it exists, displays the associated value.
//---------------------------------------------------------------------------
package ch08.apps;

import java.io.*;
import java.util.*;
import ch08.maps.*;
import java.awt.Color;

public class ColorMap
{
  public static void main(String[] args) throws IOException 
  {
Color c = new Color(5,27,45);
System.out.println(c);
/*
    // Create map
    MapInterface<String, String> map = new ArrayListMap<String, String>();

    // Set up file reading
    String fname = args[0];       // input file of text
    FileReader fin = new FileReader(fname);
    Scanner info = new Scanner(fin);
    info.useDelimiter("[,\\n]");  // delimiters are commas, line feeds

    // get information about the key and value
    String keyInfo = info.next();
    String valueInfo = info.next();
    
    // Read the info from the file and put in map
    String key, value;
    while (info.hasNext())      
    {
      key = info.next();   value = info.next();
      map.put(key, value);
    }
    
    // Interact with user, getting keys and displaying value
    Scanner scan = new Scanner(System.in);
    final String STOP = "X";     
    key = null;
    while (!STOP.equals(key))
    {
      System.out.println("Enter " + keyInfo + " (" + STOP + " to exit)");
      key = scan.next(); 
      if (!STOP.equals(key))
        if (map.contains(key))
          System.out.println("\t" + map.get(key));
        else
          System.out.println("\tNo information available.\n");
    }

*/

  } 
} 
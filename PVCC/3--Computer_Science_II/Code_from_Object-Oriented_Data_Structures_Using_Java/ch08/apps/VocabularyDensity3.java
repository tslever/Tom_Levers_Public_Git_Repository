//---------------------------------------------------------------------------
// VocabularyDensity3.java      by Dale/Joyce/Weems                 Chapter 8
//
// Displays the number of total words, unique words in the input text file,
// and the resulting vocabulary density.
// Input file indicated by a command line argument.
// Uses ArrayListMap.
//---------------------------------------------------------------------------
package ch08.apps;

import java.io.*;
import java.util.*;
import ch08.maps.*;

public class VocabularyDensity3
{
  public static void main(String[] args) throws IOException 
  {
    final int CAPACITY = 1000;   // original capacity of collection
    String fname = args[0];      // input file of text
    String word;                 // current word
    int numWords = 0;            // total number of words
    int uniqWords;               // number of unique words
    double density;              // vocabulary density

    MapInterface<String, Object> words = new HMap<String, Object>();

    // Set up file reading
    FileReader fin = new FileReader(fname);
    Scanner wordsIn = new Scanner(fin);
    wordsIn.useDelimiter("[^a-zA-Z']+");  // delimiters are nonletters,'

    while (wordsIn.hasNext())      // while more words to process
    {
      word = wordsIn.next();          
      word = word.toLowerCase();
      if (!words.contains(word))
        words.put(word, null);
      numWords++;
    }
  
    density = (double)numWords/words.size();
    System.out.println("Analyzed file " + fname);
    System.out.println("\n\tTotal words:  " + numWords);
    if (words.size() == CAPACITY)
      System.out.println("\tUnique words: at least " + words.size());
    else
    {
      System.out.println("\tUnique words: " + words.size());
      System.out.printf("\n\tVocabulary density: %.2f", density);
    }
  } 
} 
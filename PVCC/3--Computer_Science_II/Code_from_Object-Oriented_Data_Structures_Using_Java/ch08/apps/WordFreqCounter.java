//---------------------------------------------------------------------------
// WordFreqCounter.java         by Dale/Joyce/Weems                 Chapter 8
//
// Displays a word frequency list of the words listed in the input file.
// Prompts user for minSize and minFreq.
// Does not process words less than minSize in length.
// Does not output words unless their frequency is at least minFreq.
// Uses a map of String -> WordFreq.
//---------------------------------------------------------------------------
package ch08.apps;

import java.io.*;
import java.util.Scanner;
import ch08.maps.*;
import support.WordFreq;

public class WordFreqCounter
{
  public static void main(String[] args) throws IOException 
  {
    String word;
    WordFreq wordToAdd;
    WordFreq wordInMap;

    // Create map
    MapInterface<String, WordFreq> words = new ArrayListMap<String, WordFreq>();

    int numWords = 0;
    int numValidWords = 0;
    int numValidFreqs = 0;
    int minSize;
    int minFreq;
    int mapSize;

    // Set up command line reading
    Scanner scan = new Scanner(System.in);

    // Set up file reading
    String fn;
    System.out.print("File name > ");
    fn = scan.next();
    Scanner wordsIn = new Scanner(new FileReader(fn));
    wordsIn.useDelimiter("[^a-zA-Z']");  // delimiters are nonletters,'

    // Get word and frequency limits from user
    System.out.print("Minimum word size> ");
    minSize = scan.nextInt();
    System.out.print("Minimum word frequency> ");
    minFreq = scan.nextInt();

    // Process file
    while (wordsIn.hasNext())      // while more words to process
    {
      word = wordsIn.next();          
      numWords++;
      if (word.length() >= minSize)
      {
        numValidWords++;
        word = word.toLowerCase();
        wordInMap = words.get(word);
        if (wordInMap != null)
        {
          // word already in map, just increment frequency
          wordInMap.inc();
        }
        else
        {
          // insert new word into map
          wordToAdd = new WordFreq(word);
          wordToAdd.inc();               // set frequency to 1
          words.put(word, wordToAdd);
        }
      }
    }
  
    // Display results
    System.out.println("The words of length " + minSize + " and above,");
    System.out.println("with frequency counts of " + minFreq + " and above:");
    System.out.println();
    System.out.println("Freq  Word");
    System.out.println("----- -----------------");
    for (MapEntry<String, WordFreq> entry: words)
    {
      if (entry.getValue().getFreq() >= minFreq)
      {
        numValidFreqs++;
        System.out.println(entry.getKey());
      }
    }
    System.out.println();  
    System.out.println(numWords + " words in the input file.  ");
    System.out.println(numValidWords + " of them are at least " + minSize + " characters.");
    System.out.println(numValidFreqs + " of these occur at least " + minFreq + " times.");
    System.out.println("Program completed.");
  } 
} 
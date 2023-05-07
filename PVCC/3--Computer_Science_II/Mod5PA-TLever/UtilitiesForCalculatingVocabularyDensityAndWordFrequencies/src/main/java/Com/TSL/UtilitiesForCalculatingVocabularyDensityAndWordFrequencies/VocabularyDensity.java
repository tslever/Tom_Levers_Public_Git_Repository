package Com.TSL.UtilitiesForCalculatingVocabularyDensityAndWordFrequencies;


import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;


/**
 * VocabularyDensity encapsulates the entry point of this program, which scans a text file for bases for word objects,
 * increments a total word count, either adds a word corresponding to each basis to a collection of unique words or
 * increments the frequency of the word corresponding to that basis, and outputs information about the text file.
 *
 * @author Tom Lever
 * @version 1.0
 * @since 06/15/21
 */

public class VocabularyDensity 
{
	
	/**
	 * main is the entry point of this program, which scans a text file for bases for word objects, increments a total
	 * word count, either adds a word corresponding to each basis to a collection of unique words or increments the
	 * frequency of the word corresponding to that basis, and outputs information about the text file.
	 * 
	 * @param args
	 * @throws FileNotFoundException
	 */
	
    public static void main(String[] args) throws FileNotFoundException
    {
    	
    	String theFilename = args[0];
    	
    	FileReader theFileReader = new FileReader(theFilename);
    	
    	Scanner theScannerOfTheTextFile = new Scanner(theFileReader);
    	// The delimiter is any number of consecutive characters that are not in a, b, ... z, A, B, ..., Z and are not
    	// apostrophes.
    	theScannerOfTheTextFile.useDelimiter("[^a-zA-Z']+");
    	
        LinkedCollectionT<Word> theCollectionOfUniqueWords = new LinkedCollectionT<Word>();
    	
        String theBasisForTheWord;
        int theNumberOfWords = 0;
        Word theFirstEqualWordInTheCollection;
        
    	while (theScannerOfTheTextFile.hasNext()) {
    		
    		theBasisForTheWord = theScannerOfTheTextFile.next().toLowerCase();
    		theNumberOfWords++;
    		
    		theFirstEqualWordInTheCollection = theCollectionOfUniqueWords.find(new Word(theBasisForTheWord));
    		
    		if (theFirstEqualWordInTheCollection == null) {
    			theCollectionOfUniqueWords.add(new Word(theBasisForTheWord));
    		}
    		
    		else {
    			theFirstEqualWordInTheCollection.increaseFrequency();
    		}
    		
    	}
    	
    	System.out.println("Filename: " + theFilename + "\n");
    	
    	System.out.printf(
    		"Vocabulary density: %.2f\n\n", (double)theNumberOfWords / (double)theCollectionOfUniqueWords.size()
    	);
    	
    	System.out.println("The words in the file (uppercase letters were demoted) and their frequencies:");
    	theCollectionOfUniqueWords.print();
    	System.out.println();
    	
    	System.out.println("After sorting in descending order by frequency:");
    	theCollectionOfUniqueWords.sort();
    	theCollectionOfUniqueWords.print();
        
    }
    
}

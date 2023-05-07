package Com.TSL.UtilitiesForCalculatingVocabularyDensityAndWordFrequencies;


import org.junit.jupiter.api.Test;


/**
 * LinkedCollectionTTest encapsulates a JUnit test of methods of the LinkedCollectionT class.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/15/21
 */

public class LinkedCollectionTTest {

	
	/**
	 * testLinkedCollectionT tests methods of the LinkedCollectionT class by creating an empty collection of words,
	 * evaluating whether the collection is empty and displaying the collection; adding words to the collection,
	 * displaying the number of words in the collection, and displaying the collection; displaying a word equivalent to
	 * "cat"; and removing all instances of particular words from the collection.
	 */
	
	@Test
	public void testLinkedCollectionT() {
		
		LinkedCollectionT<Word> theCollectionOfWords = new LinkedCollectionT<Word>();
		
		System.out.println("Before adding any item, the collection is empty: " + theCollectionOfWords.isEmpty());
		System.out.println("Before adding any item, the collection is:");
		theCollectionOfWords.print();
		System.out.println();
		
		theCollectionOfWords.add(new Word("the"));
		theCollectionOfWords.add(new Word("cat"));
		theCollectionOfWords.add(new Word("sat"));
		theCollectionOfWords.add(new Word("on"));
		theCollectionOfWords.add(new Word("the"));
		theCollectionOfWords.add(new Word("mat"));
		System.out.println("After adding " + theCollectionOfWords.size() + " items, the collection is:");
		theCollectionOfWords.print();
		System.out.println();
		
		System.out.println("After finding a word in the collection equal to a word based on 'cat':");
		System.out.println(theCollectionOfWords.find(new Word("cat")));
		System.out.println();
		
		int theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("the"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'the':");
		theCollectionOfWords.print();
		System.out.println();
		
		theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("cat"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'cat':");
		theCollectionOfWords.print();
		System.out.println();
		
		theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("sat"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'sat':");
		theCollectionOfWords.print();
		System.out.println();
		
		theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("on"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'on':");
		theCollectionOfWords.print();
		System.out.println();
		
		theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("mat"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'mat':");
		theCollectionOfWords.print();
		System.out.println();
		
		theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("mat"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'mat':");
		theCollectionOfWords.print();
		System.out.println();
		
		System.out.println("After adding three instances of 'the':");
		theCollectionOfWords.add(new Word("the"));
		theCollectionOfWords.add(new Word("the"));
		theCollectionOfWords.add(new Word("the"));
		theCollectionOfWords.print();
		System.out.println();
		
		theNumberOfWordsRemoved = theCollectionOfWords.remove(new Word("the"));
		System.out.println("After removing " + theNumberOfWordsRemoved + " instances of 'the':");
		theCollectionOfWords.print();
		System.out.println();
		
	}
	
	
}

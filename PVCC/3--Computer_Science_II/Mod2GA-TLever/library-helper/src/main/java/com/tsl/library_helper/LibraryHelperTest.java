package com.tsl.library_helper;


import org.junit.jupiter.api.Test;


/**
 * LibraryHelperTest encapsulates JUnit tests of core methods of class LibraryHelper.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */

public class LibraryHelperTest {

	
	@Test
	public void testLibraryHelperToFillStack() {
	/**
	 * testLibraryHelperToFillStack tests LibraryHelper by correctly filling a stack of type LibraryHelper.
	 */
		
		System.out.println("Running testLibraryHelperToFillStack.");
		
		StackInterface<String> libraryStack = new LibraryHelper<String>(3);
		
		try {
			for (int i = 0; i < 3; i++) {
				libraryStack.push(String.valueOf(i));
				System.out.println(i + " pushed to libraryStack.");
			}
		}
		catch (StackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testLibraryHelperToOverfillStack() {
	/**
	 * testLibraryHelperToOverfillStack tests LibraryHelper by attempting to overfill a stack of type LibraryHelper.
	 */
		
		System.out.println("Running testLibraryHelperToOverfillStack.");
		
		StackInterface<String> libraryStack = new LibraryHelper<String>(3);
		
		try {
			for (int i = 0; i <= 3; i++) {
				libraryStack.push(String.valueOf(i));
				System.out.println(i + " pushed to libraryStack.");
			}
		}
		catch (StackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testLibraryHelperToGetTopElementFromStackOfOneString() {
	/**
	 * testLibraryHelperToGetTopElementFromStackOfOneString tests LibraryHelper by getting the top element from a
	 * stack of type LibraryHelper with one string.
	 */
		
		System.out.println("Running testLibraryHelperToGetTopElementFromStackOfOneString.");
		
		StackInterface<String> libraryStack = new LibraryHelper<String>(3);
		
		try {
			libraryStack.push(String.valueOf(0));
			System.out.println(libraryStack.top() + " found at top of stack.");
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testLibraryHelperToGetTopElementFromStackOfZeroStrings() {
	/**
	 * testLibraryHelperToGetTopElementFromStackOfZeroStrings tests LibraryHelper by attempting to get the top element
	 * from a stack of type LibraryHelper with zero strings.
	 */
		
		System.out.println("Running testLibraryHelperToGetTopElementFromStackOfZeroStrings.");
		
		StackInterface<String> libraryStack = new LibraryHelper<String>(3);
		
		try {
			System.out.println(libraryStack.top() + " found at top of stack.");
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testLibraryHelperToPopFromStackOfOneString() {
	/**
	 * testLibraryHelperToPopFromStackOfOneString tests LibraryHelper by popping the top element from a stack of type
	 * LibraryHelper with one string.
	 */
		
		System.out.println("Running testLibraryHelperToPopFromStackOfOneString.");
		
		StackInterface<String> libraryStack = new LibraryHelper<String>(3);
		
		try {
			libraryStack.push(String.valueOf(0));
			libraryStack.pop();
			System.out.println("The top element of a stack of type LibraryHelper was popped.");
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testLibraryHelperToPopFromStackOfZeroStrings() {
	/**
	 * testLibraryHelperToPopFromStackOfZeroStrings tests LibraryHelper by attempting to pop the top element from a
	 * stack of type LibraryHelper with zero strings.
	 */
		
		System.out.println("Running testLibraryHelperToPopFromStackOfZeroStrings.");
		
		StackInterface<String> libraryStack = new LibraryHelper<String>(3);
		
		try {
			libraryStack.pop();
			System.out.println("The top element of a stack of type LibraryHelper was popped.");
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}

package com.tsl.array_based_bounded_stack;


import org.junit.jupiter.api.Test;


/**
 * ImprovedArrayBoundedStackTest encapsulates JUnit tests of core methods of class ImprovedArrayBoundedStack.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */

public class ImprovedArrayBoundedStackTest {

	
	@Test
	public void testImprovedArrayBoundedStackByPoppingAllElements() {
	/**
	 * testImprovedArrayBoundedStackByPoppingAllElements tests ImprovedArrayBoundedStack by correctly popping all
	 * elements from  a stack of type ImprovedArrayBoundedStack.
	 */
	
		System.out.println("Running testImprovedArrayBoundedStackByPoppingAllElements.");
		
		ImprovedArrayBoundedStack<String> myStack = new ImprovedArrayBoundedStack<String>(10);
		
		for (int i = 0; i < 3; i++) {
			myStack.push(String.valueOf(i));
		}
		
		try {
			myStack.popSome(3);
			System.out.println("Popped three elements.");
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		
		System.out.println();
		
	}
	
	
	@Test
	public void testImprovedArrayBoundedStackByTryingToPopMoreThanSizeElements() {
	/**
	 * testImprovedArrayBoundedStackByTryingToPopMoreThanSizeElements tests ImprovedArrayBoundedStack by trying to pop
	 * more than size elements from  a stack of type ImprovedArrayBoundedStack, where size is the number of elements on
	 * the stack.
	 */
	
		System.out.println("Running testImprovedArrayBoundedStackByTryingToPopMoreThanSizeElements.");
		
		ImprovedArrayBoundedStack<String> myStack = new ImprovedArrayBoundedStack<String>(10);
		
		for (int i = 0; i < 2; i++) {
			myStack.push(String.valueOf(i));
		}
		
		try {
			myStack.popSome(3);
			System.out.println("Popped three elements.");
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testPopTopWhenStackHasTwoElements() {
	/**
	 * testPopTopWhenStackHasTwoElements tests ImprovedArrayBoundedStack by correctly displaying an element that was
	 * on top of a stack before it was popped off the stack, and the stack after the pop, when the stack had two
	 * elements.
	 */
	
		System.out.println("Running testPopTopWhenStackHasTwoElements.");
		
		ImprovedArrayBoundedStack<String> myStack = new ImprovedArrayBoundedStack<String>(10);
		
		myStack.push(String.valueOf(0));
		myStack.push(String.valueOf(1));
		
		try {
			System.out.println(myStack.poptop() + " popped from stack. Stack is now\n" + myStack.toString());
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	

	@Test
	public void testPopTopWhenStackHasZeroElements() {
	/**
	 * testPopTopWhenStackHasZeroElements tests ImprovedArrayBoundedStack by attempting to display an element that was
	 * on top of a stack before it was popped off the stack, and the stack after the pop, when the stack was empty.
	 */
	
		System.out.println("Running testPopTopWhenStackHasZeroElements.");
		
		ImprovedArrayBoundedStack<String> myStack = new ImprovedArrayBoundedStack<String>(10);
		
		try {
			System.out.println(myStack.poptop() + " popped from stack. Stack is now\n" + myStack.toString());
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}

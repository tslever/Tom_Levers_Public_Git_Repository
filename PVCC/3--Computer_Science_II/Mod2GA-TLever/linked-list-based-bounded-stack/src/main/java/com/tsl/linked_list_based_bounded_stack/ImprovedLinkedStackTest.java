package com.tsl.linked_list_based_bounded_stack;


import org.junit.jupiter.api.Test;


/**
 * ImprovedArrayBoundedStackTest encapsulates JUnit tests of core methods of class ImprovedArrayBoundedStack.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */

public class ImprovedLinkedStackTest {

	
	@Test
	public void testImprovedLinkedStackByPoppingAllElements() {
	/**
	 * testImprovedLinkedStackByPoppingAllElements tests ImprovedLinkedStack by correctly popping all elements from a
	 * stack of type ImprovedLinkedStack.
	 */
	
		System.out.println("Running testImprovedLinkedStackByPoppingAllElements.");
		
		ImprovedLinkedStack<String> myStack = new ImprovedLinkedStack<String>();
		
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
	public void testImprovedLinkedStackByTryingToPopMoreThanSizeElements() {
	/**
	 * testImprovedLinkedStackByTryingToPopMoreThanSizeElements tests ImprovedLinkedStack by trying to pop more than
	 * size elements from  a stack of type ImprovedLinkedStack, where size is the number of elements on the stack.
	 */
	
		System.out.println("Running testImprovedLinkedStackByTryingToPopMoreThanSizeElements.");
		
		ImprovedLinkedStack<String> myStack = new ImprovedLinkedStack<String>();
		
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
	 * testPopTopWhenStackHasTwoElements tests ImprovedLinkedStack by correctly displaying an element that was
	 * on top of a stack before it was popped off the stack, and the stack after the pop, when the stack had two
	 * elements.
	 */
	
		System.out.println("Running testPopTopWhenStackHasTwoElements.");
		
		ImprovedLinkedStack<String> myStack = new ImprovedLinkedStack<String>();
		
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
	 * testPopTopWhenStackHasZeroElements tests ImprovedLinkedStack by attempting to display an element that was
	 * on top of a stack before it was popped off the stack, and the stack after the pop, when the stack was empty.
	 */
	
		System.out.println("Running testPopTopWhenStackHasZeroElements.");
		
		ImprovedLinkedStack<String> myStack = new ImprovedLinkedStack<String>();
		
		try {
			System.out.println(myStack.poptop() + " popped from stack. Stack is now\n" + myStack.toString());
		}
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
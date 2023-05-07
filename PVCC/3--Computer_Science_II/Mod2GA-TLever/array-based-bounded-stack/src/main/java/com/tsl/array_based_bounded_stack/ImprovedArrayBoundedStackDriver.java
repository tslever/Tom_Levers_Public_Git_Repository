package com.tsl.array_based_bounded_stack;


import java.util.Scanner;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-05
*
* Student name:  Tom Lever
* Completion date: 05/28/21
*
*	ImprovedArrayBoundedStackDriver.java: demonstrates the use of the new methods
* 	from ImprovedArrayBoundedStack class
*
* Student tasks: complete tasks specified in the file
*/

public class ImprovedArrayBoundedStackDriver {

	public static void main(String[] args) {
	/**
	 * main is the entry point of the program. The program requests that a user enter fruit names, followed by "end".
	 * The program prints the fruit names on the stack, the size of the stack, the remaining names on the stack when
	 * three names are popped from the stack, the names on the stack when the top two names on the stack are swapped,
	 * and the top name that was on the stack before that name was popped from the stack. The program throws exceptions
	 * if the program is terminated while input is being requested from the user, a push of a fruit name onto the stack
	 * is requested when the stack is full, a pop of n names from the stack is requested when the number of names on the
	 * stack is less than n, and when a classic pop from the stack is requested when the stack is empty.
	 */

		ImprovedArrayBoundedStack<String> myStack;

		myStack = new ImprovedArrayBoundedStack<String>(10);

		// prepare the Scanner object to enter data from the user
		Scanner input = new Scanner(System.in);
		String word="";

		// read words in a loop,

		while(!word.equalsIgnoreCase("end"))
		{
			System.out.print("Enter a fruit, or type end, if you want to stop: ");
			word=input.next();
			if(!word.equalsIgnoreCase("end"))
				myStack.push(word);
		}
		
		input.close();

		//*** Task #1: test method toString()
		System.out.println("The stack contains:\n" + myStack);

		//*** Task #2: test method toString()
		int stackSize=myStack.size();
		System.out.println("The size of the stack is: " + stackSize);

		//*** Task #3: test method popSome(int count)
		//* pop three elements from the top of the stack
		//* display the resulting stack
		myStack.popSome(3);
		System.out.println("Remaining elements are: ");
		System.out.println(myStack);

		//*** Task #4: test method swapStart()
		//* and display the resulting stack
		myStack.swapStart();
		System.out.println("The stack after reversing the order of the top two elements is: ");
		System.out.println(myStack);

		//*** Task #5: test method poptop()
		String topFruit=myStack.poptop();
		System.out.println("Top element of the stack is: "+topFruit);


	}
}
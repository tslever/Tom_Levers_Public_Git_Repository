package com.tsl.library_helper;


import java.util.Scanner;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-05
*
* Student name:  Tom Lever
* Completion date: 05/27/21
*
* LibraryHelperDriver.java: demonstrates the use of LibraryHelper class
*
* Creates and pushes elements of type String in a stack,
* Retrieves the elements, and counts the ones that have code starting with R or F
*
* Student tasks: complete tasks specified in the file
*/

public class LibraryHelperDriver {

	
	public static void main(String[] args) {
	/**
	 * main is the entry point of the program. The program requests that a user enter books codes, followed by "end". The
	 * program prints the book codes, the number of book codes corresponding to reference books, and the number of book
	 * codes corresponding to fiction books. The program throws exceptions if the program is terminated while input is
	 * being requested from the user, a push of a book code onto the stack is requested when the stack is full, a look
	 * at the top book code on the stack is requested when the stack is empty, and when a pop off the stack is requested
	 * when the stack is empty.
	 * @param args
	 */
		
		//*** Task #1: define a libraryStack variable of type StackInterface with element type String
		StackInterface<String> libraryStack;

		//*** Task #2: instantiate the libraryStack of LibraryHelper type with 10 elements cap
		libraryStack = new LibraryHelper<String>(10);

		// prepare the Scanner object to enter data from the user
		Scanner input = new Scanner(System.in);
		String bookCode="";

		//*** Task #3: create a loop in which you read book codes until user enters "end"

		while(!bookCode.equalsIgnoreCase("end")) {
			System.out.print("Enter the book code, or type end, if you want to stop: ");
			
			bookCode=input.next(); // throws NoSuchElementException, IllegalStateException
			
			if(!bookCode.equalsIgnoreCase("end")) {
				libraryStack.push(bookCode);
			}
		}
		
		input.close();
		
		System.out.println("Retrieve the books, identify type, and put them in shelves.");

		//*** Task #4: define and initialize variables to count reference and fiction books
		int refNo=0;
		int fictionNo=0;

		//*** Task #5: pop the elements off the stack
		//			   count the elements that start with 'R' or 'r' and elements that start with 'F' or 'f'
		// 			   if the code starts with other letter, display a message announcing that the code is not classified

		while(!libraryStack.isEmpty())
		{
			bookCode=libraryStack.top();
			libraryStack.pop();
			System.out.println("Book code is: " + bookCode);
			if(bookCode.charAt(0)=='R' ||bookCode.charAt(0)=='r') {
				refNo++;
			}
			else {
				if(bookCode.charAt(0)=='F' ||bookCode.charAt(0)=='f') {
					fictionNo++;
				}
				else {
					System.out.println("Book with code " + bookCode + " is not classified!");
				}
			}

		}
		//*** Task #6: display the number of reference book, and the number of fiction books

		System.out.println("We received " + refNo + " reference books!");
		System.out.println("We received " + fictionNo + " fiction books!");

	}
	
}
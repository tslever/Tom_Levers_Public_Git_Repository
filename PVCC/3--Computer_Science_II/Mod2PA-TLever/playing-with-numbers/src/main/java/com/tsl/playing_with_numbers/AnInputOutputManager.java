package com.tsl.playing_with_numbers;


import java.util.InputMismatchException;
import java.util.Scanner;


/**
 * AnInputOutputManager represents the structure for an input output manager with a method to ask about and read a
 * a user's age.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 */
class AnInputOutputManager {

	
	/**
	 * scanner is a component of AnInputOutputManager.
	 */
	private static Scanner scanner;
	

	/**
	 * askAboutAndReadAUsersAge prompts a user to propose a user's age.
	 * @return
	 */
	protected static int askAboutAndReadAUsersAge() {
	
		while (true) {
		
			System.out.print("Enter your age in years: ");
			
			scanner = new Scanner(System.in);
			
			int theProposedUsersAge;
			try {
				theProposedUsersAge = scanner.nextInt();
			}
			catch (InputMismatchException theInputMismatchException) {
				System.out.println("Input mismatch exeption.");
				continue;
			}
			
			if ((theProposedUsersAge < 0) || (theProposedUsersAge > 100)) {
				System.out.println("Exception: The proposed user's age is outside of the interval 0 to 100.");
				continue;
			}
			
			return theProposedUsersAge;
			
		}
		
	}
	
}
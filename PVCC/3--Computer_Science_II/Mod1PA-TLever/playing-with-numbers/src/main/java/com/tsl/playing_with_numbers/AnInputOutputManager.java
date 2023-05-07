package com.tsl.playing_with_numbers;


import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Scanner;
import org.apache.commons.lang3.StringUtils;


/**
 * AnInputOutputManager represents the structure for an input output manager with a method to ask about and read
 * a time and a name.
 * @author Tom Lever
 * @version 1.0
 * @since 05/22/21
 */
class AnInputOutputManager {

	
	/**
	 * scanner is a component of AnInputOutputManager.
	 */
	private Scanner scanner;
	

	/**
	 * askAboutAndReadATimeAndAName prompts a user to propose a time and a name.
	 * @return
	 */
	public AnIntegerAndAString askAboutAndReadATimeAndAName()
		throws InputMismatchException, NoSuchElementException, AnInvalidTimeException, AnInvalidNameException {
		
		System.out.print("Enter a time for an office-hours appointment, between 1 and 6 inclusive: ");
		
		this.scanner = new Scanner(System.in);
		
		int theProposedTime = this.scanner.nextInt();
		
		if ((theProposedTime < 1) || (theProposedTime > 6)) {
			throw new AnInvalidTimeException("The proposed time is outside of the interval 1 to 6.");
		}
		
		System.out.print("Enter your name: ");
		
		this.scanner = new Scanner(System.in);
		
		String theProposedName = this.scanner.nextLine();
		
		if (theProposedName.equals("")) {
			throw new AnInvalidNameException("Invalid name: The proposed name is an empty string.");
		}
		
		if (!StringUtils.isAlphaSpace(theProposedName)) {
			throw new AnInvalidNameException("Invalid name: Only Unicode letters and spaces are allowed.");
		}
		
		return new AnIntegerAndAString(theProposedTime, theProposedName);
		
	}
	
}
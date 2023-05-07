package com.tsl.playing_with_numbers;


import java.util.Scanner;
import org.apache.commons.lang3.StringUtils;
import org.junit.jupiter.api.Test;


/**
 * AskAboutAndReadATimeAndANameTest encapsulates JUnit tests of core functionality of the method
 * askAboutAndReadATimeAndAName of class AnInputOutputManager.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
class AskAboutAndReadATimeAndANameTest {

	
	/**
	 * testAskAboutAndReadATimeAndANameForAValidName tests askAboutAndReadATimeAndAName by displaying a valid proposed
	 * name.
	 */
	@Test
	public void testAskAboutAndReadATimeAndANameForANameForAValidName() {
		
		System.out.println("Running testAskAboutAndReadATimeAndANameForAValidName.");
		
		String theProposedName = "Tom Lever";
		
		try {
		
			if (theProposedName.equals("") || !StringUtils.isAlphaSpace(theProposedName)) {
				throw new AnInvalidNameException("Invalid name: Only Unicode letters and spaces are allowed.");
			}
			
			System.out.println("The proposed name: " + theProposedName);
		
		}
		
		catch (AnInvalidNameException theInvalidNameException) {
			
			System.out.println(theInvalidNameException.getMessage());
			
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testAskAboutAndReadATimeAndANameForANameThatIsAnEmptyString tests
	 * askAboutAndReadATimeAndAName by displaying the message of an invalid-name exception that occurs when a proposed
	 * name is an empty string.
	 */
	@Test
	public void testAskAboutAndReadATimeAndANameForANameThatIsAnEmptyString() {
		
		System.out.println("Running testAskAboutAndReadATimeAndANameForANameThatIsAnEmptyString.");
		
		String theProposedName = "";
		
		try {
		
			if (theProposedName.equals("")) {
				throw new AnInvalidNameException("Invalid name: The proposed name is an empty string.");
			}
			
			System.out.println("The proposed name: " + theProposedName);
		
		}
		
		catch (AnInvalidNameException theInvalidNameException) {
			
			System.out.println(theInvalidNameException.getMessage());
			
		}
		
		System.out.println();
		
	}

	
	/**
	 * testAskAboutAndReadATimeAndANameForANameThatContainsAHyphen tests
	 * askAboutAndReadATimeAndAName by displaying the message of an invalid-name exception that occurs when a proposed
	 * name contains a hyphen.
	 */
	@Test
	public void testAskAboutAndReadATimeAndANameForANameThatContainsAHyphen() {
		
		System.out.println("Running testAskAboutAndReadATimeAndANameForANameThatContainsAHyphen");
		
		String theProposedName = "Tom-Lever";
		
		try {
		
			if (!StringUtils.isAlphaSpace(theProposedName)) {
				throw new AnInvalidNameException("Invalid name: Only Unicode letters and spaces are allowed.");
			}
			
			System.out.println("The proposed name: " + theProposedName);
		
		}
		
		catch (AnInvalidNameException theInvalidNameException) {
			
			System.out.println(theInvalidNameException.getMessage());
			
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testAskAboutAndReadATimeAndANameForANameThatContainsANonAlphaSpace tests
	 * askAboutAndReadATimeAndAName by displaying the message of an invalid-name exception that occurs when a proposed
	 * name contains a character that is neither alphabetic nor a space.
	 */
	@Test
	public void testAskAboutAndReadATimeAndANameForANameThatContainsANonAlphaSpace() {
		
		System.out.println("Running testAskAboutAndReadATimeAndANameForANameThatContainsANonAlphaSpace");
		
		String theProposedName = "Tom Lever!";
		
		try {
		
			if (!StringUtils.isAlphaSpace(theProposedName)) {
				throw new AnInvalidNameException("Invalid name: Only Unicode letters and spaces are allowed.");
			}
			
			System.out.println("The proposed name: " + theProposedName);
		
		}
		
		catch (AnInvalidNameException theInvalidNameException) {
			
			System.out.println(theInvalidNameException.getMessage());
			
		}
		
		System.out.println();
		
	}
	
	
}

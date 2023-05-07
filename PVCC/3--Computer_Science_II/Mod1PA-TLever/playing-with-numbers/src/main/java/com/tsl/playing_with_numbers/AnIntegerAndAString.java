package com.tsl.playing_with_numbers;


/**
 * AnIntegerAndAString provides a structure for objects that contain an integer and a string.
 * @author Tom
 *
 */
class AnIntegerAndAString {

	
	/**
	 * theInteger is a component of AnIntegerAndAString.
	 */
	private int theInteger;

	
	/**
	 * theString is a component of AnIntegerAndAString.
	 */
	private String theString;
	
	
	/**
	 * AnIntegerAndAString(int theIntegerToUse, String theStringToUse) is the zero-argument constructor for
	 * AnIntegerAndAString and sets theInteger to theIntegerToUse and theString to theStringToUse.
	 * @param theIntegerToUse
	 * @param theStringToUse
	 */
	public AnIntegerAndAString(int theIntegerToUse, String theStringToUse) {
		this.theInteger = theIntegerToUse;
		this.theString = theStringToUse;
	}
	
	
	/**
	 * getTheInteger provides the integer.
	 * @return
	 */
	public int getTheInteger() {
		return this.theInteger;
	}
	
	
	/**
	 * getTheString provides the string.
	 * @return
	 */
	public String getTheString() {
		return this.theString;
	}
	
}

package com.tsl.pez_candy;


/**
 * AColor enumerates possible colors for Pez candies, and provides a method for getting a random color from the
 * enumeration.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */
enum AColor {
	
	
	red,
	yellow,
	green,
	pink,
	blue;
	
	
	/**
	 * getARandomColor provides a random color from this enumeration.
	 * @return
	 * @throws AnIntegerOverflowException
	 */
    protected static AColor getARandomColor() throws AnIntegerOverflowException {
    	
    	return AColor.values()[ARandomNumberGenerator.getARandomIntegerInclusivelyBetween(0, AColor.values().length-1)];
    	
    }

}

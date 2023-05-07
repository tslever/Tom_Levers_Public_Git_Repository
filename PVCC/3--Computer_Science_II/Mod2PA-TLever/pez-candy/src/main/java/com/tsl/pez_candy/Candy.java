package com.tsl.pez_candy;


/**
 * Candy represents a structure for candies.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */
public class Candy {
	
	
	/**
	 * color is an attribute of Candy.
	 */
	private AColor color;
	
	
	/**
	 * Candy(AColor theColorToUse) is the one-parameter constructor for Candy that sets this candy's color to
	 * theColorToUse.
	 * @param theColorToUse
	 */
	protected Candy(AColor theColorToUse) {
		
		this.color = theColorToUse;
		
	}
	
	
	/**
	 * getsItsColor returns this candy's color.
	 * @return
	 */
	protected AColor getsItsColor() {

		return this.color;
		
	}

}

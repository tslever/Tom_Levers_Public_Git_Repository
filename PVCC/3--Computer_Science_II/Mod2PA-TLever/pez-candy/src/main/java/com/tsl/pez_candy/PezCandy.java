package com.tsl.pez_candy;


/**
 * PezCandy represents a structure for Pez-candy dispensers.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */
class PezCandy {

	
	/**
	 * THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER is an attribute of a Pez-candy dispenser.
	 */
	protected final int THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER = 12;
	
	
	/**
	 * description is an attribute of a Pez-candy dispenser.
	 */
	protected String description;

	
	/**
	 * candies is a component of a Pez-candy dispenser.
	 */
	protected Candy[] candies;
	
	/**
	 * indexOfTheCandyOnTopOfTheStack is a component of a Pez-candy dispenser.
	 */
	protected int indexOfTheCandyOnTopOfTheStack;
	
	
	/**
	 * PezCandy(String theDescriptionToUse) is the one-parameter constructor for PezCandy that sets this dispenser's
	 * description to theDescriptionToUse, candies to a new array of candies with length
	 * THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER, and indexOfTheCandyOnTopOfTheStack to -1.
	 * @param theDescriptionToUse
	 */
	protected PezCandy(String theDescriptionToUse) {
		
		this.description = theDescriptionToUse;
		this.candies = new Candy[THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER];
		this.indexOfTheCandyOnTopOfTheStack = -1;
		
	}
	
	
	/**
	 * getsItsDescription provides this dispenser's description.
	 * @return
	 */
	protected String getsItsDescription() {
		return this.description;
	}
	
	
	/**
	 * isEmpty indicates whether or not this dispenser is empty.
	 * @return
	 */
	protected boolean isEmpty() {
		
		if (this.indexOfTheCandyOnTopOfTheStack == -1) {
			return true;
		}
		
		return false;
		
	}
	
	
	/**
	 * isFull indicates whether or not this dispenser is full.
	 * @return
	 */
	// TODO: Make private before deployment.
	protected boolean isFull() {
		
		if (this.indexOfTheCandyOnTopOfTheStack == this.candies.length - 1) {
			return true;
		}
		
		return false;
		
	}
	
	
	/**
	 * receives adds a candy to this dispenser's array of candies, or throws a stack-overflow exception if this
	 * dispenser is full.
	 * @param theCandy
	 * @throws AStackOverflowException
	 */
	protected void receives(Candy theCandy) throws AStackOverflowException {
		
		if (isFull()) {
			throw new AStackOverflowException(
				"Exception: A stack of type PezCandy received a request for it to receive a candy when the stack was " +
				"full."
			);
		}
		
		this.indexOfTheCandyOnTopOfTheStack++;
		
		this.candies[this.indexOfTheCandyOnTopOfTheStack] = theCandy;
		
	}
	
	
	/**
	 * retrievesAndRemovesItsTopCandy retrieves its top candy, removes the candy from its top, and provides its
	 * retrieved candy, or throws a stack-underflow exception if this dispenser is empty.
	 * @return
	 * @throws AStackUnderflowException
	 */
	protected Candy retrievesAndRemovesItsTopCandy() throws AStackUnderflowException {
		
		if (isEmpty()) {
			throw new AStackUnderflowException(
				"Exception: A stack of type PezCandy received a request for it to retrieve and remove its top candy " +
				"when the stack was empty."
			);
		}
		
		Candy theCandyToRetrieve = this.candies[this.indexOfTheCandyOnTopOfTheStack];
		
		this.candies[this.indexOfTheCandyOnTopOfTheStack] = null;
		
		this.indexOfTheCandyOnTopOfTheStack--;
		
		return theCandyToRetrieve;
		
	}
	
	
	/**
	 * toString represents this dispenser as a indented column of strings representing the colors of its candies.
	 * @return
	 */
	@Override
	public String toString() {
		
		String theStringRepresentingTheStack = "";
		
		for (int i = this.indexOfTheCandyOnTopOfTheStack; i > 0; i--) {
			theStringRepresentingTheStack += "\t" + this.candies[i].getsItsColor() + "\n";
		}
		if (this.indexOfTheCandyOnTopOfTheStack > -1) {
			theStringRepresentingTheStack += "\t" + this.candies[0].getsItsColor();
		}
		
		return theStringRepresentingTheStack;
		
	}
	
	
}

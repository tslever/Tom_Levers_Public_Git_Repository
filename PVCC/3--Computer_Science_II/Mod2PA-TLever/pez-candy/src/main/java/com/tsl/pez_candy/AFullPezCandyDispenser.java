package com.tsl.pez_candy;


/**
 * AFullPezCandyDispenser represents a structure for full Pez-candy dispensers.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */
class AFullPezCandyDispenser extends PezCandy {
	
	
	/**
	 * AFullPezCandyDispenser(String theDescription) is the one-parameter constructor for AFullPezCandyDispenser that
	 * calls the one-parameter constructor of PezCandy and fills the full Pez-candy dispenser.
	 * @param theDescription
	 * @throws AnIntegerOverflowException
	 */
	public AFullPezCandyDispenser(String theDescription) throws AnIntegerOverflowException {
		
		super(theDescription);
		fillTheDispenser();
		
	}
	
	
	/**
	 * fillTheDispenser fills this dispenser with candies.
	 * @throws AnIntegerOverflowException
	 */
	private void fillTheDispenser() throws AnIntegerOverflowException {
		
		for (int i = 0; i < THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER; i++) {
			
			this.indexOfTheCandyOnTopOfTheStack++;
			this.candies[indexOfTheCandyOnTopOfTheStack] = new Candy(AColor.getARandomColor());
			
		}
		
	}
	
	
}

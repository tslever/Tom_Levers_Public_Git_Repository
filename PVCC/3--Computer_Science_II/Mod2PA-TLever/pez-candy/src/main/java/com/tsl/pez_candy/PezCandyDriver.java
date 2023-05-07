package com.tsl.pez_candy;


/**
 * PezCandyDriver encapsulates the entry point to this program that simulates the behavior of a child playing with a
 * Pez-candy dispenser full of twelve red, yellow, green, blue, and pink candies and an empty Pez-candy dispenser. The
 * child removes each candy from the former dispenser and eats the candy if it is red or examines it and stores it in
 * the latter dispenser if it is not red. The child then returns all remaining candies to the former dispenser in their
 * original order.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */

public class PezCandyDriver {	
	
	
	/**
	 * main is the entry point to this program that simulates a child playing with two Pez-candy dispensers. main:
	 * 1. Creates a full Pez-candy dispenser and an empty Pez-candy dispenser.
	 * 2. Outputs descriptions of the states of the created dispensers.
	 * 3. While the former dispenser contains one or more candies:
	 *    a. Retrieves and removes from the former dispenser its top candy, as the candy to eat or examine and store;
	 *       and
	 *    b. "Eats" the candy if it is red, or "examines" and stores it in the latter dispenser if it is not red.
	 * 4. Outputs descriptions of the states of the dispensers.
	 * 5. While the latter dispenser contains one or more candies, transfers each candy back to the former dispenser.
	 * 6. Outputs descriptions of the states of the dispensers.
	 * 
	 * @param args
	 * @throws AnIntegerOverflowException
	 * @throws AStackUnderflowException
	 * @throws AStackOverflowException
	 */
	
    public static void main( String[] args )
    	throws AnIntegerOverflowException, AStackUnderflowException, AStackOverflowException {
    	
    	
    	System.out.println(
    		"------------------------------------\n" +
    		"ON PLAYING WITH PEZ-CANDY DISPENSERS\n" +
    		"------------------------------------"
    	);
    	
        AFullPezCandyDispenser todaysPezCandyDispenser = new AFullPezCandyDispenser("Today's Pez-candy dispenser");
        
        PezCandy theSparePezCandyDispenser = new PezCandy("The spare Pez-candy dispenser");
        
        outputDescriptionsOfTheStatesOf(todaysPezCandyDispenser, theSparePezCandyDispenser);
        
        
    	System.out.println(
    		"--------------------------------------------------------------\n" +
    		"ON EATING ALL RED CANDIES AND EXAMINING AND STORING THE OTHERS\n" +
    		"--------------------------------------------------------------"
    	);
        
        Candy theCandyToEatOrExamineAndStore;
        while (!todaysPezCandyDispenser.isEmpty()) {
        
            theCandyToEatOrExamineAndStore = todaysPezCandyDispenser.retrievesAndRemovesItsTopCandy();
            
            if (theCandyToEatOrExamineAndStore.getsItsColor() == AColor.red) {
            	eat(theCandyToEatOrExamineAndStore);
            }
            else {
            	examine(theCandyToEatOrExamineAndStore);
            	theSparePezCandyDispenser.receives(theCandyToEatOrExamineAndStore);
            }
        	
        }
        System.out.println();
        
        outputDescriptionsOfTheStatesOf(todaysPezCandyDispenser, theSparePezCandyDispenser);
        
        
    	System.out.println(
    		"---------------------------------------------------\n" +
    		"ON RETURNING REMAINING CANDIES TO TODAY'S DISPENSER\n" +
    		"---------------------------------------------------"
    	);
        
        while (!theSparePezCandyDispenser.isEmpty()) {
        	
        	todaysPezCandyDispenser.receives(theSparePezCandyDispenser.retrievesAndRemovesItsTopCandy());
        	
        }
        
        outputDescriptionsOfTheStatesOf(todaysPezCandyDispenser, theSparePezCandyDispenser);
        
    }
    

    /**
     * eat represents a child declaring that she is eating a candy.
     * @param theCandy
     */
    private static void eat(Candy theCandy) {
    	
    	System.out.println("I am eating a " + theCandy.getsItsColor() + " candy!");
    	
    }
    
    
    /**
     * examine represents a child declaring she is examining a candy.
     * @param theCandy
     */
    private static void examine(Candy theCandy) {
    	
    	System.out.println("I am examining and about to store a " + theCandy.getsItsColor() + " candy!");
    	
    }
    
    
    /**
     * outputADescriptionOfTheStateOf outputs a description of the state of a Pez-candy dispenser.
     * @param thePezCandyDispenser
     */
    private static void outputADescriptionOfTheStateOf(PezCandy thePezCandyDispenser) {
    	
        System.out.println(
        	thePezCandyDispenser.getsItsDescription() +
        	" contains candies, from top to bottom, of the following colors: {\n" +
        	thePezCandyDispenser + "\n" +
        	"}\n"
        );
    	
    }
    
    
    /**
     * outputDescriptionsOfTheStatesOf outputs a description of the states of two Pez-candy dispensers.
     * @param theFirstPezCandyDispenser
     * @param theSecondPezCandyDispenser
     */
    private static void outputDescriptionsOfTheStatesOf(
    	PezCandy theFirstPezCandyDispenser, PezCandy theSecondPezCandyDispenser) {
    	
    	outputADescriptionOfTheStateOf(theFirstPezCandyDispenser);
    	
    	outputADescriptionOfTheStateOf(theSecondPezCandyDispenser);
    	
    }
    
    
}

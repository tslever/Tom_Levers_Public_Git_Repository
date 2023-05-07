package Com.TSL.UtilitiesForGeneratingAHistogramOfDiceRolls;


import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.random.RandomDataGenerator;


/**
 * LuckyBet encapsulates the entry point of this program, which prints a histogram of frequencies of a die face facing
 * up on a die roll versus the value of the die face.
 *
 */

public class LuckyBet 
{
	
	private static HashMap<Integer, Integer> theHashMapRepresentingAHistogramOfDieRolls;
	
	
	/**
	 * main is the entry point of this program, which prints a histogram of frequencies of a die face facing
	 * up on a die roll versus the value of the die face.
	 */
	
    public static void main( String[] args )
    {
    	final int THE_SMALLEST_DIE_VALUE = 1;
    	final int THE_LARGEST_DIE_VALUE = 6;
    	
        theHashMapRepresentingAHistogramOfDieRolls = new HashMap<Integer, Integer>();
        for (int i = THE_SMALLEST_DIE_VALUE; i <= THE_LARGEST_DIE_VALUE; i++) {
        	theHashMapRepresentingAHistogramOfDieRolls.put(i, 0);
        }
        
        RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
        for (int i = 0; i < 100; i++) {
        	theHashMapRepresentingAHistogramOfDieRolls.merge(
        		theRandomDataGenerator.nextInt(THE_SMALLEST_DIE_VALUE, THE_LARGEST_DIE_VALUE), 1, Integer::sum
        	);
        }
        
        printHashMap();
        
    }
    
    
    /**
     * printHashMap outputs a representation of the histogram 
     */
    
    private static void printHashMap() {
    	
    	System.out.println("The histogram contains:");
    	for (Map.Entry<Integer, Integer> theMapEntry : theHashMapRepresentingAHistogramOfDieRolls.entrySet()) {
    		System.out.println("The number " + theMapEntry.getKey() + " occurs " + theMapEntry.getValue() + " times.");
    	}
    	
    }
    
}

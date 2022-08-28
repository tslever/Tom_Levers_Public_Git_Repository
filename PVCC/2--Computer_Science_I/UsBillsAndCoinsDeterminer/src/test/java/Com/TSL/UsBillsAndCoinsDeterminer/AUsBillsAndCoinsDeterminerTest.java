package Com.TSL.UsBillsAndCoinsDeterminer;


import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.util.Precision;
import org.junit.jupiter.api.Test;


/** *******************************************************************************************
 * UsBillsAndCoinsDeterminerTest encapsulates a method to test the main method of this program.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/01/21
 ******************************************************************************************* */

public class AUsBillsAndCoinsDeterminerTest {

    
    /** ------------------------------------------------------------------------------------------------------------
     * testMain tests whether a calculated United-States monetary amount is not equal to a provided amount exception
     * ever occurs within the scope of method main.
     ------------------------------------------------------------------------------------------------------------ */
    
    @Test
    public void testMain()
    {
        
        int theRandomNumberOfDollars;
        int theRandomNumberOfCents;
        
        double theRandomUsMonetaryAmount;
        
        
        RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
        
        while (true)
        {
        
            try
            {
                theRandomNumberOfDollars = theRandomDataGenerator.nextInt (0, Integer.MAX_VALUE);
                
                theRandomNumberOfCents = theRandomDataGenerator.nextInt (0, 100);
                
                theRandomUsMonetaryAmount =
                    Precision.round ( (double)theRandomNumberOfDollars + (double)theRandomNumberOfCents / 100.0, 2);
                
                new AnAccountOfUsBillsAndCoins (theRandomUsMonetaryAmount);
            }
            
            catch (ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException e)
            {
                System.out.println (e.getMessage ());
            }
        
        }
        
    }
    
}

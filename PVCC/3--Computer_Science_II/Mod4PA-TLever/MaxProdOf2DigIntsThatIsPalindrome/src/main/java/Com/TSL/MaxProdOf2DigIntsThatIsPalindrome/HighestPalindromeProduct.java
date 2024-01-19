package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * HighestPalindromeProduct encapsulates the entry point of this program, which displays information about the highest
 * product of two-digit integers that is a palindrome.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 */

public class HighestPalindromeProduct 
{
	
	/**
	 * main is the entry point of this program which displays information about the highest product of two-digit integers
	 * that is a palindrome.
	 * 
	 * @param args
	 */
	
    public static void main( String[] args )
    {
    	
    	APalindromicityTester thePalindromicityTester = new APalindromicityTester();
    	
    	int j;
    	int theProduct;
    	boolean aProductWasAPalindrome = false;
    	
    	for (int i = 99; i >= 10; i--) {
    		
    		if (aProductWasAPalindrome) {
    			break;
    		}
    		
    		for (j = 99; j >= 10; j--) {
    			
    			theProduct = i * j;
    			
    			if (thePalindromicityTester.tests(theProduct)) {
    				
    				System.out.println(
    					i + " * " + j + " = " + theProduct + " is the maximum product of two digit numbers that is a " +
    					"palindrome."
    				);
    				
    				aProductWasAPalindrome = true;
    				
    				break;
    				
    			}
    			
    		}
    		
    	}
    	
    	if (!aProductWasAPalindrome) {
    		System.out.println("No product of two-digit integers was a palindrome.");
    	}
        
    }
    
}
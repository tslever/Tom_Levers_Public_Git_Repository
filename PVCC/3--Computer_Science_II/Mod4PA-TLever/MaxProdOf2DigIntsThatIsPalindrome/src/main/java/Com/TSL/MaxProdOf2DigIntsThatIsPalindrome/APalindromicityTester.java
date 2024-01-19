package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * APalindromicityTester represents the structure for a palindromicity tester, which tests whether an integer is a
 * palindrome.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 */

class APalindromicityTester {
	
	
	/**
	 * tests evaluates whether an integer is a palindrome.
	 * 
	 * @param theProduct
	 * @return
	 */
	
	protected boolean tests(int theProduct) {
		
		NumericPalindrome<Integer> theQueueOfDigits = getTheQueueOfDigitsIn(theProduct);
    	
		AnArrayBasedBoundedStack<Integer> theStackOfDigits =
			new AnArrayBasedBoundedStack<Integer>(theQueueOfDigits.providesItsNumberOfElements());
		
		int theNumberOfDigits = theQueueOfDigits.providesItsNumberOfElements();
    	for (int i = 0; i < theNumberOfDigits; i++) {
    		
    		int thePresentDigit = theQueueOfDigits.dequeue();
    		theQueueOfDigits.enqueue(thePresentDigit);
    		
    		theStackOfDigits.push(thePresentDigit);
    		
    	}
    	
    	if (isPalindromeTheIntegerRepresentedBy(theQueueOfDigits, theStackOfDigits)) {
    		return true;
    	}
    	
    	return false;
		
	}
	
	
	/**
	 * findTheNumberOfDigitsIn finds the number of digits in an integer.
	 * 
	 * @param theProduct
	 * @return
	 */
	
    private int findTheNumberOfDigitsIn(int theProduct) {
    	
        int theConsumedVersionOfTheProduct = theProduct;
    	
        int theNumberOfDigits = 0;
        
        while (theConsumedVersionOfTheProduct > 0) {
        	
        	theNumberOfDigits++;
        	
        	theConsumedVersionOfTheProduct /= 10;
        	
        }
        
        return theNumberOfDigits;
    	
    }
	
    
    /**
     * getTheQueueOfDigitsIn provides a queue of the digits in an integer.
     * 
     * @param theProduct
     * @return
     */
    
    private NumericPalindrome<Integer> getTheQueueOfDigitsIn(int theProduct) {
    
    	int theConsumedVersionOfTheProduct = theProduct;
    	
    	int theNumberOfDigits = findTheNumberOfDigitsIn(theProduct);
    	
    	NumericPalindrome<Integer> theQueueOfDigits = new NumericPalindrome<Integer>(theNumberOfDigits);
    	
    	for (int i = theNumberOfDigits - 1; i >= 0; i--) {
    		
    		theQueueOfDigits.enqueue(theConsumedVersionOfTheProduct % 10);
    		
    		theConsumedVersionOfTheProduct /= 10;
    		
    	}
    	
    	return theQueueOfDigits;
    	
    }
    
    
    /**
     * isPalindromeTheIntegerRepresentedBy indicates whether or not an integer represented by a queue and a stack is a
     * palindrome.
     * 
     * @param theQueueOfDigits
     * @param theStackOfDigits
     * @return
     */
    
    private boolean isPalindromeTheIntegerRepresentedBy(
    	NumericPalindrome<Integer> theQueueOfDigits, AnArrayBasedBoundedStack<Integer> theStackOfDigits) {
    	
    	boolean theIntegerIsAPalindrome = true;
    	
    	int theIntegerAtTheTopOfTheStack;
    	
    	int theNumberOfDigits = theQueueOfDigits.providesItsNumberOfElements();
    	
    	for (int i = 0; i < theNumberOfDigits; i++) {
    		
    		theIntegerAtTheTopOfTheStack = theStackOfDigits.top();
    		theStackOfDigits.pop();
    		
    		if (theQueueOfDigits.dequeue() != theIntegerAtTheTopOfTheStack) {
    			
    			theIntegerIsAPalindrome = false;
    			
    		}
    		
    	}
    	
    	return theIntegerIsAPalindrome;
    	
    }
	
}

package Com.TSL.StockTransactionUtilities;


/**
 * StockTransactionHistory encapsulates the entry point of this program, which creates a queue for individual stocks
 * buys and sells stocks in bulk, tracks a running capital gain, and displays that capital gain.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 */

public class StockTransactionHistory 
{
	
	/**
	 * main is the entry point of this program, which creates a queue for individual stocks, buys and sells stocks in
	 * bulk, tracks a running capital gain, and displays that capital gain.
	 * @param args
	 */
	
    public static void main( String[] args )
    {
    	
    	StockTransaction theQueueOfStocks = new StockTransaction();
    	
    	double capitalGain = 0.0;
    	
    	
    	theQueueOfStocks.buy(120, 25.00);
    	
    	theQueueOfStocks.buy(20, 75.00);
    	
    	capitalGain += theQueueOfStocks.sell(30, 65.00);
    	
    	capitalGain += theQueueOfStocks.sell(10, 65.00);
    	
    	theQueueOfStocks.buy(100, 10.00);
    	
    	theQueueOfStocks.buy(130, 4.00);
    	
    	theQueueOfStocks.buy(200, 16.00);
    	
    	capitalGain += theQueueOfStocks.sell(10, 65.00);
    	
    	capitalGain += theQueueOfStocks.sell(150, 30.00);
    	
    	System.out.println("Capital gain over stock transaction history is $" + capitalGain + ".");

    }
    
}
package Com.TSL.StockTransactionUtilities;


/**
 * StockTransaction represents the structure for a linked list based queue of stocks.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 */

public class StockTransaction {

	
	private ALinkedListNode<Stock> frontLinkedListNode;
	private ALinkedListNode<Stock> rearLinkedListNode;
	private int numberOfElements;
	
	
	/**
	 * StockTransaction is the zero-parameter constructor for StockTransaction, which sets the references to the
	 * front linked list node and the rear linked list node to null.
	 */
	
	public StockTransaction() {
		
		this.frontLinkedListNode = null;
		this.rearLinkedListNode = null;
		this.numberOfElements = 0;
		
	}
	
	
	/**
	 * buyOneShareOf enqueues one share of a stock in this queue of stocks.
	 * 
	 * @param theStock
	 */
	
	public void buyOneShareOf(Stock theStock) {
		
		ALinkedListNode<Stock> theLinkedListNodeForTheStock = new ALinkedListNode<Stock>(theStock);
		
		if (this.rearLinkedListNode == null) {
			this.frontLinkedListNode = theLinkedListNodeForTheStock;
		}
		
		else {
			this.rearLinkedListNode.setsItsReferenceTo(theLinkedListNodeForTheStock);
		}
		
		this.rearLinkedListNode = theLinkedListNodeForTheStock;
		this.numberOfElements++;
		
	}
	
	
	/**
	 * buy purchases <theNumberOfSharesToBuy> shares of a stock at the given purchase price, and outputs a completion
	 * message.
	 * 
	 * @param theNumberOfSharesToBuy
	 * @param thePurchasePrice
	 */
	
	public void buy(int theNumberOfSharesToBuy, double thePurchasePrice) {
		
    	for (int i = 0; i < theNumberOfSharesToBuy; i++) {
    		buyOneShareOf(new Stock(thePurchasePrice));
    	}
    	
    	System.out.println("Buying " + theNumberOfSharesToBuy + " shares at $" + thePurchasePrice + " each.\n");
		
	}
	
	
	/**
	 * sellOneShare dequeues one share from this queue of stocks.
	 * 
	 * @return
	 */
	
	public Stock sellsOneShare() {
		
		if (isEmpty()) {
			throw new AQueueUnderflowException("Exception: sell for an empty queue of stocks was requested.");
		}
		
		Stock theStockToSell = this.frontLinkedListNode.providesItsElement();
		this.frontLinkedListNode = this.frontLinkedListNode.providesItsReference();
		
		if (this.frontLinkedListNode == null) {
			this.rearLinkedListNode = null;
		}
		
		this.numberOfElements--;
		
		return theStockToSell;
		
	}
	
	
	/**
	 * sell sells <theNumberOfSharesToSell> shares of a stock at the given sale price, and outputs a completion
	 * message.
	 * 
	 * @param theNumberOfStocksToSell
	 * @param theSalePrice
	 * @return
	 */
	
	public double sell(int theNumberOfStocksToSell, double theSalePrice) {
		
    	double theIncomeFromSellingMultipleStocks = 0.0;
    	double theTotalCostOfTheStocks = 0.0;
    	
    	for (int i = 0; i < theNumberOfStocksToSell; i++) {
    		theIncomeFromSellingMultipleStocks += theSalePrice;
    		theTotalCostOfTheStocks += sellsOneShare().providesItsShareCost();
    	}
    	
    	double theCapitalGain = theIncomeFromSellingMultipleStocks - theTotalCostOfTheStocks; 
    	
    	System.out.println(
    		"Selling " + theNumberOfStocksToSell + " shares.\n" +
    		"Income from selling " + theNumberOfStocksToSell + " shares at $" + theSalePrice + " is $" +
    		theIncomeFromSellingMultipleStocks + ".\n" +
    		"Total cost of shares is $" + theTotalCostOfTheStocks + ".\n" +
    		"Capital gain from selling " + theNumberOfStocksToSell + " shares at $" + theSalePrice + " is $" +
    		theCapitalGain + ".\n"
    	);
    	
    	return theCapitalGain;
		
	}
	
	
	/**
	 * isEmpty indicates whether or not this queue is empty.
	 * 
	 * @return
	 */
	
	public boolean isEmpty() {
		
		return (this.frontLinkedListNode == null);
		
	}
	
	
	/**
	 * toString provides a string representation of this queue.
	 */
	
	@Override
	public String toString() {
		
		String theRepresentationOfThisQueue = "[";
		
		ALinkedListNode<Stock> theCurrentLinkedListNodeInTheQueue = this.frontLinkedListNode;
		while (theCurrentLinkedListNodeInTheQueue.providesItsReference() != null) {
			theRepresentationOfThisQueue += theCurrentLinkedListNodeInTheQueue.providesItsElement() + ", ";
			theCurrentLinkedListNodeInTheQueue = theCurrentLinkedListNodeInTheQueue.providesItsReference();
		}
		
		theRepresentationOfThisQueue += theCurrentLinkedListNodeInTheQueue.providesItsElement() + "]";
		
		return theRepresentationOfThisQueue;
		
	}
	
}

package Com.TSL.StockTransactionUtilities;


/**
 * Stock represents the structure for a stock, which has a share cost.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 */

public class Stock {

	
	private int numberToBuy;
	private double cost;
	
	
	/**
	 * Stock(double theShareCost) is the one-parameter constructor for Stock, which sets this stock's share cost to
	 * the given share cost.
	 * 
	 * @param theShareCost
	 */
	
	public Stock(double theShareCost) {
		
		this.cost = theShareCost;
		
	}
	
	
	/**
	 * setsItsShareCostTo sets the share cost of this stock to the given share cost.
	 * 
	 * @param theShareCost
	 */
	
	public void setsItsShareCostTo(double theShareCost) {
		
		this.cost = theShareCost;
		
	}
	
	
	/**
	 * providesItsShareCost provides the share cost of this stock.
	 * @return
	 */
	
	public double providesItsShareCost() {
		
		return this.cost;
		
	}
	
	
	/**
	 * toString provides a string representation of this stock.
	 */
	
	@Override
	public String toString() {
		
		return Double.toString(this.cost);
		
	}
	
}
